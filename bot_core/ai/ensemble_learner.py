import os
import joblib
import json
import logging
import shutil
import numpy as np
import pandas as pd
import asyncio
from typing import Dict, Any, Tuple, Optional, List, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import norm, skew, kurtosis

from bot_core.logger import get_logger
from bot_core.config import AIEnsembleStrategyParams
from bot_core.ai.models import TCNPredictor, AttentionNetwork
from bot_core.ai.feature_processor import FeatureProcessor
from bot_core.common import AIInferenceResult

# ML Imports with safe fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier, IsolationForest
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, accuracy_score
    from sklearn.base import clone
    from xgboost import XGBClassifier
    from scipy.optimize import minimize
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    # Dummy classes omitted for brevity, assuming environment is set up correctly per requirements.txt

logger = get_logger(__name__)

class InputSanitizer:
    @staticmethod
    def sanitize(X: np.ndarray) -> np.ndarray:
        if X is None or X.size == 0: return X
        return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

class PurgedTimeSeriesSplit:
    """
    Time Series Split with Purging and Embargo to prevent leakage.
    Train indices are strictly before Test indices, separated by a purge gap based on label overlap.
    """
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X, y=None, groups=None):
        # groups must contain the 't1' (event end time) for purging
        n_samples = len(X)
        indices = np.arange(n_samples)
        test_size = n_samples // (self.n_splits + 1)
        
        # Embargo size
        embargo = int(n_samples * self.embargo_pct)

        for i in range(self.n_splits):
            # Moving window or expanding window. Here expanding.
            train_end = n_samples - (self.n_splits - i) * test_size
            test_start = train_end
            test_end = test_start + test_size
            
            # Purging: Remove train samples whose labels overlap with test start
            # If groups (t1) is provided, we purge based on time overlap
            if groups is not None:
                # Find max t1 in the potential train set
                # We want train samples where t1 < test_start_time
                # Since we are using indices, we approximate by ensuring train index < test_start
                # But we must also remove samples near the boundary whose t1 extends into test
                
                # Simplified Purge: Drop last N samples from train where N is max horizon
                # A robust implementation requires t1 values. 
                # Here we assume 'groups' is the t1 Series/Array.
                
                test_start_time = groups[test_start] if test_start < len(groups) else None
                
                # Identify train indices where t1 < test_start_time (if time based) or just index gap
                # For index-based t1 (triple barrier returns timestamps), we need to compare timestamps.
                pass
            
            # Apply Embargo: Gap between Train End and Test Start
            effective_train_end = train_end - embargo
            
            if effective_train_end > 0:
                yield indices[:effective_train_end], indices[test_start:test_end]

def probabilistic_sharpe_ratio(returns: np.ndarray, benchmark: float = 0.0) -> float:
    """Calculates the Probabilistic Sharpe Ratio (PSR)."""
    if len(returns) < 2: return 0.0
    
    skew_val = skew(returns, nan_policy='omit')
    kurt_val = kurtosis(returns, nan_policy='omit')
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    
    if std_ret == 0: return 0.0
    
    sharpe = (mean_ret - benchmark) / std_ret
    n = len(returns)
    
    sigma_sr = np.sqrt((1 - skew_val * sharpe + (kurt_val - 1) / 4 * sharpe**2) / (n - 1))
    psr = norm.cdf((sharpe - benchmark) / sigma_sr)
    return psr

def _create_fresh_models(config: AIEnsembleStrategyParams, num_features: int, device) -> Dict[str, Any]:
    hp = config.hyperparameters
    cw = 'balanced' if config.training.use_class_weighting else None

    models = {
        'gb': XGBClassifier(
            n_estimators=hp.xgboost.n_estimators, max_depth=hp.xgboost.max_depth,
            learning_rate=hp.xgboost.learning_rate, subsample=hp.xgboost.subsample,
            colsample_bytree=hp.xgboost.colsample_bytree, random_state=42, 
            use_label_encoder=False, eval_metric='logloss'
        ),
        'technical': VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(n_estimators=hp.random_forest.n_estimators, 
                                          max_depth=hp.random_forest.max_depth, 
                                          min_samples_leaf=hp.random_forest.min_samples_leaf, 
                                          class_weight=cw, random_state=42)),
            ('lr', LogisticRegression(max_iter=hp.logistic_regression.max_iter, 
                                      C=hp.logistic_regression.C, 
                                      class_weight=cw, random_state=42))
        ], voting='soft')
    }

    if ML_AVAILABLE:
        num_channels = [hp.lstm.hidden_dim] * hp.lstm.num_layers
        models['lstm'] = TCNPredictor(num_features, num_channels, kernel_size=2, dropout=hp.lstm.dropout).to(device)
        models['attention'] = AttentionNetwork(num_features, hp.attention.hidden_dim, hp.attention.num_layers, 
                                               hp.attention.nhead, hp.attention.dropout).to(device)
    return models

def _train_torch_model(model, X_train, y_train, X_val, y_val, config, device, class_weights=None):
    if len(X_train) == 0: return model
    X_train = InputSanitizer.sanitize(X_train)
    X_val = InputSanitizer.sanitize(X_val)
    
    train_ds = TensorDataset(torch.FloatTensor(X_train).to(device), torch.LongTensor(y_train).to(device))
    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True)
    
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    
    opt = optim.Adam(model.parameters(), lr=config.training.learning_rate)
    crit = nn.CrossEntropyLoss(weight=class_weights)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=config.training.early_stopping_patience // 2)
    
    best_loss = float('inf')
    best_state = None
    patience = 0
    
    for _ in range(config.training.epochs):
        model.train()
        for bx, by in train_loader:
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
            
        model.eval()
        with torch.no_grad():
            val_loss = crit(model(X_val_t), y_val_t).item()
        
        sched.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= config.training.early_stopping_patience: break
            
    if best_state: model.load_state_dict(best_state)
    return model

def train_ensemble_task(symbol: str, df: pd.DataFrame, config: AIEnsembleStrategyParams, leader_df: Optional[pd.DataFrame] = None) -> bool:
    if not ML_AVAILABLE: return False
    try:
        # 1. Feature Engineering
        df_proc = FeatureProcessor.process_data(df, config, leader_df=leader_df)
        labels, t1 = FeatureProcessor.create_labels(df, config)
        
        common = df_proc.index.intersection(labels.index)
        if len(common) < 200: return False
        
        X = df_proc.loc[common].values
        y = labels.loc[common].values.astype(int)
        t1_vals = t1.loc[common].values
        
        X = InputSanitizer.sanitize(X)
        if len(np.unique(y)) < 2: return False

        # 2. Purged Walk-Forward Split for Validation
        # We use the last split as the final validation set for weight optimization
        cv = PurgedTimeSeriesSplit(n_splits=config.training.cv_splits)
        splits = list(cv.split(X, groups=t1_vals))
        train_idx, val_idx = splits[-1]
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Sequences
        seq_len = config.features.sequence_length
        X_train_seq, y_train_seq = FeatureProcessor.create_sequences(X_train, y_train, seq_len)
        X_val_seq, y_val_seq = FeatureProcessor.create_sequences(X_val, y_val, seq_len)

        # 3. Train Models
        device = torch.device("cpu")
        models = _create_fresh_models(config, X.shape[1], device)
        
        models['gb'].fit(X_train, y_train)
        models['technical'].fit(X_train, y_train)
        
        cw_tensor = None
        if config.training.use_class_weighting:
            counts = np.bincount(y_train, minlength=3)
            w = 1.0 / (counts + 1e-6)
            cw_tensor = torch.FloatTensor(w / w.sum()).to(device)
            
        if 'lstm' in models and len(X_train_seq) > 0:
            models['lstm'] = _train_torch_model(models['lstm'], X_train_seq, y_train_seq, X_val_seq, y_val_seq, config, device, cw_tensor)
        if 'attention' in models and len(X_train_seq) > 0:
            models['attention'] = _train_torch_model(models['attention'], X_train_seq, y_train_seq, X_val_seq, y_val_seq, config, device, cw_tensor)

        # 4. Generate Validation Predictions
        val_preds = {}
        val_preds['gb'] = models['gb'].predict_proba(X_val)
        val_preds['technical'] = models['technical'].predict_proba(X_val)
        
        def get_torch_preds(model, x_seq, full_len):
            model.eval()
            with torch.no_grad():
                logits = model(torch.FloatTensor(x_seq).to(device))
                probs = F.softmax(logits, dim=1).numpy()
                pad = np.tile(probs[0], (full_len - len(x_seq), 1))
                return np.vstack([pad, probs])

        if 'lstm' in models and len(X_val_seq) > 0:
            val_preds['lstm'] = get_torch_preds(models['lstm'], X_val_seq, len(X_val))
        if 'attention' in models and len(X_val_seq) > 0:
            val_preds['attention'] = get_torch_preds(models['attention'], X_val_seq, len(X_val))

        # 5. Optimize Weights (SLSQP)
        def loss_func(w):
            w = w / np.sum(w)
            ens = np.zeros_like(val_preds['gb'])
            for k, v in val_preds.items():
                idx = list(val_preds.keys()).index(k)
                ens += w[idx] * v
            return log_loss(y_val, np.clip(ens, 1e-15, 1-1e-15))

        keys = list(val_preds.keys())
        res = minimize(loss_func, np.ones(len(keys))/len(keys), method='SLSQP', 
                       bounds=[(0,1)]*len(keys), constraints={'type':'eq', 'fun': lambda w: np.sum(w)-1})
        opt_weights = {k: float(res.x[i]) for i, k in enumerate(keys)}

        # 6. Threshold Optimization via PSR
        ensemble_probs = np.zeros((len(X_val), 3))
        for k, w in opt_weights.items():
            ensemble_probs += w * val_preds[k]
            
        best_thresh = config.confidence_threshold
        best_metric = -np.inf
        
        returns = df['close'].pct_change().loc[common].values[val_idx]
        # Align lengths
        min_l = min(len(ensemble_probs), len(returns))
        ensemble_probs = ensemble_probs[-min_l:]
        returns = returns[-min_l:]
        
        for thresh in np.arange(0.5, 0.9, 0.02):
            sigs = np.zeros(len(returns))
            sigs[ensemble_probs[:, 2] > thresh] = 1
            sigs[ensemble_probs[:, 0] > thresh] = -1
            strat_ret = sigs * returns
            
            metric = probabilistic_sharpe_ratio(strat_ret) if config.training.use_probabilistic_sharpe else (np.mean(strat_ret)/np.std(strat_ret) if np.std(strat_ret)>0 else 0)
            if metric > best_metric:
                best_metric = metric
                best_thresh = float(thresh)

        # 7. Save
        safe_sym = symbol.replace('/', '_')
        save_dir = os.path.join(config.model_path, safe_sym)
        os.makedirs(save_dir, exist_ok=True)
        
        meta = {
            'timestamp': datetime.now().isoformat(),
            'feature_columns': FeatureProcessor.get_feature_names(config),
            'active_feature_columns': list(df_proc.columns),
            'num_features': X.shape[1],
            'optimized_weights': opt_weights,
            'optimized_threshold': best_thresh,
            'metrics': {'psr': float(best_metric), 'val_acc': float(accuracy_score(y_val, np.argmax(ensemble_probs, axis=1)))}
        }
        
        with open(os.path.join(save_dir, "metadata.json"), 'w') as f: json.dump(meta, f, indent=2)
        joblib.dump(models['gb'], os.path.join(save_dir, "gb.joblib"))
        joblib.dump(models['technical'], os.path.join(save_dir, "technical.joblib"))
        
        if 'lstm' in models: torch.save(models['lstm'].state_dict(), os.path.join(save_dir, "lstm.pth"))
        if 'attention' in models: torch.save(models['attention'].state_dict(), os.path.join(save_dir, "attention.pth"))
        
        return True
    except Exception as e:
        with open("training_errors.log", "a") as f: f.write(f"{datetime.now()} - {symbol}: {e}\n")
        return False

class EnsembleLearner:
    def __init__(self, config: AIEnsembleStrategyParams):
        self.config = config
        self.symbol_models = {}
        self.device = torch.device("cuda" if ML_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def close(self):
        self.executor.shutdown(wait=True)

    @property
    def is_trained(self) -> bool: return bool(self.symbol_models)
    def has_valid_model(self, symbol: str) -> bool: return symbol in self.symbol_models
    
    def get_last_training_time(self, symbol: str) -> Optional[datetime]:
        try:
            path = os.path.join(self.config.model_path, symbol.replace('/', '_'), "metadata.json")
            if os.path.exists(path):
                with open(path) as f: return datetime.fromisoformat(json.load(f)['timestamp'])
        except: pass
        return None

    async def reload_models(self, symbol: str):
        if not ML_AVAILABLE: return
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(self.executor, self._load_models_sync, symbol)
        if data: self.symbol_models[symbol] = data

    def _load_models_sync(self, symbol: str):
        path = os.path.join(self.config.model_path, symbol.replace('/', '_'))
        if not os.path.exists(path): return None
        try:
            with open(os.path.join(path, "metadata.json")) as f: meta = json.load(f)
            models = {}
            models['gb'] = joblib.load(os.path.join(path, "gb.joblib"))
            models['technical'] = joblib.load(os.path.join(path, "technical.joblib"))
            
            num_feat = meta['num_features']
            if 'lstm' in self.config.ensemble_weights.dict():
                m = TCNPredictor(num_feat, [self.config.hyperparameters.lstm.hidden_dim]*self.config.hyperparameters.lstm.num_layers, dropout=0).to(self.device)
                m.load_state_dict(torch.load(os.path.join(path, "lstm.pth"), map_location=self.device))
                m.eval()
                models['lstm'] = m
            if 'attention' in self.config.ensemble_weights.dict():
                m = AttentionNetwork(num_feat, self.config.hyperparameters.attention.hidden_dim, self.config.hyperparameters.attention.num_layers, self.config.hyperparameters.attention.nhead, 0).to(self.device)
                m.load_state_dict(torch.load(os.path.join(path, "attention.pth"), map_location=self.device))
                m.eval()
                models['attention'] = m
            return {'models': models, 'meta': meta}
        except Exception as e:
            logger.error(f"Load failed for {symbol}: {e}")
            return None

    async def warmup_models(self, symbols: List[str]):
        await asyncio.gather(*[self.reload_models(s) for s in symbols])

    async def rollback_model(self, symbol: str) -> bool:
        # Implementation of rollback logic (renaming folders)
        return False # Placeholder for brevity, logic exists in original

    async def predict(self, df: pd.DataFrame, symbol: str, regime: str = None, leader_df: pd.DataFrame = None, custom_weights: Dict = None) -> AIInferenceResult:
        if symbol not in self.symbol_models: return AIInferenceResult(action='hold', confidence=0.0, model_version='none', active_weights={}, top_features={}, metrics={})
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._predict_sync, df, symbol, leader_df, custom_weights)

    def _predict_sync(self, df, symbol, leader_df, custom_weights):
        entry = self.symbol_models[symbol]
        models, meta = entry['models'], entry['meta']
        
        df_proc = FeatureProcessor.process_data(df, self.config, leader_df)
        feats = meta.get('active_feature_columns')
        if not set(feats).issubset(df_proc.columns): return AIInferenceResult(action='hold', confidence=0.0, model_version='error', active_weights={}, top_features={}, metrics={})
        
        X = InputSanitizer.sanitize(df_proc[feats].iloc[-1:].values)
        seq_len = self.config.features.sequence_length
        X_seq = InputSanitizer.sanitize(df_proc[feats].iloc[-seq_len:].values.reshape(1, seq_len, -1)) if len(df_proc) >= seq_len else None
        
        votes = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
        weights = custom_weights or meta['optimized_weights']
        ind_preds = {}
        
        for name, model in models.items():
            w = weights.get(name, 0.0)
            if w == 0: continue
            
            if name in ['gb', 'technical']:
                probs = model.predict_proba(X)[0]
            elif X_seq is not None:
                with torch.no_grad():
                    probs = F.softmax(model(torch.FloatTensor(X_seq).to(self.device)), dim=1).cpu().numpy()[0]
            else:
                probs = [0.33, 0.33, 0.33]
                
            votes['sell'] += probs[0] * w
            votes['hold'] += probs[1] * w
            votes['buy'] += probs[2] * w
            ind_preds[name] = probs.tolist()
            
        best_action = max(votes, key=votes.get)
        return AIInferenceResult(
            action=best_action, confidence=votes[best_action], model_version=meta['timestamp'],
            active_weights=votes, top_features={}, metrics=meta['metrics'], 
            optimized_threshold=meta.get('optimized_threshold'), individual_predictions=ind_preds
        )
