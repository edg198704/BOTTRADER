import os
import joblib
import json
import logging
import numpy as np
import pandas as pd
import asyncio
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import norm, skew, kurtosis
from scipy.optimize import minimize

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
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, accuracy_score
    from xgboost import XGBClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = get_logger(__name__)

class InputSanitizer:
    @staticmethod
    def sanitize(X: np.ndarray) -> np.ndarray:
        if X is None or X.size == 0: return X
        return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

class PurgedTimeSeriesSplit:
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        test_size = n_samples // (self.n_splits + 1)
        embargo = int(n_samples * self.embargo_pct)

        for i in range(self.n_splits):
            train_end = n_samples - (self.n_splits - i) * test_size
            test_start = train_end
            test_end = test_start + test_size
            effective_train_end = train_end - embargo
            if effective_train_end > 0:
                yield indices[:effective_train_end], indices[test_start:test_end]

def probabilistic_sharpe_ratio(returns: np.ndarray, benchmark: float = 0.0) -> float:
    if len(returns) < 2: return 0.0
    skew_val = skew(returns, nan_policy='omit')
    kurt_val = kurtosis(returns, nan_policy='omit')
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    if std_ret == 0: return 0.0
    sharpe = (mean_ret - benchmark) / std_ret
    n = len(returns)
    sigma_sr = np.sqrt((1 - skew_val * sharpe + (kurt_val - 1) / 4 * sharpe**2) / (n - 1))
    return norm.cdf((sharpe - benchmark) / sigma_sr)

# --- Training Helpers ---

def _prepare_data(df: pd.DataFrame, config: AIEnsembleStrategyParams, leader_df: Optional[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    df_proc = FeatureProcessor.process_data(df, config, leader_df=leader_df)
    labels, t1 = FeatureProcessor.create_labels(df, config)
    common = df_proc.index.intersection(labels.index)
    
    if len(common) < 200:
        raise ValueError("Insufficient common data points for training")
        
    X = InputSanitizer.sanitize(df_proc.loc[common].values)
    y = labels.loc[common].values.astype(int)
    t1_vals = t1.loc[common].values
    return X, y, t1_vals, df_proc, list(df_proc.columns)

def _create_models(config: AIEnsembleStrategyParams, num_features: int, device) -> Dict[str, Any]:
    hp = config.hyperparameters
    cw = 'balanced' if config.training.use_class_weighting else None
    models = {
        'gb': XGBClassifier(n_estimators=hp.xgboost.n_estimators, max_depth=hp.xgboost.max_depth, learning_rate=hp.xgboost.learning_rate, subsample=hp.xgboost.subsample, colsample_bytree=hp.xgboost.colsample_bytree, random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'technical': VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(n_estimators=hp.random_forest.n_estimators, max_depth=hp.random_forest.max_depth, min_samples_leaf=hp.random_forest.min_samples_leaf, class_weight=cw, random_state=42)),
            ('lr', LogisticRegression(max_iter=hp.logistic_regression.max_iter, C=hp.logistic_regression.C, class_weight=cw, random_state=42))
        ], voting='soft')
    }
    if ML_AVAILABLE:
        models['lstm'] = TCNPredictor(num_features, [hp.lstm.hidden_dim] * hp.lstm.num_layers, kernel_size=2, dropout=hp.lstm.dropout).to(device)
        models['attention'] = AttentionNetwork(num_features, hp.attention.hidden_dim, hp.attention.num_layers, hp.attention.nhead, hp.attention.dropout).to(device)
    return models

def _train_torch_model(model, X_train, y_train, X_val, y_val, config, device):
    if len(X_train) == 0: return model
    train_ds = TensorDataset(torch.FloatTensor(X_train).to(device), torch.LongTensor(y_train).to(device))
    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    
    opt = optim.Adam(model.parameters(), lr=config.training.learning_rate)
    crit = nn.CrossEntropyLoss()
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=config.training.early_stopping_patience // 2)
    
    best_loss = float('inf')
    best_state = None
    patience = 0
    
    for _ in range(config.training.epochs):
        model.train()
        for bx, by in train_loader:
            opt.zero_grad(); loss = crit(model(bx), by); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): val_loss = crit(model(X_val_t), y_val_t).item()
        sched.step(val_loss)
        if val_loss < best_loss: best_loss = val_loss; best_state = model.state_dict(); patience = 0
        else: patience += 1
        if patience >= config.training.early_stopping_patience: break
    if best_state: model.load_state_dict(best_state)
    return model

def train_ensemble_task(symbol: str, df: pd.DataFrame, config: AIEnsembleStrategyParams, leader_df: Optional[pd.DataFrame] = None) -> bool:
    if not ML_AVAILABLE: return False
    try:
        X, y, t1_vals, df_proc, active_feats = _prepare_data(df, config, leader_df)
        if len(np.unique(y)) < 2: return False

        cv = PurgedTimeSeriesSplit(n_splits=config.training.cv_splits)
        splits = list(cv.split(X, groups=t1_vals))
        train_idx, val_idx = splits[-1]
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        seq_len = config.features.sequence_length
        X_train_seq, y_train_seq = FeatureProcessor.create_sequences(X_train, y_train, seq_len)
        X_val_seq, y_val_seq = FeatureProcessor.create_sequences(X_val, y_val, seq_len)

        device = torch.device("cpu")
        models = _create_models(config, X.shape[1], device)
        models['gb'].fit(X_train, y_train)
        models['technical'].fit(X_train, y_train)

        if 'lstm' in models and len(X_train_seq) > 0:
            models['lstm'] = _train_torch_model(models['lstm'], X_train_seq, y_train_seq, X_val_seq, y_val_seq, config, device)
        if 'attention' in models and len(X_train_seq) > 0:
            models['attention'] = _train_torch_model(models['attention'], X_train_seq, y_train_seq, X_val_seq, y_val_seq, config, device)

        # Validation Predictions & Optimization
        val_preds = {'gb': models['gb'].predict_proba(X_val), 'technical': models['technical'].predict_proba(X_val)}
        
        def get_torch_preds(model, x_seq, full_len):
            model.eval()
            with torch.no_grad():
                probs = F.softmax(model(torch.FloatTensor(x_seq).to(device)), dim=1).numpy()
                return np.vstack([np.tile(probs[0], (full_len - len(x_seq), 1)), probs])

        if 'lstm' in models: val_preds['lstm'] = get_torch_preds(models['lstm'], X_val_seq, len(X_val))
        if 'attention' in models: val_preds['attention'] = get_torch_preds(models['attention'], X_val_seq, len(X_val))

        def loss_func(w):
            ens = sum(w[i] * val_preds[k] for i, k in enumerate(val_preds.keys()))
            return log_loss(y_val, np.clip(ens, 1e-15, 1-1e-15))

        keys = list(val_preds.keys())
        res = minimize(loss_func, np.ones(len(keys))/len(keys), method='SLSQP', bounds=[(0,1)]*len(keys), constraints={'type':'eq', 'fun': lambda w: np.sum(w)-1})
        opt_weights = {k: float(res.x[i]) for i, k in enumerate(keys)}

        # Threshold Optimization
        ensemble_probs = sum(opt_weights[k] * val_preds[k] for k in keys)
        returns = df['close'].pct_change().loc[df_proc.index].values[val_idx][-len(ensemble_probs):]
        best_thresh, best_metric = config.confidence_threshold, -np.inf
        
        for thresh in np.arange(0.5, 0.9, 0.02):
            sigs = np.zeros(len(returns))
            sigs[ensemble_probs[:, 2] > thresh] = 1
            sigs[ensemble_probs[:, 0] > thresh] = -1
            metric = probabilistic_sharpe_ratio(sigs * returns) if config.training.use_probabilistic_sharpe else np.mean(sigs * returns)
            if metric > best_metric: best_metric = metric; best_thresh = float(thresh)

        # Save
        save_dir = os.path.join(config.model_path, symbol.replace('/', '_'))
        os.makedirs(save_dir, exist_ok=True)
        meta = {'timestamp': datetime.now().isoformat(), 'active_feature_columns': active_feats, 'num_features': X.shape[1], 'optimized_weights': opt_weights, 'optimized_threshold': best_thresh, 'metrics': {'psr': float(best_metric)}}
        with open(os.path.join(save_dir, "metadata.json"), 'w') as f: json.dump(meta, f, indent=2)
        joblib.dump(models['gb'], os.path.join(save_dir, "gb.joblib"))
        joblib.dump(models['technical'], os.path.join(save_dir, "technical.joblib"))
        if 'lstm' in models: torch.save(models['lstm'].state_dict(), os.path.join(save_dir, "lstm.pth"))
        if 'attention' in models: torch.save(models['attention'].state_dict(), os.path.join(save_dir, "attention.pth"))
        return True
    except Exception as e:
        logger.error(f"Training failed for {symbol}: {e}")
        return False

class EnsembleLearner:
    def __init__(self, config: AIEnsembleStrategyParams):
        self.config = config
        self.symbol_models = {}
        self.device = torch.device("cuda" if ML_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def close(self): self.executor.shutdown(wait=True)
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
            models = {
                'gb': joblib.load(os.path.join(path, "gb.joblib")),
                'technical': joblib.load(os.path.join(path, "technical.joblib"))
            }
            num_feat = meta['num_features']
            if 'lstm' in self.config.ensemble_weights.dict():
                m = TCNPredictor(num_feat, [self.config.hyperparameters.lstm.hidden_dim]*self.config.hyperparameters.lstm.num_layers, dropout=0).to(self.device)
                m.load_state_dict(torch.load(os.path.join(path, "lstm.pth"), map_location=self.device))
                m.eval(); models['lstm'] = m
            if 'attention' in self.config.ensemble_weights.dict():
                m = AttentionNetwork(num_feat, self.config.hyperparameters.attention.hidden_dim, self.config.hyperparameters.attention.num_layers, self.config.hyperparameters.attention.nhead, 0).to(self.device)
                m.load_state_dict(torch.load(os.path.join(path, "attention.pth"), map_location=self.device))
                m.eval(); models['attention'] = m
            return {'models': models, 'meta': meta}
        except Exception as e:
            logger.error(f"Load failed for {symbol}: {e}")
            return None

    async def warmup_models(self, symbols: List[str]):
        await asyncio.gather(*[self.reload_models(s) for s in symbols])

    async def predict(self, df: pd.DataFrame, symbol: str, regime: str = None, leader_df: pd.DataFrame = None, custom_weights: Dict = None) -> AIInferenceResult:
        if symbol not in self.symbol_models: return AIInferenceResult(action='hold', confidence=0.0, model_version='none', active_weights={}, top_features={}, metrics={})
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._predict_sync, df, symbol, leader_df, custom_weights)

    def _predict_sync(self, df, symbol, leader_df, custom_weights):
        entry = self.symbol_models[symbol]
        models, meta = entry['models'], entry['meta']
        
        # Optimization: Only process necessary history for feature generation
        # We need enough history for rolling windows (normalization_window) + sequence_length
        required_history = self.config.features.normalization_window + self.config.features.sequence_length + 50
        if len(df) > required_history:
            df_subset = df.iloc[-required_history:].copy()
            if leader_df is not None and len(leader_df) > required_history:
                leader_subset = leader_df.iloc[-required_history:].copy()
            else:
                leader_subset = leader_df
        else:
            df_subset = df
            leader_subset = leader_df

        df_proc = FeatureProcessor.process_data(df_subset, self.config, leader_subset)
        feats = meta.get('active_feature_columns')
        if not set(feats).issubset(df_proc.columns): 
            return AIInferenceResult(action='hold', confidence=0.0, model_version='error', active_weights={}, top_features={}, metrics={})
        
        X = InputSanitizer.sanitize(df_proc[feats].iloc[-1:].values)
        seq_len = self.config.features.sequence_length
        X_seq = InputSanitizer.sanitize(df_proc[feats].iloc[-seq_len:].values.reshape(1, seq_len, -1)) if len(df_proc) >= seq_len else None
        
        votes = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
        weights = custom_weights or meta['optimized_weights']
        ind_preds = {}
        
        for name, model in models.items():
            w = weights.get(name, 0.0)
            if w == 0: continue
            if name in ['gb', 'technical']: probs = model.predict_proba(X)[0]
            elif X_seq is not None:
                with torch.no_grad(): probs = F.softmax(model(torch.FloatTensor(X_seq).to(self.device)), dim=1).cpu().numpy()[0]
            else: probs = [0.33, 0.33, 0.33]
            votes['sell'] += probs[0] * w; votes['hold'] += probs[1] * w; votes['buy'] += probs[2] * w
            ind_preds[name] = probs.tolist()
            
        best_action = max(votes, key=votes.get)
        return AIInferenceResult(action=best_action, confidence=votes[best_action], model_version=meta['timestamp'], active_weights=votes, top_features={}, metrics=meta['metrics'], optimized_threshold=meta.get('optimized_threshold'), individual_predictions=ind_preds)
