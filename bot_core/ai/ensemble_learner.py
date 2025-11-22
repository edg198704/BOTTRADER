import os
import joblib
import json
import logging
import numpy as np
import pandas as pd
import asyncio
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List, Union
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
    from sklearn.metrics import log_loss, accuracy_score, precision_score, f1_score
    from xgboost import XGBClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = get_logger(__name__)

# --- Utility Classes ---

class InputSanitizer:
    @staticmethod
    def sanitize(X: np.ndarray) -> np.ndarray:
        if X is None or X.size == 0: return X
        return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

class PurgedTimeSeriesSplit:
    """
    Time Series Split with Embargo to prevent leakage.
    """
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

# --- Model Adapters ---

class BaseModelAdapter(ABC):
    """Abstract base class for unified model interaction."""
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        return None

class SklearnAdapter(BaseModelAdapter):
    def __init__(self, model: Any):
        self.model = model

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
        self.model.fit(X_train, y_train)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
        
    def get_feature_importance(self) -> Optional[np.ndarray]:
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        # Handle VotingClassifier
        if hasattr(self.model, 'estimators_'):
            imps = []
            for _, est in self.model.estimators_:
                if hasattr(est, 'feature_importances_'):
                    imps.append(est.feature_importances_)
                elif hasattr(est, 'coef_'):
                    imps.append(np.abs(est.coef_[0]))
            if imps:
                return np.mean(imps, axis=0)
        return None

class TorchAdapter(BaseModelAdapter):
    def __init__(self, model_class: Any, input_dim: int, config: Any, device: torch.device):
        self.model_class = model_class
        self.input_dim = input_dim
        self.config = config
        self.device = device
        self.model = None
        self.seq_len = config.features.sequence_length

    def _init_model(self):
        hp = self.config.hyperparameters
        if self.model_class == TCNPredictor:
            self.model = TCNPredictor(self.input_dim, [hp.lstm.hidden_dim] * hp.lstm.num_layers, kernel_size=2, dropout=hp.lstm.dropout).to(self.device)
        elif self.model_class == AttentionNetwork:
            self.model = AttentionNetwork(self.input_dim, hp.attention.hidden_dim, hp.attention.num_layers, hp.attention.nhead, hp.attention.dropout).to(self.device)

    def _create_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if len(X) <= self.seq_len:
            return np.array([]), np.array([]) if y is not None else np.array([])
        
        shape = (len(X) - self.seq_len + 1, self.seq_len, X.shape[1])
        strides = (X.strides[0], X.strides[0], X.strides[1])
        X_seq = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
        
        if y is not None:
            y_seq = y[self.seq_len-1:]
            return X_seq, y_seq
        return X_seq

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
        self._init_model()
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self._create_sequences(X_val, y_val)

        if len(X_train_seq) == 0: return

        train_ds = TensorDataset(torch.FloatTensor(X_train_seq).to(self.device), torch.LongTensor(y_train_seq).to(self.device))
        train_loader = DataLoader(train_ds, batch_size=self.config.training.batch_size, shuffle=True)
        X_val_t = torch.FloatTensor(X_val_seq).to(self.device)
        y_val_t = torch.LongTensor(y_val_seq).to(self.device)

        opt = optim.Adam(self.model.parameters(), lr=self.config.training.learning_rate)
        crit = nn.CrossEntropyLoss()
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=self.config.training.early_stopping_patience // 2)

        best_loss = float('inf')
        best_state = None
        patience = 0

        for _ in range(self.config.training.epochs):
            self.model.train()
            for bx, by in train_loader:
                opt.zero_grad()
                loss = crit(self.model(bx), by)
                loss.backward()
                opt.step()
            
            self.model.eval()
            with torch.no_grad():
                val_loss = crit(self.model(X_val_t), y_val_t).item()
            
            sched.step(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = self.model.state_dict()
                patience = 0
            else:
                patience += 1
            
            if patience >= self.config.training.early_stopping_patience:
                break
        
        if best_state:
            self.model.load_state_dict(best_state)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None: self._init_model()
        self.model.eval()
        X_seq = self._create_sequences(X)
        if len(X_seq) == 0:
            # Fallback for insufficient history: return uniform prob
            return np.full((len(X), 3), 1.0/3.0)

        with torch.no_grad():
            probs = F.softmax(self.model(torch.FloatTensor(X_seq).to(self.device)), dim=1).cpu().numpy()
        
        # Pad the beginning to match input length (since sequences consume history)
        pad_len = len(X) - len(probs)
        if pad_len > 0:
            padding = np.tile(probs[0], (pad_len, 1))
            return np.vstack([padding, probs])
        return probs

    def save(self, path: str) -> None:
        if self.model:
            torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        self._init_model()
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

# --- Main Classes ---

class EnsembleModel:
    """Container for loaded model artifacts."""
    def __init__(self, adapters: Dict[str, BaseModelAdapter], meta: Dict[str, Any], meta_model: Any = None):
        self.adapters = adapters
        self.meta = meta
        self.meta_model = meta_model

class EnsembleTrainer:
    """
    Encapsulates the training logic for the ensemble model.
    """
    def __init__(self, config: AIEnsembleStrategyParams):
        self.config = config
        self.device = torch.device("cpu") # Train on CPU in background process to avoid CUDA context issues

    def _prepare_data(self, df: pd.DataFrame, leader_df: Optional[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
        df_proc = FeatureProcessor.process_data(df, self.config, leader_df=leader_df)
        labels, t1 = FeatureProcessor.create_labels(df, self.config)
        common = df_proc.index.intersection(labels.index)
        
        if len(common) < 200:
            raise ValueError("Insufficient common data points for training")
            
        X = InputSanitizer.sanitize(df_proc.loc[common].values)
        y = labels.loc[common].values.astype(int)
        t1_vals = t1.loc[common].values
        return X, y, t1_vals, df_proc, list(df_proc.columns)

    def _create_adapters(self, num_features: int) -> Dict[str, BaseModelAdapter]:
        hp = self.config.hyperparameters
        cw = 'balanced' if self.config.training.use_class_weighting else None
        
        adapters = {}
        
        # XGBoost
        xgb = XGBClassifier(n_estimators=hp.xgboost.n_estimators, max_depth=hp.xgboost.max_depth, 
                            learning_rate=hp.xgboost.learning_rate, subsample=hp.xgboost.subsample, 
                            colsample_bytree=hp.xgboost.colsample_bytree, random_state=42, 
                            use_label_encoder=False, eval_metric='logloss')
        adapters['gb'] = SklearnAdapter(xgb)

        # Technical Ensemble
        rf = RandomForestClassifier(n_estimators=hp.random_forest.n_estimators, max_depth=hp.random_forest.max_depth, 
                                    min_samples_leaf=hp.random_forest.min_samples_leaf, class_weight=cw, random_state=42)
        lr = LogisticRegression(max_iter=hp.logistic_regression.max_iter, C=hp.logistic_regression.C, 
                                class_weight=cw, random_state=42)
        voting = VotingClassifier(estimators=[('rf', rf), ('lr', lr)], voting='soft')
        adapters['technical'] = SklearnAdapter(voting)

        # Deep Learning
        if ML_AVAILABLE:
            adapters['lstm'] = TorchAdapter(TCNPredictor, num_features, self.config, self.device)
            adapters['attention'] = TorchAdapter(AttentionNetwork, num_features, self.config, self.device)
            
        return adapters

    def train(self, symbol: str, df: pd.DataFrame, leader_df: Optional[pd.DataFrame] = None) -> bool:
        if not ML_AVAILABLE: return False
        try:
            X, y, t1_vals, df_proc, active_feats = self._prepare_data(df, leader_df)
            if len(np.unique(y)) < 2: return False

            cv = PurgedTimeSeriesSplit(n_splits=self.config.training.cv_splits)
            splits = list(cv.split(X, groups=t1_vals))
            train_idx, val_idx = splits[-1]
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            adapters = self._create_adapters(X.shape[1])
            val_preds = {}

            # Train all models
            for name, adapter in adapters.items():
                adapter.fit(X_train, y_train, X_val, y_val)
                val_preds[name] = adapter.predict_proba(X_val)

            # Optimize Weights
            def loss_func(w):
                ens = sum(w[i] * val_preds[k] for i, k in enumerate(val_preds.keys()))
                return log_loss(y_val, np.clip(ens, 1e-15, 1-1e-15))

            keys = list(val_preds.keys())
            res = minimize(loss_func, np.ones(len(keys))/len(keys), method='SLSQP', 
                           bounds=[(0,1)]*len(keys), constraints={'type':'eq', 'fun': lambda w: np.sum(w)-1})
            opt_weights = {k: float(res.x[i]) for i, k in enumerate(keys)}

            # Meta-Model Training
            meta_model = None
            meta_metrics = {}
            if self.config.meta_labeling.enabled:
                meta_feats_list = [val_preds[k] for k in keys]
                regime_cols = ['regime_volatility', 'regime_trend']
                regime_indices = [active_feats.index(c) for c in regime_cols if c in active_feats]
                if regime_indices: meta_feats_list.append(X_val[:, regime_indices])
                
                X_meta = np.hstack(meta_feats_list)
                ensemble_probs = sum(opt_weights[k] * val_preds[k] for k in keys)
                ensemble_preds = np.argmax(ensemble_probs, axis=1)
                
                y_meta = np.zeros_like(y_val)
                y_meta[(ensemble_preds == 2) & (y_val == 2)] = 1
                y_meta[(ensemble_preds == 0) & (y_val == 0)] = 1
                
                if len(np.unique(y_meta)) > 1:
                    meta_model = RandomForestClassifier(n_estimators=self.config.meta_labeling.n_estimators, 
                                                        max_depth=self.config.meta_labeling.max_depth, 
                                                        random_state=42, class_weight='balanced')
                    meta_model.fit(X_meta, y_meta)
                    meta_val_preds = meta_model.predict(X_meta)
                    meta_metrics = {'precision': float(precision_score(y_meta, meta_val_preds, zero_division=0)), 
                                    'f1': float(f1_score(y_meta, meta_val_preds, zero_division=0))}

            # Threshold Optimization
            ensemble_probs = sum(opt_weights[k] * val_preds[k] for k in keys)
            returns = df['close'].pct_change().loc[df_proc.index].values[val_idx][-len(ensemble_probs):]
            best_thresh, best_metric = self.config.confidence_threshold, -np.inf
            
            for thresh in np.arange(0.5, 0.9, 0.02):
                sigs = np.zeros(len(returns))
                sigs[ensemble_probs[:, 2] > thresh] = 1
                sigs[ensemble_probs[:, 0] > thresh] = -1
                metric = probabilistic_sharpe_ratio(sigs * returns) if self.config.training.use_probabilistic_sharpe else np.mean(sigs * returns)
                if metric > best_metric: best_metric = metric; best_thresh = float(thresh)

            # Save Artifacts
            save_dir = os.path.join(self.config.model_path, symbol.replace('/', '_'))
            os.makedirs(save_dir, exist_ok=True)
            
            meta = {
                'timestamp': datetime.now().isoformat(), 'active_feature_columns': active_feats, 
                'num_features': X.shape[1], 'optimized_weights': opt_weights, 
                'optimized_threshold': best_thresh, 'metrics': {'psr': float(best_metric), 'meta': meta_metrics},
                'weight_momentum': {k: 0.0 for k in opt_weights}
            }
            
            with open(os.path.join(save_dir, "metadata.json"), 'w') as f: json.dump(meta, f, indent=2)
            
            for name, adapter in adapters.items():
                adapter.save(os.path.join(save_dir, f"{name}.model"))
            
            if meta_model:
                joblib.dump(meta_model, os.path.join(save_dir, "meta_model.joblib"))
                
            return True
        except Exception as e:
            logger.error(f"Training failed for {symbol}: {e}", exc_info=True)
            return False

def train_ensemble_task(symbol: str, df: pd.DataFrame, config: AIEnsembleStrategyParams, leader_df: Optional[pd.DataFrame] = None) -> bool:
    trainer = EnsembleTrainer(config)
    return trainer.train(symbol, df, leader_df)

class EnsembleLearner:
    def __init__(self, config: AIEnsembleStrategyParams):
        self.config = config
        self.symbol_models: Dict[str, EnsembleModel] = {}
        self.device = torch.device("cuda" if ML_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.RLock()

    async def close(self): self.executor.shutdown(wait=True)
    
    @property
    def is_trained(self) -> bool: 
        with self._lock:
            return bool(self.symbol_models)
            
    def has_valid_model(self, symbol: str) -> bool: 
        with self._lock:
            return symbol in self.symbol_models
    
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
        model_data = await loop.run_in_executor(self.executor, self._load_models_sync, symbol)
        if model_data: 
            with self._lock:
                self.symbol_models[symbol] = model_data

    def _load_models_sync(self, symbol: str) -> Optional[EnsembleModel]:
        path = os.path.join(self.config.model_path, symbol.replace('/', '_'))
        if not os.path.exists(path): return None
        try:
            with open(os.path.join(path, "metadata.json")) as f: meta = json.load(f)
            
            adapters = {}
            num_feat = meta['num_features']
            
            if os.path.exists(os.path.join(path, "gb.model")):
                adapters['gb'] = SklearnAdapter(None)
                adapters['gb'].load(os.path.join(path, "gb.model"))
                
            if os.path.exists(os.path.join(path, "technical.model")):
                adapters['technical'] = SklearnAdapter(None)
                adapters['technical'].load(os.path.join(path, "technical.model"))
                
            if os.path.exists(os.path.join(path, "lstm.model")):
                adapters['lstm'] = TorchAdapter(TCNPredictor, num_feat, self.config, self.device)
                adapters['lstm'].load(os.path.join(path, "lstm.model"))
                
            if os.path.exists(os.path.join(path, "attention.model")):
                adapters['attention'] = TorchAdapter(AttentionNetwork, num_feat, self.config, self.device)
                adapters['attention'].load(os.path.join(path, "attention.model"))

            meta_model = None
            if os.path.exists(os.path.join(path, "meta_model.joblib")):
                meta_model = joblib.load(os.path.join(path, "meta_model.joblib"))
                
            return EnsembleModel(adapters, meta, meta_model)
        except Exception as e:
            logger.error(f"Load failed for {symbol}: {e}")
            return None

    async def warmup_models(self, symbols: List[str]):
        await asyncio.gather(*[self.reload_models(s) for s in symbols])

    async def predict(self, df: pd.DataFrame, symbol: str, regime: str = None, leader_df: pd.DataFrame = None, custom_weights: Dict = None) -> AIInferenceResult:
        entry = None
        with self._lock:
            entry = self.symbol_models.get(symbol)
            
        if not entry:
            return AIInferenceResult(action='hold', confidence=0.0, model_version='none', active_weights={}, top_features={}, metrics={})
        
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._predict_sync, df, entry, leader_df, custom_weights)

    def _predict_sync(self, df: pd.DataFrame, entry: EnsembleModel, leader_df: pd.DataFrame, custom_weights: Dict) -> AIInferenceResult:
        adapters, meta, meta_model = entry.adapters, entry.meta, entry.meta_model
        
        # Data Prep
        required_history = self.config.features.normalization_window + self.config.features.sequence_length + 50
        if len(df) > required_history:
            df_subset = df.iloc[-required_history:].copy()
            leader_subset = leader_df.iloc[-required_history:].copy() if leader_df is not None and len(leader_df) > required_history else leader_df
        else:
            df_subset = df
            leader_subset = leader_df

        df_proc = FeatureProcessor.process_data(df_subset, self.config, leader_subset)
        if df_proc.empty: 
             return AIInferenceResult(action='hold', confidence=0.0, model_version='error', active_weights={}, top_features={}, metrics={})

        feats = meta.get('active_feature_columns')
        if not set(feats).issubset(df_proc.columns): 
            return AIInferenceResult(action='hold', confidence=0.0, model_version='error', active_weights={}, top_features={}, metrics={})
        
        X = InputSanitizer.sanitize(df_proc[feats].iloc[-1:].values)
        
        votes = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
        weights = custom_weights or meta['optimized_weights']
        ind_preds = {}
        feature_importances = {}

        for name, adapter in adapters.items():
            w = weights.get(name, 0.0)
            if w == 0: continue
            
            probs = adapter.predict_proba(X)[0]
            votes['sell'] += probs[0] * w; votes['hold'] += probs[1] * w; votes['buy'] += probs[2] * w
            ind_preds[name] = probs.tolist()
            
            # Aggregate Feature Importance
            imp = adapter.get_feature_importance()
            if imp is not None and len(imp) == len(feats):
                for i, feat_name in enumerate(feats):
                    feature_importances[feat_name] = feature_importances.get(feat_name, 0.0) + (imp[i] * w)
            
        best_action = max(votes, key=votes.get)
        base_confidence = votes[best_action]
        
        meta_prob = 1.0
        if meta_model and self.config.meta_labeling.enabled:
            meta_feats_list = [np.array(ind_preds[k]).reshape(1, -1) for k in sorted(ind_preds.keys())]
            regime_cols = ['regime_volatility', 'regime_trend']
            regime_indices = [feats.index(c) for c in regime_cols if c in feats]
            if regime_indices: meta_feats_list.append(X[:, regime_indices])
            
            X_meta = np.hstack(meta_feats_list)
            meta_prob = meta_model.predict_proba(X_meta)[0, 1]
            if meta_prob < self.config.meta_labeling.probability_threshold:
                base_confidence *= 0.5 

        # Sort top features
        top_features = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)[:5])

        return AIInferenceResult(
            action=best_action, confidence=base_confidence, model_version=meta['timestamp'], 
            active_weights=weights, top_features=top_features, metrics=meta['metrics'], 
            optimized_threshold=meta.get('optimized_threshold'), individual_predictions=ind_preds, meta_probability=meta_prob
        )

    def update_weights(self, symbol: str, trade_pnl: float, individual_predictions: Dict[str, List[float]], side: str):
        with self._lock:
            if symbol not in self.symbol_models or not individual_predictions: return
            entry = self.symbol_models[symbol]
            meta = entry.meta
            current_weights = meta['optimized_weights']
            momentum = meta.get('weight_momentum', {k: 0.0 for k in current_weights})
            
            learning_rate = self.config.ensemble_weights.adaptive_weight_learning_rate
            beta = 0.9
            target_idx = 2 if (side == 'BUY' and trade_pnl > 0) or (side == 'SELL' and trade_pnl < 0) else 0
            if side == 'SELL' and trade_pnl > 0: target_idx = 0

            new_weights = current_weights.copy()
            total_weight = 0.0

            for model_name, probs in individual_predictions.items():
                if model_name not in current_weights: continue
                prob_correct = probs[target_idx]
                grad = learning_rate * (prob_correct - 0.5)
                v = momentum.get(model_name, 0.0)
                new_v = beta * v + (1 - beta) * grad
                momentum[model_name] = new_v
                w = max(0.01, min(current_weights[model_name] + new_v, 1.0))
                new_weights[model_name] = w
                total_weight += w

            if total_weight > 0:
                for k in new_weights: new_weights[k] /= total_weight

            meta['optimized_weights'] = new_weights
            meta['weight_momentum'] = momentum
            logger.info(f"Updated ensemble weights for {symbol} (Momentum)", weights=new_weights, pnl=trade_pnl)
