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

from bot_core.logger import get_logger
from bot_core.config import AIEnsembleStrategyParams
from bot_core.ai.models import LSTMPredictor, AttentionNetwork
from bot_core.ai.feature_processor import FeatureProcessor
from bot_core.ai.regime_detector import MarketRegimeDetector

# ML Imports with safe fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier, IsolationForest
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_score, classification_report, log_loss
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_predict
    from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
    from sklearn.isotonic import IsotonicRegression
    from xgboost import XGBClassifier
    from scipy.optimize import minimize
    from scipy.stats import skew, kurtosis, norm
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    # Dummy classes to prevent ImportErrors
    class VotingClassifier: pass
    class RandomForestClassifier: pass
    class LogisticRegression: pass
    class XGBClassifier: pass
    class RandomizedSearchCV: pass
    class TimeSeriesSplit: pass
    class IsotonicRegression: pass
    class IsolationForest: pass
    def precision_score(*args, **kwargs): return 0.0
    def log_loss(*args, **kwargs): return 0.0
    def classification_report(*args, **kwargs): return ""
    def compute_class_weight(*args, **kwargs): return []
    def compute_sample_weight(*args, **kwargs): return []
    def minimize(*args, **kwargs): return None
    def skew(*args, **kwargs): return 0.0
    def kurtosis(*args, **kwargs): return 0.0
    def cross_val_predict(*args, **kwargs): return []
    class norm:
        @staticmethod
        def cdf(x): return 0.5
    class nn:
        class Module: pass
    class optim: pass
    class TensorDataset: pass
    class DataLoader: pass
    class F:
        @staticmethod
        def softmax(x, dim=1): return x

logger = get_logger(__name__)

# --- Helper Classes ---

class InputSanitizer:
    """Ensures data fed into models is clean, finite, and shaped correctly."""
    @staticmethod
    def sanitize(X: np.ndarray) -> np.ndarray:
        if X is None or X.size == 0:
            return X
        # Replace NaNs with 0.0 (assuming Z-score normalization centered at 0)
        # Replace Infs with large finite numbers
        X_clean = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        return X_clean

class MulticlassCalibrator:
    """Calibrates probabilities for multiclass classification using One-vs-Rest Isotonic Regression."""
    def __init__(self, method='isotonic'):
        self.method = method
        self.calibrators = {} # class_idx -> regressor

    def fit(self, X_probs, y):
        n_classes = X_probs.shape[1]
        for i in range(n_classes):
            y_binary = (y == i).astype(int)
            X_col = X_probs[:, i]
            reg = IsotonicRegression(out_of_bounds='clip')
            reg.fit(X_col, y_binary)
            self.calibrators[i] = reg
            
    def predict(self, X_probs):
        n_samples, n_classes = X_probs.shape
        calibrated = np.zeros_like(X_probs)
        for i in range(n_classes):
            if i in self.calibrators:
                calibrated[:, i] = self.calibrators[i].predict(X_probs[:, i])
            else:
                calibrated[:, i] = X_probs[:, i]
        
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return calibrated / row_sums

# --- Standalone Helper Functions (Pickle-safe) ---

def _create_fresh_models(config: AIEnsembleStrategyParams, num_features: int, device) -> Dict[str, Any]:
    hp = config.hyperparameters
    xgb_config = hp.xgboost
    rf_config = hp.random_forest
    lr_config = hp.logistic_regression
    lstm_config = hp.lstm
    attn_config = hp.attention

    cw_option = 'balanced' if config.training.use_class_weighting else None

    models = {
        'gb': XGBClassifier(
            n_estimators=xgb_config.n_estimators,
            max_depth=xgb_config.max_depth,
            learning_rate=xgb_config.learning_rate,
            subsample=xgb_config.subsample,
            colsample_bytree=xgb_config.colsample_bytree,
            random_state=42, use_label_encoder=False, eval_metric='logloss'
        ),
        'technical': VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(
                n_estimators=rf_config.n_estimators,
                max_depth=rf_config.max_depth,
                min_samples_leaf=rf_config.min_samples_leaf,
                class_weight=cw_option,
                random_state=42
            )),
            ('lr', LogisticRegression(
                max_iter=lr_config.max_iter,
                C=lr_config.C,
                class_weight=cw_option,
                random_state=42
            ))
        ], voting='soft')
    }

    if ML_AVAILABLE and torch.cuda.is_available():
        models['lstm'] = LSTMPredictor(
            num_features, lstm_config.hidden_dim, lstm_config.num_layers, lstm_config.dropout
        ).to(device)
        models['attention'] = AttentionNetwork(
            num_features, attn_config.hidden_dim, attn_config.num_layers, attn_config.nhead, attn_config.dropout
        ).to(device)
    
    return models

def _train_torch_model(model, X_train, y_train, X_val, y_val, config, device, class_weights=None):
    if len(X_train) == 0 or len(X_val) == 0:
        return model

    epochs = config.training.epochs
    batch_size = config.training.batch_size
    lr = config.training.learning_rate
    patience = config.training.early_stopping_patience
    
    # Sanitize inputs
    X_train = InputSanitizer.sanitize(X_train)
    X_val = InputSanitizer.sanitize(X_val)
    
    train_tensor = TensorDataset(torch.FloatTensor(X_train).to(device), torch.LongTensor(y_train).to(device))
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=max(1, patience // 2))

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for b_x, b_y in train_loader:
            optimizer.zero_grad()
            outputs = model(b_x)
            loss = criterion(outputs, b_y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
        
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model

def _optimize_ensemble_weights(predictions: Dict[str, np.ndarray], y_true: np.ndarray, method: str = 'slsqp', sample_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    model_names = list(predictions.keys())
    if not model_names:
        return {}
    
    n_models = len(model_names)
    pred_stack = np.array([predictions[name] for name in model_names])
    best_weights = {name: 1.0/n_models for name in model_names}

    def loss_func(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        ensemble_probs = np.tensordot(w, pred_stack, axes=([0],[0]))
        ensemble_probs = np.clip(ensemble_probs, 1e-15, 1 - 1e-15)
        return log_loss(y_true, ensemble_probs, sample_weight=sample_weights)

    if method == 'slsqp' and ML_AVAILABLE:
        try:
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            bounds = [(0.0, 1.0)] * n_models
            x0 = np.ones(n_models) / n_models
            res = minimize(loss_func, x0, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-4)
            if res.success:
                optimized_w = res.x / np.sum(res.x)
                best_weights = {name: float(optimized_w[i]) for i, name in enumerate(model_names)}
                return best_weights
        except Exception:
            pass

    # Fallback: Random Search
    best_score = float('inf')
    for _ in range(500):
        w = np.random.dirichlet(np.ones(n_models))
        score = loss_func(w)
        if score < best_score:
            best_score = score
            best_weights = {name: float(w[i]) for i, name in enumerate(model_names)}
            
    return best_weights

def _load_saved_models(symbol: str, model_path: str, config: AIEnsembleStrategyParams, device) -> Optional[Dict[str, Any]]:
    safe_symbol = symbol.replace('/', '_')
    load_dir = os.path.join(model_path, safe_symbol)
    if not os.path.exists(load_dir):
        return None

    try:
        meta_path = os.path.join(load_dir, "metadata.json")
        if not os.path.exists(meta_path):
            return None

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        saved_active_features = meta.get('active_feature_columns', meta.get('feature_columns', []))
        current_available_features = FeatureProcessor.get_feature_names(config)
        
        if not set(saved_active_features).issubset(set(current_available_features)):
            return None
        
        saved_num_features = meta.get('num_features', len(saved_active_features))
        models = {}
        for name in ['gb', 'technical']:
            path = os.path.join(load_dir, f"{name}.joblib")
            if os.path.exists(path):
                models[name] = joblib.load(path)

        calibrator = None
        calib_path = os.path.join(load_dir, "calibrator.joblib")
        if os.path.exists(calib_path):
            calibrator = joblib.load(calib_path)

        drift_detector = None
        drift_path = os.path.join(load_dir, "drift_detector.joblib")
        if os.path.exists(drift_path):
            drift_detector = joblib.load(drift_path)
            
        meta_model = None
        meta_path = os.path.join(load_dir, "meta_model.joblib")
        if os.path.exists(meta_path):
            meta_model = joblib.load(meta_path)

        if 'lstm' in config.ensemble_weights.dict():
            path = os.path.join(load_dir, "lstm.pth")
            if os.path.exists(path):
                lstm = LSTMPredictor(saved_num_features, config.hyperparameters.lstm.hidden_dim, 
                                     config.hyperparameters.lstm.num_layers, 
                                     config.hyperparameters.lstm.dropout).to(device)
                lstm.load_state_dict(torch.load(path, map_location=device))
                lstm.eval()
                models['lstm'] = lstm

        if 'attention' in config.ensemble_weights.dict():
            path = os.path.join(load_dir, "attention.pth")
            if os.path.exists(path):
                attn = AttentionNetwork(saved_num_features, config.hyperparameters.attention.hidden_dim,
                                        config.hyperparameters.attention.num_layers,
                                        config.hyperparameters.attention.nhead,
                                        config.hyperparameters.attention.dropout).to(device)
                attn.load_state_dict(torch.load(path, map_location=device))
                attn.eval()
                models['attention'] = attn

        return {
            'models': models,
            'calibrator': calibrator,
            'drift_detector': drift_detector,
            'meta_model': meta_model,
            'meta': meta
        }
    except Exception:
        return None

# --- Main Class ---

class EnsembleLearner:
    """
    Manages the lifecycle of AI models: loading, inference, and state management.
    Uses a ThreadPoolExecutor to prevent blocking the main asyncio loop during CPU-bound operations.
    """
    def __init__(self, config: AIEnsembleStrategyParams):
        self.config = config
        self.symbol_models: Dict[str, Dict[str, Any]] = {}
        self.device = torch.device("cuda" if ML_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.executor = ThreadPoolExecutor(max_workers=4)
        logger.info(f"EnsembleLearner initialized. Device: {self.device}")

    async def close(self):
        self.executor.shutdown(wait=True)
        logger.info("EnsembleLearner executor shut down.")

    @property
    def is_trained(self) -> bool:
        return bool(self.symbol_models)

    def has_valid_model(self, symbol: str) -> bool:
        return symbol in self.symbol_models

    def get_last_training_time(self, symbol: str) -> Optional[datetime]:
        try:
            safe_symbol = symbol.replace('/', '_')
            meta_path = os.path.join(self.config.model_path, safe_symbol, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    data = json.load(f)
                    return datetime.fromisoformat(data['timestamp'])
        except Exception:
            pass
        return None

    async def reload_models(self, symbol: str):
        if not ML_AVAILABLE:
            return

        try:
            loop = asyncio.get_running_loop()
            loaded_data = await loop.run_in_executor(
                self.executor, 
                _load_saved_models, 
                symbol, 
                self.config.model_path, 
                self.config, 
                self.device
            )
            
            if loaded_data:
                self.symbol_models[symbol] = loaded_data
                logger.info(f"Models reloaded successfully for {symbol}")
            else:
                if symbol in self.symbol_models:
                    del self.symbol_models[symbol]
                logger.warning(f"Could not load valid models for {symbol} (possible config mismatch).")

        except Exception as e:
            logger.error(f"Failed to reload models for {symbol}: {e}")

    async def warmup_models(self, symbols: List[str]):
        logger.info(f"Warming up models for {len(symbols)} symbols...")
        tasks = [self.reload_models(symbol) for symbol in symbols]
        await asyncio.gather(*tasks)
        loaded_count = sum(1 for s in symbols if s in self.symbol_models)
        logger.info(f"Warmup complete. Loaded {loaded_count}/{len(symbols)} models.")

    async def rollback_model(self, symbol: str) -> bool:
        safe_symbol = symbol.replace('/', '_')
        final_dir = os.path.join(self.config.model_path, safe_symbol)
        backup_dir = os.path.join(self.config.model_path, f"{safe_symbol}_backup")
        
        if not os.path.exists(backup_dir):
            logger.warning(f"No backup found for {symbol}. Cannot rollback.")
            return False
            
        try:
            temp_hold = os.path.join(self.config.model_path, f"{safe_symbol}_bad_hold")
            if os.path.exists(final_dir):
                os.rename(final_dir, temp_hold)
            
            os.rename(backup_dir, final_dir)
            
            if os.path.exists(temp_hold):
                shutil.rmtree(temp_hold)
                
            logger.info(f"Rolled back model for {symbol} from backup.")
            await self.reload_models(symbol)
            return True
        except Exception as e:
            logger.error(f"Rollback failed for {symbol}: {e}")
            return False

    def _predict_sync(self, df: pd.DataFrame, symbol: str, regime: Optional[str] = None, leader_df: Optional[pd.DataFrame] = None, custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        entry = self.symbol_models.get(symbol)
        if not entry:
            return {}

        models = entry['models']
        meta = entry['meta']
        calibrator = entry.get('calibrator')
        drift_detector = entry.get('drift_detector')
        meta_model = entry.get('meta_model')
        
        try:
            df_proc = FeatureProcessor.process_data(df, self.config, leader_df=leader_df)
            active_features = meta.get('active_feature_columns', meta.get('feature_columns'))
            
            if not set(active_features).issubset(set(df_proc.columns)):
                logger.error(f"Inference failed: Missing features for {symbol}. Model expects {active_features}")
                return {}
                
            df_proc = df_proc[active_features]
            X = df_proc.iloc[-1:].values
            X = InputSanitizer.sanitize(X)
            
            is_anomaly = False
            anomaly_score = 0.0
            if drift_detector and self.config.drift.enabled:
                try:
                    pred = drift_detector.predict(X)[0]
                    is_anomaly = (pred == -1)
                    anomaly_score = drift_detector.decision_function(X)[0]
                except Exception as e:
                    logger.error(f"Drift detection failed: {e}")

            seq_len = self.config.features.sequence_length
            if len(df_proc) >= seq_len:
                X_seq = df_proc.iloc[-seq_len:].values.reshape(1, seq_len, -1)
                X_seq = InputSanitizer.sanitize(X_seq)
            else:
                X_seq = None

            votes = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
            config_weights = self.config.ensemble_weights
            
            optimized_weights = meta.get('optimized_weights')
            regime_weights = meta.get('regime_weights', {})
            
            active_weight_map = {}
            if custom_weights:
                active_weight_map = custom_weights
            elif config_weights.use_regime_specific_weights and regime and regime in regime_weights:
                active_weight_map = regime_weights[regime]
            elif config_weights.auto_tune and optimized_weights:
                active_weight_map = optimized_weights
            
            def get_weight(name, default_weight):
                if active_weight_map:
                    return active_weight_map.get(name, 0.0)
                return default_weight

            model_predictions = []
            individual_preds = {}
            
            # --- Partial Ensemble Tolerance ---
            # If one model fails, we continue with the others.
            
            if 'gb' in models:
                try:
                    probs = models['gb'].predict_proba(X)[0]
                    w = get_weight('gb', config_weights.xgboost)
                    votes['sell'] += probs[0] * w
                    votes['hold'] += probs[1] * w
                    votes['buy'] += probs[2] * w
                    model_predictions.append((w, probs))
                    individual_preds['gb'] = probs
                except Exception as e:
                    logger.warning(f"GB inference failed for {symbol}: {e}")

            if 'technical' in models:
                try:
                    probs = models['technical'].predict_proba(X)[0]
                    w = get_weight('technical', config_weights.technical_ensemble)
                    votes['sell'] += probs[0] * w
                    votes['hold'] += probs[1] * w
                    votes['buy'] += probs[2] * w
                    model_predictions.append((w, probs))
                    individual_preds['technical'] = probs
                except Exception as e:
                    logger.warning(f"Technical inference failed for {symbol}: {e}")

            if X_seq is not None:
                with torch.no_grad():
                    tensor_in = torch.FloatTensor(X_seq).to(self.device)
                    
                    if 'lstm' in models:
                        try:
                            logits = models['lstm'](tensor_in)
                            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                            w = get_weight('lstm', config_weights.lstm)
                            votes['sell'] += probs[0] * w
                            votes['hold'] += probs[1] * w
                            votes['buy'] += probs[2] * w
                            model_predictions.append((w, probs))
                            individual_preds['lstm'] = probs
                        except Exception as e:
                            logger.warning(f"LSTM inference failed for {symbol}: {e}")
                        
                    if 'attention' in models:
                        try:
                            logits = models['attention'](tensor_in)
                            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                            w = get_weight('attention', config_weights.attention)
                            votes['sell'] += probs[0] * w
                            votes['hold'] += probs[1] * w
                            votes['buy'] += probs[2] * w
                            model_predictions.append((w, probs))
                            individual_preds['attention'] = probs
                        except Exception as e:
                            logger.warning(f"Attention inference failed for {symbol}: {e}")

            total_weight = sum(votes.values())
            if total_weight > 0:
                for k in votes:
                    votes[k] /= total_weight
            else:
                # Fallback if all models failed
                return {'action': 'hold', 'confidence': 0.0}

            if calibrator:
                try:
                    raw_probs = np.array([[votes['sell'], votes['hold'], votes['buy']]])
                    calibrated_probs = calibrator.predict(raw_probs)[0]
                    votes['sell'] = calibrated_probs[0]
                    votes['hold'] = calibrated_probs[1]
                    votes['buy'] = calibrated_probs[2]
                except Exception as e:
                    logger.warning(f"Calibration failed for {symbol}: {e}")

            best_action = max(votes, key=votes.get)
            confidence = votes[best_action]

            if model_predictions and config_weights.disagreement_penalty > 0:
                action_map = {'sell': 0, 'hold': 1, 'buy': 2}
                best_action_idx = action_map[best_action]
                relevant_probs = [p[best_action_idx] for w, p in model_predictions if w > 0]
                if len(relevant_probs) > 1:
                    std_dev = np.std(relevant_probs)
                    penalty = std_dev * config_weights.disagreement_penalty
                    confidence = max(0.0, confidence - penalty)

            # --- Meta-Labeling Inference ---
            meta_prob = None
            if meta_model and self.config.meta_labeling.enabled and best_action != 'hold':
                try:
                    X_meta = X
                    if self.config.meta_labeling.use_primary_confidence_feature:
                        raw_conf = votes[best_action]
                        X_meta = np.hstack([X, np.array([[raw_conf]])])
                    
                    meta_probs = meta_model.predict_proba(X_meta)[0]
                    meta_prob = meta_probs[1] # Probability of success (Class 1)
                    
                    if meta_prob < self.config.meta_labeling.probability_threshold:
                        best_action = 'hold'
                        confidence = 0.0
                except Exception as e:
                    logger.error(f"Meta-model inference failed: {e}")

            top_features = {}
            if 'gb' in models and hasattr(models['gb'], 'feature_importances_'):
                imps = models['gb'].feature_importances_
                cols = active_features
                indices = np.argsort(imps)[::-1][:5]
                for idx in indices:
                    if idx < len(cols):
                        top_features[cols[idx]] = float(imps[idx])

            optimized_threshold = meta.get('optimized_threshold')
            regime_thresholds = meta.get('regime_thresholds', {})
            
            if regime and regime in regime_thresholds:
                optimized_threshold = regime_thresholds[regime]

            return {
                'action': best_action,
                'confidence': round(confidence, 4),
                'model_version': meta['timestamp'],
                'active_weights': votes,
                'top_features': top_features,
                'metrics': meta.get('metrics', {}),
                'optimized_weights': active_weight_map if active_weight_map else None,
                'is_anomaly': is_anomaly,
                'anomaly_score': float(anomaly_score),
                'optimized_threshold': optimized_threshold,
                'individual_predictions': individual_preds,
                'meta_probability': float(meta_prob) if meta_prob is not None else None
            }
        except Exception as e:
            logger.error(f"Critical inference error for {symbol}: {e}", exc_info=True)
            return {'action': 'hold', 'confidence': 0.0}

    async def predict(self, df: pd.DataFrame, symbol: str, regime: Optional[str] = None, leader_df: Optional[pd.DataFrame] = None, custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        if symbol not in self.symbol_models:
            await self.reload_models(symbol)
            if symbol not in self.symbol_models:
                return {}

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._predict_sync, df, symbol, regime, leader_df, custom_weights)
