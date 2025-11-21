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
    from sklearn.metrics import precision_score, log_loss, accuracy_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.isotonic import IsotonicRegression
    from xgboost import XGBClassifier
    from scipy.optimize import minimize
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    # Dummy classes to prevent ImportErrors
    class VotingClassifier: pass
    class RandomForestClassifier: pass
    class LogisticRegression: pass
    class XGBClassifier: pass
    class TimeSeriesSplit: pass
    class IsotonicRegression: pass
    class IsolationForest: pass
    def precision_score(*args, **kwargs): return 0.0
    def log_loss(*args, **kwargs): return 0.0
    def accuracy_score(*args, **kwargs): return 0.0
    def minimize(*args, **kwargs): return None
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

# --- Standalone Helper Functions (Pickle-safe for ProcessPoolExecutor) ---

def _create_fresh_models(config: AIEnsembleStrategyParams, num_features: int, device) -> Dict[str, Any]:
    hp = config.hyperparameters
    xgb_config = hp.xgboost
    rf_config = hp.random_forest
    lr_config = hp.logistic_regression
    lstm_config = hp.lstm # Reusing config struct for TCN params
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

    if ML_AVAILABLE:
        # TCN Configuration: Create channel list [hidden, hidden, ...] based on num_layers
        num_channels = [lstm_config.hidden_dim] * lstm_config.num_layers
        
        models['lstm'] = TCNPredictor(
            num_inputs=num_features, 
            num_channels=num_channels,
            kernel_size=2,
            dropout=lstm_config.dropout
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

def _optimize_ensemble_weights(predictions: Dict[str, np.ndarray], y_true: np.ndarray, method: str = 'slsqp') -> Dict[str, float]:
    model_names = list(predictions.keys())
    if not model_names:
        return {}
    
    n_models = len(model_names)
    pred_stack = np.array([predictions[name] for name in model_names])
    
    # Default to equal weights
    best_weights = {name: 1.0/n_models for name in model_names}

    def loss_func(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        # Weighted average of probabilities
        ensemble_probs = np.tensordot(w, pred_stack, axes=([0],[0]))
        # Clip to avoid log(0)
        ensemble_probs = np.clip(ensemble_probs, 1e-15, 1 - 1e-15)
        return log_loss(y_true, ensemble_probs)

    if method == 'slsqp' and ML_AVAILABLE:
        try:
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            bounds = [(0.0, 1.0)] * n_models
            x0 = np.ones(n_models) / n_models
            res = minimize(loss_func, x0, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-4)
            if res.success:
                optimized_w = res.x / np.sum(res.x)
                # Ensure non-negative and sum to 1 (numerical stability)
                optimized_w = np.maximum(optimized_w, 0)
                optimized_w /= optimized_w.sum()
                best_weights = {name: float(optimized_w[i]) for i, name in enumerate(model_names)}
                return best_weights
        except Exception:
            pass

    return best_weights

# --- The Critical Training Task (Process Safe) ---

def train_ensemble_task(symbol: str, df: pd.DataFrame, config: AIEnsembleStrategyParams, leader_df: Optional[pd.DataFrame] = None) -> bool:
    """
    Top-level function to be run in a ProcessPoolExecutor.
    Performs the full training pipeline: Feature Eng -> Labeling -> Training -> Optimization -> Saving.
    """
    if not ML_AVAILABLE:
        return False

    try:
        # 1. Feature Engineering
        df_proc = FeatureProcessor.process_data(df, config, leader_df=leader_df)
        labels = FeatureProcessor.create_labels(df, config)
        
        # Align features and labels
        common_index = df_proc.index.intersection(labels.index)
        if len(common_index) < 200:
            with open("training_errors.log", "a") as f:
                f.write(f"{datetime.now()} - Insufficient data for {symbol}: {len(common_index)} rows\n")
            return False
            
        X = df_proc.loc[common_index].values
        y = labels.loc[common_index].values.astype(int)
        
        X = InputSanitizer.sanitize(X)

        # CRITICAL: Check for class diversity. 
        # If the market has been flat, we might only have 'HOLD' labels.
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            with open("training_errors.log", "a") as f:
                f.write(f"{datetime.now()} - Single class detected for {symbol} (Classes: {unique_classes}). Aborting training.\n")
            return False
        
        # 2. Walk-Forward Split (Last 20% for Validation/Optimization)
        split_idx = int(len(X) * (1 - config.training.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Sequence generation for LSTM/Attention
        seq_len = config.features.sequence_length
        X_train_seq, y_train_seq = FeatureProcessor.create_sequences(X_train, y_train, seq_len)
        X_val_seq, y_val_seq = FeatureProcessor.create_sequences(X_val, y_val, seq_len)

        # 3. Model Initialization
        # Force CPU for background process to avoid CUDA context issues during multiprocessing
        device = torch.device("cpu") 
        num_features = X.shape[1]
        models = _create_fresh_models(config, num_features, device)
        
        # 4. Training Loop
        # Train XGBoost
        models['gb'].fit(X_train, y_train)
        
        # Train Technical Ensemble
        models['technical'].fit(X_train, y_train)
        
        # Train PyTorch Models
        class_weights = None
        if config.training.use_class_weighting:
            counts = np.bincount(y_train)
            # Handle missing classes in bincount if any
            if len(counts) < 3:
                # Pad counts to length 3 (Sell, Hold, Buy)
                padded_counts = np.zeros(3)
                padded_counts[:len(counts)] = counts
                counts = padded_counts
            
            weights = 1.0 / (counts + 1e-6)
            class_weights = torch.FloatTensor(weights / weights.sum()).to(device)

        if 'lstm' in models and len(X_train_seq) > 0:
            models['lstm'] = _train_torch_model(models['lstm'], X_train_seq, y_train_seq, X_val_seq, y_val_seq, config, device, class_weights)
            
        if 'attention' in models and len(X_train_seq) > 0:
            models['attention'] = _train_torch_model(models['attention'], X_train_seq, y_train_seq, X_val_seq, y_val_seq, config, device, class_weights)

        # 5. Validation & Weight Optimization
        val_preds = {}
        val_preds['gb'] = models['gb'].predict_proba(X_val)
        val_preds['technical'] = models['technical'].predict_proba(X_val)
        
        if 'lstm' in models and len(X_val_seq) > 0:
            models['lstm'].eval()
            with torch.no_grad():
                logits = models['lstm'](torch.FloatTensor(X_val_seq).to(device))
                # Pad predictions to match X_val length (sequences consume start data)
                pad_len = len(X_val) - len(X_val_seq)
                probs = F.softmax(logits, dim=1).numpy()
                # Simple padding: repeat first prediction
                pad_arr = np.tile(probs[0], (pad_len, 1))
                val_preds['lstm'] = np.vstack([pad_arr, probs])

        if 'attention' in models and len(X_val_seq) > 0:
            models['attention'].eval()
            with torch.no_grad():
                logits = models['attention'](torch.FloatTensor(X_val_seq).to(device))
                pad_len = len(X_val) - len(X_val_seq)
                probs = F.softmax(logits, dim=1).numpy()
                pad_arr = np.tile(probs[0], (pad_len, 1))
                val_preds['attention'] = np.vstack([pad_arr, probs])

        # Optimize Weights
        optimized_weights = _optimize_ensemble_weights(val_preds, y_val, method=config.ensemble_weights.optimization_method)
        
        # 6. Threshold Optimization (Maximize Sharpe on Validation)
        # Calculate weighted ensemble probabilities
        ensemble_probs = np.zeros((len(X_val), 3))
        for name, w in optimized_weights.items():
            if name in val_preds:
                ensemble_probs += w * val_preds[name]
        
        # Find best threshold
        best_threshold = config.confidence_threshold
        best_score = -float('inf')
        
        if config.training.optimize_entry_threshold:
            returns = df['close'].pct_change().loc[common_index].values[split_idx:]
            # Handle length mismatch due to sequence generation or alignment
            min_len = min(len(ensemble_probs), len(returns))
            ensemble_probs = ensemble_probs[-min_len:]
            returns = returns[-min_len:]
            
            for thresh in np.arange(0.5, 0.9, 0.05):
                signals = np.zeros(len(ensemble_probs))
                signals[ensemble_probs[:, 2] > thresh] = 1 # Buy
                signals[ensemble_probs[:, 0] > thresh] = -1 # Sell
                
                strategy_returns = signals * returns
                if np.std(strategy_returns) > 0:
                    sharpe = np.mean(strategy_returns) / np.std(strategy_returns)
                    if sharpe > best_score:
                        best_score = sharpe
                        best_threshold = float(thresh)

        # 7. Drift Detector Training (Isolation Forest)
        drift_detector = None
        if config.drift.enabled:
            drift_detector = IsolationForest(contamination=config.drift.contamination, random_state=42)
            drift_detector.fit(X_train)

        # 8. Persistence
        safe_symbol = symbol.replace('/', '_')
        save_dir = os.path.join(config.model_path, safe_symbol)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save Metadata
        meta = {
            'timestamp': datetime.now().isoformat(),
            'feature_columns': FeatureProcessor.get_feature_names(config),
            'active_feature_columns': list(df_proc.columns),
            'num_features': num_features,
            'optimized_weights': optimized_weights,
            'optimized_threshold': best_threshold,
            'metrics': {
                'validation_accuracy': float(accuracy_score(y_val, np.argmax(ensemble_probs, axis=1))),
                'validation_sharpe': float(best_score) if best_score != -float('inf') else 0.0
            }
        }
        
        with open(os.path.join(save_dir, "metadata.json"), 'w') as f:
            json.dump(meta, f, indent=2)
            
        # Save Models
        joblib.dump(models['gb'], os.path.join(save_dir, "gb.joblib"))
        joblib.dump(models['technical'], os.path.join(save_dir, "technical.joblib"))
        if drift_detector:
            joblib.dump(drift_detector, os.path.join(save_dir, "drift_detector.joblib"))
            
        if 'lstm' in models:
            # Move to CPU before saving to ensure portability
            models['lstm'].to('cpu')
            torch.save(models['lstm'].state_dict(), os.path.join(save_dir, "lstm.pth"))
        if 'attention' in models:
            models['attention'].to('cpu')
            torch.save(models['attention'].state_dict(), os.path.join(save_dir, "attention.pth"))
            
        return True

    except Exception as e:
        # Log to a file since we are in a separate process
        with open("training_errors.log", "a") as f:
            f.write(f"{datetime.now()} - Error training {symbol}: {str(e)}\n")
        return False

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
        saved_num_features = meta.get('num_features', len(saved_active_features))
        
        models = {}
        for name in ['gb', 'technical']:
            path = os.path.join(load_dir, f"{name}.joblib")
            if os.path.exists(path):
                models[name] = joblib.load(path)

        drift_detector = None
        drift_path = os.path.join(load_dir, "drift_detector.joblib")
        if os.path.exists(drift_path):
            drift_detector = joblib.load(drift_path)

        if 'lstm' in config.ensemble_weights.dict():
            path = os.path.join(load_dir, "lstm.pth")
            if os.path.exists(path):
                # Reconstruct TCN architecture
                num_channels = [config.hyperparameters.lstm.hidden_dim] * config.hyperparameters.lstm.num_layers
                tcn = TCNPredictor(saved_num_features, num_channels, 
                                   kernel_size=2, 
                                   dropout=config.hyperparameters.lstm.dropout).to(device)
                tcn.load_state_dict(torch.load(path, map_location=device))
                tcn.eval()
                
                # Dynamic Quantization for CPU inference speedup
                if device.type == 'cpu':
                    try:
                        tcn = torch.quantization.quantize_dynamic(
                            tcn, {torch.nn.Linear, torch.nn.Conv1d}, dtype=torch.qint8
                        )
                    except Exception as e:
                        logger.warning(f"Quantization failed for TCN: {e}")
                
                models['lstm'] = tcn

        if 'attention' in config.ensemble_weights.dict():
            path = os.path.join(load_dir, "attention.pth")
            if os.path.exists(path):
                attn = AttentionNetwork(saved_num_features, config.hyperparameters.attention.hidden_dim,
                                        config.hyperparameters.attention.num_layers,
                                        config.hyperparameters.attention.nhead,
                                        config.hyperparameters.attention.dropout).to(device)
                attn.load_state_dict(torch.load(path, map_location=device))
                attn.eval()
                
                if device.type == 'cpu':
                    try:
                        attn = torch.quantization.quantize_dynamic(
                            attn, {torch.nn.Linear}, dtype=torch.qint8
                        )
                    except Exception as e:
                        logger.warning(f"Quantization failed for Attention: {e}")

                models['attention'] = attn

        return {
            'models': models,
            'drift_detector': drift_detector,
            'meta': meta
        }
    except Exception as e:
        logger.error(f"Failed to load models for {symbol}: {e}")
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

    def _prepare_features(self, df: pd.DataFrame, symbol: str, leader_df: Optional[pd.DataFrame]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        """Prepares features for inference."""
        entry = self.symbol_models.get(symbol)
        if not entry: return None, None, []
        
        meta = entry['meta']
        df_proc = FeatureProcessor.process_data(df, self.config, leader_df=leader_df)
        active_features = meta.get('active_feature_columns', meta.get('feature_columns'))
        
        # Ensure columns match training
        missing = set(active_features) - set(df_proc.columns)
        if missing:
            logger.error(f"Inference failed: Missing features for {symbol}. Missing: {missing}")
            return None, None, []
            
        df_proc = df_proc[active_features]
        X = df_proc.iloc[-1:].values
        X = InputSanitizer.sanitize(X)
        
        seq_len = self.config.features.sequence_length
        if len(df_proc) >= seq_len:
            X_seq = df_proc.iloc[-seq_len:].values.reshape(1, seq_len, -1)
            X_seq = InputSanitizer.sanitize(X_seq)
        else:
            X_seq = None
            
        return X, X_seq, active_features

    def _get_model_votes(self, models: Dict[str, Any], X: np.ndarray, X_seq: Optional[np.ndarray], 
                         config_weights: Any, active_weight_map: Dict[str, float]) -> Tuple[Dict[str, float], List[Tuple[float, np.ndarray]], Dict[str, Any]]:
        """Aggregates predictions from all ensemble models."""
        votes = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
        model_predictions = []
        individual_preds = {}
        
        def get_weight(name, default_weight):
            if active_weight_map:
                return active_weight_map.get(name, 0.0)
            return default_weight

        # 1. XGBoost
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
                logger.warning(f"GB inference failed: {e}")

        # 2. Technical Ensemble
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
                logger.warning(f"Technical inference failed: {e}")

        # 3. Deep Learning (LSTM/Attention)
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
                        logger.warning(f"LSTM inference failed: {e}")
                    
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
                        logger.warning(f"Attention inference failed: {e}")

        return votes, model_predictions, individual_preds

    def _predict_sync(self, df: pd.DataFrame, symbol: str, regime: Optional[str] = None, leader_df: Optional[pd.DataFrame] = None, custom_weights: Optional[Dict[str, float]] = None) -> AIInferenceResult:
        entry = self.symbol_models.get(symbol)
        if not entry:
            return AIInferenceResult(action='hold', confidence=0.0, model_version='none', active_weights={}, top_features={}, metrics={})

        models = entry['models']
        meta = entry['meta']
        drift_detector = entry.get('drift_detector')
        
        try:
            # 1. Feature Preparation
            X, X_seq, active_features = self._prepare_features(df, symbol, leader_df)
            if X is None:
                return AIInferenceResult(action='hold', confidence=0.0, model_version='none', active_weights={}, top_features={}, metrics={})

            # 2. Drift Detection
            is_anomaly = False
            anomaly_score = 0.0
            if drift_detector and self.config.drift.enabled:
                try:
                    pred = drift_detector.predict(X)[0]
                    is_anomaly = (pred == -1)
                    anomaly_score = drift_detector.decision_function(X)[0]
                except Exception as e:
                    logger.error(f"Drift detection failed: {e}")

            # 3. Weight Resolution
            config_weights = self.config.ensemble_weights
            optimized_weights = meta.get('optimized_weights')
            
            active_weight_map = {}
            if custom_weights:
                active_weight_map = custom_weights
            elif config_weights.auto_tune and optimized_weights:
                active_weight_map = optimized_weights

            # 4. Model Inference & Voting
            votes, model_predictions, individual_preds = self._get_model_votes(models, X, X_seq, config_weights, active_weight_map)

            total_weight = sum(votes.values())
            if total_weight > 0:
                for k in votes:
                    votes[k] /= total_weight
            else:
                return AIInferenceResult(action='hold', confidence=0.0, model_version=meta['timestamp'], active_weights={}, top_features={}, metrics={})

            best_action = max(votes, key=votes.get)
            confidence = votes[best_action]

            # 5. Disagreement Penalty
            if model_predictions and config_weights.disagreement_penalty > 0:
                action_map = {'sell': 0, 'hold': 1, 'buy': 2}
                best_action_idx = action_map[best_action]
                relevant_probs = [p[best_action_idx] for w, p in model_predictions if w > 0]
                if len(relevant_probs) > 1:
                    std_dev = np.std(relevant_probs)
                    penalty = std_dev * config_weights.disagreement_penalty
                    confidence = max(0.0, confidence - penalty)

            # 6. Feature Importance Extraction
            top_features = {}
            if 'gb' in models and hasattr(models['gb'], 'feature_importances_'):
                imps = models['gb'].feature_importances_
                cols = active_features
                indices = np.argsort(imps)[::-1][:5]
                for idx in indices:
                    if idx < len(cols):
                        top_features[cols[idx]] = float(imps[idx])

            optimized_threshold = meta.get('optimized_threshold')

            # Convert numpy arrays in individual_preds to lists for Pydantic serialization
            serializable_preds = {}
            for k, v in individual_preds.items():
                if isinstance(v, np.ndarray):
                    serializable_preds[k] = v.tolist()
                else:
                    serializable_preds[k] = v

            return AIInferenceResult(
                action=best_action,
                confidence=round(confidence, 4),
                model_version=meta['timestamp'],
                active_weights=votes,
                top_features=top_features,
                metrics=meta.get('metrics', {}),
                optimized_weights=active_weight_map if active_weight_map else None,
                is_anomaly=is_anomaly,
                anomaly_score=float(anomaly_score),
                optimized_threshold=optimized_threshold,
                individual_predictions=serializable_preds
            )
        except Exception as e:
            logger.error(f"Critical inference error for {symbol}: {e}", exc_info=True)
            return AIInferenceResult(action='hold', confidence=0.0, model_version='error', active_weights={}, top_features={}, metrics={})

    async def predict(self, df: pd.DataFrame, symbol: str, regime: Optional[str] = None, leader_df: Optional[pd.DataFrame] = None, custom_weights: Optional[Dict[str, float]] = None) -> AIInferenceResult:
        if symbol not in self.symbol_models:
            await self.reload_models(symbol)
            if symbol not in self.symbol_models:
                return AIInferenceResult(action='hold', confidence=0.0, model_version='none', active_weights={}, top_features={}, metrics={})

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._predict_sync, df, symbol, regime, leader_df, custom_weights)
