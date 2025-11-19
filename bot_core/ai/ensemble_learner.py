import os
import joblib
import asyncio
import copy
import json
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from bot_core.logger import get_logger
from bot_core.config import AIEnsembleStrategyParams
from bot_core.ai.models import LSTMPredictor, AttentionNetwork
from bot_core.ai.feature_processor import FeatureProcessor

# ML Imports with safe fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_score, classification_report
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    from xgboost import XGBClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    # Define dummy classes if ML libraries are not available
    class VotingClassifier: pass
    class RandomForestClassifier: pass
    class LogisticRegression: pass
    class XGBClassifier: pass
    class RandomizedSearchCV: pass
    class TimeSeriesSplit: pass
    def precision_score(*args, **kwargs): return 0.0
    def classification_report(*args, **kwargs): return ""
    class nn:
        class Module: pass
    class optim: pass
    class TensorDataset: pass
    class DataLoader: pass

logger = get_logger(__name__)

# --- Standalone Helper Functions (Pickle-safe for Multiprocessing) ---

def _create_fresh_models(config: AIEnsembleStrategyParams, device) -> Dict[str, Any]:
    """Creates a fresh set of untrained model instances based on configuration."""
    num_features = len(config.feature_columns)
    hp = config.hyperparameters
    xgb_config = hp.xgboost
    rf_config = hp.random_forest
    lr_config = hp.logistic_regression
    lstm_config = hp.lstm
    attn_config = hp.attention

    return {
        'lstm': LSTMPredictor(
            num_features, lstm_config.hidden_dim, lstm_config.num_layers, lstm_config.dropout
        ).to(device),
        'gb': XGBClassifier(
            n_estimators=xgb_config.n_estimators,
            max_depth=xgb_config.max_depth,
            learning_rate=xgb_config.learning_rate,
            subsample=xgb_config.subsample,
            colsample_bytree=xgb_config.colsample_bytree,
            random_state=42, use_label_encoder=False, eval_metric='logloss'
        ),
        'attention': AttentionNetwork(
            num_features, attn_config.hidden_dim, attn_config.num_layers, attn_config.nhead, attn_config.dropout
        ).to(device),
        'technical': VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(
                n_estimators=rf_config.n_estimators,
                max_depth=rf_config.max_depth,
                min_samples_leaf=rf_config.min_samples_leaf,
                random_state=42
            )),
            ('lr', LogisticRegression(
                max_iter=lr_config.max_iter,
                C=lr_config.C,
                random_state=42
            ))
        ], voting='soft')
    }

def _optimize_hyperparameters(model, X, y, param_dist, n_iter, logger_instance, symbol):
    """Runs RandomizedSearchCV to find better hyperparameters."""
    try:
        # Use TimeSeriesSplit to prevent data leakage (future peeking)
        tscv = TimeSeriesSplit(n_splits=3)
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='neg_log_loss',
            cv=tscv,
            n_jobs=1, # Already in a subprocess
            random_state=42,
            verbose=0
        )
        search.fit(X, y)
        logger_instance.info("Hyperparameter optimization complete", symbol=symbol, best_params=search.best_params_)
        return search.best_estimator_
    except Exception as e:
        logger_instance.warning("Hyperparameter optimization failed, using default.", symbol=symbol, error=str(e))
        return model

def _train_pytorch_model(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor, config: AIEnsembleStrategyParams):
    train_cfg = config.training
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size)

    optimizer = optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(train_cfg.epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= train_cfg.early_stopping_patience:
                break
    
    if best_model_state:
        model.load_state_dict(best_model_state)

def _get_ensemble_prediction_static(models: Dict[str, Any], X_flat: np.ndarray, X_seq: torch.Tensor, weights: List[float]) -> np.ndarray:
    """
    Static version of prediction logic for validation during training.
    X_flat: Flattened sequence history for tree models (N, seq_len * features)
    X_seq: 3D Tensor for deep learning models (N, seq_len, features)
    weights: List of weights [xgboost, technical, lstm, attention]
    """
    predictions = []
    
    # Tree models now receive the full flattened history
    gb_pred = models['gb'].predict_proba(X_flat)
    predictions.append(gb_pred)
    tech_pred = models['technical'].predict_proba(X_flat)
    predictions.append(tech_pred)

    with torch.no_grad():
        models['lstm'].eval()
        models['attention'].eval()
        if X_seq.dim() == 2:
             X_seq = X_seq.unsqueeze(0)
        lstm_pred = models['lstm'](X_seq).cpu().numpy()
        predictions.append(lstm_pred)
        attn_pred = models['attention'](X_seq).cpu().numpy()
        predictions.append(attn_pred)

    stacked_preds = np.array(predictions)
    ensemble_pred = np.average(stacked_preds, weights=weights, axis=0)
    return ensemble_pred

def _atomic_save(obj: Any, path: str, method: str = 'joblib'):
    """Saves an object to a temporary file and then atomically renames it."""
    temp_path = f"{path}.tmp"
    try:
        if method == 'joblib':
            joblib.dump(obj, temp_path)
        elif method == 'torch':
            torch.save(obj, temp_path)
        elif method == 'json':
            with open(temp_path, 'w') as f:
                json.dump(obj, f)
        
        # Atomic replacement
        os.replace(temp_path, path)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e

def _save_models_to_disk(symbol: str, models: Dict[str, Any], config: AIEnsembleStrategyParams, learned_weights: Optional[List[float]] = None, extra_meta: Dict[str, Any] = None):
    symbol_path_str = symbol.replace('/', '_')
    save_path = os.path.join(config.model_path, symbol_path_str)
    os.makedirs(save_path, exist_ok=True)
    
    metadata = {
        'feature_columns': config.feature_columns,
        'hyperparameters': config.hyperparameters.dict(),
        'timestamp': str(pd.Timestamp.utcnow()),
        'learned_weights': learned_weights
    }
    if extra_meta:
        metadata.update(extra_meta)
    
    # Use atomic saves to prevent corruption if read occurs during write
    _atomic_save(metadata, os.path.join(save_path, "metadata.json"), method='json')
    _atomic_save(models['gb'], os.path.join(save_path, "gb_model.pkl"), method='joblib')
    _atomic_save(models['technical'], os.path.join(save_path, "technical_model.pkl"), method='joblib')
    _atomic_save(models['lstm'].state_dict(), os.path.join(save_path, "lstm_model.pth"), method='torch')
    _atomic_save(models['attention'].state_dict(), os.path.join(save_path, "attention_model.pth"), method='torch')

# --- Main Training Task (Executed in ProcessPool) ---

def train_ensemble_task(symbol: str, df: pd.DataFrame, config: AIEnsembleStrategyParams) -> bool:
    """
    Standalone function to train models in a separate process.
    Returns True if training was successful and models were saved.
    """
    # Re-initialize logger for the worker process
    worker_logger = get_logger(f"trainer_{symbol.replace('/', '_')}")
    worker_logger.info("Starting model training task in separate process", symbol=symbol)
    
    try:
        if not ML_AVAILABLE:
            worker_logger.error("ML libraries not available in worker process.")
            return False

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. Create labels using FeatureProcessor
        labels = FeatureProcessor.create_labels(df, config)
        
        # 2. Normalize features using FeatureProcessor
        normalized_df = FeatureProcessor.normalize(df, config)
        
        # Align features and labels
        valid_indices = normalized_df.dropna().index.intersection(labels.index)
        if len(valid_indices) < 100:
            worker_logger.warning("Insufficient data after normalization.", symbol=symbol)
            return False

        X_normalized = normalized_df.loc[valid_indices].values
        y = labels.loc[valid_indices].values

        if len(np.unique(y)) < 3:
            worker_logger.warning("Training data has fewer than 3 unique labels.", symbol=symbol)
            return False

        # 3. Create sequences using FeatureProcessor
        X_seq, y_seq = FeatureProcessor.create_sequences(X_normalized, y, config.features.sequence_length)
        if len(X_seq) == 0:
            return False

        # 4. Sequential Split
        split_idx = int(len(X_seq) * (1 - config.training.validation_split))
        X_train_seq = X_seq[:split_idx]
        y_train_seq = y_seq[:split_idx]
        X_val_seq = X_seq[split_idx:]
        y_val_seq = y_seq[split_idx:]

        if len(X_train_seq) == 0 or len(X_val_seq) == 0:
            return False

        # Flatten sequences for tree models: (N, seq_len, features) -> (N, seq_len * features)
        X_train_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
        y_train_flat = y_train_seq
        X_val_flat = X_val_seq.reshape(X_val_seq.shape[0], -1)
        y_val_flat = y_val_seq
        
        # 5. Create Models
        models = _create_fresh_models(config, device)

        # --- Hyperparameter Optimization (Optional) ---
        if config.training.auto_tune_models:
            worker_logger.info("Running hyperparameter optimization...", symbol=symbol)
            
            # XGBoost Optimization
            xgb_params = {
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'n_estimators': [50, 100, 200],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
            models['gb'] = _optimize_hyperparameters(
                models['gb'], X_train_flat, y_train_flat, xgb_params, 
                config.training.n_iter_search, worker_logger, symbol
            )

            # Random Forest Optimization (Standalone first)
            rf_params = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_leaf': [1, 2, 4, 10]
            }
            # Extract RF from VotingClassifier or create new
            rf_base = RandomForestClassifier(random_state=42)
            optimized_rf = _optimize_hyperparameters(
                rf_base, X_train_flat, y_train_flat, rf_params,
                config.training.n_iter_search, worker_logger, symbol
            )
            
            # Reconstruct VotingClassifier with optimized RF
            lr_base = models['technical'].estimators[1][1] # Get existing LR
            models['technical'] = VotingClassifier(estimators=[
                ('rf', optimized_rf),
                ('lr', lr_base)
            ], voting='soft')

        # 6. Train Models
        worker_logger.info("Training scikit-learn models with flattened history...", symbol=symbol)
        # Note: If HPO ran, models['gb'] is already fitted on X_train_flat by RandomizedSearchCV (refit=True)
        # But VotingClassifier needs to be fitted.
        if not config.training.auto_tune_models:
            models['gb'].fit(X_train_flat, y_train_flat)
        
        models['technical'].fit(X_train_flat, y_train_flat)
        
        worker_logger.info("Training PyTorch models...", symbol=symbol)
        X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
        y_train_tensor = torch.LongTensor(y_train_seq).to(device)
        X_val_tensor = torch.FloatTensor(X_val_seq).to(device)
        y_val_tensor = torch.LongTensor(y_val_seq).to(device)
        
        _train_pytorch_model(models['lstm'], X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, config)
        _train_pytorch_model(models['attention'], X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, config)

        # 7. Auto-Tune Ensemble Weights (Auto-ML)
        weights_config = config.ensemble_weights
        default_weights = [
            weights_config.xgboost,
            weights_config.technical_ensemble,
            weights_config.lstm,
            weights_config.attention
        ]
        
        final_weights = default_weights
        
        if weights_config.auto_tune:
            worker_logger.info("Auto-tuning ensemble weights based on validation performance...", symbol=symbol)
            
            # Get individual predictions on validation set
            preds_map = {}
            preds_map['gb'] = models['gb'].predict_proba(X_val_flat)
            preds_map['technical'] = models['technical'].predict_proba(X_val_flat)
            
            with torch.no_grad():
                models['lstm'].eval()
                models['attention'].eval()
                preds_map['lstm'] = models['lstm'](X_val_tensor).cpu().numpy()
                preds_map['attention'] = models['attention'](X_val_tensor).cpu().numpy()
            
            # Calculate score for each model (Average Precision of Buy/Sell classes)
            model_scores = []
            model_order = ['gb', 'technical', 'lstm', 'attention']
            
            for name in model_order:
                probs = preds_map[name]
                y_pred_cls = np.argmax(probs, axis=1)
                # Calculate precision for each class
                prec = precision_score(y_val_flat, y_pred_cls, average=None, labels=[0, 1, 2], zero_division=0)
                # Score is average of Sell(0) and Buy(2) precision. We care less about Hold(1).
                score = (prec[0] + prec[2]) / 2
                model_scores.append(score)
            
            # Calculate weights: Square the score to punish weak models, then normalize
            total_score_sq = sum(s**2 for s in model_scores) 
            if total_score_sq > 0:
                final_weights = [(s**2)/total_score_sq for s in model_scores]
                worker_logger.info("Learned optimal weights", symbol=symbol, weights=final_weights, scores=model_scores)
            else:
                worker_logger.warning("Validation scores too low, reverting to default weights.", symbol=symbol)

        # 8. Evaluation with Final Weights
        worker_logger.info("Evaluating ensemble...", symbol=symbol)
        ensemble_probs = _get_ensemble_prediction_static(models, X_val_flat, X_val_tensor, final_weights)
        y_pred = np.argmax(ensemble_probs, axis=1)
        
        # Detailed Metrics
        precision = precision_score(y_val_flat, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
        avg_action_precision = (precision[0] + precision[2]) / 2
        
        threshold = config.training.min_precision_threshold
        worker_logger.info("Model Performance", symbol=symbol, precision=avg_action_precision, threshold=threshold)

        if avg_action_precision < threshold:
            worker_logger.warning("Model failed validation threshold.", symbol=symbol)
            return False

        # 9. Feature Importance Extraction
        # We aggregate importance across the sequence length for each feature
        feature_importance_map = {}
        try:
            # XGBoost Importance
            gb_imp = models['gb'].feature_importances_
            # RandomForest Importance (inside VotingClassifier)
            rf_model = None
            if hasattr(models['technical'], 'named_estimators_'):
                rf_model = models['technical'].named_estimators_.get('rf')
            elif hasattr(models['technical'], 'estimators_'):
                # Fallback for older sklearn or if named_estimators_ is not populated yet
                rf_model = models['technical'].estimators_[0]
            
            rf_imp = rf_model.feature_importances_ if rf_model else np.zeros_like(gb_imp)
            
            # Average the two tree models
            avg_imp = (gb_imp + rf_imp) / 2.0
            
            # Reshape to (seq_len, num_features) and sum across time
            num_feats = len(config.feature_columns)
            seq_len = config.features.sequence_length
            
            if len(avg_imp) == seq_len * num_feats:
                reshaped_imp = avg_imp.reshape(seq_len, num_feats)
                total_imp = reshaped_imp.sum(axis=0)
                
                # Map to feature names
                for i, col in enumerate(config.feature_columns):
                    feature_importance_map[col] = float(total_imp[i])
            
            # Sort and keep top 5
            sorted_feats = sorted(feature_importance_map.items(), key=lambda x: x[1], reverse=True)[:5]
            feature_importance_map = dict(sorted_feats)
            
        except Exception as e:
            worker_logger.warning("Could not extract feature importance", error=str(e))

        # Prepare extra metadata
        extra_meta = {
            'metrics': {
                'precision_sell': float(precision[0]),
                'precision_hold': float(precision[1]),
                'precision_buy': float(precision[2]),
                'avg_action_precision': float(avg_action_precision)
            },
            'top_features': feature_importance_map
        }

        # 10. Save Models & Weights
        _save_models_to_disk(symbol, models, config, learned_weights=final_weights, extra_meta=extra_meta)
        worker_logger.info("Models saved successfully.", symbol=symbol)
        return True

    except Exception as e:
        worker_logger.critical("Error in training task", symbol=symbol, error=str(e), exc_info=True)
        return False

# --- Main Class (Inference & Management) ---

class EnsembleLearner:
    """
    Manages AI/ML models for inference. 
    Delegates training to `train_ensemble_task` via external executor.
    Performs inference in a ThreadPoolExecutor to avoid blocking the asyncio loop.
    """

    def __init__(self, config: AIEnsembleStrategyParams):
        if not ML_AVAILABLE:
            raise ImportError("Required ML libraries not installed for EnsembleLearner.")
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Stores models AND metadata: {symbol: {'models': {...}, 'meta': {...}, 'timestamp': datetime}}
        self.symbol_models: Dict[str, Dict[str, Any]] = {}
        self.is_trained = False
        
        # Executor for inference to prevent blocking the main loop
        self.inference_executor = ThreadPoolExecutor(max_workers=config.inference_workers)
        
        logger.info("EnsembleLearner initialized.", device=str(self.device), inference_workers=config.inference_workers)
        
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.load_all_models())
        except RuntimeError:
            logger.warning("No running event loop found in EnsembleLearner init.")

    async def load_all_models(self):
        """Loads models for all symbols found in the model directory."""
        base_path = self.config.model_path
        if not os.path.isdir(base_path):
            return
        for symbol_dir in os.listdir(base_path):
            symbol = symbol_dir.replace('_', '/')
            await self.reload_models(symbol)

    async def reload_models(self, symbol: str):
        """Reloads models for a specific symbol from disk."""
        base_path = self.config.model_path
        symbol_path = os.path.join(base_path, symbol.replace('/', '_'))
        
        if not os.path.isdir(symbol_path):
            return

        try:
            # Run IO-bound load in thread pool (using default executor for IO)
            loop = asyncio.get_running_loop()
            loaded_data = await loop.run_in_executor(None, self._load_models_sync, symbol_path)
            
            if loaded_data:
                self.symbol_models[symbol] = loaded_data
                self.is_trained = True
                logger.info("Models reloaded successfully.", symbol=symbol, version=loaded_data['meta'].get('timestamp'))
        except Exception as e:
            logger.error("Failed to reload models", symbol=symbol, error=str(e))

    def _load_models_sync(self, symbol_path: str) -> Optional[Dict[str, Any]]:
        try:
            meta_path = os.path.join(symbol_path, "metadata.json")
            if not os.path.exists(meta_path):
                return None
            
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                if meta.get('feature_columns') != self.config.feature_columns:
                    return None
            
            # Parse timestamp for logic checks
            timestamp_str = meta.get('timestamp')
            timestamp = pd.to_datetime(timestamp_str) if timestamp_str else None

            models = _create_fresh_models(self.config, self.device)
            models['gb'] = joblib.load(os.path.join(symbol_path, "gb_model.pkl"))
            models['technical'] = joblib.load(os.path.join(symbol_path, "technical_model.pkl"))
            models['lstm'].load_state_dict(torch.load(os.path.join(symbol_path, "lstm_model.pth"), map_location=self.device))
            models['attention'].load_state_dict(torch.load(os.path.join(symbol_path, "attention_model.pth"), map_location=self.device))
            
            return {
                'models': models,
                'meta': meta,
                'timestamp': timestamp
            }
        except Exception:
            return None

    def get_last_training_time(self, symbol: str) -> Optional[datetime]:
        """Returns the timestamp of the currently loaded model for a symbol."""
        entry = self.symbol_models.get(symbol)
        if entry:
            return entry.get('timestamp')
        return None

    async def predict(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Async wrapper for prediction logic."""
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                self.inference_executor, 
                self._predict_sync, 
                df, 
                symbol
            )
        except Exception as e:
            logger.error("Error during async prediction", symbol=symbol, error=str(e))
            return {'action': 'hold', 'confidence': 0.0}

    def _predict_sync(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Synchronous prediction logic to be run in a thread."""
        entry = self.symbol_models.get(symbol)
        if not entry:
            return {'action': 'hold', 'confidence': 0.0}
        
        models = entry['models']
        meta = entry['meta']
        
        # Determine weights to use: Learned or Config
        weights = meta.get('learned_weights')
        if not weights:
            weights_config = self.config.ensemble_weights
            weights = [
                weights_config.xgboost,
                weights_config.technical_ensemble,
                weights_config.lstm,
                weights_config.attention
            ]

        seq_len = self.config.features.sequence_length
        norm_window = self.config.features.normalization_window
        required_len = seq_len + norm_window

        if len(df) < required_len:
            return {'action': 'hold', 'confidence': 0.0}

        try:
            # Use FeatureProcessor for normalization
            normalized_df = FeatureProcessor.normalize(df, self.config)
            sequence_df = normalized_df.tail(seq_len)
            
            if sequence_df.isnull().values.any():
                return {'action': 'hold', 'confidence': 0.0}

            features = np.nan_to_num(sequence_df.values, nan=0.0)
            
            # Flatten the sequence for tree models: (1, seq_len * features)
            flattened_features = features.reshape(1, -1)
            
            # Tensor for Deep Learning models: (1, seq_len, features)
            sequence_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

            # Use the static prediction logic with explicit weights
            ensemble_pred = _get_ensemble_prediction_static(models, flattened_features, sequence_tensor, weights)
            
            action_idx = np.argmax(ensemble_pred)
            confidence = float(ensemble_pred[action_idx])
            action_map = {0: 'sell', 1: 'hold', 2: 'buy'}
            
            return {
                'action': action_map.get(action_idx, 'hold'), 
                'confidence': confidence,
                'model_version': meta.get('timestamp'),
                'model_type': 'ensemble',
                'active_weights': weights,
                'top_features': meta.get('top_features', {}),
                'metrics': meta.get('metrics', {})
            }
        except Exception as e:
            logger.error("Error during sync prediction", symbol=symbol, error=str(e))
            return {'action': 'hold', 'confidence': 0.0}

    def shutdown(self):
        """Shuts down the inference executor."""
        self.inference_executor.shutdown(wait=False)
