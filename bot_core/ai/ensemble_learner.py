import os
import joblib
import asyncio
import copy
import json
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np

from bot_core.logger import get_logger
from bot_core.config import AIEnsembleStrategyParams
from bot_core.ai.models import LSTMPredictor, AttentionNetwork

# ML Imports with safe fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_score, classification_report
    from xgboost import XGBClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    # Define dummy classes if ML libraries are not available
    class VotingClassifier: pass
    class RandomForestClassifier: pass
    class LogisticRegression: pass
    class XGBClassifier: pass
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

def _normalize(df: pd.DataFrame, config: AIEnsembleStrategyParams) -> pd.DataFrame:
    """Applies rolling Z-score normalization."""
    window = config.features.normalization_window
    cols = config.feature_columns
    
    subset = df[cols].copy()
    rolling_mean = subset.rolling(window=window).mean()
    rolling_std = subset.rolling(window=window).std()
    
    epsilon = 1e-8
    normalized = (subset - rolling_mean) / (rolling_std + epsilon)
    return normalized

def _create_triple_barrier_labels(df: pd.DataFrame, config: AIEnsembleStrategyParams) -> pd.Series:
    """
    Implements the Triple Barrier Method for labeling.
    Labels: 0 (SELL), 1 (HOLD), 2 (BUY)
    """
    horizon = config.features.labeling_horizon
    tp_mult = config.features.triple_barrier_tp_multiplier
    sl_mult = config.features.triple_barrier_sl_multiplier
    
    # Use ATR for volatility if available, else fallback to percentage
    atr_col = config.market_regime.volatility_col
    
    # Prepare arrays for speed
    close_arr = df['close'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    
    if atr_col in df.columns:
        vol_arr = df[atr_col].values
    else:
        # Fallback volatility: 1% of price if ATR is missing
        vol_arr = close_arr * 0.01
    
    labels = np.ones(len(df), dtype=int) # Default 1 (HOLD)
    
    # Iterate up to len - horizon
    limit = len(df) - horizon
    
    for i in range(limit):
        current_close = close_arr[i]
        vol = vol_arr[i]
        
        if np.isnan(vol) or vol == 0:
            continue
            
        upper_barrier = current_close + (vol * tp_mult)
        lower_barrier = current_close - (vol * sl_mult)
        
        # Slice the future window (path dependency)
        window_high = high_arr[i+1 : i+1+horizon]
        window_low = low_arr[i+1 : i+1+horizon]
        
        # Check hits
        hit_upper_mask = window_high >= upper_barrier
        hit_lower_mask = window_low <= lower_barrier
        
        has_upper = hit_upper_mask.any()
        has_lower = hit_lower_mask.any()
        
        if has_upper and not has_lower:
            labels[i] = 2 # BUY
        elif has_lower and not has_upper:
            labels[i] = 0 # SELL
        elif has_upper and has_lower:
            # Both hit. Which was first?
            first_upper_idx = np.argmax(hit_upper_mask)
            first_lower_idx = np.argmax(hit_lower_mask)
            
            if first_upper_idx < first_lower_idx:
                labels[i] = 2 # BUY
            elif first_lower_idx < first_upper_idx:
                labels[i] = 0 # SELL
            else:
                # Same bar hit both barriers (extreme volatility). Default to HOLD/Avoid.
                labels[i] = 1
        else:
            labels[i] = 1 # HOLD
            
    return pd.Series(labels, index=df.index).iloc[:-horizon]

def _create_labels(df: pd.DataFrame, config: AIEnsembleStrategyParams) -> pd.Series:
    if config.features.use_triple_barrier:
        return _create_triple_barrier_labels(df, config)

    horizon = config.features.labeling_horizon
    future_price = df['close'].shift(-horizon)
    price_change_pct = (future_price - df['close']) / df['close']

    labels = pd.Series(1, index=df.index)  # Default to 'hold' (1)

    if config.features.use_dynamic_labeling:
        atr_col = config.market_regime.volatility_col
        if atr_col in df.columns:
            multiplier = config.features.labeling_atr_multiplier
            dynamic_threshold = (df[atr_col] * multiplier) / df['close']
            labels[price_change_pct > dynamic_threshold] = 2  # 'buy'
            labels[price_change_pct < -dynamic_threshold] = 0 # 'sell'
        else:
            threshold = config.features.labeling_threshold
            labels[price_change_pct > threshold] = 2
            labels[price_change_pct < -threshold] = 0
    else:
        threshold = config.features.labeling_threshold
        labels[price_change_pct > threshold] = 2
        labels[price_change_pct < -threshold] = 0
    
    return labels.iloc[:-horizon]

def _create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    X_seq, y_seq = [], []
    if len(X) <= seq_length:
        return np.array([]), np.array([])
        
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length-1])
        
    return np.array(X_seq), np.array(y_seq)

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

def _get_ensemble_prediction_static(models: Dict[str, Any], X_flat: np.ndarray, X_seq: torch.Tensor, config: AIEnsembleStrategyParams) -> np.ndarray:
    """
    Static version of prediction logic for validation during training.
    X_flat: Flattened sequence history for tree models (N, seq_len * features)
    X_seq: 3D Tensor for deep learning models (N, seq_len, features)
    """
    predictions = []
    weights_config = config.ensemble_weights
    weights = [
        weights_config.xgboost,
        weights_config.technical_ensemble,
        weights_config.lstm,
        weights_config.attention
    ]

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

def _save_models_to_disk(symbol: str, models: Dict[str, Any], config: AIEnsembleStrategyParams):
    symbol_path_str = symbol.replace('/', '_')
    save_path = os.path.join(config.model_path, symbol_path_str)
    os.makedirs(save_path, exist_ok=True)
    
    metadata = {
        'feature_columns': config.feature_columns,
        'hyperparameters': config.hyperparameters.dict(),
        'timestamp': str(pd.Timestamp.utcnow())
    }
    with open(os.path.join(save_path, "metadata.json"), 'w') as f:
        json.dump(metadata, f)

    joblib.dump(models['gb'], os.path.join(save_path, "gb_model.pkl"))
    joblib.dump(models['technical'], os.path.join(save_path, "technical_model.pkl"))
    torch.save(models['lstm'].state_dict(), os.path.join(save_path, "lstm_model.pth"))
    torch.save(models['attention'].state_dict(), os.path.join(save_path, "attention_model.pth"))

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

        # 1. Create labels
        labels = _create_labels(df, config)
        
        # 2. Normalize features
        normalized_df = _normalize(df, config)
        
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

        # 3. Create sequences
        X_seq, y_seq = _create_sequences(X_normalized, y, config.features.sequence_length)
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
        
        # 5. Create and Train Models
        models = _create_fresh_models(config, device)

        worker_logger.info("Training scikit-learn models with flattened history...", symbol=symbol)
        models['gb'].fit(X_train_flat, y_train_flat)
        models['technical'].fit(X_train_flat, y_train_flat)
        
        worker_logger.info("Training PyTorch models...", symbol=symbol)
        X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
        y_train_tensor = torch.LongTensor(y_train_seq).to(device)
        X_val_tensor = torch.FloatTensor(X_val_seq).to(device)
        y_val_tensor = torch.LongTensor(y_val_seq).to(device)
        
        _train_pytorch_model(models['lstm'], X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, config)
        _train_pytorch_model(models['attention'], X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, config)

        # 6. Evaluation
        worker_logger.info("Evaluating models...", symbol=symbol)
        ensemble_probs = _get_ensemble_prediction_static(models, X_val_flat, X_val_tensor, config)
        y_pred = np.argmax(ensemble_probs, axis=1)
        
        precision = precision_score(y_val_flat, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
        avg_action_precision = (precision[0] + precision[2]) / 2
        
        threshold = config.training.min_precision_threshold
        worker_logger.info("Model Performance", symbol=symbol, precision=avg_action_precision, threshold=threshold)

        if avg_action_precision < threshold:
            worker_logger.warning("Model failed validation threshold.", symbol=symbol)
            return False

        # 7. Save Models
        _save_models_to_disk(symbol, models, config)
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
    """

    def __init__(self, config: AIEnsembleStrategyParams):
        if not ML_AVAILABLE:
            raise ImportError("Required ML libraries not installed for EnsembleLearner.")
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Stores models AND metadata: {symbol: {'models': {...}, 'meta': {...}}}
        self.symbol_models: Dict[str, Dict[str, Any]] = {}
        self.is_trained = False
        
        logger.info("EnsembleLearner initialized.", device=str(self.device))
        
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
            # Run IO-bound load in thread pool
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

            models = _create_fresh_models(self.config, self.device)
            models['gb'] = joblib.load(os.path.join(symbol_path, "gb_model.pkl"))
            models['technical'] = joblib.load(os.path.join(symbol_path, "technical_model.pkl"))
            models['lstm'].load_state_dict(torch.load(os.path.join(symbol_path, "lstm_model.pth"), map_location=self.device))
            models['attention'].load_state_dict(torch.load(os.path.join(symbol_path, "attention_model.pth"), map_location=self.device))
            
            return {
                'models': models,
                'meta': meta
            }
        except Exception:
            return None

    async def predict(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        entry = self.symbol_models.get(symbol)
        if not entry:
            return {'action': 'hold', 'confidence': 0.0}
        
        models = entry['models']
        meta = entry['meta']
        
        seq_len = self.config.features.sequence_length
        norm_window = self.config.features.normalization_window
        required_len = seq_len + norm_window

        if len(df) < required_len:
            return {'action': 'hold', 'confidence': 0.0}

        try:
            # Use the standalone normalize function
            normalized_df = _normalize(df, self.config)
            sequence_df = normalized_df.tail(seq_len)
            
            if sequence_df.isnull().values.any():
                return {'action': 'hold', 'confidence': 0.0}

            features = np.nan_to_num(sequence_df.values, nan=0.0)
            
            # Flatten the sequence for tree models: (1, seq_len * features)
            flattened_features = features.reshape(1, -1)
            
            # Tensor for Deep Learning models: (1, seq_len, features)
            sequence_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

            # Use the static prediction logic
            ensemble_pred = _get_ensemble_prediction_static(models, flattened_features, sequence_tensor, self.config)
            
            action_idx = np.argmax(ensemble_pred)
            confidence = float(ensemble_pred[action_idx])
            action_map = {0: 'sell', 1: 'hold', 2: 'buy'}
            
            return {
                'action': action_map.get(action_idx, 'hold'), 
                'confidence': confidence,
                'model_version': meta.get('timestamp'),
                'model_type': 'ensemble'
            }
        except Exception as e:
            logger.error("Error during prediction", symbol=symbol, error=str(e))
            return {'action': 'hold', 'confidence': 0.0}
