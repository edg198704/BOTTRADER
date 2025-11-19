import os
import joblib
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime
import logging

from bot_core.logger import get_logger
from bot_core.config import AIEnsembleStrategyParams
from bot_core.ai.models import LSTMPredictor, AttentionNetwork
from bot_core.ai.feature_processor import FeatureProcessor
from bot_core.utils import Clock

# ML Imports with safe fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_score, classification_report, log_loss
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    from xgboost import XGBClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    # Dummy classes to prevent ImportErrors if ML libraries are not available
    class VotingClassifier: pass
    class RandomForestClassifier: pass
    class LogisticRegression: pass
    class XGBClassifier: pass
    class RandomizedSearchCV: pass
    class TimeSeriesSplit: pass
    def precision_score(*args, **kwargs): return 0.0
    def classification_report(*args, **kwargs): return ""
    def log_loss(*args, **kwargs): return 0.0
    class nn:
        class Module: pass
    class optim: pass
    class TensorDataset: pass
    class DataLoader: pass
    torch = None

logger = get_logger(__name__)

def train_ensemble_task(symbol: str, df: pd.DataFrame, config: AIEnsembleStrategyParams) -> bool:
    """
    Standalone function to be run in a separate process for training.
    Instantiates a temporary learner to perform the heavy lifting.
    """
    # Re-initialize logger for the worker process
    worker_logger = get_logger(f"trainer_{symbol}")
    worker_logger.info("Starting training task in worker process", symbol=symbol, rows=len(df))
    
    if not ML_AVAILABLE:
        worker_logger.error("ML libraries not installed. Cannot train.")
        return False

    try:
        learner = EnsembleLearner(config)
        return learner.train(symbol, df)
    except Exception as e:
        worker_logger.error("Training task failed", symbol=symbol, error=str(e), exc_info=True)
        return False

class EnsembleLearner:
    def __init__(self, config: AIEnsembleStrategyParams):
        self.config = config
        self.models = {}
        self.is_trained = False
        self.symbol_models = {} # Cache for loaded models: {symbol: {'models': ..., 'meta': ...}}
        
        # Determine device for PyTorch
        self.device = "cpu"
        if ML_AVAILABLE and torch and torch.cuda.is_available():
            self.device = "cuda"
        
        # Ensure model directory exists
        os.makedirs(self.config.model_path, exist_ok=True)

    def get_last_training_time(self, symbol: str) -> Optional[datetime]:
        """Returns the timestamp of the last successful training for the symbol."""
        meta_path = os.path.join(self.config.model_path, f"{symbol.replace('/', '_')}_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    data = json.load(f)
                    # Parse ISO format string back to datetime
                    return datetime.fromisoformat(data.get('timestamp'))
            except Exception:
                return None
        return None

    async def reload_models(self, symbol: str):
        """Reloads models from disk into memory for the given symbol."""
        self._load_models(symbol)

    def train(self, symbol: str, df: pd.DataFrame) -> bool:
        """
        Main training pipeline:
        1. Feature Engineering & Labeling
        2. Splitting (Train/Val)
        3. Training Individual Models (ML & DL)
        4. Evaluation & Threshold Check
        5. Persistence
        """
        if len(df) < self.config.features.sequence_length + 50:
            logger.warning("Insufficient data for training", symbol=symbol)
            return False

        # 1. Prepare Data
        # Normalize features
        X_df = FeatureProcessor.normalize(df, self.config)
        # Create labels
        y_series = FeatureProcessor.create_labels(df, self.config)
        
        # Align X and y (drop NaNs created by shifting/rolling)
        common_index = X_df.index.intersection(y_series.index)
        X_df = X_df.loc[common_index]
        y_series = y_series.loc[common_index]
        
        # Drop any remaining NaNs
        valid_mask = ~X_df.isna().any(axis=1) & ~y_series.isna()
        X_df = X_df[valid_mask]
        y_series = y_series[valid_mask]

        if len(X_df) < 100:
            logger.warning("Data too small after preprocessing", symbol=symbol)
            return False

        # Split Data (Time Series Split - No Shuffle)
        split_idx = int(len(X_df) * (1 - self.config.training.validation_split))
        X_train_df, X_val_df = X_df.iloc[:split_idx], X_df.iloc[split_idx:]
        y_train, y_val = y_series.iloc[:split_idx].values.astype(int), y_series.iloc[split_idx:].values.astype(int)

        # Convert to numpy for ML models
        X_train = X_train_df.values
        X_val = X_val_df.values

        trained_models = {}
        metrics = {}

        # --- Train Technical Models (XGBoost, Voting) ---
        if self.config.training.auto_tune_models:
            trained_models['gb'] = self._optimize_xgboost(X_train, y_train)
        else:
            xgb = XGBClassifier(**self.config.hyperparameters.xgboost.dict())
            xgb.fit(X_train, y_train)
            trained_models['gb'] = xgb

        # Voting Classifier (RF + LR)
        rf = RandomForestClassifier(**self.config.hyperparameters.random_forest.dict())
        lr = LogisticRegression(**self.config.hyperparameters.logistic_regression.dict())
        voting = VotingClassifier(estimators=[('rf', rf), ('lr', lr)], voting='soft')
        voting.fit(X_train, y_train)
        trained_models['technical'] = voting

        # --- Train Deep Learning Models (LSTM, Attention) ---
        seq_len = self.config.features.sequence_length
        
        # Create sequences
        X_train_seq, y_train_seq = FeatureProcessor.create_sequences(X_train, y_train, seq_len)
        X_val_seq, y_val_seq = FeatureProcessor.create_sequences(X_val, y_val, seq_len)

        if len(X_train_seq) > 0:
            # LSTM
            lstm = LSTMPredictor(
                input_dim=X_train.shape[1],
                hidden_dim=self.config.hyperparameters.lstm.hidden_dim,
                num_layers=self.config.hyperparameters.lstm.num_layers,
                dropout=self.config.hyperparameters.lstm.dropout
            ).to(self.device)
            self._train_torch_model(lstm, X_train_seq, y_train_seq, X_val_seq, y_val_seq, "LSTM")
            trained_models['lstm'] = lstm

            # Attention
            attn = AttentionNetwork(
                input_dim=X_train.shape[1],
                hidden_dim=self.config.hyperparameters.attention.hidden_dim,
                num_layers=self.config.hyperparameters.attention.num_layers,
                nhead=self.config.hyperparameters.attention.nhead,
                dropout=self.config.hyperparameters.attention.dropout
            ).to(self.device)
            self._train_torch_model(attn, X_train_seq, y_train_seq, X_val_seq, y_val_seq, "Attention")
            trained_models['attention'] = attn

        # --- Evaluation ---
        # We evaluate the ensemble on the validation set
        val_preds = self._ensemble_predict_internal(trained_models, X_val, X_val_seq)
        
        # Calculate Precision for Buy (2) and Sell (0)
        precision_buy = precision_score(y_val[seq_len-1:], val_preds, labels=[2], average='micro', zero_division=0)
        precision_sell = precision_score(y_val[seq_len-1:], val_preds, labels=[0], average='micro', zero_division=0)
        avg_precision = (precision_buy + precision_sell) / 2

        metrics = {
            'precision_buy': float(precision_buy),
            'precision_sell': float(precision_sell),
            'avg_action_precision': float(avg_precision)
        }

        logger.info("Training completed", symbol=symbol, metrics=metrics)

        # Check threshold
        if avg_precision < self.config.training.min_precision_threshold:
            logger.warning("Model failed precision threshold. Discarding.", symbol=symbol, metrics=metrics)
            return False

        # --- Persistence ---
        self._save_models(symbol, trained_models, metrics, list(X_df.columns))
        return True

    async def predict(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Generates a prediction using the loaded ensemble for the symbol."""
        if symbol not in self.symbol_models:
            self._load_models(symbol)
        
        entry = self.symbol_models.get(symbol)
        if not entry:
            return {'action': 'hold', 'confidence': 0.0, 'model_version': 'none'}

        models = entry['models']
        meta = entry['meta']
        feature_cols = meta.get('feature_columns', [])

        # Ensure columns match training
        # Normalize using the same logic (rolling Z-score is stateless per window)
        df_norm = FeatureProcessor.normalize(df, self.config)
        
        # Check for missing columns
        missing = set(feature_cols) - set(df_norm.columns)
        if missing:
            logger.error("Missing features for inference", symbol=symbol, missing=missing)
            return {'action': 'hold', 'confidence': 0.0}

        # Prepare Input
        # ML models take the last row
        X_last = df_norm[feature_cols].iloc[-1:].values
        
        # DL models take the last sequence
        seq_len = self.config.features.sequence_length
        if len(df_norm) < seq_len:
            return {'action': 'hold', 'confidence': 0.0}
            
        X_seq = df_norm[feature_cols].iloc[-seq_len:].values
        X_seq = np.expand_dims(X_seq, axis=0) # (1, seq_len, features)

        # Get Probabilities
        probs = []
        weights = self.config.ensemble_weights
        
        # XGBoost
        if 'gb' in models:
            p = models['gb'].predict_proba(X_last)[0]
            probs.append(p * weights.xgboost)
        
        # Technical Voting
        if 'technical' in models:
            p = models['technical'].predict_proba(X_last)[0]
            probs.append(p * weights.technical_ensemble)
            
        # LSTM
        if 'lstm' in models:
            with torch.no_grad():
                t_in = torch.FloatTensor(X_seq).to(self.device)
                p = models['lstm'](t_in).cpu().numpy()[0]
                probs.append(p * weights.lstm)

        # Attention
        if 'attention' in models:
            with torch.no_grad():
                t_in = torch.FloatTensor(X_seq).to(self.device)
                p = models['attention'](t_in).cpu().numpy()[0]
                probs.append(p * weights.attention)

        # Aggregate
        if not probs:
            return {'action': 'hold', 'confidence': 0.0}
            
        avg_probs = np.sum(probs, axis=0) / sum(weights.dict().values())
        
        # 0: Sell, 1: Hold, 2: Buy
        best_idx = np.argmax(avg_probs)
        confidence = avg_probs[best_idx]
        
        actions = {0: 'sell', 1: 'hold', 2: 'buy'}
        action = actions.get(best_idx, 'hold')

        # Get Feature Importance (from XGBoost if available)
        top_features = {}
        if 'gb' in models:
            try:
                importances = models['gb'].feature_importances_
                indices = np.argsort(importances)[::-1][:5]
                for i in indices:
                    if i < len(feature_cols):
                        top_features[feature_cols[i]] = float(importances[i])
            except: 
                pass

        return {
            'action': action,
            'confidence': float(confidence),
            'model_version': meta.get('timestamp'),
            'active_weights': weights.dict(),
            'top_features': top_features,
            'metrics': meta.get('metrics')
        }

    def _ensemble_predict_internal(self, models, X_flat, X_seq):
        """Internal helper to predict on validation set for evaluation."""
        # This is a simplified voting mechanism for validation metrics
        # We just use XGBoost + Technical for speed in validation check if DL is slow
        # Or we can do full ensemble. Let's do full ensemble.
        
        preds_accum = np.zeros((len(X_seq), 3))
        weights = self.config.ensemble_weights
        
        # Align X_flat to X_seq (X_seq starts at index seq_len-1 of X_flat)
        start_idx = self.config.features.sequence_length - 1
        X_flat_aligned = X_flat[start_idx:]
        
        if 'gb' in models:
            p = models['gb'].predict_proba(X_flat_aligned)
            preds_accum += p * weights.xgboost
            
        if 'technical' in models:
            p = models['technical'].predict_proba(X_flat_aligned)
            preds_accum += p * weights.technical_ensemble
            
        # For DL, we batch process if needed, but for validation set size it's usually fine
        if 'lstm' in models or 'attention' in models:
            t_in = torch.FloatTensor(X_seq).to(self.device)
            with torch.no_grad():
                if 'lstm' in models:
                    p = models['lstm'](t_in).cpu().numpy()
                    preds_accum += p * weights.lstm
                if 'attention' in models:
                    p = models['attention'](t_in).cpu().numpy()
                    preds_accum += p * weights.attention
                    
        return np.argmax(preds_accum, axis=1)

    def _optimize_xgboost(self, X, y):
        """Runs RandomizedSearchCV to find better hyperparameters for XGBoost."""
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        tscv = TimeSeriesSplit(n_splits=3)
        search = RandomizedSearchCV(
            xgb, param_dist, n_iter=self.config.training.n_iter_search, 
            scoring='neg_log_loss', cv=tscv, n_jobs=1, verbose=0
        )
        search.fit(X, y)
        return search.best_estimator_

    def _train_torch_model(self, model, X_seq, y_seq, X_val, y_val, name):
        """Standard PyTorch training loop."""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.training.learning_rate)
        
        # Create DataLoaders
        train_data = TensorDataset(torch.FloatTensor(X_seq), torch.LongTensor(y_seq))
        train_loader = DataLoader(train_data, batch_size=self.config.training.batch_size, shuffle=False)
        
        best_loss = float('inf')
        patience_counter = 0
        
        model.train()
        for epoch in range(self.config.training.epochs):
            epoch_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_in = torch.FloatTensor(X_val).to(self.device)
                val_target = torch.LongTensor(y_val).to(self.device)
                val_out = model(val_in)
                val_loss = criterion(val_out, val_target).item()
            model.train()
            
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.training.early_stopping_patience:
                    break
        
        model.eval()

    def _save_models(self, symbol: str, models: Dict, metrics: Dict, feature_cols: List[str]):
        """Saves models and metadata to disk."""
        safe_symbol = symbol.replace('/', '_')
        base_path = os.path.join(self.config.model_path, safe_symbol)
        
        # Save Sklearn/XGB models
        if 'gb' in models: joblib.dump(models['gb'], f"{base_path}_gb.joblib")
        if 'technical' in models: joblib.dump(models['technical'], f"{base_path}_tech.joblib")
        
        # Save PyTorch models
        if 'lstm' in models: torch.save(models['lstm'].state_dict(), f"{base_path}_lstm.pth")
        if 'attention' in models: torch.save(models['attention'].state_dict(), f"{base_path}_attn.pth")
        
        # Save Metadata
        meta = {
            'timestamp': Clock.now().isoformat(),
            'metrics': metrics,
            'feature_columns': feature_cols,
            'config_hash': str(hash(str(self.config.dict())))
        }
        with open(f"{base_path}_meta.json", 'w') as f:
            json.dump(meta, f, indent=2)

    def _load_models(self, symbol: str):
        """Loads models from disk."""
        safe_symbol = symbol.replace('/', '_')
        base_path = os.path.join(self.config.model_path, safe_symbol)
        meta_path = f"{base_path}_meta.json"
        
        if not os.path.exists(meta_path):
            return
            
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            models = {}
            feature_cols = meta.get('feature_columns', [])
            num_features = len(feature_cols)
            
            # Load Sklearn/XGB
            if os.path.exists(f"{base_path}_gb.joblib"):
                models['gb'] = joblib.load(f"{base_path}_gb.joblib")
            if os.path.exists(f"{base_path}_tech.joblib"):
                models['technical'] = joblib.load(f"{base_path}_tech.joblib")
                
            # Load PyTorch
            if os.path.exists(f"{base_path}_lstm.pth"):
                lstm = LSTMPredictor(
                    num_features, 
                    self.config.hyperparameters.lstm.hidden_dim, 
                    self.config.hyperparameters.lstm.num_layers, 
                    self.config.hyperparameters.lstm.dropout
                ).to(self.device)
                lstm.load_state_dict(torch.load(f"{base_path}_lstm.pth", map_location=self.device))
                lstm.eval()
                models['lstm'] = lstm
                
            if os.path.exists(f"{base_path}_attn.pth"):
                attn = AttentionNetwork(
                    num_features,
                    self.config.hyperparameters.attention.hidden_dim,
                    self.config.hyperparameters.attention.num_layers,
                    self.config.hyperparameters.attention.nhead,
                    self.config.hyperparameters.attention.dropout
                ).to(self.device)
                attn.load_state_dict(torch.load(f"{base_path}_attn.pth", map_location=self.device))
                attn.eval()
                models['attention'] = attn
            
            self.symbol_models[symbol] = {'models': models, 'meta': meta}
            self.is_trained = True
            logger.info("Models loaded successfully", symbol=symbol, timestamp=meta['timestamp'])
            
        except Exception as e:
            logger.error("Failed to load models", symbol=symbol, error=str(e))
