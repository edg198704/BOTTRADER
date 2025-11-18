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
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MinMaxScaler
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
    class MinMaxScaler: pass
    def precision_score(*args, **kwargs): return 0.0
    def classification_report(*args, **kwargs): return ""
    class nn:
        class Module: pass
    class optim: pass
    def train_test_split(*args, **kwargs): pass
    class TensorDataset: pass
    class DataLoader: pass

logger = get_logger(__name__)

class EnsembleLearner:
    """
    Manages a suite of AI/ML models for generating trading predictions.
    Implements atomic model updates to ensure thread safety during retraining.
    """

    def __init__(self, config: AIEnsembleStrategyParams):
        if not ML_AVAILABLE:
            raise ImportError("Required ML libraries not installed for EnsembleLearner.")
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # symbol_models maps symbol -> Dict of model objects
        self.symbol_models: Dict[str, Dict[str, Any]] = {}
        self.symbol_scalers: Dict[str, MinMaxScaler] = {}
        self.is_trained = False
        
        logger.info("EnsembleLearner initialized.", device=str(self.device))
        
        # Attempt to load models in the background
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._load_models())
        except RuntimeError:
            logger.warning("No running event loop found in EnsembleLearner init. Models will not auto-load.")

    def _create_fresh_models(self) -> Dict[str, Any]:
        """Creates a fresh set of untrained model instances based on configuration."""
        num_features = len(self.config.feature_columns)
        hp = self.config.hyperparameters
        xgb_config = hp.xgboost
        rf_config = hp.random_forest
        lr_config = hp.logistic_regression
        lstm_config = hp.lstm
        attn_config = hp.attention

        return {
            'lstm': LSTMPredictor(
                num_features, lstm_config.hidden_dim, lstm_config.num_layers, lstm_config.dropout
            ).to(self.device),
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
            ).to(self.device),
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

    async def _load_models(self):
        base_path = self.config.model_path
        if not os.path.isdir(base_path):
            logger.warning("Model path does not exist, cannot load models.", path=base_path)
            return

        for symbol_dir in os.listdir(base_path):
            symbol_path = os.path.join(base_path, symbol_dir)
            if os.path.isdir(symbol_path):
                symbol = symbol_dir.replace('_', '/')
                try:
                    # 1. Validate Metadata
                    meta_path = os.path.join(symbol_path, "metadata.json")
                    if os.path.exists(meta_path):
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                            if meta.get('feature_columns') != self.config.feature_columns:
                                logger.warning("Model feature mismatch. Skipping load to force retrain.", symbol=symbol)
                                continue
                    else:
                        logger.warning("No metadata found for model. Skipping load to be safe.", symbol=symbol)
                        continue

                    # 2. Load Models into fresh instances
                    models = self._create_fresh_models()
                    scaler = joblib.load(os.path.join(symbol_path, "scaler.pkl"))
                    
                    models['gb'] = joblib.load(os.path.join(symbol_path, "gb_model.pkl"))
                    models['technical'] = joblib.load(os.path.join(symbol_path, "technical_model.pkl"))
                    models['lstm'].load_state_dict(torch.load(os.path.join(symbol_path, "lstm_model.pth"), map_location=self.device))
                    models['attention'].load_state_dict(torch.load(os.path.join(symbol_path, "attention_model.pth"), map_location=self.device))
                    
                    # Atomic update
                    self.symbol_models[symbol] = models
                    self.symbol_scalers[symbol] = scaler
                    self.is_trained = True
                    logger.info("Loaded models and scaler for symbol", symbol=symbol)
                except Exception as e:
                    logger.warning("Could not load models for symbol", symbol=symbol, error=str(e))

    async def predict(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        # Thread-safe retrieval of the model dict reference
        models = self.symbol_models.get(symbol)
        scaler = self.symbol_scalers.get(symbol)

        if not models or not scaler:
            logger.debug("Predict called but models are not loaded for symbol.", symbol=symbol)
            return {'action': 'hold', 'confidence': 0.0}
        
        seq_len = self.config.features.sequence_length
        if len(df) < seq_len:
            logger.warning("Not enough data for a full sequence prediction.", symbol=symbol, data_len=len(df), required=seq_len)
            return {'action': 'hold', 'confidence': 0.0}

        try:
            sequence_df = df.tail(seq_len)
            features = np.nan_to_num(sequence_df[self.config.feature_columns].values, nan=0.0)
            features_scaled = scaler.transform(features)

            # Prepare inputs
            last_step_features = features_scaled[-1, :].reshape(1, -1)
            sequence_tensor = torch.FloatTensor(features_scaled).unsqueeze(0).to(self.device)

            ensemble_pred = self._get_ensemble_prediction(models, last_step_features, sequence_tensor)
            
            action_idx = np.argmax(ensemble_pred)
            confidence = float(ensemble_pred[action_idx])
            action_map = {0: 'sell', 1: 'hold', 2: 'buy'}
            
            return {'action': action_map.get(action_idx, 'hold'), 'confidence': confidence}
        except Exception as e:
            logger.error("Error during prediction", symbol=symbol, error=str(e), exc_info=True)
            return {'action': 'hold', 'confidence': 0.0}

    def _get_ensemble_prediction(self, models: Dict[str, Any], X_flat: np.ndarray, X_seq: torch.Tensor) -> np.ndarray:
        """Calculates the weighted average prediction from all models."""
        predictions = []
        weights_config = self.config.ensemble_weights
        weights = [
            weights_config.xgboost,
            weights_config.technical_ensemble,
            weights_config.lstm,
            weights_config.attention
        ]

        # Scikit-learn models
        gb_pred = models['gb'].predict_proba(X_flat)
        predictions.append(gb_pred)
        tech_pred = models['technical'].predict_proba(X_flat)
        predictions.append(tech_pred)

        # PyTorch models
        with torch.no_grad():
            models['lstm'].eval()
            models['attention'].eval()
            
            if X_seq.dim() == 2:
                 X_seq = X_seq.unsqueeze(0)

            lstm_pred = models['lstm'](X_seq).cpu().numpy()
            predictions.append(lstm_pred)
            attn_pred = models['attention'](X_seq).cpu().numpy()
            predictions.append(attn_pred)

        stacked_preds = np.array(predictions) # Shape: (4, N, 3)
        ensemble_pred = np.average(stacked_preds, weights=weights, axis=0) # Shape: (N, 3)
        
        if ensemble_pred.shape[0] == 1:
            return ensemble_pred[0]
        return ensemble_pred

    def _create_labels(self, df: pd.DataFrame) -> pd.Series:
        horizon = self.config.features.labeling_horizon
        threshold = self.config.features.labeling_threshold
        
        future_price = df['close'].shift(-horizon)
        price_change_pct = (future_price - df['close']) / df['close']

        labels = pd.Series(1, index=df.index)  # Default to 'hold'
        labels[price_change_pct > threshold] = 2  # 'buy'
        labels[price_change_pct < -threshold] = 0 # 'sell'
        
        return labels.iloc[:-horizon]

    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        seq_length = self.config.features.sequence_length
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length + 1):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length-1])
        return np.array(X_seq), np.array(y_seq)

    def train(self, symbol: str, df: pd.DataFrame) -> bool:
        logger.info("Starting model training", symbol=symbol)
        try:
            # 1. Create features and labels
            labels = self._create_labels(df)
            features_df = df[self.config.feature_columns].loc[labels.index]
            
            X = np.nan_to_num(features_df.values, nan=0.0)
            y = labels.values

            if len(np.unique(y)) < 3:
                logger.warning("Training data has fewer than 3 unique labels. Skipping training.", symbol=symbol, labels=np.unique(y))
                return False

            # 2. Split data BEFORE scaling to prevent data leakage
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.config.training.validation_split, random_state=42, stratify=y
            )

            # 3. Fit scaler on training data and transform both sets
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # 4. Create sequences from SCALED data
            X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_train)
            X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val)

            # 5. Prepare flat data for scikit-learn models
            X_train_flat_scaled = X_train_seq[:, -1, :]
            y_train_flat = y_train_seq
            X_val_flat_scaled = X_val_seq[:, -1, :]
            y_val_flat = y_val_seq
            
            # 6. Create FRESH models for this training session
            # This ensures we don't corrupt the live models if training fails or is poor
            models = self._create_fresh_models()

            # 7. Train scikit-learn models
            logger.info("Training scikit-learn models...", symbol=symbol)
            models['gb'].fit(X_train_flat_scaled, y_train_flat)
            models['technical'].fit(X_train_flat_scaled, y_train_flat)
            
            # Log Feature Importance
            try:
                if hasattr(models['gb'], 'feature_importances_'):
                    importances = models['gb'].feature_importances_
                    indices = np.argsort(importances)[::-1][:5]
                    top_features = {self.config.feature_columns[i]: float(importances[i]) for i in indices}
                    logger.info("XGBoost Top 5 Features", symbol=symbol, features=top_features)
            except Exception as e:
                logger.warning("Could not log feature importance", error=str(e))

            # 8. Train PyTorch models
            logger.info("Training PyTorch models...", symbol=symbol)
            X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
            y_train_tensor = torch.LongTensor(y_train_seq).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
            y_val_tensor = torch.LongTensor(y_val_seq).to(self.device)
            
            self._train_pytorch_model(models['lstm'], X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)
            self._train_pytorch_model(models['attention'], X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)

            # 9. EVALUATION & VALIDATION
            logger.info("Evaluating new models on validation set...", symbol=symbol)
            
            ensemble_probs = self._get_ensemble_prediction(models, X_val_flat_scaled, X_val_tensor)
            y_pred = np.argmax(ensemble_probs, axis=1)
            
            precision = precision_score(y_val_flat, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
            sell_precision = precision[0]
            buy_precision = precision[2]
            avg_action_precision = (sell_precision + buy_precision) / 2
            
            report = classification_report(y_val_flat, y_pred, target_names=['Sell', 'Hold', 'Buy'], zero_division=0)
            logger.info("Validation Classification Report", symbol=symbol, report=report)
            
            threshold = self.config.training.min_precision_threshold
            logger.info("Model Performance Check", 
                        symbol=symbol, 
                        buy_precision=buy_precision, 
                        sell_precision=sell_precision, 
                        avg_action_precision=avg_action_precision,
                        threshold=threshold)

            if avg_action_precision < threshold:
                logger.warning("New model failed validation threshold. Discarding.", 
                               symbol=symbol, 
                               achieved=avg_action_precision, 
                               required=threshold)
                return False

            # 10. Atomic Update & Save
            # Only now do we update the live models
            self.symbol_models[symbol] = models
            self.symbol_scalers[symbol] = scaler
            self.is_trained = True
            
            self._save_models(symbol, models, scaler)
            logger.info("All models trained, validated, and promoted to production.", symbol=symbol)
            return True

        except Exception as e:
            logger.critical("An error occurred during model training", symbol=symbol, error=str(e), exc_info=True)
            return False

    def _train_pytorch_model(self, model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor):
        train_cfg = self.config.training
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

    def _save_models(self, symbol: str, models: Dict[str, Any], scaler: Any):
        symbol_path_str = symbol.replace('/', '_')
        save_path = os.path.join(self.config.model_path, symbol_path_str)
        os.makedirs(save_path, exist_ok=True)
        
        # Save Metadata
        metadata = {
            'feature_columns': self.config.feature_columns,
            'timestamp': str(pd.Timestamp.utcnow())
        }
        with open(os.path.join(save_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f)

        joblib.dump(scaler, os.path.join(save_path, "scaler.pkl"))
        joblib.dump(models['gb'], os.path.join(save_path, "gb_model.pkl"))
        joblib.dump(models['technical'], os.path.join(save_path, "technical_model.pkl"))
        torch.save(models['lstm'].state_dict(), os.path.join(save_path, "lstm_model.pth"))
        torch.save(models['attention'].state_dict(), os.path.join(save_path, "attention_model.pth"))
        logger.info("Saved models, scaler, and metadata to disk", path=save_path)
