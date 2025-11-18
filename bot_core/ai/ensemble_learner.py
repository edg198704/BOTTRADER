import os
import joblib
import asyncio
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from bot_core.logger import get_logger
from bot_core.config import AIStrategyConfig

# ML Imports with safe fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    # Define dummy classes if ML libraries are not available
    class VotingClassifier: pass
    class RandomForestClassifier: pass
    class GradientBoostingClassifier: pass
    class LogisticRegression: pass
    class XGBClassifier: pass
    class nn:
        class Module: pass
    class optim: pass
    def train_test_split(*args, **kwargs): pass

logger = get_logger(__name__)

class EnsembleLearner:
    """
    Manages a suite of AI/ML models for generating trading predictions.
    This includes loading, training, and predicting using an ensemble of models.
    """
    class LSTMPredictor(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 3) # 0: sell, 1: hold, 2: buy

        def forward(self, x):
            _, (hn, _) = self.lstm(x)
            return F.softmax(self.fc(hn[0]), dim=1)

    class AttentionNetwork(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.embedding = nn.Linear(input_dim, hidden_dim)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True), 
                num_layers=2
            )
            self.fc = nn.Linear(hidden_dim, 3) # 0: sell, 1: hold, 2: buy

        def forward(self, x):
            x = self.embedding(x)
            x = self.transformer(x)
            x = x.mean(dim=1)
            return F.softmax(self.fc(x), dim=1)

    def __init__(self, config: AIStrategyConfig):
        if not ML_AVAILABLE:
            raise ImportError("Required ML libraries not installed for EnsembleLearner.")
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.symbol_models: Dict[str, Dict[str, Any]] = {}
        self.is_trained = False
        logger.info("EnsembleLearner initialized.", device=str(self.device))
        asyncio.create_task(self._load_models())

    def _get_or_create_symbol_models(self, symbol: str) -> Dict[str, Any]:
        if symbol not in self.symbol_models:
            logger.info("Creating new model set for symbol", symbol=symbol)
            num_features = len(self.config.feature_columns)
            
            xgb_config = self.config.xgboost
            rf_config = self.config.random_forest
            lr_config = self.config.logistic_regression

            self.symbol_models[symbol] = {
                'lstm': self.LSTMPredictor(num_features, 64).to(self.device),
                'gb': XGBClassifier(
                    n_estimators=xgb_config.n_estimators,
                    max_depth=xgb_config.max_depth,
                    learning_rate=xgb_config.learning_rate,
                    subsample=xgb_config.subsample,
                    colsample_bytree=xgb_config.colsample_bytree,
                    random_state=42, use_label_encoder=False, eval_metric='logloss'
                ),
                'attention': self.AttentionNetwork(num_features, 64).to(self.device),
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
        return self.symbol_models[symbol]

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
                    models = self._get_or_create_symbol_models(symbol)
                    models['gb'] = joblib.load(os.path.join(symbol_path, "gb_model.pkl"))
                    models['technical'] = joblib.load(os.path.join(symbol_path, "technical_model.pkl"))
                    models['lstm'].load_state_dict(torch.load(os.path.join(symbol_path, "lstm_model.pth"), map_location=self.device))
                    models['attention'].load_state_dict(torch.load(os.path.join(symbol_path, "attention_model.pth"), map_location=self.device))
                    self.is_trained = True
                    logger.info("Loaded models for symbol", symbol=symbol)
                except Exception as e:
                    logger.warning("Could not load models for symbol", symbol=symbol, error=str(e))

    async def predict(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        if not self.is_trained:
            logger.debug("Predict called but models are not trained or loaded.")
            return {'action': 'hold', 'confidence': 0.0}
        
        if len(df) < self.config.sequence_length:
            logger.warning("Not enough data for a full sequence prediction.", symbol=symbol, data_len=len(df), required=self.config.sequence_length)
            return {'action': 'hold', 'confidence': 0.0}

        try:
            models = self._get_or_create_symbol_models(symbol)
            
            # Get the last sequence of data
            sequence_df = df.tail(self.config.sequence_length)
            features = np.nan_to_num(sequence_df[self.config.feature_columns].values, nan=0.0)

            predictions = []
            weights_config = self.config.ensemble_weights
            weights = [
                weights_config.xgboost,
                weights_config.technical_ensemble,
                weights_config.lstm,
                weights_config.attention
            ]

            # Scikit-learn models predict on the last timestep
            last_step_features = features[-1, :].reshape(1, -1)
            gb_pred = models['gb'].predict_proba(last_step_features)[0]
            predictions.append(gb_pred)
            tech_pred = models['technical'].predict_proba(last_step_features)[0]
            predictions.append(tech_pred)

            # PyTorch models predict on the full sequence
            sequence_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device) # Add batch dimension
            with torch.no_grad():
                lstm_pred = models['lstm'](sequence_tensor).cpu().numpy()[0]
                predictions.append(lstm_pred)
                attn_pred = models['attention'](sequence_tensor).cpu().numpy()[0]
                predictions.append(attn_pred)

            ensemble_pred = np.average(predictions, weights=weights, axis=0)
            action_idx = np.argmax(ensemble_pred)
            confidence = float(ensemble_pred[action_idx])
            action_map = {0: 'sell', 1: 'hold', 2: 'buy'}
            
            return {'action': action_map.get(action_idx, 'hold'), 'confidence': confidence}
        except Exception as e:
            logger.error("Error during prediction", symbol=symbol, error=str(e), exc_info=True)
            return {'action': 'hold', 'confidence': 0.0}

    def _create_labels(self, df: pd.DataFrame) -> pd.Series:
        """Creates labels for supervised learning."""
        horizon = self.config.labeling_horizon
        threshold = self.config.labeling_threshold
        
        future_price = df['close'].shift(-horizon)
        price_change_pct = (future_price - df['close']) / df['close']

        labels = pd.Series(1, index=df.index)  # Default to 'hold'
        labels[price_change_pct > threshold] = 2  # 'buy'
        labels[price_change_pct < -threshold] = 0 # 'sell'
        
        return labels.iloc[:-horizon] # Drop last rows where future is unknown

    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms flat data into sequences for time-series models."""
        seq_length = self.config.sequence_length
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length + 1):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length-1]) # Label corresponds to the last element
        return np.array(X_seq), np.array(y_seq)

    def train(self, symbol: str, df: pd.DataFrame):
        """Trains all models in the ensemble for a given symbol."""
        logger.info("Starting model training", symbol=symbol)
        try:
            # 1. Prepare flat data
            labels = self._create_labels(df)
            features = df[self.config.feature_columns].loc[labels.index]
            
            X_flat = np.nan_to_num(features.values, nan=0.0)
            y_flat = labels.values

            if len(np.unique(y_flat)) < 3:
                logger.warning("Training data has fewer than 3 unique labels. Skipping training.", symbol=symbol, labels=np.unique(y_flat))
                return

            # 2. Create sequences for PyTorch models
            X_seq, y_seq = self._create_sequences(X_flat, y_flat)
            
            # 3. Create flat data for scikit-learn models (using last timestep of each sequence)
            X_flat_from_seq = X_seq[:, -1, :]
            y_flat_from_seq = y_seq

            # 4. Split data
            # Scikit-learn split
            X_train_flat, X_test_flat, y_train_flat, y_test_flat = train_test_split(
                X_flat_from_seq, y_flat_from_seq, test_size=0.2, random_state=42, stratify=y_flat_from_seq
            )
            # PyTorch split
            X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
                X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
            )
            
            models = self._get_or_create_symbol_models(symbol)

            # 5. Train scikit-learn models
            logger.info("Training scikit-learn models...", symbol=symbol)
            models['gb'].fit(X_train_flat, y_train_flat)
            models['technical'].fit(X_train_flat, y_train_flat)
            logger.info("Scikit-learn models trained.", symbol=symbol)

            # 6. Train PyTorch models
            logger.info("Training PyTorch models...", symbol=symbol)
            X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
            y_train_tensor = torch.LongTensor(y_train_seq).to(self.device)
            
            self._train_pytorch_model(models['lstm'], X_train_tensor, y_train_tensor)
            self._train_pytorch_model(models['attention'], X_train_tensor, y_train_tensor)
            logger.info("PyTorch models trained.", symbol=symbol)

            # 7. Save models
            self._save_models(symbol)
            self.is_trained = True
            logger.info("All models trained and saved successfully.", symbol=symbol)

        except Exception as e:
            logger.critical("An error occurred during model training", symbol=symbol, error=str(e), exc_info=True)
            self.is_trained = False # Mark as not trained if it fails

    def _train_pytorch_model(self, model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor):
        """Generic training loop for a PyTorch model."""
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.config.training_epochs):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 5 == 0:
                logger.debug(f"PyTorch model training", model=model.__class__.__name__, epoch=epoch+1, loss=loss.item())

    def _save_models(self, symbol: str):
        """Saves all trained models for a symbol to disk."""
        symbol_path_str = symbol.replace('/', '_')
        save_path = os.path.join(self.config.model_path, symbol_path_str)
        os.makedirs(save_path, exist_ok=True)
        
        models = self.symbol_models[symbol]
        joblib.dump(models['gb'], os.path.join(save_path, "gb_model.pkl"))
        joblib.dump(models['technical'], os.path.join(save_path, "technical_model.pkl"))
        torch.save(models['lstm'].state_dict(), os.path.join(save_path, "lstm_model.pth"))
        torch.save(models['attention'].state_dict(), os.path.join(save_path, "attention_model.pth"))
        logger.info("Saved models to disk", path=save_path)
