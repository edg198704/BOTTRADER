import os
import joblib
import asyncio
from typing import Dict, Any
import pandas as pd
import numpy as np

from bot_core.logger import get_logger
from bot_core.config import AIStrategyConfig

# ML Imports with safe fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
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
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
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
        try:
            models = self._get_or_create_symbol_models(symbol)
            feature_cols = self.config.feature_columns
            
            current_features = {col: df[col].iloc[-1] if col in df.columns and not df[col].empty else 0 for col in feature_cols}
            features_df = pd.DataFrame([current_features], columns=feature_cols)
            features = np.nan_to_num(features_df.values, nan=0.0)

            predictions = []
            weights_config = self.config.ensemble_weights
            weights = [
                weights_config.xgboost,
                weights_config.technical_ensemble,
                weights_config.lstm,
                weights_config.attention
            ]

            # Scikit-learn models
            gb_pred = models['gb'].predict_proba(features)[0]
            predictions.append(gb_pred)
            tech_pred = models['technical'].predict_proba(features)[0]
            predictions.append(tech_pred)

            # PyTorch models
            features_tensor = torch.FloatTensor(features).unsqueeze(1).to(self.device)
            with torch.no_grad():
                lstm_pred = models['lstm'](features_tensor).cpu().numpy()[0]
                predictions.append(lstm_pred)
                attn_pred = models['attention'](features_tensor).cpu().numpy()[0]
                predictions.append(attn_pred)

            ensemble_pred = np.average(predictions, weights=weights, axis=0)
            action_idx = np.argmax(ensemble_pred)
            confidence = float(ensemble_pred[action_idx])
            action_map = {0: 'sell', 1: 'hold', 2: 'buy'}
            
            return {'action': action_map.get(action_idx, 'hold'), 'confidence': confidence}
        except Exception as e:
            logger.error("Error during prediction", symbol=symbol, error=str(e), exc_info=True)
            return {'action': 'hold', 'confidence': 0.0}
