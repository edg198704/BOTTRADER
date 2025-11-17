import abc
import logging
import os
import joblib
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np

# ML Imports with safe fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    # Define dummy classes if ML libraries are not available
    class VotingClassifier: pass
    class RandomForestClassifier: pass
    class LogisticRegression: pass
    class XGBClassifier: pass
    class nn:
        class Module: pass

logger = logging.getLogger(__name__)

# --- Strategy Interface ---

class TradingStrategy(abc.ABC):
    """Abstract Base Class for a trading strategy."""
    def __init__(self, config: Dict[str, Any]):
        self.symbol = config.get("symbol", "BTC/USDT")
        self.interval_seconds = config.get("interval_seconds", 60)
        logger.info(f"{self.__class__.__name__} initialized for {self.symbol}.")

    @abc.abstractmethod
    async def analyze_market(self, ohlcv_df: pd.DataFrame, open_positions: List[Any]) -> Optional[Dict[str, Any]]:
        """Analyzes market data to generate a new trade signal (BUY/SELL)."""
        pass

    @abc.abstractmethod
    async def manage_positions(self, ohlcv_df: pd.DataFrame, open_positions: List[Any]) -> List[Dict[str, Any]]:
        """Manages existing open positions (e.g., signals to close)."""
        pass

# --- Simple MA Crossover Strategy ---

class SimpleMACrossoverStrategy(TradingStrategy):
    """A simple Moving Average Crossover strategy."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.fast_ma_period = config.get("fast_ma_period", 10)
        self.slow_ma_period = config.get("slow_ma_period", 20)

    async def analyze_market(self, ohlcv_df: pd.DataFrame, open_positions: List[Any]) -> Optional[Dict[str, Any]]:
        if ohlcv_df.empty or len(ohlcv_df) < self.slow_ma_period:
            return None

        df = ohlcv_df.copy()
        df['fast_ma'] = df['close'].rolling(window=self.fast_ma_period).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_ma_period).mean()
        
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        has_open_position = any(p.symbol == self.symbol for p in open_positions)

        # Buy signal: Fast MA crosses above Slow MA
        if not has_open_position and last_row['fast_ma'] > last_row['slow_ma'] and prev_row['fast_ma'] <= prev_row['slow_ma']:
            logger.info(f"Strategy: BUY signal for {self.symbol} at {last_row['close']:.2f}")
            return {'action': 'BUY', 'confidence': 0.75} # Static confidence for simple strategy

        return None

    async def manage_positions(self, ohlcv_df: pd.DataFrame, open_positions: List[Any]) -> List[Dict[str, Any]]:
        actions = []
        position_to_manage = next((p for p in open_positions if p.symbol == self.symbol), None)
        if ohlcv_df.empty or not position_to_manage:
            return actions

        df = ohlcv_df.copy()
        df['fast_ma'] = df['close'].rolling(window=self.fast_ma_period).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_ma_period).mean()
        
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        # Close signal: Fast MA crosses below Slow MA
        if position_to_manage.side == 'BUY' and last_row['fast_ma'] < last_row['slow_ma'] and prev_row['fast_ma'] >= prev_row['slow_ma']:
            logger.info(f"Strategy: Closing BUY position {position_to_manage.id} for {position_to_manage.symbol}.")
            actions.append({'action': 'CLOSE', 'position_id': position_to_manage.id})
        
        return actions

# --- AI Ensemble Strategy ---

class AIEnsembleStrategy(TradingStrategy):
    """Trading strategy based on an ensemble of AI models."""

    class LSTMPredictor(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 3)

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
            self.fc = nn.Linear(hidden_dim, 3)

        def forward(self, x):
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            x = self.embedding(x)
            x = self.transformer(x)
            x = x.mean(dim=1)
            return F.softmax(self.fc(x), dim=1)

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not ML_AVAILABLE:
            raise ImportError("Required machine learning libraries are not installed for AIEnsembleStrategy.")
        
        self.feature_columns = config.get('feature_columns', ['close', 'rsi', 'macd', 'volume'])
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.model_path = config.get('model_path', 'models/ensemble')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models: Dict[str, Any] = {}
        self.is_trained = False
        self._initialize_models()
        self._load_models()

    def _initialize_models(self):
        """Initializes the ensemble model structure."""
        try:
            self.models['lstm'] = self.LSTMPredictor(len(self.feature_columns), 64).to(self.device)
            self.models['attention'] = self.AttentionNetwork(len(self.feature_columns), 64).to(self.device)
            self.models['gb'] = XGBClassifier(n_estimators=10, max_depth=5, random_state=42, verbosity=0, use_label_encoder=False, eval_metric='logloss')
            self.models['technical'] = VotingClassifier(estimators=[
                ('rf', RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)),
                ('lr', LogisticRegression(max_iter=100, random_state=42))
            ], voting='soft')
            logger.info("AI ensemble models initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
            self.models = {}

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
        rs = gain / (loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df.dropna(inplace=True)
        return df

    async def analyze_market(self, ohlcv_df: pd.DataFrame, open_positions: List[Any]) -> Optional[Dict[str, Any]]:
        df_with_features = self._calculate_features(ohlcv_df)
        if df_with_features is None or df_with_features.empty:
            return None

        if not self.is_trained:
            logger.warning("AI model is not trained. Cannot generate signals.")
            return None

        if any(p.symbol == self.symbol for p in open_positions):
            return None # Don't open a new position if one already exists for this symbol

        prediction = self._predict(df_with_features)
        action = prediction['action']
        confidence = prediction['confidence']

        logger.debug(f"AI prediction for {self.symbol}: action={action}, confidence={confidence:.2f}")

        if action != 'HOLD' and confidence > self.confidence_threshold:
            # Map to BUY/SELL for the bot core
            final_action = 'BUY' if action == 'buy' else 'SELL'
            return {'action': final_action, 'confidence': confidence}
        
        return None

    async def manage_positions(self, ohlcv_df: pd.DataFrame, open_positions: List[Any]) -> List[Dict[str, Any]]:
        actions = []
        position_to_manage = next((p for p in open_positions if p.symbol == self.symbol), None)
        if ohlcv_df.empty or not position_to_manage or not self.is_trained:
            return actions

        df_with_features = self._calculate_features(ohlcv_df)
        if df_with_features is None or df_with_features.empty:
            return actions

        prediction = self._predict(df_with_features)
        current_signal = prediction['action']

        is_opposing_signal = (position_to_manage.side == 'BUY' and current_signal == 'sell') or \
                               (position_to_manage.side == 'SELL' and current_signal == 'buy')

        if is_opposing_signal and prediction['confidence'] > self.confidence_threshold:
            logger.info(f"Strategy: Closing {position_to_manage.side} position {position_to_manage.id} for {self.symbol} due to opposing signal '{current_signal}'.")
            actions.append({'action': 'CLOSE', 'position_id': position_to_manage.id})
        
        return actions

    def _predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.is_trained or not self.models:
            return {'action': 'HOLD', 'confidence': 0.0}

        try:
            latest_features = df[self.feature_columns].iloc[-1:].values
            latest_features = np.nan_to_num(latest_features, nan=0.0, posinf=0.0, neginf=0.0)

            predictions = []
            weights = []

            # LSTM prediction
            if 'lstm' in self.models:
                features_lstm = latest_features.reshape(1, 1, -1)
                with torch.no_grad():
                    lstm_out = self.models['lstm'](torch.FloatTensor(features_lstm).to(self.device))
                    predictions.append(F.softmax(lstm_out, dim=1).cpu().numpy()[0])
                    weights.append(0.3)
            
            # Attention prediction
            if 'attention' in self.models:
                features_attn = latest_features.reshape(1, 1, -1)
                with torch.no_grad():
                    attn_out = self.models['attention'](torch.FloatTensor(features_attn).to(self.device))
                    predictions.append(F.softmax(attn_out, dim=1).cpu().numpy()[0])
                    weights.append(0.3)

            # Gradient Boosting prediction
            if 'gb' in self.models:
                predictions.append(self.models['gb'].predict_proba(latest_features)[0])
                weights.append(0.25)

            # Technical models prediction
            if 'technical' in self.models:
                predictions.append(self.models['technical'].predict_proba(latest_features)[0])
                weights.append(0.15)

            if not predictions:
                return {'action': 'HOLD', 'confidence': 0.0}

            ensemble_pred = np.average(predictions, weights=weights, axis=0)
            action_idx = np.argmax(ensemble_pred)
            confidence = float(ensemble_pred[action_idx])

            action_map = {0: 'sell', 1: 'hold', 2: 'buy'} # Assumes classes: 0=sell, 1=hold, 2=buy
            final_action = action_map.get(action_idx, 'hold')

            return {'action': final_action, 'confidence': confidence}
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {'action': 'HOLD', 'confidence': 0.0}

    def _save_models(self):
        os.makedirs(self.model_path, exist_ok=True)
        symbol_safe = self.symbol.replace('/', '_')
        try:
            joblib.dump(self.models['gb'], os.path.join(self.model_path, f"{symbol_safe}_gb.joblib"))
            joblib.dump(self.models['technical'], os.path.join(self.model_path, f"{symbol_safe}_technical.joblib"))
            torch.save(self.models['lstm'].state_dict(), os.path.join(self.model_path, f"{symbol_safe}_lstm.pth"))
            torch.save(self.models['attention'].state_dict(), os.path.join(self.model_path, f"{symbol_safe}_attention.pth"))
            logger.info(f"AI models for {self.symbol} saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving AI models: {e}")

    def _load_models(self):
        symbol_safe = self.symbol.replace('/', '_')
        try:
            gb_path = os.path.join(self.model_path, f"{symbol_safe}_gb.joblib")
            tech_path = os.path.join(self.model_path, f"{symbol_safe}_technical.joblib")
            lstm_path = os.path.join(self.model_path, f"{symbol_safe}_lstm.pth")
            attn_path = os.path.join(self.model_path, f"{symbol_safe}_attention.pth")

            if not all(os.path.exists(p) for p in [gb_path, tech_path, lstm_path, attn_path]):
                logger.warning(f"No pre-trained AI models found for {self.symbol} in {self.model_path}. Model needs training.")
                return

            self.models['gb'] = joblib.load(gb_path)
            self.models['technical'] = joblib.load(tech_path)
            self.models['lstm'].load_state_dict(torch.load(lstm_path, map_location=self.device))
            self.models['attention'].load_state_dict(torch.load(attn_path, map_location=self.device))
            
            self.is_trained = True
            logger.info(f"AI models for {self.symbol} loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading AI models: {e}")
            self.is_trained = False
