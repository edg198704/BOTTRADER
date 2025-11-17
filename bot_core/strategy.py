import abc
import logging
import os
import joblib
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np

# ML Imports with safe fallbacks
try:
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

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not ML_AVAILABLE:
            raise ImportError("Required machine learning libraries are not installed for AIEnsembleStrategy.")
        
        self.feature_columns = config.get('feature_columns', ['close', 'rsi', 'macd', 'volume'])
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.model_path = config.get('model_path', 'models/ensemble')
        
        self.model: Optional[VotingClassifier] = None
        self.is_trained = False
        self._initialize_model()
        self._load_model()

    def _initialize_model(self):
        """Initializes the ensemble model structure."""
        clf1 = RandomForestClassifier(n_estimators=50, random_state=1)
        clf2 = XGBClassifier(n_estimators=50, random_state=1, use_label_encoder=False, eval_metric='logloss')
        clf3 = LogisticRegression(random_state=1)
        self.model = VotingClassifier(estimators=[('rf', clf1), ('xgb', clf2), ('lr', clf3)], voting='soft')
        logger.info("AI ensemble model initialized.")

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
        df = self._calculate_features(ohlcv_df)
        if df is None or df.empty:
            return None

        if not self.is_trained:
            logger.warning("AI model is not trained. Cannot generate signals. Please train the model first.")
            return None

        if any(p.symbol == self.symbol for p in open_positions):
            return None # Don't open a new position if one already exists for this symbol

        prediction = self._predict(df)
        action = prediction['action']
        confidence = prediction['confidence']

        logger.debug(f"AI prediction for {self.symbol}: action={action}, confidence={confidence:.2f}")

        if action != 'HOLD' and confidence > self.confidence_threshold:
            return {'action': action, 'confidence': confidence}
        
        return None

    async def manage_positions(self, ohlcv_df: pd.DataFrame, open_positions: List[Any]) -> List[Dict[str, Any]]:
        actions = []
        position_to_manage = next((p for p in open_positions if p.symbol == self.symbol), None)
        if ohlcv_df.empty or not position_to_manage or not self.is_trained:
            return actions

        df = self._calculate_features(ohlcv_df)
        if df is None or df.empty:
            return actions

        prediction = self._predict(df)
        current_signal = prediction['action']

        # Close if signal is opposite to current position
        is_opposing_signal = (position_to_manage.side == 'BUY' and current_signal == 'SELL') or \
                               (position_to_manage.side == 'SELL' and current_signal == 'BUY')

        if is_opposing_signal and prediction['confidence'] > self.confidence_threshold:
            logger.info(f"Strategy: Closing {position_to_manage.side} position {position_to_manage.id} for {self.symbol} due to opposing signal '{current_signal}'.")
            actions.append({'action': 'CLOSE', 'position_id': position_to_manage.id})
        
        return actions

    def _predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        latest_features = df[self.feature_columns].iloc[-1:]
        
        pred_proba = self.model.predict_proba(latest_features)[0]
        
        # Classes are assumed to be trained as: 0=SELL, 1=HOLD, 2=BUY
        if len(pred_proba) != 3:
            logger.error("Model prediction shape is incorrect. Expected 3 classes.")
            return {'action': 'HOLD', 'confidence': 0.0}

        confidence = np.max(pred_proba)
        signal_idx = np.argmax(pred_proba)

        action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        final_action = action_map.get(signal_idx, 'HOLD')

        return {'action': final_action, 'confidence': confidence}

    def _save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        model_file = os.path.join(self.model_path, f"{self.symbol.replace('/', '_')}_ensemble.joblib")
        try:
            joblib.dump(self.model, model_file)
            logger.info(f"AI model saved to {model_file}")
        except Exception as e:
            logger.error(f"Error saving AI model: {e}")

    def _load_model(self):
        model_file = os.path.join(self.model_path, f"{self.symbol.replace('/', '_')}_ensemble.joblib")
        if not os.path.exists(model_file):
            logger.warning(f"No pre-trained AI model found at {model_file}. Model needs training.")
            return
        try:
            self.model = joblib.load(model_file)
            self.is_trained = True
            logger.info(f"AI model loaded from {model_file}")
        except Exception as e:
            logger.error(f"Error loading AI model: {e}")
