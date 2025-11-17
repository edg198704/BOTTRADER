import abc
import logging
import os
import pickle
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

    async def analyze_market(self, ohlcv_df: pd.DataFrame, open_positions: List[Any]) -> Optional[Dict[str, Any]]:
        df = self._calculate_technical_indicators(ohlcv_df)
        if df is None or df.empty:
            return None

        if not self.is_trained:
            logger.warning("AI model is not trained. Cannot generate signals. Please train the model first.")
            # In a real system, you might trigger training here.
            return None

        if any(p.symbol == self.symbol for p in open_positions):
            return None # Don't open a new position if one already exists for this symbol

        prediction = self._predict(df)
        signal = prediction['signal']
        confidence = prediction['confidence']

        logger.debug(f"AI prediction for {self.symbol}: signal={signal}, confidence={confidence:.2f}")

        if signal == 1 and confidence > self.confidence_threshold: # BUY
            return {'action': 'BUY', 'confidence': confidence}
        elif signal == -1 and confidence > self.confidence_threshold: # SELL
            return {'action': 'SELL', 'confidence': confidence}
        
        return None

    async def manage_positions(self, ohlcv_df: pd.DataFrame, open_positions: List[Any]) -> List[Dict[str, Any]]:
        # This strategy relies on the RiskManager for SL/TP, so no specific management logic is needed here.
        return []

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def _predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        latest_features = df[self.feature_columns].iloc[-1:]
        
        pred_proba = self.model.predict_proba(latest_features)[0]
        
        # Assuming classes are ordered: 0=SELL (-1), 1=HOLD (0), 2=BUY (1)
        if len(pred_proba) != 3:
            logger.error("Model prediction shape is incorrect. Expected 3 classes.")
            return {'signal': 0, 'confidence': 0.0}

        confidence = np.max(pred_proba)
        signal_idx = np.argmax(pred_proba)

        signal_map = {-1: 0, 0: 1, 1: 2} # Our signal to class index
        reverse_map = {0: -1, 1: 0, 2: 1} # Class index to our signal
        final_signal = reverse_map.get(signal_idx, 0)

        return {'signal': final_signal, 'confidence': confidence}

    def _save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        model_file = os.path.join(self.model_path, f"{self.symbol.replace('/', '_')}_ensemble.pkl")
        try:
            with open(model_file, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"AI model saved to {model_file}")
        except Exception as e:
            logger.error(f"Error saving AI model: {e}")

    def _load_model(self):
        model_file = os.path.join(self.model_path, f"{self.symbol.replace('/', '_')}_ensemble.pkl")
        if not os.path.exists(model_file):
            logger.warning(f"No pre-trained AI model found at {model_file}. Model needs training.")
            return
        try:
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            self.is_trained = True
            logger.info(f"AI model loaded from {model_file}")
        except Exception as e:
            logger.error(f"Error loading AI model: {e}")
