import abc
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# --- Helper Functions for AI Strategy ---
def _calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) < 26: # Need enough data for MACD
        return df
    df = df.copy()
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df.fillna(method='bfill', inplace=True)
    return df

class AdvancedEnsembleLearner:
    """A self-contained ML model for the AI strategy."""
    def __init__(self, feature_columns: List[str]):
        self.feature_columns = feature_columns
        self.models = {
            'rf': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42)
        }
        self.is_trained = False

    def train(self, df: pd.DataFrame, buy_threshold: float, sell_threshold: float):
        logger.info("Starting AI model training...")
        df_with_indicators = _calculate_technical_indicators(df)
        
        # Create target variable
        future_returns = df_with_indicators['close'].pct_change(periods=-1).shift(1)
        df_with_indicators['target'] = 0 # HOLD
        df_with_indicators.loc[future_returns > buy_threshold, 'target'] = 1 # BUY
        df_with_indicators.loc[future_returns < sell_threshold, 'target'] = -1 # SELL
        df_with_indicators.dropna(inplace=True)

        if len(df_with_indicators) < 100:
            logger.warning("Not enough data to train AI model.")
            return

        X = df_with_indicators[self.feature_columns]
        y = df_with_indicators['target']

        if len(y.unique()) < 2:
            logger.warning("Not enough class diversity to train AI model.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            logger.info(f"Model '{name}' trained with accuracy: {score:.2f}")
        
        self.is_trained = True
        logger.info("AI model training complete.")

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.is_trained:
            return {'signal': 0, 'confidence': 0.0}
        
        df_with_indicators = _calculate_technical_indicators(df)
        latest_features = df_with_indicators[self.feature_columns].iloc[-1:]

        predictions = []
        confidences = []
        for model in self.models.values():
            pred = model.predict(latest_features)[0]
            proba = model.predict_proba(latest_features)[0]
            predictions.append(pred)
            confidences.append(np.max(proba))

        # Simple majority vote
        final_prediction = int(np.sign(sum(predictions)))
        avg_confidence = np.mean(confidences)

        return {'signal': final_prediction, 'confidence': avg_confidence}

# --- Strategy Interfaces ---

class TradingStrategy(abc.ABC):
    """Abstract Base Class for a trading strategy."""
    def __init__(self, config: Dict[str, Any]):
        self.symbol = config.get("symbol", "BTC/USDT")
        self.interval_seconds = config.get("interval_seconds", 60)
        self.trade_quantity = config.get("trade_quantity", 0.001)
        logger.info(f"{self.__class__.__name__} initialized for {self.symbol}.")

    @abc.abstractmethod
    async def analyze_market(self, ohlcv: List[List[float]], open_positions: List[Any]) -> Optional[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    async def manage_positions(self, ohlcv: List[List[float]], open_positions: List[Any]) -> List[Dict[str, Any]]:
        pass

class AIEnsembleStrategy(TradingStrategy):
    """Trading strategy based on an ensemble of AI models."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = AdvancedEnsembleLearner(config.get('feature_columns', []))
        self.buy_threshold = config.get('buy_threshold', 0.002)
        self.sell_threshold = config.get('sell_threshold', -0.0015)
        self.is_training = False

    async def _ensure_model_trained(self, df: pd.DataFrame):
        if not self.model.is_trained and not self.is_training:
            self.is_training = True
            try:
                # Run training in a separate thread to not block the event loop
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.model.train, df, self.buy_threshold, self.sell_threshold)
            finally:
                self.is_training = False

    async def analyze_market(self, ohlcv: List[List[float]], open_positions: List[Any]) -> Optional[Dict[str, Any]]:
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        await self._ensure_model_trained(df)

        if not self.model.is_trained or len(open_positions) > 0:
            return None

        prediction = self.model.predict(df)
        signal = prediction['signal']
        confidence = prediction['confidence']

        if signal == 1 and confidence > 0.6: # BUY signal
            return {'action': 'BUY', 'quantity': self.trade_quantity, 'order_type': 'MARKET'}
        elif signal == -1 and confidence > 0.6: # SELL signal
            return {'action': 'SELL', 'quantity': self.trade_quantity, 'order_type': 'MARKET'}
        
        return None

    async def manage_positions(self, ohlcv: List[List[float]], open_positions: List[Any]) -> List[Dict[str, Any]]:
        # For this strategy, we close positions based on an opposite signal or risk management (handled elsewhere)
        # A more advanced version could use the model to decide when to hold or close.
        return []

class SimpleMACrossoverStrategy(TradingStrategy):
    """A simple Moving Average Crossover strategy."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.fast_ma_period = config.get("fast_ma_period", 10)
        self.slow_ma_period = config.get("slow_ma_period", 20)

    async def analyze_market(self, ohlcv: List[List[float]], open_positions: List[Any]) -> Optional[Dict[str, Any]]:
        if not ohlcv or len(ohlcv) < self.slow_ma_period:
            return None

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['fast_ma'] = df['close'].rolling(window=self.fast_ma_period).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_ma_period).mean()
        
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        has_open_position = len(open_positions) > 0

        # Buy signal: Fast MA crosses above Slow MA
        if last_row['fast_ma'] > last_row['slow_ma'] and prev_row['fast_ma'] <= prev_row['slow_ma'] and not has_open_position:
            logger.info(f"Strategy: BUY signal for {self.symbol} at {last_row['close']:.2f}")
            return {'action': 'BUY', 'quantity': self.trade_quantity, 'order_type': 'MARKET'}

        return None

    async def manage_positions(self, ohlcv: List[List[float]], open_positions: List[Any]) -> List[Dict[str, Any]]:
        actions = []
        if not ohlcv or len(open_positions) == 0:
            return actions

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['fast_ma'] = df['close'].rolling(window=self.fast_ma_period).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_ma_period).mean()
        
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        for position in open_positions:
            # Close signal: Fast MA crosses below Slow MA
            if position.side == 'BUY' and last_row['fast_ma'] < last_row['slow_ma'] and prev_row['fast_ma'] >= prev_row['slow_ma']:
                logger.info(f"Strategy: Closing BUY position {position.id} for {position.symbol}.")
                actions.append({'action': 'CLOSE', 'position_id': position.id, 'order_type': 'MARKET'})
        
        return actions
