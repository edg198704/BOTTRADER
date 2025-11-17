from enum import Enum
import pandas as pd
from typing import Dict, Any

from bot_core.logger import get_logger
from bot_core.config import AIStrategyConfig

logger = get_logger(__name__)

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"

class MarketRegimeDetector:
    """Detects the current market regime based on technical indicators."""
    def __init__(self, config: AIStrategyConfig):
        self.config = config
        logger.info("MarketRegimeDetector initialized.")

    async def detect_regime(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes the provided DataFrame to determine the market regime.

        Args:
            symbol: The trading symbol.
            df: A DataFrame containing OHLCV data and technical indicators.

        Returns:
            A dictionary containing the detected 'regime' and 'confidence'.
        """
        if len(df) < 50:
            return {'regime': MarketRegime.UNKNOWN.value, 'confidence': 0.0}
        
        try:
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
            rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50

            trend_strength = (sma_20 - sma_50) / sma_50 if sma_50 > 0 else 0

            if trend_strength > 0.01:
                regime = MarketRegime.BULL
            elif trend_strength < -0.01:
                regime = MarketRegime.BEAR
            elif volatility > df['close'].pct_change().rolling(50).std().mean() * 1.5:
                regime = MarketRegime.VOLATILE
            else:
                regime = MarketRegime.SIDEWAYS

            confidence = min(1.0, abs(trend_strength) * 20 + (abs(rsi - 50) / 50))
            
            result = {'regime': regime.value, 'confidence': confidence}
            logger.debug("Market regime detected", symbol=symbol, result=result)
            return result
            
        except Exception as e:
            logger.error("Error in regime detection", symbol=symbol, error=str(e))
            return {'regime': MarketRegime.UNKNOWN.value, 'confidence': 0.0}
