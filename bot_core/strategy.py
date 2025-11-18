import abc
import pandas as pd
from typing import Dict, Any, Optional

from bot_core.logger import get_logger
from bot_core.config import StrategyConfig
from bot_core.ai.ensemble_learner import EnsembleLearner
from bot_core.ai.regime_detector import MarketRegimeDetector
from bot_core.position_manager import Position

logger = get_logger(__name__)

class TradingStrategy(abc.ABC):
    def __init__(self, config: StrategyConfig):
        self.config = config

    @abc.abstractmethod
    async def analyze_market(self, symbol: str, df: pd.DataFrame, position: Optional[Position]) -> Optional[Dict[str, Any]]:
        """Analyzes market data and returns a trading signal or None."""
        pass

class SimpleMACrossoverStrategy(TradingStrategy):
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.fast_ma_period = config.simple_ma.fast_ma_period
        self.slow_ma_period = config.simple_ma.slow_ma_period
        logger.info("SimpleMACrossoverStrategy initialized", fast_ma=self.fast_ma_period, slow_ma=self.slow_ma_period)

    async def analyze_market(self, symbol: str, df: pd.DataFrame, position: Optional[Position]) -> Optional[Dict[str, Any]]:
        if 'sma_fast' not in df.columns or 'sma_slow' not in df.columns:
            logger.warning("SMA indicators not found in DataFrame.", symbol=symbol)
            return None

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        is_bullish_cross = last_row['sma_fast'] > last_row['sma_slow'] and prev_row['sma_fast'] <= prev_row['sma_slow']
        is_bearish_cross = last_row['sma_fast'] < last_row['sma_slow'] and prev_row['sma_fast'] >= prev_row['sma_slow']

        if position:
            # Logic to close existing position
            if position.side == 'BUY' and is_bearish_cross:
                logger.info("Close Long signal detected (MA Crossover)", symbol=symbol)
                return {'action': 'CLOSE', 'symbol': symbol}
            if position.side == 'SELL' and is_bullish_cross:
                logger.info("Close Short signal detected (MA Crossover)", symbol=symbol)
                return {'action': 'CLOSE', 'symbol': symbol}
        else:
            # Logic to open new position
            if is_bullish_cross:
                logger.info("Open Long signal detected (MA Crossover)", symbol=symbol)
                return {'action': 'BUY', 'symbol': symbol}
            if is_bearish_cross:
                logger.info("Open Short signal detected (MA Crossover)", symbol=symbol)
                return {'action': 'SELL', 'symbol': symbol}

        return None

class AIEnsembleStrategy(TradingStrategy):
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.ai_config = config.ai_ensemble
        try:
            self.ensemble_learner = EnsembleLearner(self.ai_config)
            self.regime_detector = MarketRegimeDetector(self.ai_config)
            logger.info("AIEnsembleStrategy initialized")
        except ImportError as e:
            logger.critical("Failed to initialize AIEnsembleStrategy due to missing ML libraries", error=str(e))
            raise

    async def analyze_market(self, symbol: str, df: pd.DataFrame, position: Optional[Position]) -> Optional[Dict[str, Any]]:
        if not self.ensemble_learner.is_trained:
            logger.debug("AI models not trained yet, skipping analysis.", symbol=symbol)
            return None

        if self.ai_config.use_regime_filter:
            regime_result = await self.regime_detector.detect_regime(symbol, df)
            regime = regime_result.get('regime')
            if regime in ['sideways', 'unknown']:
                logger.debug("Market regime is sideways or unknown, holding position.", regime=regime, symbol=symbol)
                return None

        prediction = await self.ensemble_learner.predict(df, symbol)
        action = prediction.get('action')
        confidence = prediction.get('confidence', 0.0)

        logger.debug("AI prediction received", symbol=symbol, **prediction)

        if confidence < self.ai_config.confidence_threshold:
            return None

        if position:
            # Have a position, look for close signals
            if position.side == 'BUY' and action == 'sell':
                logger.info("AI Close Long signal detected", symbol=symbol, confidence=confidence)
                return {'action': 'CLOSE', 'symbol': symbol, 'confidence': confidence}
            if position.side == 'SELL' and action == 'buy':
                logger.info("AI Close Short signal detected", symbol=symbol, confidence=confidence)
                return {'action': 'CLOSE', 'symbol': symbol, 'confidence': confidence}
        else:
            # No position, look for open signals
            if action == 'buy':
                logger.info("AI Open Long signal detected", symbol=symbol, confidence=confidence)
                return {'action': 'BUY', 'symbol': symbol, 'confidence': confidence}
            if action == 'sell':
                logger.info("AI Open Short signal, detected", symbol=symbol, confidence=confidence)
                return {'action': 'SELL', 'symbol': symbol, 'confidence': confidence}

        return None
