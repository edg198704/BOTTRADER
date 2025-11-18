import abc
import pandas as pd
from typing import Dict, Any, List, Optional

from bot_core.logger import get_logger
from bot_core.config import StrategyConfig
from bot_core.ai.ensemble_learner import EnsembleLearner
from bot_core.ai.regime_detector import MarketRegimeDetector
from bot_core.position_manager import Position

logger = get_logger(__name__)

class TradingStrategy(abc.ABC):
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.symbol = config.symbol

    @abc.abstractmethod
    async def analyze_market(self, df: pd.DataFrame, open_positions: List[Position]) -> Optional[Dict[str, Any]]:
        """Analyzes market data and returns a trading signal or None."""
        pass

class SimpleMACrossoverStrategy(TradingStrategy):
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.fast_ma_period = config.simple_ma.fast_ma_period
        self.slow_ma_period = config.simple_ma.slow_ma_period
        logger.info("SimpleMACrossoverStrategy initialized", fast_ma=self.fast_ma_period, slow_ma=self.slow_ma_period)

    async def analyze_market(self, df: pd.DataFrame, open_positions: List[Position]) -> Optional[Dict[str, Any]]:
        if 'sma_fast' not in df.columns or 'sma_slow' not in df.columns:
            logger.warning("SMA indicators not found in DataFrame.")
            return None

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        position_open = any(p.symbol == self.symbol for p in open_positions)

        # Buy signal: fast MA crosses above slow MA
        if last_row['sma_fast'] > last_row['sma_slow'] and prev_row['sma_fast'] <= prev_row['sma_slow'] and not position_open:
            logger.info("Buy signal detected (MA Crossover)", symbol=self.symbol)
            return {'action': 'BUY', 'symbol': self.symbol}

        # Sell signal: fast MA crosses below slow MA
        if last_row['sma_fast'] < last_row['sma_slow'] and prev_row['sma_fast'] >= prev_row['sma_slow'] and position_open:
            logger.info("Sell signal detected (MA Crossover)", symbol=self.symbol)
            return {'action': 'SELL', 'symbol': self.symbol}

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

    async def analyze_market(self, df: pd.DataFrame, open_positions: List[Position]) -> Optional[Dict[str, Any]]:
        if not self.ensemble_learner.is_trained:
            logger.debug("AI models not trained yet, skipping analysis.")
            return None

        position_open = any(p.symbol == self.symbol for p in open_positions)

        if self.ai_config.use_regime_filter:
            regime_result = await self.regime_detector.detect_regime(self.symbol, df)
            regime = regime_result.get('regime')
            if regime in ['sideways', 'unknown']:
                logger.debug("Market regime is sideways or unknown, holding position.", regime=regime)
                return None

        prediction = await self.ensemble_learner.predict(df, self.symbol)
        action = prediction.get('action')
        confidence = prediction.get('confidence', 0.0)

        logger.debug("AI prediction received", **prediction)

        if confidence < self.ai_config.confidence_threshold:
            return None

        if action == 'buy' and not position_open:
            logger.info("AI Buy signal detected", symbol=self.symbol, confidence=confidence)
            return {'action': 'BUY', 'symbol': self.symbol, 'confidence': confidence}
        
        if action == 'sell' and position_open:
            logger.info("AI Sell signal detected", symbol=self.symbol, confidence=confidence)
            return {'action': 'SELL', 'symbol': self.symbol, 'confidence': confidence}

        return None
