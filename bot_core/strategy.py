from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional, List

from bot_core.logger import get_logger
from bot_core.config import StrategyConfig
from bot_core.position_manager import Position

logger = get_logger(__name__)

class TradingStrategy(ABC):
    """Abstract base class for all trading strategies."""
    def __init__(self, config: StrategyConfig):
        self.config = config

    @abstractmethod
    async def analyze_market(self, df: pd.DataFrame, open_positions: List[Position]) -> Optional[Dict]:
        """
        Analyzes market data and returns a trading signal if conditions are met.
        Returns a dictionary with 'action' ('BUY' or 'SELL') or None.
        """
        pass

class SimpleMACrossoverStrategy(TradingStrategy):
    """A simple moving average crossover strategy."""
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.fast_ma_period = config.simple_ma.fast_ma_period
        self.slow_ma_period = config.simple_ma.slow_ma_period
        logger.info("SimpleMACrossoverStrategy initialized", fast_period=self.fast_ma_period, slow_period=self.slow_ma_period)

    async def analyze_market(self, df: pd.DataFrame, open_positions: List[Position]) -> Optional[Dict]:
        symbol = self.config.symbol
        position_open = any(p.symbol == symbol for p in open_positions)

        if 'sma_fast' not in df.columns or 'sma_slow' not in df.columns:
            logger.warning("Required SMA columns not in DataFrame. Skipping analysis.")
            return None

        # Get the last two candles for crossover detection
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]

        # Golden Cross (Buy Signal)
        is_golden_cross = prev_candle['sma_fast'] <= prev_candle['sma_slow'] and last_candle['sma_fast'] > last_candle['sma_slow']
        if is_golden_cross and not position_open:
            logger.info("Golden Cross detected. Generating BUY signal.", symbol=symbol)
            return {'action': 'BUY', 'symbol': symbol}

        # Death Cross (Sell Signal)
        is_death_cross = prev_candle['sma_fast'] >= prev_candle['sma_slow'] and last_candle['sma_fast'] < last_candle['sma_slow']
        if is_death_cross and position_open:
            logger.info("Death Cross detected. Generating SELL signal to close position.", symbol=symbol)
            return {'action': 'SELL', 'symbol': symbol}

        return None

class AIEnsembleStrategy(TradingStrategy):
    """Placeholder for the advanced AI ensemble strategy."""
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        logger.info("AIEnsembleStrategy initialized (placeholder).")
        # In a future iteration, this would initialize the EnsembleLearner, etc.

    async def analyze_market(self, df: pd.DataFrame, open_positions: List[Position]) -> Optional[Dict]:
        logger.debug("AIEnsembleStrategy analyze_market called. No logic implemented yet.")
        # TODO: Implement AI-based signal generation using EnsembleLearner and RegimeDetector
        return None
