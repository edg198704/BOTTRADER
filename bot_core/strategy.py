import abc
from typing import Dict, Any, Optional
import pandas as pd

from bot_core.logger import get_logger
from bot_core.config import StrategyConfig
from bot_core.ai.ensemble_learner import EnsembleLearner
from bot_core.ai.regime_detector import MarketRegimeDetector, MarketRegime

logger = get_logger(__name__)

class TradingStrategy(abc.ABC):
    """Abstract Base Class for all trading strategies."""

    @abc.abstractmethod
    async def analyze_market(self, df: pd.DataFrame, open_positions: Dict) -> Optional[Dict[str, Any]]:
        """Analyzes market data and returns a trading signal or None."""
        pass

class AIEnsembleStrategy(TradingStrategy):
    """An advanced strategy using an ensemble of ML models and market regime detection."""
    def __init__(self, config: StrategyConfig):
        self.config = config.ai_ensemble
        self.symbol = config.symbol
        self.ensemble_learner = EnsembleLearner(self.config)
        self.regime_detector = MarketRegimeDetector(self.config)
        logger.info("AIEnsembleStrategy initialized.")

    async def analyze_market(self, df: pd.DataFrame, open_positions: Dict) -> Optional[Dict[str, Any]]:
        # 1. Detect Market Regime
        regime_result = await self.regime_detector.detect_regime(self.symbol, df)
        regime = regime_result.get('regime')

        # 2. Regime Filtering
        if self.config.use_regime_filter and regime in [MarketRegime.SIDEWAYS.value, MarketRegime.UNKNOWN.value]:
            logger.debug("Holding due to market regime", regime=regime)
            return None

        # 3. Get AI Prediction
        prediction = await self.ensemble_learner.predict(df, self.symbol)
        action = prediction.get('action')
        confidence = prediction.get('confidence', 0.0)

        if not action or action == 'hold' or confidence < self.config.confidence_threshold:
            return None

        # 4. Generate Signal
        signal = {
            'symbol': self.symbol,
            'action': action.upper(),
            'confidence': confidence,
            'strategy': 'AIEnsembleStrategy',
            'regime': regime
        }
        logger.info("Generated AI signal", **signal)
        return signal

class SimpleMACrossoverStrategy(TradingStrategy):
    """A simple moving average crossover strategy for testing and as a baseline."""
    def __init__(self, config: StrategyConfig):
        self.config = config.simple_ma
        self.symbol = config.symbol
        logger.info("SimpleMACrossoverStrategy initialized.")

    async def analyze_market(self, df: pd.DataFrame, open_positions: Dict) -> Optional[Dict[str, Any]]:
        if 'sma_fast' not in df.columns or 'sma_slow' not in df.columns:
            logger.warning("MA indicators not found in DataFrame.")
            return None

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        # Check for crossover
        fast_above_slow_now = last_row['sma_fast'] > last_row['sma_slow']
        fast_above_slow_before = prev_row['sma_fast'] > prev_row['sma_slow']

        action = None
        if fast_above_slow_now and not fast_above_slow_before:
            action = 'BUY'
        elif not fast_above_slow_now and fast_above_slow_before:
            action = 'SELL'

        if action:
            signal = {
                'symbol': self.symbol,
                'action': action,
                'confidence': 0.8, # Fixed confidence for simple strategy
                'strategy': 'SimpleMACrossoverStrategy'
            }
            logger.info("Generated MA Crossover signal", **signal)
            return signal
        
        return None
