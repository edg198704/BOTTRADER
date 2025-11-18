import abc
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from bot_core.logger import get_logger
from bot_core.config import AIEnsembleStrategyParams, SimpleMACrossoverStrategyParams, StrategyParamsBase
from bot_core.ai.ensemble_learner import EnsembleLearner
from bot_core.ai.regime_detector import MarketRegimeDetector
from bot_core.position_manager import Position

logger = get_logger(__name__)

class TradingStrategy(abc.ABC):
    def __init__(self, config: StrategyParamsBase):
        self.config = config

    @abc.abstractmethod
    async def analyze_market(self, symbol: str, df: pd.DataFrame, position: Optional[Position]) -> Optional[Dict[str, Any]]:
        """Analyzes market data and returns a trading signal or None."""
        pass

    @abc.abstractmethod
    def retrain(self, symbol: str, df: pd.DataFrame):
        """Triggers the retraining of the strategy's underlying models."""
        pass

    @abc.abstractmethod
    def needs_retraining(self, symbol: str) -> bool:
        """Checks if the strategy needs to be retrained."""
        pass

    @abc.abstractmethod
    def get_training_data_limit(self) -> int:
        """Returns the number of historical candles needed for training."""
        pass

class SimpleMACrossoverStrategy(TradingStrategy):
    def __init__(self, config: SimpleMACrossoverStrategyParams):
        super().__init__(config)
        self.fast_ma_period = config.fast_ma_period
        self.slow_ma_period = config.slow_ma_period
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
            # Logic to close existing position with an opposing signal
            if position.side == 'BUY' and is_bearish_cross:
                logger.info("Close Long signal detected (MA Crossover)", symbol=symbol)
                return {'action': 'SELL', 'symbol': symbol}
            if position.side == 'SELL' and is_bullish_cross:
                logger.info("Close Short signal detected (MA Crossover)", symbol=symbol)
                return {'action': 'BUY', 'symbol': symbol}
        else:
            # Logic to open new position
            if is_bullish_cross:
                logger.info("Open Long signal detected (MA Crossover)", symbol=symbol)
                return {'action': 'BUY', 'symbol': symbol}
            if is_bearish_cross:
                logger.info("Open Short signal detected (MA Crossover)", symbol=symbol)
                return {'action': 'SELL', 'symbol': symbol}

        return None

    def retrain(self, symbol: str, df: pd.DataFrame):
        # No models to retrain for this strategy
        pass

    def needs_retraining(self, symbol: str) -> bool:
        return False

    def get_training_data_limit(self) -> int:
        return 0 # This strategy does not require training data.

class AIEnsembleStrategy(TradingStrategy):
    def __init__(self, config: AIEnsembleStrategyParams):
        super().__init__(config)
        self.ai_config = config # The config is already the specific AI config
        self.last_retrained_at: Dict[str, datetime] = {}
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

        regime_result = await self.regime_detector.detect_regime(symbol, df)
        regime = regime_result.get('regime')

        if self.ai_config.use_regime_filter:
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
            # Have a position, look for close signals (opposing action)
            if position.side == 'BUY' and action == 'sell':
                logger.info("AI Close Long signal detected", symbol=symbol, confidence=confidence)
                return {'action': 'SELL', 'symbol': symbol, 'confidence': confidence, 'regime': regime}
            if position.side == 'SELL' and action == 'buy':
                logger.info("AI Close Short signal detected", symbol=symbol, confidence=confidence)
                return {'action': 'BUY', 'symbol': symbol, 'confidence': confidence, 'regime': regime}
        else:
            # No position, look for open signals
            if action == 'buy':
                logger.info("AI Open Long signal detected", symbol=symbol, confidence=confidence)
                return {'action': 'BUY', 'symbol': symbol, 'confidence': confidence, 'regime': regime}
            if action == 'sell':
                logger.info("AI Open Short signal detected", symbol=symbol, confidence=confidence)
                return {'action': 'SELL', 'symbol': symbol, 'confidence': confidence, 'regime': regime}

        return None

    def retrain(self, symbol: str, df: pd.DataFrame):
        """Initiates the retraining process for the ensemble learner."""
        if not hasattr(self, 'ensemble_learner'):
            logger.warning("Ensemble learner not available, cannot retrain.")
            return
        
        logger.info("Strategy is triggering model retraining.", symbol=symbol)
        self.ensemble_learner.train(symbol, df)
        self.last_retrained_at[symbol] = datetime.utcnow()
        logger.info("Model retraining process completed.", symbol=symbol)

    def needs_retraining(self, symbol: str) -> bool:
        """Checks if enough time has passed since the last retraining."""
        if self.ai_config.retrain_interval_hours <= 0:
            return False # Retraining is disabled
        
        last_retrained = self.last_retrained_at.get(symbol)
        if not last_retrained:
            return True # Never been trained, so it needs it
        
        time_since_retrain = datetime.utcnow() - last_retrained
        return time_since_retrain >= timedelta(hours=self.ai_config.retrain_interval_hours)

    def get_training_data_limit(self) -> int:
        """Returns the number of candles required for retraining the AI models."""
        return self.ai_config.training_data_limit
