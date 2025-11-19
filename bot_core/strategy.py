import abc
import asyncio
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from concurrent.futures import Executor

from bot_core.logger import get_logger
from bot_core.config import AIEnsembleStrategyParams, SimpleMACrossoverStrategyParams, StrategyParamsBase
from bot_core.ai.ensemble_learner import EnsembleLearner, train_ensemble_task
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
    async def retrain(self, symbol: str, df: pd.DataFrame, executor: Executor):
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
        self.fast_ma_col = f"SMA_{self.fast_ma_period}"
        self.slow_ma_col = f"SMA_{self.slow_ma_period}"
        logger.info("SimpleMACrossoverStrategy initialized", fast_ma=self.fast_ma_period, slow_ma=self.slow_ma_period)

    async def analyze_market(self, symbol: str, df: pd.DataFrame, position: Optional[Position]) -> Optional[Dict[str, Any]]:
        if self.fast_ma_col not in df.columns or self.slow_ma_col not in df.columns:
            logger.warning("Required SMA indicators not found in DataFrame.", 
                         symbol=symbol, required=[self.fast_ma_col, self.slow_ma_col])
            return None

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        is_bullish_cross = last_row[self.fast_ma_col] > last_row[self.slow_ma_col] and prev_row[self.fast_ma_col] <= prev_row[self.slow_ma_col]
        is_bearish_cross = last_row[self.fast_ma_col] < last_row[self.slow_ma_col] and prev_row[self.fast_ma_col] >= prev_row[self.slow_ma_col]

        if position:
            if position.side == 'BUY' and is_bearish_cross:
                logger.info("Close Long signal detected (MA Crossover)", symbol=symbol)
                return {'action': 'SELL', 'symbol': symbol}
            if position.side == 'SELL' and is_bullish_cross:
                logger.info("Close Short signal detected (MA Crossover)", symbol=symbol)
                return {'action': 'BUY', 'symbol': symbol}
        else:
            if is_bullish_cross:
                logger.info("Open Long signal detected (MA Crossover)", symbol=symbol)
                return {'action': 'BUY', 'symbol': symbol}
            if is_bearish_cross:
                logger.info("Open Short signal detected (MA Crossover)", symbol=symbol)
                return {'action': 'SELL', 'symbol': symbol}

        return None

    async def retrain(self, symbol: str, df: pd.DataFrame, executor: Executor):
        pass

    def needs_retraining(self, symbol: str) -> bool:
        return False

    def get_training_data_limit(self) -> int:
        return 0

class AIEnsembleStrategy(TradingStrategy):
    def __init__(self, config: AIEnsembleStrategyParams):
        super().__init__(config)
        self.ai_config = config
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
        model_version = prediction.get('model_version')
        active_weights = prediction.get('active_weights')

        logger.debug("AI prediction received", symbol=symbol, **prediction)

        if confidence < self.ai_config.confidence_threshold:
            return None

        # Construct metadata for the signal
        strategy_metadata = {
            'model_version': model_version,
            'confidence': confidence,
            'regime': regime,
            'regime_confidence': regime_result.get('confidence'),
            'model_type': prediction.get('model_type'),
            'active_weights': active_weights
        }

        if position:
            if position.side == 'BUY' and action == 'sell':
                logger.info("AI Close Long signal detected", symbol=symbol, confidence=confidence)
                return {'action': 'SELL', 'symbol': symbol, 'strategy_metadata': strategy_metadata}
            if position.side == 'SELL' and action == 'buy':
                logger.info("AI Close Short signal detected", symbol=symbol, confidence=confidence)
                return {'action': 'BUY', 'symbol': symbol, 'strategy_metadata': strategy_metadata}
        else:
            if action == 'buy':
                logger.info("AI Open Long signal detected", symbol=symbol, confidence=confidence)
                return {'action': 'BUY', 'symbol': symbol, 'strategy_metadata': strategy_metadata}
            if action == 'sell':
                logger.info("AI Open Short signal detected", symbol=symbol, confidence=confidence)
                return {'action': 'SELL', 'symbol': symbol, 'strategy_metadata': strategy_metadata}

        return None

    async def retrain(self, symbol: str, df: pd.DataFrame, executor: Executor):
        """Initiates the retraining process using the provided executor."""
        logger.info("Strategy is triggering model retraining via ProcessPool.", symbol=symbol)
        
        loop = asyncio.get_running_loop()
        try:
            # Offload the heavy training task to a separate process
            success = await loop.run_in_executor(
                executor, 
                train_ensemble_task, 
                symbol, 
                df, 
                self.ai_config
            )
            
            if success:
                self.last_retrained_at[symbol] = datetime.utcnow()
                logger.info("Model training successful. Reloading models...", symbol=symbol)
                await self.ensemble_learner.reload_models(symbol)
            else:
                logger.warning("Model training failed or rejected.", symbol=symbol)
                
        except Exception as e:
            logger.error("Error during async model retraining", symbol=symbol, error=str(e))

    def needs_retraining(self, symbol: str) -> bool:
        if self.ai_config.retrain_interval_hours <= 0:
            return False
        
        last_retrained = self.last_retrained_at.get(symbol)
        if not last_retrained:
            return True
        
        time_since_retrain = datetime.utcnow() - last_retrained
        return time_since_retrain >= timedelta(hours=self.ai_config.retrain_interval_hours)

    def get_training_data_limit(self) -> int:
        return self.ai_config.training_data_limit
