import abc
import asyncio
from typing import Dict, Any, List

from bot_core.logger import get_logger
from bot_core.data_handler import MarketEvent, SignalEvent, FillEvent
from bot_core.ai.regime_detector import MarketRegimeDetector, MarketRegime
from bot_core.ai.ensemble_learner import EnsembleLearner

logger = get_logger(__name__)

class TradingStrategy(abc.ABC):
    """Abstract Base Class for a trading strategy."""
    def __init__(self, event_queue: asyncio.Queue, config: Dict[str, Any]):
        self.event_queue = event_queue
        self.symbol = config.get("symbol", "BTC/USDT")
        self.interval_seconds = config.get("interval_seconds", 60)
        logger.info("Strategy initialized", strategy_name=self.__class__.__name__, symbol=self.symbol)

    @abc.abstractmethod
    async def on_market_event(self, event: MarketEvent, open_positions: List[Any]):
        """Reacts to new market data to generate trade signals."""
        pass

    @abc.abstractmethod
    async def on_fill_event(self, event: FillEvent):
        """Reacts to fill events to manage position state within the strategy."""
        pass

class SimpleMACrossoverStrategy(TradingStrategy):
    """A simple Moving Average Crossover strategy."""
    def __init__(self, event_queue: asyncio.Queue, config: Dict[str, Any]):
        super().__init__(event_queue, config)
        from bot_core.config import SimpleMAStrategyConfig
        ma_config = SimpleMAStrategyConfig(**config.get('simple_ma', {}))
        self.fast_ma_period = ma_config.fast_ma_period
        self.slow_ma_period = ma_config.slow_ma_period

    async def on_market_event(self, event: MarketEvent, open_positions: List[Any]):
        if event.symbol != self.symbol:
            return

        ohlcv_df = event.ohlcv_df
        if ohlcv_df.empty or len(ohlcv_df) < self.slow_ma_period:
            return

        df = ohlcv_df.copy()
        df['fast_ma'] = df['close'].rolling(window=self.fast_ma_period).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_ma_period).mean()
        
        if df['fast_ma'].isna().any() or df['slow_ma'].isna().any():
            return

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        has_open_position = any(p.symbol == self.symbol for p in open_positions)

        # Golden Cross (Buy Signal)
        if not has_open_position and last_row['fast_ma'] > last_row['slow_ma'] and prev_row['fast_ma'] <= prev_row['slow_ma']:
            logger.info("BUY signal generated", strategy="SimpleMACrossover", symbol=self.symbol, price=last_row['close'])
            signal = SignalEvent(symbol=self.symbol, action='BUY', confidence=0.75)
            await self.event_queue.put(signal)

        # Death Cross (Sell Signal to close position)
        position_to_manage = next((p for p in open_positions if p.symbol == self.symbol), None)
        if position_to_manage and position_to_manage.side == 'BUY' and last_row['fast_ma'] < last_row['slow_ma'] and prev_row['fast_ma'] >= prev_row['slow_ma']:
            logger.info("CLOSE signal generated for BUY position", strategy="SimpleMACrossover", position_id=position_to_manage.id)
            signal = SignalEvent(symbol=self.symbol, action='SELL', confidence=0.75) # Signal to sell/close
            await self.event_queue.put(signal)

    async def on_fill_event(self, event: FillEvent):
        pass

class AIEnsembleStrategy(TradingStrategy):
    """
    Generates trading signals using an ensemble of AI models and a market regime filter.
    Dependencies (EnsembleLearner, MarketRegimeDetector) are injected for modularity.
    """
    def __init__(self, event_queue: asyncio.Queue, config: Dict[str, Any], 
                 ensemble_learner: EnsembleLearner, regime_detector: MarketRegimeDetector):
        super().__init__(event_queue, config)
        from bot_core.config import AIStrategyConfig
        self.ai_config = AIStrategyConfig(**config.get('ai_ensemble', {}))
        
        self.ensemble_learner = ensemble_learner
        self.regime_detector = regime_detector
        
        self.confidence_threshold = self.ai_config.confidence_threshold
        self.use_regime_filter = self.ai_config.use_regime_filter

    async def on_market_event(self, event: MarketEvent, open_positions: List[Any]):
        if event.symbol != self.symbol or not self.ensemble_learner.is_trained:
            return

        if any(p.symbol == self.symbol for p in open_positions):
            logger.debug("Skipping signal generation, position already open.", symbol=self.symbol)
            return

        ohlcv_df = event.ohlcv_df
        regime = await self.regime_detector.detect_regime(self.symbol, ohlcv_df)
        
        ensemble_prediction = await self.ensemble_learner.predict(ohlcv_df, self.symbol)
        
        ensemble_action = ensemble_prediction['action']
        ensemble_confidence = ensemble_prediction['confidence']

        if ensemble_confidence < self.confidence_threshold:
            logger.debug("Signal ignored due to low confidence.", confidence=ensemble_confidence, threshold=self.confidence_threshold)
            return

        final_action = ensemble_action
        final_confidence = ensemble_confidence
        if self.use_regime_filter:
            current_regime = regime.get('regime')
            if (current_regime == MarketRegime.BULL.value and ensemble_action == 'sell') or \
               (current_regime == MarketRegime.BEAR.value and ensemble_action == 'buy'):
                logger.info("Regime filter overrides action", regime=current_regime, action=ensemble_action)
                return
            final_confidence *= (0.5 + regime.get('confidence', 0.5)) # Weight by regime confidence

        if final_action != 'hold':
            logger.info("Final AI signal generated", symbol=self.symbol, action=final_action.upper(), confidence=final_confidence)
            signal = SignalEvent(symbol=self.symbol, action=final_action.upper(), confidence=final_confidence)
            await self.event_queue.put(signal)

    async def on_fill_event(self, event: FillEvent):
        logger.debug("AI Strategy received fill event", symbol=event.symbol)
