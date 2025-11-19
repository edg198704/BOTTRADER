import abc
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from concurrent.futures import Executor
from collections import deque

from bot_core.logger import get_logger
from bot_core.config import AIEnsembleStrategyParams, SimpleMACrossoverStrategyParams, StrategyParamsBase
from bot_core.ai.ensemble_learner import EnsembleLearner, train_ensemble_task
from bot_core.ai.regime_detector import MarketRegimeDetector
from bot_core.ai.feature_processor import FeatureProcessor
from bot_core.position_manager import Position
from bot_core.utils import Clock, parse_timeframe_to_seconds

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
        self.last_signal_time: Dict[str, datetime] = {}
        
        # Performance Monitoring State
        # Map: symbol -> list of (timestamp, predicted_label_int)
        self.prediction_logs: Dict[str, List[Tuple[datetime, int]]] = {}
        # Map: symbol -> deque of recent accuracy (1 for correct, 0 for incorrect)
        self.accuracy_history: Dict[str, deque] = {}
        # Map: symbol -> bool
        self.force_retrain_flags: Dict[str, bool] = {}

        try:
            self.ensemble_learner = EnsembleLearner(self.ai_config)
            self.regime_detector = MarketRegimeDetector(self.ai_config)
            logger.info("AIEnsembleStrategy initialized")
        except ImportError as e:
            logger.critical("Failed to initialize AIEnsembleStrategy due to missing ML libraries", error=str(e))
            raise

    def _in_cooldown(self, symbol: str) -> bool:
        """Checks if the symbol is in a signal cooldown period."""
        last_sig = self.last_signal_time.get(symbol)
        if not last_sig:
            return False
        
        # Assume 5m candles (300s) for cooldown calculation if timeframe not explicitly available
        seconds_per_candle = 300 
        cooldown_seconds = self.ai_config.signal_cooldown_candles * seconds_per_candle
        
        time_since = Clock.now() - last_sig
        return time_since.total_seconds() < cooldown_seconds

    def _get_confidence_threshold(self, regime: str, is_exit: bool = False) -> float:
        """Determines the confidence threshold based on the current market regime and action type."""
        regime_config = self.ai_config.market_regime
        
        if is_exit:
            # Use exit-specific thresholds
            base = self.ai_config.exit_confidence_threshold
            if regime == 'bull' and regime_config.bull_exit_threshold is not None:
                return regime_config.bull_exit_threshold
            elif regime == 'bear' and regime_config.bear_exit_threshold is not None:
                return regime_config.bear_exit_threshold
            elif regime == 'volatile' and regime_config.volatile_exit_threshold is not None:
                return regime_config.volatile_exit_threshold
            elif regime == 'sideways' and regime_config.sideways_exit_threshold is not None:
                return regime_config.sideways_exit_threshold
            return base
        else:
            # Use entry thresholds
            base = self.ai_config.confidence_threshold
            if regime == 'bull' and regime_config.bull_confidence_threshold is not None:
                return regime_config.bull_confidence_threshold
            elif regime == 'bear' and regime_config.bear_confidence_threshold is not None:
                return regime_config.bear_confidence_threshold
            elif regime == 'volatile' and regime_config.volatile_confidence_threshold is not None:
                return regime_config.volatile_confidence_threshold
            elif regime == 'sideways' and regime_config.sideways_confidence_threshold is not None:
                return regime_config.sideways_confidence_threshold
            return base

    async def analyze_market(self, symbol: str, df: pd.DataFrame, position: Optional[Position]) -> Optional[Dict[str, Any]]:
        if not self.ensemble_learner.is_trained:
            logger.debug("AI models not trained yet, skipping analysis.", symbol=symbol)
            return None

        # 1. Monitor Performance of past predictions
        if self.ai_config.performance.enabled:
            self._monitor_performance(symbol, df)

        # 2. Enrich Data with Regime Features
        # This injects 'regime_trend' and 'regime_volatility' into the DF so the model can use them
        df_enriched = self.regime_detector.add_regime_features(df)

        # 3. Detect Regime (Optimized: uses the columns we just added)
        regime_result = await self.regime_detector.detect_regime(symbol, df_enriched)
        regime = regime_result.get('regime')

        if self.ai_config.use_regime_filter:
            if regime in ['sideways', 'unknown']:
                # Check if we have a specific threshold for sideways to allow trading
                if not (regime == 'sideways' and self.ai_config.market_regime.sideways_confidence_threshold is not None):
                    logger.debug("Market regime is sideways or unknown, holding position.", regime=regime, symbol=symbol)
                    return None

        # Check Cooldown ONLY if we are looking to enter (position is None)
        if position is None and self._in_cooldown(symbol):
            logger.debug("Signal ignored due to cooldown.", symbol=symbol)
            return None

        # 4. Predict using Enriched Data
        prediction = await self.ensemble_learner.predict(df_enriched, symbol)
        action = prediction.get('action')
        confidence = prediction.get('confidence', 0.0)
        model_version = prediction.get('model_version')
        active_weights = prediction.get('active_weights')
        top_features = prediction.get('top_features')
        metrics = prediction.get('metrics')

        # Log prediction for future evaluation
        if self.ai_config.performance.enabled and action:
            action_map = {'sell': 0, 'hold': 1, 'buy': 2}
            pred_int = action_map.get(action, 1)
            current_time = df.index[-1]
            
            if symbol not in self.prediction_logs:
                self.prediction_logs[symbol] = []
            self.prediction_logs[symbol].append((current_time, pred_int))

        logger.debug("AI prediction received", symbol=symbol, **prediction)

        # --- Dynamic Threshold Check (Asymmetric) ---
        # Determine if this is an exit signal based on current position
        is_exit = False
        if position:
            if (position.side == 'BUY' and action == 'sell') or \
               (position.side == 'SELL' and action == 'buy'):
                is_exit = True
        
        required_threshold = self._get_confidence_threshold(regime, is_exit)
        
        if confidence < required_threshold:
            logger.debug("Confidence below threshold", symbol=symbol, confidence=confidence, required=required_threshold, regime=regime, is_exit=is_exit)
            return None

        # Construct metadata for the signal
        strategy_metadata = {
            'model_version': model_version,
            'confidence': confidence,
            'regime': regime,
            'regime_confidence': regime_result.get('confidence'),
            'model_type': prediction.get('model_type'),
            'active_weights': active_weights,
            'top_features': top_features,
            'metrics': metrics
        }

        # NOTE: We explicitly pass 'regime' at the top level so RiskManager can see it.
        signal = {
            'symbol': symbol,
            'regime': regime,
            'strategy_metadata': strategy_metadata
        }

        signal_generated = False

        if position:
            if position.side == 'BUY' and action == 'sell':
                logger.info("AI Close Long signal detected", symbol=symbol, confidence=confidence)
                signal['action'] = 'SELL'
                signal_generated = True
            elif position.side == 'SELL' and action == 'buy':
                logger.info("AI Close Short signal detected", symbol=symbol, confidence=confidence)
                signal['action'] = 'BUY'
                signal_generated = True
        else:
            if action == 'buy':
                logger.info("AI Open Long signal detected", symbol=symbol, confidence=confidence)
                signal['action'] = 'BUY'
                signal_generated = True
            elif action == 'sell':
                logger.info("AI Open Short signal detected", symbol=symbol, confidence=confidence)
                signal['action'] = 'SELL'
                signal_generated = True

        if signal_generated:
            # Update last signal time to enforce cooldown on next check
            self.last_signal_time[symbol] = Clock.now()
            return signal

        return None

    def _monitor_performance(self, symbol: str, df: pd.DataFrame):
        """Evaluates past predictions against realized market moves."""
        logs = self.prediction_logs.get(symbol, [])
        if not logs:
            return

        # We only need to evaluate predictions that are older than the labeling horizon
        horizon = self.ai_config.features.labeling_horizon
        
        # Use FeatureProcessor to generate ground truth labels for the recent history
        # We take a slice large enough to cover the oldest unevaluated prediction + horizon
        # But for efficiency, we just take the last 200 rows
        eval_df = df.tail(200)
        if len(eval_df) < horizon + 1:
            return

        try:
            # Generate actual labels (0, 1, 2) based on future price movement
            actual_labels = FeatureProcessor.create_labels(eval_df, self.ai_config)
        except Exception as e:
            logger.error("Failed to generate labels for performance monitoring", error=str(e))
            return

        if symbol not in self.accuracy_history:
            self.accuracy_history[symbol] = deque(maxlen=self.ai_config.performance.window_size)

        remaining_logs = []
        evaluated_count = 0

        for ts, pred_int in logs:
            if ts in actual_labels.index:
                actual = actual_labels.loc[ts]
                # Check if actual is valid (not NaN due to horizon)
                if not np.isnan(actual):
                    is_correct = 1 if int(actual) == pred_int else 0
                    self.accuracy_history[symbol].append(is_correct)
                    evaluated_count += 1
                else:
                    # Horizon not reached yet
                    remaining_logs.append((ts, pred_int))
            else:
                # Timestamp fell out of the evaluation window or mismatch
                # If it's very old, discard. If recent, keep.
                if ts > df.index[0]:
                    remaining_logs.append((ts, pred_int))
        
        self.prediction_logs[symbol] = remaining_logs

        # Check accuracy threshold
        if len(self.accuracy_history[symbol]) >= 10: # Minimum samples to judge
            accuracy = sum(self.accuracy_history[symbol]) / len(self.accuracy_history[symbol])
            if accuracy < self.ai_config.performance.min_accuracy:
                if not self.force_retrain_flags.get(symbol, False):
                    logger.warning("Model accuracy dropped below threshold. Forcing retrain.", 
                                   symbol=symbol, accuracy=f"{accuracy:.2%}", threshold=self.ai_config.performance.min_accuracy)
                    self.force_retrain_flags[symbol] = True

    async def retrain(self, symbol: str, df: pd.DataFrame, executor: Executor):
        """Initiates the retraining process using the provided executor."""
        logger.info("Strategy is triggering model retraining via ProcessPool.", symbol=symbol)
        
        loop = asyncio.get_running_loop()
        try:
            # Enrich Data with Regime Features before sending to worker
            df_enriched = self.regime_detector.add_regime_features(df)

            # Offload the heavy training task to a separate process
            success = await loop.run_in_executor(
                executor, 
                train_ensemble_task, 
                symbol, 
                df_enriched, 
                self.ai_config
            )
            
            if success:
                self.last_retrained_at[symbol] = Clock.now()
                # Reset performance metrics for the new model
                self.force_retrain_flags[symbol] = False
                self.prediction_logs[symbol] = []
                if symbol in self.accuracy_history:
                    self.accuracy_history[symbol].clear()
                
                logger.info("Model training successful. Reloading models and resetting performance stats.", symbol=symbol)
                await self.ensemble_learner.reload_models(symbol)
            else:
                logger.warning("Model training failed or rejected.", symbol=symbol)
                
        except Exception as e:
            logger.error("Error during async model retraining", symbol=symbol, error=str(e))

    def needs_retraining(self, symbol: str) -> bool:
        # 0. Check if we even have a valid model loaded
        # This handles the case where config changed and old models were rejected
        if not self.ensemble_learner.has_valid_model(symbol):
            logger.info(f"No valid model found for {symbol} (or feature mismatch). Retraining required.")
            return True

        # 1. Check forced retrain flag (Performance based)
        if self.force_retrain_flags.get(symbol, False):
            return True

        # 2. Check scheduled retrain
        if self.ai_config.retrain_interval_hours <= 0:
            return False
        
        # Sync with learner state if local state is empty (e.g. after restart)
        if symbol not in self.last_retrained_at:
            last_train_time = self.ensemble_learner.get_last_training_time(symbol)
            if last_train_time:
                self.last_retrained_at[symbol] = last_train_time
                logger.info("Synced last training time from loaded model.", symbol=symbol, timestamp=last_train_time)
            else:
                # If learner has no record, we must train.
                return True
        
        last_retrained = self.last_retrained_at.get(symbol)
        time_since_retrain = Clock.now() - last_retrained
        return time_since_retrain >= timedelta(hours=self.ai_config.retrain_interval_hours)

    def get_training_data_limit(self) -> int:
        return self.ai_config.training_data_limit
