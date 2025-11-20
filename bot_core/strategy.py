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

    async def close(self):
        """Clean up strategy resources."""
        pass

    async def warmup(self, symbols: List[str]):
        """Optional warmup hook."""
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
        # Map: symbol -> int (consecutive anomalies)
        self.drift_counters: Dict[str, int] = {}
        
        # Reference to DataHandler (injected via property or method if needed, but usually accessed via bot)
        self.data_fetcher = None # Will be set by Bot

        try:
            self.ensemble_learner = EnsembleLearner(self.ai_config)
            self.regime_detector = MarketRegimeDetector(self.ai_config)
            logger.info("AIEnsembleStrategy initialized")
        except ImportError as e:
            logger.critical("Failed to initialize AIEnsembleStrategy due to missing ML libraries", error=str(e))
            raise

    async def close(self):
        """Shuts down the ensemble learner resources."""
        await self.ensemble_learner.close()

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

    def _get_confidence_threshold(self, regime: str, is_exit: bool = False, optimized_base: Optional[float] = None) -> float:
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
            # Prioritize optimized base threshold if available
            base = optimized_base if optimized_base is not None else self.ai_config.confidence_threshold
            
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
            await self._monitor_performance(symbol, df)

        # --- Performance Circuit Breaker ---
        if position is None and self.force_retrain_flags.get(symbol, False):
            logger.warning("Skipping entry signal due to poor model performance (Circuit Breaker Active).", symbol=symbol)
            return None

        # 2. Enrich Data with Regime Features
        df_enriched = self.regime_detector.add_regime_features(df)

        # 3. Detect Regime
        regime_result = await self.regime_detector.detect_regime(symbol, df_enriched)
        regime = regime_result.get('regime')

        if self.ai_config.use_regime_filter:
            if regime in ['sideways', 'unknown']:
                if not (regime == 'sideways' and self.ai_config.market_regime.sideways_confidence_threshold is not None):
                    logger.debug("Market regime is sideways or unknown, holding position.", regime=regime, symbol=symbol)
                    return None

        # Check Cooldown
        if position is None and self._in_cooldown(symbol):
            logger.debug("Signal ignored due to cooldown.", symbol=symbol)
            return None

        # 4. Fetch Market Leader Data (if configured)
        leader_df = None
        if self.ai_config.market_leader_symbol and self.data_fetcher:
            # Use the injected data fetcher (DataHandler)
            leader_df = self.data_fetcher.get_market_data(self.ai_config.market_leader_symbol)

        # 5. Predict using Enriched Data + Leader Data
        prediction = await self.ensemble_learner.predict(df_enriched, symbol, regime=regime, leader_df=leader_df)
        
        action = prediction.get('action')
        confidence = prediction.get('confidence', 0.0)
        model_version = prediction.get('model_version')
        active_weights = prediction.get('active_weights')
        top_features = prediction.get('top_features')
        metrics = prediction.get('metrics')
        is_anomaly = prediction.get('is_anomaly', False)
        anomaly_score = prediction.get('anomaly_score', 0.0)
        optimized_threshold = prediction.get('optimized_threshold')

        # --- DRIFT DETECTION CHECK ---
        if is_anomaly:
            # Increment drift counter
            self.drift_counters[symbol] = self.drift_counters.get(symbol, 0) + 1
            
            # Check if we need to trigger a retrain due to sustained drift
            if self.drift_counters[symbol] >= self.ai_config.drift.max_consecutive_anomalies:
                logger.warning("Sustained data drift detected. Forcing retrain.", symbol=symbol, count=self.drift_counters[symbol])
                self.force_retrain_flags[symbol] = True
                self.drift_counters[symbol] = 0 # Reset counter
            
            if self.ai_config.drift.block_trade:
                logger.warning("Drift detected (Anomaly). Blocking trade.", symbol=symbol, score=anomaly_score)
                return None
            else:
                penalty = self.ai_config.drift.confidence_penalty
                confidence = max(0.0, confidence - penalty)
                logger.info("Drift detected. Penalizing confidence.", symbol=symbol, penalty=penalty, new_conf=confidence, score=anomaly_score)
        else:
            # Reset drift counter if normal
            self.drift_counters[symbol] = 0

        # Log prediction
        if self.ai_config.performance.enabled and action:
            action_map = {'sell': 0, 'hold': 1, 'buy': 2}
            pred_int = action_map.get(action, 1)
            current_time = df.index[-1]
            
            if symbol not in self.prediction_logs:
                self.prediction_logs[symbol] = []
            self.prediction_logs[symbol].append((current_time, pred_int))

        logger.debug("AI prediction received", symbol=symbol, **prediction)

        # --- Dynamic Threshold Check (Asymmetric) ---
        is_exit = False
        if position:
            if (position.side == 'BUY' and action == 'sell') or \
               (position.side == 'SELL' and action == 'buy'):
                is_exit = True
        
        # Pass the optimized threshold to the helper
        required_threshold = self._get_confidence_threshold(regime, is_exit, optimized_base=optimized_threshold)
        
        if confidence < required_threshold:
            logger.debug("Confidence below threshold", symbol=symbol, confidence=confidence, required=required_threshold, regime=regime, is_exit=is_exit)
            return None

        # Construct metadata
        strategy_metadata = {
            'model_version': model_version,
            'confidence': confidence,
            'regime': regime,
            'regime_confidence': regime_result.get('confidence'),
            'model_type': prediction.get('model_type'),
            'active_weights': active_weights,
            'top_features': top_features,
            'metrics': metrics,
            'is_anomaly': is_anomaly,
            'optimized_threshold': optimized_threshold
        }

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
            self.last_signal_time[symbol] = Clock.now()
            return signal

        return None

    async def _monitor_performance(self, symbol: str, df: pd.DataFrame):
        """Evaluates past predictions against realized market moves."""
        logs = self.prediction_logs.get(symbol, [])
        if not logs:
            return

        horizon = self.ai_config.features.labeling_horizon
        eval_df = df.tail(200)
        if len(eval_df) < horizon + 1:
            return

        try:
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
                if not np.isnan(actual):
                    is_correct = 1 if int(actual) == pred_int else 0
                    self.accuracy_history[symbol].append(is_correct)
                    evaluated_count += 1
                else:
                    remaining_logs.append((ts, pred_int))
            else:
                if ts > df.index[0]:
                    remaining_logs.append((ts, pred_int))
        
        self.prediction_logs[symbol] = remaining_logs

        if len(self.accuracy_history[symbol]) >= 10:
            accuracy = sum(self.accuracy_history[symbol]) / len(self.accuracy_history[symbol])
            
            # --- CRITICAL FAILURE CHECK (ROLLBACK) ---
            if self.ai_config.performance.auto_rollback and accuracy < self.ai_config.performance.critical_accuracy_threshold:
                logger.critical("Model accuracy CRITICAL. Initiating ROLLBACK.", symbol=symbol, accuracy=f"{accuracy:.2%}")
                success = await self.ensemble_learner.rollback_model(symbol)
                if success:
                    # Clear history as we are now on a different model
                    self.accuracy_history[symbol].clear()
                    self.prediction_logs[symbol] = []
                    self.force_retrain_flags[symbol] = False # Clear flag if it was set
                    return

            # --- STANDARD FAILURE CHECK (RETRAIN) ---
            if accuracy < self.ai_config.performance.min_accuracy:
                if not self.force_retrain_flags.get(symbol, False):
                    logger.warning("Model accuracy dropped below threshold. Forcing retrain and blocking entries.", 
                                   symbol=symbol, accuracy=f"{accuracy:.2%}", threshold=self.ai_config.performance.min_accuracy)
                    self.force_retrain_flags[symbol] = True

    async def retrain(self, symbol: str, df: pd.DataFrame, executor: Executor):
        """Initiates the retraining process using the provided executor."""
        logger.info("Strategy is triggering model retraining via ProcessPool.", symbol=symbol)
        
        loop = asyncio.get_running_loop()
        try:
            # Enrich Data with Regime Features
            df_enriched = self.regime_detector.add_regime_features(df)
            
            # Fetch Leader Data for Training
            leader_df = None
            if self.ai_config.market_leader_symbol and self.data_fetcher:
                # We need full history for training, so we ask DataHandler for it
                # Note: This is an async call, but we are in an async method.
                # However, data_fetcher is the DataHandler instance.
                # We need to fetch the leader history corresponding to 'df'.
                # Since 'df' is fetched via fetch_full_history_for_symbol, we should do the same for leader.
                limit = len(df)
                leader_df = await self.data_fetcher.fetch_full_history_for_symbol(self.ai_config.market_leader_symbol, limit)

            # Offload the heavy training task to a separate process
            success = await loop.run_in_executor(
                executor, 
                train_ensemble_task, 
                symbol, 
                df_enriched, 
                self.ai_config,
                leader_df # Pass leader data
            )
            
            if success:
                self.last_retrained_at[symbol] = Clock.now()
                self.force_retrain_flags[symbol] = False
                self.drift_counters[symbol] = 0
                self.prediction_logs[symbol] = []
                if symbol in self.accuracy_history:
                    self.accuracy_history[symbol].clear()
                
                logger.info("Model training successful. Reloading models.", symbol=symbol)
                await self.ensemble_learner.reload_models(symbol)
            else:
                logger.warning("Model training failed or rejected.", symbol=symbol)
                
        except Exception as e:
            logger.error("Error during async model retraining", symbol=symbol, error=str(e))

    def needs_retraining(self, symbol: str) -> bool:
        if not self.ensemble_learner.has_valid_model(symbol):
            logger.info(f"No valid model found for {symbol}. Retraining required.")
            return True

        if self.force_retrain_flags.get(symbol, False):
            return True

        if self.ai_config.retrain_interval_hours <= 0:
            return False
        
        if symbol not in self.last_retrained_at:
            last_train_time = self.ensemble_learner.get_last_training_time(symbol)
            if last_train_time:
                self.last_retrained_at[symbol] = last_train_time
            else:
                return True
        
        last_retrained = self.last_retrained_at.get(symbol)
        time_since = Clock.now() - last_retrained
        return time_since >= timedelta(hours=self.ai_config.retrain_interval_hours)

    def get_training_data_limit(self) -> int:
        return self.ai_config.training_data_limit
