import abc
import asyncio
import pandas as pd
import numpy as np
import json
import os
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

    @abc.abstractmethod
    def get_latest_regime(self, symbol: str) -> Optional[str]:
        """Returns the last detected market regime for the symbol."""
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

    def get_latest_regime(self, symbol: str) -> Optional[str]:
        return None

class AIEnsembleStrategy(TradingStrategy):
    def __init__(self, config: AIEnsembleStrategyParams):
        super().__init__(config)
        self.ai_config = config
        self.last_retrained_at: Dict[str, datetime] = {}
        self.last_signal_time: Dict[str, datetime] = {}
        self.last_detected_regime: Dict[str, str] = {}
        
        # Performance Monitoring State
        self.prediction_logs: Dict[str, List[Tuple[datetime, int]]] = {}
        self.accuracy_history: Dict[str, deque] = {}
        self.force_retrain_flags: Dict[str, bool] = {}
        self.drift_counters: Dict[str, int] = {}
        
        # Dynamic Weighting State
        # symbol -> list of (timestamp, {model_name: probs_array})
        self.individual_model_logs: Dict[str, List[Tuple[datetime, Dict[str, Any]]]] = {}
        # symbol -> {model_name: deque(correct_bools)}
        self.model_performance_stats: Dict[str, Dict[str, deque]] = {}
        
        self.data_fetcher = None # Will be set by Bot

        # State Persistence Path
        self.state_path = os.path.join(self.ai_config.model_path, "strategy_state.json")

        try:
            self.ensemble_learner = EnsembleLearner(self.ai_config)
            self.regime_detector = MarketRegimeDetector(self.ai_config)
            self._load_state()
            logger.info("AIEnsembleStrategy initialized")
        except ImportError as e:
            logger.critical("Failed to initialize AIEnsembleStrategy due to missing ML libraries", error=str(e))
            raise

    async def close(self):
        """Shuts down the ensemble learner resources."""
        self._save_state()
        await self.ensemble_learner.close()

    async def warmup(self, symbols: List[str]):
        """
        Pre-loads models into memory to avoid latency on the first tick.
        """
        logger.info("Warming up AI Ensemble Strategy...")
        await self.ensemble_learner.warmup_models(symbols)

    def _save_state(self):
        """Persists runtime state to disk."""
        try:
            state = {
                'last_retrained_at': {k: v.isoformat() for k, v in self.last_retrained_at.items()},
                'last_signal_time': {k: v.isoformat() for k, v in self.last_signal_time.items()},
                'force_retrain_flags': self.force_retrain_flags,
                'drift_counters': self.drift_counters,
                # Convert deque to list for JSON
                'accuracy_history': {k: list(v) for k, v in self.accuracy_history.items()},
                # Persist logs (timestamps as strings)
                'prediction_logs': {k: [(ts.isoformat(), p) for ts, p in v] for k, v in self.prediction_logs.items()},
                # Persist individual model logs (simplified for storage)
                # We only store the last N logs to avoid huge files
                'individual_model_logs': {
                    k: [(ts.isoformat(), {m: p.tolist() if isinstance(p, np.ndarray) else p for m, p in preds.items()}) 
                        for ts, preds in v[-50:]] 
                    for k, v in self.individual_model_logs.items()
                },
                'model_performance_stats': {
                    k: {m: list(d) for m, d in v.items()} for k, v in self.model_performance_stats.items()
                }
            }
            
            # Ensure dir exists
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            
            with open(self.state_path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error("Failed to save strategy state", error=str(e))

    def _load_state(self):
        """Loads runtime state from disk."""
        if not os.path.exists(self.state_path):
            return

        try:
            with open(self.state_path, 'r') as f:
                state = json.load(f)
            
            if 'last_retrained_at' in state:
                self.last_retrained_at = {k: datetime.fromisoformat(v) for k, v in state['last_retrained_at'].items()}
            
            if 'last_signal_time' in state:
                self.last_signal_time = {k: datetime.fromisoformat(v) for k, v in state['last_signal_time'].items()}

            self.force_retrain_flags = state.get('force_retrain_flags', {})
            self.drift_counters = state.get('drift_counters', {})
            
            if 'accuracy_history' in state:
                for k, v in state['accuracy_history'].items():
                    self.accuracy_history[k] = deque(v, maxlen=self.ai_config.performance.window_size)
            
            if 'prediction_logs' in state:
                for k, v in state['prediction_logs'].items():
                    self.prediction_logs[k] = [(datetime.fromisoformat(ts), p) for ts, p in v]
            
            if 'individual_model_logs' in state:
                for k, v in state['individual_model_logs'].items():
                    # Reconstruct numpy arrays from lists
                    self.individual_model_logs[k] = [
                        (datetime.fromisoformat(ts), {m: np.array(p) for m, p in preds.items()}) 
                        for ts, preds in v
                    ]
            
            if 'model_performance_stats' in state:
                window = self.ai_config.ensemble_weights.dynamic_window
                for k, v in state['model_performance_stats'].items():
                    self.model_performance_stats[k] = {m: deque(d, maxlen=window) for m, d in v.items()}
            
            logger.info("Restored AI Strategy state from disk.")
        except Exception as e:
            logger.error("Failed to load strategy state", error=str(e))

    def _in_cooldown(self, symbol: str) -> bool:
        """Checks if the symbol is in a signal cooldown period."""
        last_sig = self.last_signal_time.get(symbol)
        if not last_sig:
            return False
        
        seconds_per_candle = 300 
        cooldown_seconds = self.ai_config.signal_cooldown_candles * seconds_per_candle
        
        time_since = Clock.now() - last_sig
        return time_since.total_seconds() < cooldown_seconds

    def _get_confidence_threshold(self, regime: str, is_exit: bool = False, optimized_base: Optional[float] = None) -> float:
        """Determines the confidence threshold based on the current market regime and action type."""
        regime_config = self.ai_config.market_regime
        
        if is_exit:
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
            # If we have an optimized base (which is now regime-specific from the learner),
            # we prioritize it unless the config explicitly disables optimization.
            if optimized_base is not None and self.ai_config.training.optimize_entry_threshold:
                return optimized_base

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

    def get_latest_regime(self, symbol: str) -> Optional[str]:
        return self.last_detected_regime.get(symbol)

    def _calculate_dynamic_weights(self, symbol: str) -> Optional[Dict[str, float]]:
        """Calculates ensemble weights based on recent individual model accuracy."""
        if not self.ai_config.ensemble_weights.use_dynamic_weighting:
            return None
            
        stats = self.model_performance_stats.get(symbol)
        if not stats:
            return None
            
        accuracies = {}
        for model, history in stats.items():
            if len(history) >= 5: # Minimum samples to consider
                acc = sum(history) / len(history)
                accuracies[model] = acc
        
        if not accuracies:
            return None
            
        # Filter out poor performers
        min_acc = self.ai_config.ensemble_weights.min_model_accuracy
        valid_models = {m: acc for m, acc in accuracies.items() if acc > min_acc}
        
        if not valid_models:
            return None
            
        # Normalize to sum to 1
        total_acc = sum(valid_models.values())
        weights = {m: acc / total_acc for m, acc in valid_models.items()}
        
        # Apply smoothing if previous weights exist (not implemented here for simplicity, direct calc)
        return weights

    async def analyze_market(self, symbol: str, df: pd.DataFrame, position: Optional[Position]) -> Optional[Dict[str, Any]]:
        if not self.ensemble_learner.is_trained:
            logger.debug("AI models not trained yet, skipping analysis.", symbol=symbol)
            return None

        if self.ai_config.performance.enabled:
            await self._monitor_performance(symbol, df)

        if position is None and self.force_retrain_flags.get(symbol, False):
            logger.warning("Skipping entry signal due to poor model performance (Circuit Breaker Active).", symbol=symbol)
            return None

        df_enriched = self.regime_detector.add_regime_features(df)

        regime_result = await self.regime_detector.detect_regime(symbol, df_enriched)
        regime = regime_result.get('regime')
        
        # --- Regime Shift Detection ---
        prev_regime = self.last_detected_regime.get(symbol)
        if prev_regime and prev_regime != regime:
            logger.info("Market regime shift detected.", symbol=symbol, old=prev_regime, new=regime)
            # Trigger proactive retrain if not recently retrained
            # We use the force flag to bypass the standard interval check in needs_retraining
            self.force_retrain_flags[symbol] = True
            logger.info("Triggering proactive retrain due to regime shift.", symbol=symbol)

        self.last_detected_regime[symbol] = regime

        if self.ai_config.use_regime_filter:
            if regime in ['sideways', 'unknown']:
                if not (regime == 'sideways' and self.ai_config.market_regime.sideways_confidence_threshold is not None):
                    logger.debug("Market regime is sideways or unknown, holding position.", regime=regime, symbol=symbol)
                    return None

        if position is None and self._in_cooldown(symbol):
            logger.debug("Signal ignored due to cooldown.", symbol=symbol)
            return None

        leader_df = None
        if self.ai_config.market_leader_symbol and self.data_fetcher:
            leader_df = self.data_fetcher.get_market_data(self.ai_config.market_leader_symbol)

        # --- Dynamic Weighting ---
        dynamic_weights = self._calculate_dynamic_weights(symbol)
        if dynamic_weights:
            logger.debug("Using dynamic ensemble weights", symbol=symbol, weights=dynamic_weights)

        prediction = await self.ensemble_learner.predict(df_enriched, symbol, regime=regime, leader_df=leader_df, custom_weights=dynamic_weights)
        
        action = prediction.get('action')
        confidence = prediction.get('confidence', 0.0)
        model_version = prediction.get('model_version')
        active_weights = prediction.get('active_weights')
        top_features = prediction.get('top_features')
        metrics = prediction.get('metrics')
        is_anomaly = prediction.get('is_anomaly', False)
        anomaly_score = prediction.get('anomaly_score', 0.0)
        optimized_threshold = prediction.get('optimized_threshold')
        individual_preds = prediction.get('individual_predictions', {})

        state_changed = False

        if is_anomaly:
            self.drift_counters[symbol] = self.drift_counters.get(symbol, 0) + 1
            state_changed = True
            if self.drift_counters[symbol] >= self.ai_config.drift.max_consecutive_anomalies:
                logger.warning("Sustained data drift detected. Forcing retrain.", symbol=symbol, count=self.drift_counters[symbol])
                self.force_retrain_flags[symbol] = True
                self.drift_counters[symbol] = 0
            
            if self.ai_config.drift.block_trade:
                logger.warning("Drift detected (Anomaly). Blocking trade.", symbol=symbol, score=anomaly_score)
                self._save_state()
                return None
            else:
                penalty = self.ai_config.drift.confidence_penalty
                confidence = max(0.0, confidence - penalty)
                logger.info("Drift detected. Penalizing confidence.", symbol=symbol, penalty=penalty, new_conf=confidence, score=anomaly_score)
        else:
            if self.drift_counters.get(symbol, 0) > 0:
                self.drift_counters[symbol] = 0
                state_changed = True

        current_time = df.index[-1]
        
        # Log Ensemble Prediction
        if self.ai_config.performance.enabled and action:
            action_map = {'sell': 0, 'hold': 1, 'buy': 2}
            pred_int = action_map.get(action, 1)
            
            if symbol not in self.prediction_logs:
                self.prediction_logs[symbol] = []
            self.prediction_logs[symbol].append((current_time, pred_int))
            state_changed = True

        # Log Individual Predictions for Dynamic Weighting
        if individual_preds:
            if symbol not in self.individual_model_logs:
                self.individual_model_logs[symbol] = []
            self.individual_model_logs[symbol].append((current_time, individual_preds))
            state_changed = True

        if state_changed:
            self._save_state()

        logger.debug("AI prediction received", symbol=symbol, **prediction)

        is_exit = False
        if position:
            if (position.side == 'BUY' and action == 'sell') or \
               (position.side == 'SELL' and action == 'buy'):
                is_exit = True
        
        required_threshold = self._get_confidence_threshold(regime, is_exit, optimized_base=optimized_threshold)
        
        if confidence < required_threshold:
            logger.debug("Confidence below threshold", symbol=symbol, confidence=confidence, required=required_threshold, regime=regime, is_exit=is_exit)
            return None

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
            self._save_state()
            return signal

        return None

    async def _monitor_performance(self, symbol: str, df: pd.DataFrame):
        # --- 1. Monitor Ensemble Accuracy ---
        logs = self.prediction_logs.get(symbol, [])
        if logs:
            horizon = self.ai_config.features.labeling_horizon
            eval_df = df.tail(200)
            if len(eval_df) >= horizon + 1:
                try:
                    actual_labels = FeatureProcessor.create_labels(eval_df, self.ai_config)
                    
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

                    if evaluated_count > 0:
                        self._save_state()

                    if len(self.accuracy_history[symbol]) >= 10:
                        accuracy = sum(self.accuracy_history[symbol]) / len(self.accuracy_history[symbol])
                        
                        if self.ai_config.performance.auto_rollback and accuracy < self.ai_config.performance.critical_accuracy_threshold:
                            logger.critical("Model accuracy CRITICAL. Initiating ROLLBACK.", symbol=symbol, accuracy=f"{accuracy:.2%}")
                            success = await self.ensemble_learner.rollback_model(symbol)
                            if success:
                                self.accuracy_history[symbol].clear()
                                self.prediction_logs[symbol] = []
                                self.force_retrain_flags[symbol] = False
                                self._save_state()
                                return

                        if accuracy < self.ai_config.performance.min_accuracy:
                            if not self.force_retrain_flags.get(symbol, False):
                                logger.warning("Model accuracy dropped below threshold. Forcing retrain and blocking entries.", 
                                            symbol=symbol, accuracy=f"{accuracy:.2%}", threshold=self.ai_config.performance.min_accuracy)
                                self.force_retrain_flags[symbol] = True
                                self._save_state()
                except Exception as e:
                    logger.error("Failed to monitor ensemble performance", error=str(e))

        # --- 2. Monitor Individual Model Accuracy (Dynamic Weighting) ---
        if self.ai_config.ensemble_weights.use_dynamic_weighting:
            ind_logs = self.individual_model_logs.get(symbol, [])
            if ind_logs:
                try:
                    # Re-generate labels if not done above (optimization: reuse if possible, but safe to redo)
                    horizon = self.ai_config.features.labeling_horizon
                    eval_df = df.tail(200)
                    if len(eval_df) >= horizon + 1:
                        actual_labels = FeatureProcessor.create_labels(eval_df, self.ai_config)
                        
                        if symbol not in self.model_performance_stats:
                            self.model_performance_stats[symbol] = {}

                        remaining_ind_logs = []
                        
                        for ts, preds_dict in ind_logs:
                            if ts in actual_labels.index:
                                actual = actual_labels.loc[ts]
                                if not np.isnan(actual):
                                    actual_int = int(actual)
                                    for model_name, probs in preds_dict.items():
                                        # Determine model prediction (argmax)
                                        model_pred = np.argmax(probs)
                                        is_correct = 1 if model_pred == actual_int else 0
                                        
                                        if model_name not in self.model_performance_stats[symbol]:
                                            self.model_performance_stats[symbol][model_name] = deque(maxlen=self.ai_config.ensemble_weights.dynamic_window)
                                        
                                        self.model_performance_stats[symbol][model_name].append(is_correct)
                                else:
                                    remaining_ind_logs.append((ts, preds_dict))
                            else:
                                if ts > df.index[0]:
                                    remaining_ind_logs.append((ts, preds_dict))
                        
                        self.individual_model_logs[symbol] = remaining_ind_logs
                except Exception as e:
                    logger.error("Failed to monitor individual model performance", error=str(e))

    async def retrain(self, symbol: str, df: pd.DataFrame, executor: Executor):
        logger.info("Strategy is triggering model retraining via ProcessPool.", symbol=symbol)
        
        loop = asyncio.get_running_loop()
        try:
            df_enriched = self.regime_detector.add_regime_features(df)
            
            leader_df = None
            if self.ai_config.market_leader_symbol and self.data_fetcher:
                limit = len(df)
                leader_df = await self.data_fetcher.fetch_full_history_for_symbol(self.ai_config.market_leader_symbol, limit)

            success = await loop.run_in_executor(
                executor, 
                train_ensemble_task, 
                symbol, 
                df_enriched, 
                self.ai_config,
                leader_df
            )
            
            if success:
                self.last_retrained_at[symbol] = Clock.now()
                self.force_retrain_flags[symbol] = False
                self.drift_counters[symbol] = 0
                self.prediction_logs[symbol] = []
                if symbol in self.accuracy_history:
                    self.accuracy_history[symbol].clear()
                
                # Clear dynamic weighting history on retrain to start fresh with new models
                self.individual_model_logs[symbol] = []
                if symbol in self.model_performance_stats:
                    self.model_performance_stats[symbol].clear()
                
                self._save_state()
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
