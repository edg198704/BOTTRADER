import abc
import asyncio
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Any, Optional, List, Tuple, Literal, Deque
from datetime import datetime, timedelta, timezone
from concurrent.futures import Executor
from collections import deque
from pydantic import BaseModel, Field

from bot_core.logger import get_logger
from bot_core.config import AIEnsembleStrategyParams, SimpleMACrossoverStrategyParams, StrategyParamsBase
from bot_core.ai.ensemble_learner import EnsembleLearner, train_ensemble_task
from bot_core.ai.regime_detector import MarketRegimeDetector
from bot_core.ai.feature_processor import FeatureProcessor
from bot_core.position_manager import Position
from bot_core.utils import Clock, AtomicJsonStore
from bot_core.common import TradeSignal, AIInferenceResult

logger = get_logger(__name__)

class StrategyStateManager:
    """
    Manages persistence and in-memory state for the AI Strategy.
    Handles accuracy history, prediction logs, and retraining flags.
    """
    def __init__(self, config: AIEnsembleStrategyParams):
        self.config = config
        self.persistence = AtomicJsonStore(os.path.join(config.model_path, "strategy_state.json"))
        
        # State Containers
        self.last_retrained_at: Dict[str, datetime] = {}
        self.last_signal_time: Dict[str, datetime] = {}
        self.last_detected_regime: Dict[str, str] = {}
        self.force_retrain_flags: Dict[str, bool] = {}
        self.drift_counters: Dict[str, int] = {}
        
        # Performance Metrics
        self.accuracy_history: Dict[str, Deque[int]] = {}
        self.prediction_logs: Dict[str, List[Tuple[datetime, int]]] = {}
        self.individual_model_logs: Dict[str, List[Tuple[datetime, Dict[str, Any]]]] = {}
        self.model_performance_stats: Dict[str, Dict[str, Deque[int]]] = {}

        self._load()

    def _load(self):
        state = self.persistence.load()
        if not state: return

        def parse_dt_dict(d): return {k: datetime.fromisoformat(v) for k, v in d.items()}
        
        self.last_retrained_at = parse_dt_dict(state.get('last_retrained_at', {}))
        self.last_signal_time = parse_dt_dict(state.get('last_signal_time', {}))
        self.force_retrain_flags = state.get('force_retrain_flags', {})
        self.drift_counters = state.get('drift_counters', {})
        
        if 'accuracy_history' in state:
            for k, v in state['accuracy_history'].items():
                self.accuracy_history[k] = deque(v, maxlen=self.config.performance.window_size)
        
        if 'prediction_logs' in state:
            for k, v in state['prediction_logs'].items():
                self.prediction_logs[k] = [(datetime.fromisoformat(ts), p) for ts, p in v]

        if 'individual_model_logs' in state:
            for k, v in state['individual_model_logs'].items():
                self.individual_model_logs[k] = [
                    (datetime.fromisoformat(ts), {m: np.array(p) for m, p in preds.items()}) 
                    for ts, preds in v
                ]

        if 'model_performance_stats' in state:
            window = self.config.ensemble_weights.dynamic_window
            for k, v in state['model_performance_stats'].items():
                self.model_performance_stats[k] = {m: deque(d, maxlen=window) for m, d in v.items()}

    def save(self):
        def _serialize_pred(p):
            if isinstance(p, np.ndarray): return p.tolist()
            if isinstance(p, list): return p
            return float(p)

        state = {
            'last_retrained_at': {k: v.isoformat() for k, v in self.last_retrained_at.items()},
            'last_signal_time': {k: v.isoformat() for k, v in self.last_signal_time.items()},
            'force_retrain_flags': self.force_retrain_flags,
            'drift_counters': self.drift_counters,
            'accuracy_history': {k: list(v) for k, v in self.accuracy_history.items()},
            'prediction_logs': {k: [(ts.isoformat(), int(p)) for ts, p in v] for k, v in self.prediction_logs.items()},
            'individual_model_logs': {
                k: [(ts.isoformat(), {m: _serialize_pred(p) for m, p in preds.items()}) 
                    for ts, preds in v[-50:]] 
                for k, v in self.individual_model_logs.items()
            },
            'model_performance_stats': {
                k: {m: list(d) for m, d in v.items()} for k, v in self.model_performance_stats.items()
            }
        }
        self.persistence.save(state)

    def log_prediction(self, symbol: str, timestamp: datetime, action: str, individual_preds: Optional[Dict] = None):
        action_map = {'sell': 0, 'hold': 1, 'buy': 2}
        if symbol not in self.prediction_logs: self.prediction_logs[symbol] = []
        self.prediction_logs[symbol].append((timestamp, action_map.get(action, 1)))
        
        if individual_preds:
            if symbol not in self.individual_model_logs: self.individual_model_logs[symbol] = []
            self.individual_model_logs[symbol].append((timestamp, individual_preds))
        self.save()

    def clear_symbol_state(self, symbol: str):
        self.force_retrain_flags[symbol] = False
        self.drift_counters[symbol] = 0
        self.prediction_logs[symbol] = []
        if symbol in self.accuracy_history: self.accuracy_history[symbol].clear()
        self.individual_model_logs[symbol] = []
        if symbol in self.model_performance_stats: self.model_performance_stats[symbol].clear()
        self.save()

class TradingStrategy(abc.ABC):
    def __init__(self, config: StrategyParamsBase):
        self.config = config
        self.data_fetcher = None # Injected by Bot

    @abc.abstractmethod
    async def analyze_market(self, symbol: str, df: pd.DataFrame, position: Optional[Position]) -> Optional[TradeSignal]:
        pass

    @abc.abstractmethod
    async def retrain(self, symbol: str, df: pd.DataFrame, executor: Executor):
        pass

    @abc.abstractmethod
    def needs_retraining(self, symbol: str) -> bool:
        pass

    @abc.abstractmethod
    def get_latest_regime(self, symbol: str) -> Optional[str]:
        pass

    async def close(self):
        pass

    async def warmup(self, symbols: List[str]):
        pass

class SimpleMACrossoverStrategy(TradingStrategy):
    def __init__(self, config: SimpleMACrossoverStrategyParams):
        super().__init__(config)
        self.fast_ma_period = config.fast_ma_period
        self.slow_ma_period = config.slow_ma_period
        self.fast_ma_col = f"SMA_{self.fast_ma_period}"
        self.slow_ma_col = f"SMA_{self.slow_ma_period}"
        logger.info("SimpleMACrossoverStrategy initialized", fast_ma=self.fast_ma_period, slow_ma=self.slow_ma_period)

    async def analyze_market(self, symbol: str, df: pd.DataFrame, position: Optional[Position]) -> Optional[TradeSignal]:
        if self.fast_ma_col not in df.columns or self.slow_ma_col not in df.columns:
            return None

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        is_bullish_cross = last_row[self.fast_ma_col] > last_row[self.slow_ma_col] and prev_row[self.fast_ma_col] <= prev_row[self.slow_ma_col]
        is_bearish_cross = last_row[self.fast_ma_col] < last_row[self.slow_ma_col] and prev_row[self.fast_ma_col] >= prev_row[self.slow_ma_col]

        action = None
        if position:
            if position.side == 'BUY' and is_bearish_cross: action = 'SELL'
            elif position.side == 'SELL' and is_bullish_cross: action = 'BUY'
        else:
            if is_bullish_cross: action = 'BUY'
            elif is_bearish_cross: action = 'SELL'

        if action:
            return TradeSignal(symbol=symbol, action=action, strategy_name=self.config.name, confidence=1.0)
        return None

    async def retrain(self, symbol: str, df: pd.DataFrame, executor: Executor): pass
    def needs_retraining(self, symbol: str) -> bool: return False
    def get_latest_regime(self, symbol: str) -> Optional[str]: return None

class AIEnsembleStrategy(TradingStrategy):
    def __init__(self, config: AIEnsembleStrategyParams):
        super().__init__(config)
        self.ai_config = config
        self.state = StrategyStateManager(config)
        
        try: # Lazy import to avoid circular dependency issues during init if not used
            self.ensemble_learner = EnsembleLearner(self.ai_config)
            self.regime_detector = MarketRegimeDetector(self.ai_config)
            logger.info("AIEnsembleStrategy initialized")
        except ImportError as e:
            logger.critical("Failed to initialize AIEnsembleStrategy due to missing ML libraries", error=str(e))
            raise

    async def close(self):
        self.state.save()
        await self.ensemble_learner.close()

    async def warmup(self, symbols: List[str]):
        logger.info("Warming up AI Ensemble Strategy...")
        await self.ensemble_learner.warmup_models(symbols)

    def _in_cooldown(self, symbol: str) -> bool:
        last_sig = self.state.last_signal_time.get(symbol)
        if not last_sig:
            return False
        # Assuming 5m candles (300s) - ideally fetch from config timeframe
        cooldown_seconds = self.ai_config.signal_cooldown_candles * 300 
        time_since = Clock.now() - last_sig
        return time_since.total_seconds() < cooldown_seconds

    def get_latest_regime(self, symbol: str) -> Optional[str]:
        return self.state.last_detected_regime.get(symbol)

    async def analyze_market(self, symbol: str, df: pd.DataFrame, position: Optional[Position]) -> Optional[TradeSignal]:
        if not self.ensemble_learner.is_trained:
            return None

        # 1. Performance Monitoring (Async check on past predictions)
        if self.ai_config.performance.enabled:
            await self._monitor_performance(symbol, df)

        # 2. Circuit Breaker Check
        if position is None and self.state.force_retrain_flags.get(symbol, False):
            logger.debug("Skipping entry due to force_retrain flag.", symbol=symbol)
            return None

        # 3. Regime Detection
        df_enriched = self.regime_detector.add_regime_features(df)
        regime_result = await self.regime_detector.detect_regime(symbol, df_enriched)
        regime = regime_result.get('regime')
        
        prev_regime = self.state.last_detected_regime.get(symbol)
        if prev_regime and prev_regime != regime:
            logger.info("Market regime shift.", symbol=symbol, old=prev_regime, new=regime)
            self.state.force_retrain_flags[symbol] = True
        self.state.last_detected_regime[symbol] = regime

        # 4. Regime Filtering
        if self.ai_config.use_regime_filter and regime in ['sideways', 'unknown']:
            # Allow sideways only if explicitly configured
            if not (regime == 'sideways' and self.ai_config.market_regime.sideways_confidence_threshold is not None):
                return None

        # 5. Cooldown Check
        if position is None and self._in_cooldown(symbol):
            return None

        # 6. Prediction
        leader_df = None
        if self.ai_config.market_leader_symbol and self.data_fetcher:
            leader_df = self.data_fetcher.get_market_data(self.ai_config.market_leader_symbol)

        dynamic_weights = self._calculate_dynamic_weights(symbol)
        prediction: AIInferenceResult = await self.ensemble_learner.predict(df_enriched, symbol, regime=regime, leader_df=leader_df, custom_weights=dynamic_weights)
        
        if not prediction.action: # Empty result
            return None

        action = prediction.action
        confidence = prediction.confidence
        is_anomaly = prediction.is_anomaly

        # 7. Drift Handling
        if is_anomaly:
            self.state.drift_counters[symbol] = self.state.drift_counters.get(symbol, 0) + 1
            if self.state.drift_counters[symbol] >= self.ai_config.drift.max_consecutive_anomalies:
                self.state.force_retrain_flags[symbol] = True
                self.state.drift_counters[symbol] = 0
            if self.ai_config.drift.block_trade:
                self.state.save()
                return None
            confidence = max(0.0, confidence - self.ai_config.drift.confidence_penalty)
        else:
            self.state.drift_counters[symbol] = 0

        # 8. Log Prediction
        if self.ai_config.performance.enabled and action != 'hold':
            self.state.log_prediction(symbol, df.index[-1], action, prediction.individual_predictions)

        # 9. Threshold Check
        is_exit = False
        if position:
            if (position.side == 'BUY' and action == 'sell') or (position.side == 'SELL' and action == 'buy'):
                is_exit = True
        
        required_threshold = self._get_confidence_threshold(regime, is_exit, optimized_base=prediction.optimized_threshold)
        if confidence < required_threshold:
            return None

        # 10. Signal Generation
        final_action = None
        if position:
            if position.side == 'BUY' and action == 'sell': final_action = 'SELL'
            elif position.side == 'SELL' and action == 'buy': final_action = 'BUY'
        else:
            if action == 'buy': final_action = 'BUY'
            elif action == 'sell': final_action = 'SELL'

        if final_action:
            self.state.last_signal_time[symbol] = Clock.now()
            self.state.save()
            
            strategy_metadata = {
                'model_version': prediction.model_version,
                'confidence': float(confidence),
                'effective_threshold': float(required_threshold),
                'regime': regime,
                'regime_confidence': float(regime_result.get('confidence', 0.0)),
                'active_weights': {k: float(v) for k, v in prediction.active_weights.items()},
                'top_features': {k: float(v) for k, v in prediction.top_features.items()},
                'metrics': prediction.metrics,
                'is_anomaly': is_anomaly,
                'optimized_threshold': float(prediction.optimized_threshold) if prediction.optimized_threshold else None
            }

            logger.info(f"AI Signal: {final_action} {symbol} (Conf: {confidence:.2f} | Regime: {regime})")
            return TradeSignal(
                symbol=symbol,
                action=final_action,
                regime=regime,
                confidence=confidence,
                strategy_name=self.config.name,
                metadata=strategy_metadata
            )

        return None

    def _get_confidence_threshold(self, regime: str, is_exit: bool = False, optimized_base: Optional[float] = None) -> float:
        regime_config = self.ai_config.market_regime
        if is_exit:
            base = self.ai_config.exit_confidence_threshold
            if regime == 'bull' and regime_config.bull_exit_threshold is not None: return regime_config.bull_exit_threshold
            if regime == 'bear' and regime_config.bear_exit_threshold is not None: return regime_config.bear_exit_threshold
            if regime == 'volatile' and regime_config.volatile_exit_threshold is not None: return regime_config.volatile_exit_threshold
            if regime == 'sideways' and regime_config.sideways_exit_threshold is not None: return regime_config.sideways_exit_threshold
            return base
        else:
            ai_threshold = self.ai_config.confidence_threshold
            if optimized_base is not None and self.ai_config.training.optimize_entry_threshold:
                ai_threshold = optimized_base

            manager_threshold = self.ai_config.confidence_threshold
            if regime == 'bull' and regime_config.bull_confidence_threshold is not None: manager_threshold = regime_config.bull_confidence_threshold
            elif regime == 'bear' and regime_config.bear_confidence_threshold is not None: manager_threshold = regime_config.bear_confidence_threshold
            elif regime == 'volatile' and regime_config.volatile_confidence_threshold is not None: manager_threshold = regime_config.volatile_confidence_threshold
            elif regime == 'sideways' and regime_config.sideways_confidence_threshold is not None: manager_threshold = regime_config.sideways_confidence_threshold
            
            return max(ai_threshold, manager_threshold)

    def _calculate_dynamic_weights(self, symbol: str) -> Optional[Dict[str, float]]:
        if not self.ai_config.ensemble_weights.use_dynamic_weighting:
            return None
        stats = self.state.model_performance_stats.get(symbol)
        if not stats: return None
        accuracies = {m: sum(h)/len(h) for m, h in stats.items() if len(h) >= 5}
        if not accuracies: return None
        valid_models = {m: acc for m, acc in accuracies.items() if acc > 0.45}
        if not valid_models: return None
        total_acc = sum(valid_models.values())
        return {m: acc / total_acc for m, acc in valid_models.items()}

    async def _monitor_performance(self, symbol: str, df: pd.DataFrame):
        logs = self.state.prediction_logs.get(symbol, [])
        if not logs: return

        horizon = self.ai_config.features.labeling_horizon
        eval_df = df.tail(200)
        if len(eval_df) < horizon + 1: return

        try:
            actual_labels = FeatureProcessor.create_labels(eval_df, self.ai_config)
            if symbol not in self.state.accuracy_history: 
                self.state.accuracy_history[symbol] = deque(maxlen=self.ai_config.performance.window_size)
            
            remaining_logs = []
            evaluated_count = 0
            for ts, pred_int in logs:
                if ts in actual_labels.index:
                    actual = actual_labels.loc[ts]
                    if not np.isnan(actual):
                        is_correct = 1 if int(actual) == pred_int else 0
                        self.state.accuracy_history[symbol].append(is_correct)
                        evaluated_count += 1
                    else:
                        remaining_logs.append((ts, pred_int))
                elif ts > df.index[0]:
                    remaining_logs.append((ts, pred_int))
            
            self.state.prediction_logs[symbol] = remaining_logs
            if evaluated_count > 0: self.state.save()

            # Check Accuracy Thresholds
            if len(self.state.accuracy_history[symbol]) >= 10:
                accuracy = sum(self.state.accuracy_history[symbol]) / len(self.state.accuracy_history[symbol])
                
                # Critical Rollback
                if self.ai_config.performance.auto_rollback and accuracy < self.ai_config.performance.critical_accuracy_threshold:
                    logger.critical("Model accuracy CRITICAL. Initiating ROLLBACK.", symbol=symbol, accuracy=f"{accuracy:.2%}")
                    if await self.ensemble_learner.rollback_model(symbol):
                        self.state.clear_symbol_state(symbol)
                        return
                
                # Retrain Trigger
                if accuracy < self.ai_config.performance.min_accuracy and not self.state.force_retrain_flags.get(symbol, False):
                    logger.warning("Model accuracy dropped below threshold. Forcing retrain.", symbol=symbol, accuracy=f"{accuracy:.2%}")
                    self.state.force_retrain_flags[symbol] = True
                    self.state.save()

            # Dynamic Weights Monitoring
            if self.ai_config.ensemble_weights.use_dynamic_weighting:
                await self._monitor_individual_models(symbol, df, actual_labels)

        except Exception as e:
            logger.error("Failed to monitor ensemble performance", error=str(e))

    async def _monitor_individual_models(self, symbol: str, df: pd.DataFrame, actual_labels: pd.Series):
        ind_logs = self.state.individual_model_logs.get(symbol, [])
        if not ind_logs: return
        
        if symbol not in self.state.model_performance_stats: 
            self.state.model_performance_stats[symbol] = {}
        
        remaining_ind_logs = []
        for ts, preds_dict in ind_logs:
            if ts in actual_labels.index:
                actual = actual_labels.loc[ts]
                if not np.isnan(actual):
                    actual_int = int(actual)
                    for model_name, probs in preds_dict.items():
                        is_correct = 1 if np.argmax(probs) == actual_int else 0
                        if model_name not in self.state.model_performance_stats[symbol]:
                            self.state.model_performance_stats[symbol][model_name] = deque(maxlen=self.ai_config.ensemble_weights.dynamic_window)
                        self.state.model_performance_stats[symbol][model_name].append(is_correct)
                else:
                    remaining_ind_logs.append((ts, preds_dict))
            elif ts > df.index[0]:
                remaining_ind_logs.append((ts, preds_dict))
        self.state.individual_model_logs[symbol] = remaining_ind_logs

    async def retrain(self, symbol: str, df: pd.DataFrame, executor: Executor):
        logger.info("Strategy is triggering model retraining via ProcessPool.", symbol=symbol)
        loop = asyncio.get_running_loop()
        try:
            df_enriched = self.regime_detector.add_regime_features(df)
            leader_df = None
            if self.ai_config.market_leader_symbol and self.data_fetcher:
                leader_df = await self.data_fetcher.fetch_full_history_for_symbol(self.ai_config.market_leader_symbol, len(df))

            success = await loop.run_in_executor(executor, train_ensemble_task, symbol, df_enriched, self.ai_config, leader_df)
            if success:
                self.state.last_retrained_at[symbol] = Clock.now()
                self.state.clear_symbol_state(symbol)
                logger.info("Model training successful. Reloading models.", symbol=symbol)
                await self.ensemble_learner.reload_models(symbol)
            else:
                logger.warning("Model training failed or rejected.", symbol=symbol)
        except Exception as e:
            logger.error("Error during async model retraining", symbol=symbol, error=str(e))

    def needs_retraining(self, symbol: str) -> bool:
        if not self.ensemble_learner.has_valid_model(symbol):
            return True
        if self.state.force_retrain_flags.get(symbol, False):
            return True
        if self.ai_config.retrain_interval_hours <= 0:
            return False
        if symbol not in self.state.last_retrained_at:
            last_train_time = self.ensemble_learner.get_last_training_time(symbol)
            if last_train_time: self.state.last_retrained_at[symbol] = last_train_time
            else: return True
        return (Clock.now() - self.state.last_retrained_at.get(symbol)) >= timedelta(hours=self.ai_config.retrain_interval_hours)

    def get_training_data_limit(self) -> int:
        return self.ai_config.training_data_limit
