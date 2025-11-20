import abc
import asyncio
import pandas as pd
import numpy as np
import json
import os
import tempfile
from typing import Dict, Any, Optional, List, Tuple, Literal
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
from bot_core.utils import Clock

logger = get_logger(__name__)

class TradeSignal(BaseModel):
    """
    A standardized signal object enforcing type safety across the trading pipeline.
    """
    symbol: str
    action: Literal['BUY', 'SELL']
    regime: Optional[str] = None
    confidence: float = 0.0
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    strategy_name: str
    metadata: Dict[str, Any] = {}

class AtomicStateWriter:
    """Helper class to handle atomic JSON state persistence."""
    def __init__(self, path: str):
        self.path = path

    def save(self, state: Dict[str, Any]):
        try:
            dir_name = os.path.dirname(self.path)
            os.makedirs(dir_name, exist_ok=True)
            
            # Write to temp file first
            with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False) as tf:
                json.dump(state, tf, indent=2)
                temp_name = tf.name
            
            # Atomic rename
            os.replace(temp_name, self.path)
        except Exception as e:
            logger.error("Failed to save strategy state atomically", error=str(e))
            if 'temp_name' in locals() and os.path.exists(temp_name):
                os.remove(temp_name)

    def load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load strategy state", error=str(e))
            return {}

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
    def get_training_data_limit(self) -> int:
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
            logger.warning("Required SMA indicators not found in DataFrame.", 
                         symbol=symbol, required=[self.fast_ma_col, self.slow_ma_col])
            return None

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        is_bullish_cross = last_row[self.fast_ma_col] > last_row[self.slow_ma_col] and prev_row[self.fast_ma_col] <= prev_row[self.slow_ma_col]
        is_bearish_cross = last_row[self.fast_ma_col] < last_row[self.slow_ma_col] and prev_row[self.fast_ma_col] >= prev_row[self.slow_ma_col]

        action = None
        if position:
            if position.side == 'BUY' and is_bearish_cross:
                action = 'SELL'
            elif position.side == 'SELL' and is_bullish_cross:
                action = 'BUY'
        else:
            if is_bullish_cross:
                action = 'BUY'
            elif is_bearish_cross:
                action = 'SELL'

        if action:
            logger.info(f"Signal detected: {action} ({symbol}) via MA Crossover")
            return TradeSignal(
                symbol=symbol,
                action=action,
                strategy_name=self.config.name,
                confidence=1.0
            )

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
        self.individual_model_logs: Dict[str, List[Tuple[datetime, Dict[str, Any]]]] = {}
        self.model_performance_stats: Dict[str, Dict[str, deque]] = {}
        
        self.persistence = AtomicStateWriter(os.path.join(self.ai_config.model_path, "strategy_state.json"))

        try:
            self.ensemble_learner = EnsembleLearner(self.ai_config)
            self.regime_detector = MarketRegimeDetector(self.ai_config)
            self._load_state()
            logger.info("AIEnsembleStrategy initialized")
        except ImportError as e:
            logger.critical("Failed to initialize AIEnsembleStrategy due to missing ML libraries", error=str(e))
            raise

    async def close(self):
        self._save_state()
        await self.ensemble_learner.close()

    async def warmup(self, symbols: List[str]):
        logger.info("Warming up AI Ensemble Strategy...")
        await self.ensemble_learner.warmup_models(symbols)

    def _save_state(self):
        # Convert deques and numpy types to standard python types for JSON serialization
        state = {
            'last_retrained_at': {k: v.isoformat() for k, v in self.last_retrained_at.items()},
            'last_signal_time': {k: v.isoformat() for k, v in self.last_signal_time.items()},
            'force_retrain_flags': self.force_retrain_flags,
            'drift_counters': self.drift_counters,
            'accuracy_history': {k: list(v) for k, v in self.accuracy_history.items()},
            'prediction_logs': {k: [(ts.isoformat(), int(p)) for ts, p in v] for k, v in self.prediction_logs.items()},
            'individual_model_logs': {
                k: [(ts.isoformat(), {m: p.tolist() if isinstance(p, np.ndarray) else float(p) for m, p in preds.items()}) 
                    for ts, preds in v[-50:]] 
                for k, v in self.individual_model_logs.items()
            },
            'model_performance_stats': {
                k: {m: list(d) for m, d in v.items()} for k, v in self.model_performance_stats.items()
            }
        }
        self.persistence.save(state)

    def _load_state(self):
        state = self.persistence.load()
        if not state:
            return

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
                self.individual_model_logs[k] = [
                    (datetime.fromisoformat(ts), {m: np.array(p) for m, p in preds.items()}) 
                    for ts, preds in v
                ]
        
        if 'model_performance_stats' in state:
            window = self.ai_config.ensemble_weights.dynamic_window
            for k, v in state['model_performance_stats'].items():
                self.model_performance_stats[k] = {m: deque(d, maxlen=window) for m, d in v.items()}
        
        logger.info("Restored AI Strategy state from disk.")

    def _in_cooldown(self, symbol: str) -> bool:
        last_sig = self.last_signal_time.get(symbol)
        if not last_sig:
            return False
        seconds_per_candle = 300 # Assuming 5m candles, ideally fetch from config
        cooldown_seconds = self.ai_config.signal_cooldown_candles * seconds_per_candle
        time_since = Clock.now() - last_sig
        return time_since.total_seconds() < cooldown_seconds

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

    def get_latest_regime(self, symbol: str) -> Optional[str]:
        return self.last_detected_regime.get(symbol)

    def _calculate_dynamic_weights(self, symbol: str) -> Optional[Dict[str, float]]:
        if not self.ai_config.ensemble_weights.use_dynamic_weighting:
            return None
        stats = self.model_performance_stats.get(symbol)
        if not stats: return None
        accuracies = {m: sum(h)/len(h) for m, h in stats.items() if len(h) >= 5}
        if not accuracies: return None
        valid_models = {m: acc for m, acc in accuracies.items() if acc > 0.45}
        if not valid_models: return None
        total_acc = sum(valid_models.values())
        return {m: acc / total_acc for m, acc in valid_models.items()}

    async def analyze_market(self, symbol: str, df: pd.DataFrame, position: Optional[Position]) -> Optional[TradeSignal]:
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
        
        prev_regime = self.last_detected_regime.get(symbol)
        if prev_regime and prev_regime != regime:
            logger.info("Market regime shift detected.", symbol=symbol, old=prev_regime, new=regime)
            self.force_retrain_flags[symbol] = True

        self.last_detected_regime[symbol] = regime

        if self.ai_config.use_regime_filter and regime in ['sideways', 'unknown']:
            if not (regime == 'sideways' and self.ai_config.market_regime.sideways_confidence_threshold is not None):
                return None

        if position is None and self._in_cooldown(symbol):
            return None

        leader_df = None
        if self.ai_config.market_leader_symbol and self.data_fetcher:
            leader_df = self.data_fetcher.get_market_data(self.ai_config.market_leader_symbol)

        dynamic_weights = self._calculate_dynamic_weights(symbol)
        prediction = await self.ensemble_learner.predict(df_enriched, symbol, regime=regime, leader_df=leader_df, custom_weights=dynamic_weights)
        
        action = prediction.get('action')
        confidence = prediction.get('confidence', 0.0)
        is_anomaly = prediction.get('is_anomaly', False)
        
        if is_anomaly:
            self.drift_counters[symbol] = self.drift_counters.get(symbol, 0) + 1
            if self.drift_counters[symbol] >= self.ai_config.drift.max_consecutive_anomalies:
                self.force_retrain_flags[symbol] = True
                self.drift_counters[symbol] = 0
            if self.ai_config.drift.block_trade:
                self._save_state()
                return None
            confidence = max(0.0, confidence - self.ai_config.drift.confidence_penalty)
        else:
            if self.drift_counters.get(symbol, 0) > 0: self.drift_counters[symbol] = 0

        # Log Prediction for Performance Monitoring
        if self.ai_config.performance.enabled and action:
            action_map = {'sell': 0, 'hold': 1, 'buy': 2}
            if symbol not in self.prediction_logs: self.prediction_logs[symbol] = []
            self.prediction_logs[symbol].append((df.index[-1], action_map.get(action, 1)))
            if prediction.get('individual_predictions'):
                if symbol not in self.individual_model_logs: self.individual_model_logs[symbol] = []
                self.individual_model_logs[symbol].append((df.index[-1], prediction['individual_predictions']))
            self._save_state()

        is_exit = False
        if position:
            if (position.side == 'BUY' and action == 'sell') or (position.side == 'SELL' and action == 'buy'):
                is_exit = True
        
        required_threshold = self._get_confidence_threshold(regime, is_exit, optimized_base=prediction.get('optimized_threshold'))
        
        if confidence < required_threshold:
            return None

        # Ensure metadata is JSON serializable
        strategy_metadata = {
            'model_version': prediction.get('model_version'),
            'confidence': float(confidence),
            'effective_threshold': float(required_threshold),
            'regime': regime,
            'regime_confidence': float(regime_result.get('confidence', 0.0)),
            'model_type': prediction.get('model_type'),
            'active_weights': {k: float(v) for k, v in prediction.get('active_weights', {}).items()},
            'top_features': {k: float(v) for k, v in prediction.get('top_features', {}).items()},
            'metrics': prediction.get('metrics'),
            'is_anomaly': is_anomaly,
            'optimized_threshold': float(prediction.get('optimized_threshold')) if prediction.get('optimized_threshold') else None
        }

        final_action = None
        if position:
            if position.side == 'BUY' and action == 'sell': final_action = 'SELL'
            elif position.side == 'SELL' and action == 'buy': final_action = 'BUY'
        else:
            if action == 'buy': final_action = 'BUY'
            elif action == 'sell': final_action = 'SELL'

        if final_action:
            self.last_signal_time[symbol] = Clock.now()
            self._save_state()
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

    async def _monitor_performance(self, symbol: str, df: pd.DataFrame):
        logs = self.prediction_logs.get(symbol, [])
        if logs:
            horizon = self.ai_config.features.labeling_horizon
            eval_df = df.tail(200)
            if len(eval_df) >= horizon + 1:
                try:
                    actual_labels = FeatureProcessor.create_labels(eval_df, self.ai_config)
                    if symbol not in self.accuracy_history: self.accuracy_history[symbol] = deque(maxlen=self.ai_config.performance.window_size)
                    
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
                        elif ts > df.index[0]:
                            remaining_logs.append((ts, pred_int))
                    
                    self.prediction_logs[symbol] = remaining_logs
                    if evaluated_count > 0: self._save_state()

                    if len(self.accuracy_history[symbol]) >= 10:
                        accuracy = sum(self.accuracy_history[symbol]) / len(self.accuracy_history[symbol])
                        if self.ai_config.performance.auto_rollback and accuracy < self.ai_config.performance.critical_accuracy_threshold:
                            logger.critical("Model accuracy CRITICAL. Initiating ROLLBACK.", symbol=symbol, accuracy=f"{accuracy:.2%}")
                            if await self.ensemble_learner.rollback_model(symbol):
                                self.accuracy_history[symbol].clear()
                                self.prediction_logs[symbol] = []
                                self.force_retrain_flags[symbol] = False
                                self._save_state()
                                return
                        if accuracy < self.ai_config.performance.min_accuracy and not self.force_retrain_flags.get(symbol, False):
                            logger.warning("Model accuracy dropped below threshold. Forcing retrain.", symbol=symbol, accuracy=f"{accuracy:.2%}")
                            self.force_retrain_flags[symbol] = True
                            self._save_state()
                except Exception as e:
                    logger.error("Failed to monitor ensemble performance", error=str(e))

        if self.ai_config.ensemble_weights.use_dynamic_weighting:
            ind_logs = self.individual_model_logs.get(symbol, [])
            if ind_logs:
                try:
                    horizon = self.ai_config.features.labeling_horizon
                    eval_df = df.tail(200)
                    if len(eval_df) >= horizon + 1:
                        actual_labels = FeatureProcessor.create_labels(eval_df, self.ai_config)
                        if symbol not in self.model_performance_stats: self.model_performance_stats[symbol] = {}
                        remaining_ind_logs = []
                        for ts, preds_dict in ind_logs:
                            if ts in actual_labels.index:
                                actual = actual_labels.loc[ts]
                                if not np.isnan(actual):
                                    actual_int = int(actual)
                                    for model_name, probs in preds_dict.items():
                                        is_correct = 1 if np.argmax(probs) == actual_int else 0
                                        if model_name not in self.model_performance_stats[symbol]:
                                            self.model_performance_stats[symbol][model_name] = deque(maxlen=self.ai_config.ensemble_weights.dynamic_window)
                                        self.model_performance_stats[symbol][model_name].append(is_correct)
                                else:
                                    remaining_ind_logs.append((ts, preds_dict))
                            elif ts > df.index[0]:
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
                leader_df = await self.data_fetcher.fetch_full_history_for_symbol(self.ai_config.market_leader_symbol, len(df))

            success = await loop.run_in_executor(executor, train_ensemble_task, symbol, df_enriched, self.ai_config, leader_df)
            if success:
                self.last_retrained_at[symbol] = Clock.now()
                self.force_retrain_flags[symbol] = False
                self.drift_counters[symbol] = 0
                self.prediction_logs[symbol] = []
                if symbol in self.accuracy_history: self.accuracy_history[symbol].clear()
                self.individual_model_logs[symbol] = []
                if symbol in self.model_performance_stats: self.model_performance_stats[symbol].clear()
                self._save_state()
                logger.info("Model training successful. Reloading models.", symbol=symbol)
                await self.ensemble_learner.reload_models(symbol)
            else:
                logger.warning("Model training failed or rejected.", symbol=symbol)
        except Exception as e:
            logger.error("Error during async model retraining", symbol=symbol, error=str(e))

    def needs_retraining(self, symbol: str) -> bool:
        if not self.ensemble_learner.has_valid_model(symbol):
            return True
        if self.force_retrain_flags.get(symbol, False):
            return True
        if self.ai_config.retrain_interval_hours <= 0:
            return False
        if symbol not in self.last_retrained_at:
            last_train_time = self.ensemble_learner.get_last_training_time(symbol)
            if last_train_time: self.last_retrained_at[symbol] = last_train_time
            else: return True
        return (Clock.now() - self.last_retrained_at.get(symbol)) >= timedelta(hours=self.ai_config.retrain_interval_hours)

    def get_training_data_limit(self) -> int:
        return self.ai_config.training_data_limit
