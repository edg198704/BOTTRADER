import abc
import asyncio
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime, timedelta
from concurrent.futures import Executor
from collections import deque

from bot_core.logger import get_logger
from bot_core.config import AIEnsembleStrategyParams, SimpleMACrossoverStrategyParams, StrategyParamsBase
from bot_core.ai.ensemble_learner import EnsembleLearner, train_ensemble_task
from bot_core.ai.regime_detector import MarketRegimeDetector
from bot_core.position_manager import Position
from bot_core.utils import Clock, AsyncAtomicJsonStore
from bot_core.common import TradeSignal, AIInferenceResult
from bot_core.event_system import TradeCompletedEvent

logger = get_logger(__name__)

class StrategyStateManager:
    def __init__(self, config: AIEnsembleStrategyParams):
        self.config = config
        self.persistence = AsyncAtomicJsonStore(os.path.join(config.model_path, "strategy_state.json"))
        self.last_retrained_at = {}
        self.last_signal_time = {}
        self.force_retrain_flags = {}
        self.accuracy_history = {} # symbol -> deque of (is_correct, timestamp)

    async def load(self):
        state = await self.persistence.load()
        if not state: return
        self.last_retrained_at = {k: datetime.fromisoformat(v) for k,v in state.get('last_retrained_at', {}).items()}
        self.last_signal_time = {k: datetime.fromisoformat(v) for k,v in state.get('last_signal_time', {}).items()}
        self.force_retrain_flags = state.get('force_retrain_flags', {})
        # Load accuracy history (convert lists back to deques)
        hist = state.get('accuracy_history', {})
        for sym, data in hist.items():
            self.accuracy_history[sym] = deque(data, maxlen=self.config.performance.window_size)

    async def save(self):
        state = {
            'last_retrained_at': {k: v.isoformat() for k,v in self.last_retrained_at.items()},
            'last_signal_time': {k: v.isoformat() for k,v in self.last_signal_time.items()},
            'force_retrain_flags': self.force_retrain_flags,
            'accuracy_history': {k: list(v) for k,v in self.accuracy_history.items()}
        }
        await self.persistence.save(state)

class TradingStrategy(abc.ABC):
    def __init__(self, config: StrategyParamsBase):
        self.config = config
        self.data_fetcher = None

    @abc.abstractmethod
    async def analyze_market(self, symbol: str, df: pd.DataFrame, position: Optional[Position]) -> Optional[TradeSignal]: pass
    @abc.abstractmethod
    async def retrain(self, symbol: str, df: pd.DataFrame, executor: Executor): pass
    @abc.abstractmethod
    def needs_retraining(self, symbol: str) -> bool: pass
    @abc.abstractmethod
    def get_latest_regime(self, symbol: str) -> Optional[str]: pass
    async def close(self): pass
    async def warmup(self, symbols: List[str]): pass
    async def on_trade_complete(self, event: TradeCompletedEvent): pass
    def get_training_data_limit(self) -> int: return 0

class SimpleMACrossoverStrategy(TradingStrategy):
    def __init__(self, config: SimpleMACrossoverStrategyParams):
        super().__init__(config)
        self.fast_col = f"SMA_{config.fast_ma_period}"
        self.slow_col = f"SMA_{config.slow_ma_period}"

    async def analyze_market(self, symbol: str, df: pd.DataFrame, position: Optional[Position]) -> Optional[TradeSignal]:
        if self.fast_col not in df.columns or self.slow_col not in df.columns: return None
        last, prev = df.iloc[-1], df.iloc[-2]
        bull = last[self.fast_col] > last[self.slow_col] and prev[self.fast_col] <= prev[self.slow_col]
        bear = last[self.fast_col] < last[self.slow_col] and prev[self.fast_col] >= prev[self.slow_col]
        
        action = None
        if position:
            if position.side == 'BUY' and bear: action = 'SELL'
            elif position.side == 'SELL' and bull: action = 'BUY'
        else:
            if bull: action = 'BUY'
            elif bear: action = 'SELL'
            
        return TradeSignal(symbol=symbol, action=action, strategy_name=self.config.name, confidence=1.0) if action else None

    async def retrain(self, symbol, df, executor): pass
    def needs_retraining(self, symbol): return False
    def get_latest_regime(self, symbol): return None

class AIEnsembleStrategy(TradingStrategy):
    def __init__(self, config: AIEnsembleStrategyParams):
        super().__init__(config)
        self.ai_config = config
        self.state = StrategyStateManager(config)
        self._training_in_progress = set()
        self.ensemble_learner = EnsembleLearner(config)
        self.regime_detector = MarketRegimeDetector(config)
        self.last_regime = {}

    async def close(self):
        await self.state.save()
        await self.ensemble_learner.close()

    async def warmup(self, symbols: List[str]):
        await self.state.load()
        await self.ensemble_learner.warmup_models(symbols)

    def get_latest_regime(self, symbol: str) -> Optional[str]:
        return self.last_regime.get(symbol)

    def get_training_data_limit(self) -> int:
        return self.ai_config.training_data_limit

    def needs_retraining(self, symbol: str) -> bool:
        if symbol in self._training_in_progress: return False
        if not self.ensemble_learner.has_valid_model(symbol): return True
        if self.state.force_retrain_flags.get(symbol, False): return True
        last = self.state.last_retrained_at.get(symbol)
        if not last: 
            last = self.ensemble_learner.get_last_training_time(symbol)
            if last: self.state.last_retrained_at[symbol] = last
            else: return True
        return (Clock.now() - last) >= timedelta(hours=self.ai_config.retrain_interval_hours)

    async def retrain(self, symbol: str, df: pd.DataFrame, executor: Executor):
        if symbol in self._training_in_progress: return
        self._training_in_progress.add(symbol)
        try:
            # Enrich data before sending to process to ensure consistency
            regime_res = await self.regime_detector.detect_regime(symbol, df)
            df_enriched = regime_res.get('enriched_df', df)
            
            leader_df = None
            if self.ai_config.market_leader_symbol and self.data_fetcher:
                leader_df = await self.data_fetcher.fetch_full_history_for_symbol(self.ai_config.market_leader_symbol, len(df))

            loop = asyncio.get_running_loop()
            success = await loop.run_in_executor(executor, train_ensemble_task, symbol, df_enriched, self.ai_config, leader_df)
            
            if success:
                self.state.last_retrained_at[symbol] = Clock.now()
                self.state.force_retrain_flags[symbol] = False
                # Clear history on retrain to give new model a fresh start
                if symbol in self.state.accuracy_history:
                    self.state.accuracy_history[symbol].clear()
                await self.ensemble_learner.reload_models(symbol)
                logger.info(f"Retraining successful for {symbol}")
        finally:
            self._training_in_progress.discard(symbol)
            asyncio.create_task(self.state.save())

    def _check_technical_guardrails(self, symbol: str, df: pd.DataFrame, action: str) -> bool:
        """
        Configuration-driven safety checks to prevent AI from trading into extreme conditions.
        """
        if not self.ai_config.guardrails.enabled:
            return True
            
        if df is None or df.empty: return False
        last_row = df.iloc[-1]
        guards = self.ai_config.guardrails
        
        # 1. RSI Guardrail
        if 'rsi' in df.columns:
            rsi = last_row['rsi']
            if action == 'BUY' and rsi > guards.rsi_overbought:
                logger.info(f"Signal rejected by RSI Guardrail (Overbought: {rsi:.2f})", symbol=symbol)
                return False
            if action == 'SELL' and rsi < guards.rsi_oversold:
                logger.info(f"Signal rejected by RSI Guardrail (Oversold: {rsi:.2f})", symbol=symbol)
                return False
        
        # 2. ADX Trend Strength
        if guards.adx_min_strength > 0 and 'adx' in df.columns:
            adx = last_row['adx']
            if adx < guards.adx_min_strength:
                logger.info(f"Signal rejected by ADX Guardrail (Weak Trend: {adx:.2f})", symbol=symbol)
                return False

        # 3. Volume Percentile Check (Liquidity)
        if guards.min_volume_percentile > 0 and 'volume' in df.columns:
            # Check if current volume is extremely low compared to recent history
            recent_vol = df['volume'].iloc[-50:]
            if not recent_vol.empty:
                vol_percentile = (last_row['volume'] >= recent_vol).mean()
                if vol_percentile < guards.min_volume_percentile:
                    logger.info(f"Signal rejected by Volume Guardrail (Low Liquidity)", symbol=symbol)
                    return False

        return True

    async def analyze_market(self, symbol: str, df: pd.DataFrame, position: Optional[Position]) -> Optional[TradeSignal]:
        if not self.ensemble_learner.is_trained: return None
        
        # Detect Regime
        regime_res = await self.regime_detector.detect_regime(symbol, df)
        regime = regime_res.get('regime')
        self.last_regime[symbol] = regime
        df_enriched = regime_res.get('enriched_df', df)

        # Cooldown check
        last_sig = self.state.last_signal_time.get(symbol)
        if last_sig and (Clock.now() - last_sig).total_seconds() < (self.ai_config.signal_cooldown_candles * 300):
            return None

        # Predict
        leader_df = None
        if self.ai_config.market_leader_symbol and self.data_fetcher:
            leader_df = self.data_fetcher.get_market_data(self.ai_config.market_leader_symbol)
            
        pred = await self.ensemble_learner.predict(df_enriched, symbol, regime, leader_df)
        if not pred.action or pred.action == 'hold': return None

        # Threshold Logic
        is_exit = position and ((position.side == 'BUY' and pred.action == 'sell') or (position.side == 'SELL' and pred.action == 'buy'))
        threshold = self.ai_config.exit_confidence_threshold if is_exit else self.ai_config.confidence_threshold
        
        # Use optimized threshold if available and higher
        if pred.optimized_threshold and not is_exit:
            threshold = max(threshold, pred.optimized_threshold)
            
        if pred.confidence < threshold: return None

        # Meta-Model Check (Double Confirmation)
        if pred.meta_probability is not None and pred.meta_probability < self.ai_config.meta_labeling.probability_threshold:
            logger.info(f"Signal rejected by Meta-Model for {symbol}", meta_prob=pred.meta_probability)
            return None

        # Signal Generation
        final_action = None
        if position:
            if position.side == 'BUY' and pred.action == 'sell': final_action = 'SELL'
            elif position.side == 'SELL' and pred.action == 'buy': final_action = 'BUY'
        else:
            if pred.action == 'buy': final_action = 'BUY'
            elif pred.action == 'sell': final_action = 'SELL'
            
        if final_action:
            # Apply Technical Guardrails
            if not self._check_technical_guardrails(symbol, df, final_action):
                return None

            self.state.last_signal_time[symbol] = Clock.now()
            asyncio.create_task(self.state.save())
            meta = {
                'model_version': pred.model_version, 'confidence': pred.confidence, 
                'regime': regime, 'metrics': pred.metrics, 'effective_threshold': threshold,
                'meta_prob': pred.meta_probability,
                'individual_predictions': pred.individual_predictions, # Pass for online learning
                'top_features': pred.top_features # Pass for explainability
            }
            return TradeSignal(symbol=symbol, action=final_action, regime=regime, confidence=pred.confidence, strategy_name=self.config.name, metadata=meta)
        return None

    async def on_trade_complete(self, event: TradeCompletedEvent):
        """Callback for Online Learning and Drift Detection"""
        pos = event.position
        symbol = pos.symbol
        
        # 1. Online Learning (Weight Updates)
        if self.ai_config.ensemble_weights.use_dynamic_weighting and pos.strategy_metadata:
            try:
                meta = json.loads(pos.strategy_metadata)
                ind_preds = meta.get('individual_predictions')
                if ind_preds:
                    self.ensemble_learner.update_weights(symbol, pos.pnl, ind_preds, pos.side)
            except Exception as e:
                logger.error(f"Failed to process trade completion for AI learning: {e}")

        # 2. Drift Detection (Performance Monitoring)
        if self.ai_config.performance.enabled:
            is_win = pos.pnl > 0
            if symbol not in self.state.accuracy_history:
                self.state.accuracy_history[symbol] = deque(maxlen=self.ai_config.performance.window_size)
            
            self.state.accuracy_history[symbol].append(is_win)
            
            # Check accuracy
            history = self.state.accuracy_history[symbol]
            if len(history) >= 10: # Minimum sample
                accuracy = sum(history) / len(history)
                if accuracy < self.ai_config.performance.min_accuracy:
                    logger.warning(f"Drift Detected for {symbol}. Accuracy: {accuracy:.2f}. Triggering Retrain.")
                    self.state.force_retrain_flags[symbol] = True
                    asyncio.create_task(self.state.save())
