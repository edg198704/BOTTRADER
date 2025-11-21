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

logger = get_logger(__name__)

class StrategyStateManager:
    def __init__(self, config: AIEnsembleStrategyParams):
        self.config = config
        self.persistence = AsyncAtomicJsonStore(os.path.join(config.model_path, "strategy_state.json"))
        self.last_retrained_at = {}
        self.last_signal_time = {}
        self.force_retrain_flags = {}
        self.accuracy_history = {}

    async def load(self):
        state = await self.persistence.load()
        if not state: return
        self.last_retrained_at = {k: datetime.fromisoformat(v) for k,v in state.get('last_retrained_at', {}).items()}
        self.last_signal_time = {k: datetime.fromisoformat(v) for k,v in state.get('last_signal_time', {}).items()}
        self.force_retrain_flags = state.get('force_retrain_flags', {})

    async def save(self):
        state = {
            'last_retrained_at': {k: v.isoformat() for k,v in self.last_retrained_at.items()},
            'last_signal_time': {k: v.isoformat() for k,v in self.last_signal_time.items()},
            'force_retrain_flags': self.force_retrain_flags
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
                await self.ensemble_learner.reload_models(symbol)
                logger.info(f"Retraining successful for {symbol}")
        finally:
            self._training_in_progress.discard(symbol)
            asyncio.create_task(self.state.save())

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
        # If meta_probability is available and low, we might want to skip even if base confidence is high
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
            self.state.last_signal_time[symbol] = Clock.now()
            asyncio.create_task(self.state.save())
            meta = {
                'model_version': pred.model_version, 'confidence': pred.confidence, 
                'regime': regime, 'metrics': pred.metrics, 'effective_threshold': threshold,
                'meta_prob': pred.meta_probability
            }
            return TradeSignal(symbol=symbol, action=final_action, regime=regime, confidence=pred.confidence, strategy_name=self.config.name, metadata=meta)
        return None
