import time
import pandas as pd
from typing import Dict, Any, Optional
from dataclasses import dataclass

from bot_core.logger import get_logger
from bot_core.position_manager import PositionManager, Position
from bot_core.strategy import TradingStrategy
from bot_core.trade_executor import TradeExecutor, TradeExecutionResult
from bot_core.data_handler import DataHandler
from bot_core.monitoring import Watchdog, InfluxDBMetrics
from bot_core.common import TradeSignal

logger = get_logger(__name__)

@dataclass
class TickContext:
    """Encapsulates the state of a single tick processing pipeline."""
    symbol: str
    start_time: float
    df: Optional[pd.DataFrame] = None
    position: Optional[Position] = None
    signal: Optional[TradeSignal] = None
    execution_result: Optional[TradeExecutionResult] = None
    stages: Dict[str, float] = None

    def __post_init__(self):
        self.stages = {}

    def mark_stage(self, name: str):
        self.stages[name] = (time.perf_counter() - self.start_time) * 1000

class TickPipeline:
    """
    Encapsulates the logic for processing a single tick for a single symbol.
    Orchestrates Data -> Position -> Strategy -> Execution.
    """
    def __init__(self, 
                 data_handler: DataHandler,
                 position_manager: PositionManager,
                 strategy: TradingStrategy,
                 trade_executor: TradeExecutor,
                 watchdog: Watchdog,
                 metrics_writer: Optional[InfluxDBMetrics],
                 latest_prices: Dict[str, float]):
        self.data_handler = data_handler
        self.position_manager = position_manager
        self.strategy = strategy
        self.trade_executor = trade_executor
        self.watchdog = watchdog
        self.metrics_writer = metrics_writer
        self.latest_prices = latest_prices
        self.processed_candles: Dict[str, pd.Timestamp] = {}

    async def run(self, symbol: str, start_time: float):
        ctx = TickContext(symbol=symbol, start_time=start_time)
        
        try:
            # Stage 1: Data Fetch
            ctx.df = self.data_handler.get_market_data(symbol, include_forming=False)
            if ctx.df is None or ctx.df.empty: 
                return
            
            last_ts = ctx.df.index[-1]
            if self.processed_candles.get(symbol) == last_ts:
                return # Already processed this candle
            
            if symbol not in self.latest_prices:
                return

            ctx.mark_stage('data_fetch')

            # Stage 2: Position Fetch
            ctx.position = await self.position_manager.get_open_position(symbol)
            ctx.mark_stage('position_fetch')

            # Stage 3: Strategy Analysis
            ctx.signal = await self.strategy.analyze_market(symbol, ctx.df, ctx.position)
            ctx.mark_stage('strategy_analysis')

            # Stage 4: Execution
            if ctx.signal: 
                ctx.execution_result = await self.trade_executor.execute_trade_signal(ctx.signal, ctx.df, ctx.position)
                if ctx.execution_result and self.metrics_writer:
                    await self.metrics_writer.write_metric('trade_execution', fields=ctx.execution_result.dict())
                ctx.mark_stage('execution')
            
            self.processed_candles[symbol] = last_ts
            ctx.mark_stage('total_duration')

        except Exception as e:
            logger.error(f"Critical error processing tick for {symbol}", error=str(e), exc_info=True)
            self.watchdog.record_error("TickError")
        finally:
            self.watchdog.register_heartbeat(symbol)
            if self.metrics_writer and ctx.stages:
                await self.metrics_writer.write_metric('tick_pipeline', fields=ctx.stages, tags={'symbol': symbol})
