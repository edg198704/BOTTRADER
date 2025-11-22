import time
import asyncio
import pandas as pd
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

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
    symbol: str
    start_time: float
    df: Optional[pd.DataFrame] = None
    position: Optional[Position] = None
    signal: Optional[TradeSignal] = None
    execution_result: Optional[TradeExecutionResult] = None
    stages: Dict[str, float] = field(default_factory=dict)

    def mark_stage(self, name: str):
        self.stages[name] = (time.perf_counter() - self.start_time) * 1000

class TickPipeline:
    """
    Encapsulates the logic for processing a single tick for a single symbol.
    Orchestrates Data -> Position -> Strategy -> Execution (Async Initiation).
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

            # Stage 4: Execution Initiation
            if ctx.signal: 
                # This is now non-blocking (returns INITIATED immediately)
                ctx.execution_result = await self.trade_executor.execute_trade_signal(ctx.signal, ctx.df, ctx.position)
                if ctx.execution_result and self.metrics_writer:
                    # Log the initiation of the trade
                    await self.metrics_writer.write_metric('trade_initiated', fields=ctx.execution_result.dict())
                ctx.mark_stage('execution_init')
            
            self.processed_candles[symbol] = last_ts
            ctx.mark_stage('total_duration')

        except Exception as e:
            logger.error(f"Critical error processing tick for {symbol}", error=str(e), exc_info=True)
            self.watchdog.record_error("TickError")
        finally:
            self.watchdog.register_heartbeat(symbol)
            if self.metrics_writer and ctx.stages:
                await self.metrics_writer.write_metric('tick_pipeline', fields=ctx.stages, tags={'symbol': symbol})

class SymbolProcessor:
    """
    Manages the processing loop for a specific symbol using an Actor-like pattern.
    Decouples tick arrival from processing to handle backpressure via conflation.
    """
    def __init__(self, symbol: str, pipeline: TickPipeline):
        self.symbol = symbol
        self.pipeline = pipeline
        self._queue = asyncio.Queue(maxsize=1) 
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        if self._running: return
        self._running = True
        self._task = asyncio.create_task(self._process_loop(), name=f"Processor-{self.symbol}")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try: await self._task
            except asyncio.CancelledError: pass

    def on_tick(self):
        if not self._running: return
        try:
            # Conflation: If queue is full, drop the old tick and replace with new one
            if self._queue.full():
                try: self._queue.get_nowait(); self._queue.task_done()
                except asyncio.QueueEmpty: pass
            self._queue.put_nowait(time.perf_counter())
        except Exception: pass

    async def _process_loop(self):
        while self._running:
            try:
                start_time = await self._queue.get()
                await self.pipeline.run(self.symbol, start_time)
                self._queue.task_done()
            except asyncio.CancelledError: break
            except Exception as e:
                logger.error(f"Error in processor loop for {self.symbol}", error=str(e))
