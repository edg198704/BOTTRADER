import asyncio
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from bot_core.logger import get_logger, set_correlation_id
from bot_core.exchange_api import ExchangeAPI
from bot_core.position_manager import PositionManager, Position
from bot_core.risk_manager import RiskManager
from bot_core.strategy import TradingStrategy, AIEnsembleStrategy
from bot_core.config import BotConfig
from bot_core.data_handler import DataHandler
from bot_core.monitoring import HealthChecker, InfluxDBMetrics, AlertSystem, Watchdog
from bot_core.position_monitor import PositionMonitor
from bot_core.trade_executor import TradeExecutor, TradeExecutionResult
from bot_core.optimizer import StrategyOptimizer
from bot_core.event_system import EventBus, MarketDataEvent, TradeCompletedEvent
from bot_core.services import ServiceManager
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

class TradingBot:
    """
    Central Orchestrator. Manages the lifecycle of all sub-components and the main event loop.
    Delegates specific responsibilities to dedicated managers (Risk, Position, Watchdog).
    """
    def __init__(self, config: BotConfig, exchange_api: ExchangeAPI, data_handler: DataHandler, 
                 strategy: TradingStrategy, position_manager: PositionManager, risk_manager: RiskManager,
                 health_checker: HealthChecker, position_monitor: PositionMonitor,
                 trade_executor: TradeExecutor, alert_system: AlertSystem, 
                 shared_latest_prices: Dict[str, float], event_bus: EventBus,
                 metrics_writer: Optional[InfluxDBMetrics] = None,
                 shared_bot_state: Optional[Dict[str, Any]] = None):
        self.config = config
        self.exchange_api = exchange_api
        self.data_handler = data_handler
        self.strategy = strategy
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.health_checker = health_checker
        self.position_monitor = position_monitor
        self.trade_executor = trade_executor
        self.alert_system = alert_system
        self.event_bus = event_bus
        self.metrics_writer = metrics_writer
        self.shared_bot_state = shared_bot_state if shared_bot_state is not None else {}
        
        self.optimizer = StrategyOptimizer(config, position_manager)
        self.service_manager = ServiceManager()
        
        self.start_time = datetime.now(timezone.utc)
        self.latest_prices = shared_latest_prices
        self.market_details: Dict[str, Dict[str, Any]] = {}
        self.processed_candles: Dict[str, pd.Timestamp] = {}
        
        self._symbol_locks: Dict[str, asyncio.Lock] = {}
        
        self.process_executor = ProcessPoolExecutor(max_workers=2)
        
        # Initialize Watchdog
        self.watchdog = Watchdog(
            symbols=config.strategy.symbols,
            alert_system=alert_system,
            stop_callback=self.stop,
            health_checker=health_checker,
            timeout_seconds=300
        )
        
        self._initialize_shared_state()
        logger.info("TradingBot orchestrator initialized.")

    def _initialize_shared_state(self):
        self.shared_bot_state.update({
            'status': 'initializing',
            'start_time': self.start_time,
            'position_manager': self.position_manager,
            'risk_manager': self.risk_manager,
            'latest_prices': self.latest_prices,
            'config': self.config,
            'strategy': self.strategy
        })

    def _get_symbol_lock(self, symbol: str) -> asyncio.Lock:
        if symbol not in self._symbol_locks:
            self._symbol_locks[symbol] = asyncio.Lock()
        return self._symbol_locks[symbol]

    async def setup(self):
        logger.info("Setting up TradingBot components...")
        if isinstance(self.strategy, AIEnsembleStrategy):
            self.strategy.data_fetcher = self.data_handler

        self.position_monitor.set_strategy(self.strategy)
        await self._load_market_details()
        
        # Initial Reconciliation
        await self.position_manager.reconcile_positions(self.exchange_api, self.latest_prices)
        
        logger.info("Warming up strategy...")
        await self.strategy.warmup(self.config.strategy.symbols)
        
        # Event Subscriptions
        self.event_bus.subscribe(MarketDataEvent, self.on_market_data)
        self.event_bus.subscribe(TradeCompletedEvent, self.strategy.on_trade_complete)
        
        logger.info("TradingBot setup complete.")

    async def _load_market_details(self):
        for symbol in self.config.strategy.symbols:
            try:
                details = await self.exchange_api.fetch_market_details(symbol)
                if details: 
                    self.market_details[symbol] = details
            except Exception as e:
                logger.error("Could not load market details", symbol=symbol, error=str(e))
        self.trade_executor.market_details = self.market_details

    async def run(self):
        self.shared_bot_state['status'] = 'running'
        logger.info("Starting TradingBot...", symbols=self.config.strategy.symbols)
        
        await self.setup()

        self.service_manager.register("DataHandler", self.data_handler.run(), critical=True)
        self.service_manager.register("PositionMonitor", self.position_monitor.run(), critical=True)
        self.service_manager.register("Watchdog", self.watchdog.run(), critical=True)
        self.service_manager.register("Monitoring", self._monitoring_loop(), critical=False)
        self.service_manager.register("Retraining", self._retraining_loop(), critical=False)
        self.service_manager.register("Optimizer", self.optimizer.run(), critical=False)

        self.service_manager.start_all()
        await self.service_manager.monitor()
        logger.info("Service Manager loop ended. Shutting down.")

    async def on_market_data(self, event: MarketDataEvent):
        set_correlation_id()
        # Fire and forget tick processing
        asyncio.create_task(self.process_symbol_tick(event.symbol))

    async def process_symbol_tick(self, symbol: str):
        # 1. Fast Fail on Risk Halt or Circuit Breaker
        if self.risk_manager.is_halted or self.watchdog._circuit_breaker.tripped:
            return

        # 2. Latency Guard
        latency = self.data_handler.get_latency(symbol)
        if latency > 10.0:
            logger.warning("Skipping tick due to high latency", symbol=symbol, latency=f"{latency:.2f}s")
            if self.metrics_writer:
                await self.metrics_writer.write_metric('data_latency_skip', fields={'latency': latency}, tags={'symbol': symbol})
            return

        lock = self._get_symbol_lock(symbol)
        if lock.locked():
            # Drop tick if busy (Backpressure)
            return

        ctx = TickContext(symbol=symbol, start_time=time.perf_counter())
        
        try:
            async with lock:
                await asyncio.wait_for(self._process_tick_pipeline(ctx), timeout=15.0)
        except asyncio.TimeoutError:
            logger.error("Tick processing timed out", symbol=symbol)
            self.watchdog.record_error("TickTimeout")
        except Exception as e:
            logger.error(f"Critical error processing tick for {symbol}", error=str(e), exc_info=True)
            self.watchdog.record_error("TickError")
        finally:
            self.watchdog.register_heartbeat(symbol)
            if self.metrics_writer and ctx.stages:
                await self.metrics_writer.write_metric('tick_pipeline', fields=ctx.stages, tags={'symbol': symbol})

    async def _process_tick_pipeline(self, ctx: TickContext):
        # Stage 1: Data Fetch
        ctx.df = self.data_handler.get_market_data(ctx.symbol, include_forming=False)
        if ctx.df is None or ctx.df.empty: return
        
        last_ts = ctx.df.index[-1]
        if self.processed_candles.get(ctx.symbol) == last_ts:
            return # Already processed this candle
        
        if ctx.symbol not in self.latest_prices:
            return

        ctx.mark_stage('data_fetch')

        # Stage 2: Position Fetch
        ctx.position = await self.position_manager.get_open_position(ctx.symbol)
        ctx.mark_stage('position_fetch')

        # Stage 3: Strategy Analysis
        ctx.signal = await self.strategy.analyze_market(ctx.symbol, ctx.df, ctx.position)
        ctx.mark_stage('strategy_analysis')

        # Stage 4: Execution
        if ctx.signal: 
            ctx.execution_result = await self.trade_executor.execute_trade_signal(ctx.signal, ctx.df, ctx.position)
            if ctx.execution_result and self.metrics_writer:
                await self.metrics_writer.write_metric('trade_execution', fields=ctx.execution_result.dict())
            ctx.mark_stage('execution')
        
        self.processed_candles[ctx.symbol] = last_ts
        ctx.mark_stage('total_duration')

    async def _monitoring_loop(self):
        reconcile_counter = 0
        summary_counter = 0
        while not self.service_manager._shutdown_event.is_set():
            try:
                # Periodic Reconciliation (every 2 minutes)
                reconcile_counter += 1
                if reconcile_counter >= 2:
                    await self.position_manager.reconcile_positions(self.exchange_api, self.latest_prices)
                    reconcile_counter = 0

                if not self.latest_prices:
                    await asyncio.sleep(5)
                    continue

                open_pos = await self.position_manager.get_all_open_positions()
                val = self.position_manager.get_portfolio_value(self.latest_prices, open_pos)
                pnl = await self.position_manager.get_daily_realized_pnl()
                
                await self.risk_manager.update_portfolio_risk(val, pnl)
                
                if self.risk_manager.liquidation_needed:
                    await self._liquidate_all_positions("Emergency Risk Halt")
                    self.risk_manager.liquidation_needed = False

                self.shared_bot_state['portfolio_equity'] = val
                self.shared_bot_state['open_positions_count'] = len(open_pos)
                self.shared_bot_state['daily_pnl'] = pnl
                
                if self.metrics_writer:
                    await self.metrics_writer.write_metric('portfolio', fields={'equity': val, 'daily_pnl': pnl, 'open_positions': len(open_pos)})

                # System Summary Log (Every 5 minutes)
                summary_counter += 1
                if summary_counter >= 5:
                    logger.info("System Summary", equity=f"${val:.2f}", daily_pnl=f"${pnl:.2f}", positions=len(open_pos))
                    summary_counter = 0

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                self.watchdog.record_error("MonitoringLoop")
            await asyncio.sleep(60)

    async def _liquidate_all_positions(self, reason: str):
        logger.warning("Initiating portfolio liquidation...", reason=reason)
        open_pos = await self.position_manager.get_all_open_positions()
        tasks = [self.trade_executor.close_position(pos, reason) for pos in open_pos]
        await asyncio.gather(*tasks)

    async def _retraining_loop(self):
        logger.info("Starting model retraining loop.")
        await asyncio.sleep(60)
        while not self.service_manager._shutdown_event.is_set():
            try:
                for symbol in self.config.strategy.symbols:
                    if self.strategy.needs_retraining(symbol):
                        limit = self.strategy.get_training_data_limit()
                        if limit > 0:
                            df = await self.data_handler.fetch_full_history_for_symbol(symbol, limit)
                            if df is not None and not df.empty:
                                await self.strategy.retrain(symbol, df, self.process_executor)
                await asyncio.sleep(300)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in retraining loop", error=str(e))
                self.watchdog.record_error("RetrainingLoop")
                await asyncio.sleep(300)

    async def _close_position(self, position: Position, reason: str):
        await self.trade_executor.close_position(position, reason)

    async def stop(self):
        logger.info("Stopping TradingBot...")
        self.shared_bot_state['status'] = 'stopping'
        await self.service_manager.stop_all()
        await self.watchdog.stop()
        await self.data_handler.stop()
        await self.position_monitor.stop()
        await self.optimizer.stop()
        await self.strategy.close()
        if self.exchange_api: await self.exchange_api.close()
        if self.position_manager: self.position_manager.close()
        if self.metrics_writer: await self.metrics_writer.close()
        self.process_executor.shutdown(wait=True)
        logger.info("TradingBot stopped gracefully.")
        self.shared_bot_state['status'] = 'stopped'
