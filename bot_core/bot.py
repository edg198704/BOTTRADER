import asyncio
import time
from typing import Dict, Any, Optional
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
from bot_core.trade_executor import TradeExecutor
from bot_core.optimizer import StrategyOptimizer
from bot_core.event_system import EventBus, MarketDataEvent, TradeCompletedEvent
from bot_core.services import ServiceManager
from bot_core.tick_pipeline import TickPipeline

logger = get_logger(__name__)

class SymbolProcessor:
    """
    Actor-like class responsible for processing ticks for a specific symbol sequentially.
    Implements Conflation: If the system is busy, older ticks in the queue are discarded 
    in favor of the newest one to ensure real-time relevance.
    """
    def __init__(self, symbol: str, pipeline: TickPipeline):
        self.symbol = symbol
        self.pipeline = pipeline
        # Conflation queue: Size 1 ensures we only ever hold the LATEST tick pending.
        self._queue = asyncio.Queue(maxsize=1) 
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._process_loop(), name=f"Processor-{self.symbol}")
        logger.info(f"SymbolProcessor started for {self.symbol}")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"SymbolProcessor stopped for {self.symbol}")

    def on_tick(self):
        """
        Called when a new tick is available.
        Non-blocking. Drops old tick if busy (Conflation).
        """
        if not self._running:
            return

        try:
            # If queue is full, remove the old item (conflation)
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except asyncio.QueueEmpty:
                    pass
            
            self._queue.put_nowait(time.perf_counter())
        except Exception as e:
            logger.error(f"Error queuing tick for {self.symbol}", error=str(e))

    async def _process_loop(self):
        while self._running:
            try:
                # Wait for a tick signal
                start_time = await self._queue.get()
                
                # Process using the pipeline
                await self.pipeline.run(self.symbol, start_time)
                
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in processor loop for {self.symbol}", error=str(e), exc_info=True)

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
        
        # Initialize Watchdog
        self.watchdog = Watchdog(
            symbols=config.strategy.symbols,
            alert_system=alert_system,
            stop_callback=self.stop,
            health_checker=health_checker,
            timeout_seconds=300
        )

        # Initialize Tick Pipeline
        self.tick_pipeline = TickPipeline(
            data_handler=data_handler,
            position_manager=position_manager,
            strategy=strategy,
            trade_executor=trade_executor,
            watchdog=self.watchdog,
            metrics_writer=metrics_writer,
            latest_prices=shared_latest_prices
        )
        
        # Symbol Processors (Actors)
        self.processors: Dict[str, SymbolProcessor] = {}
        
        self.process_executor = ProcessPoolExecutor(max_workers=2)
        
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
        
        # Initialize Processors
        for symbol in self.config.strategy.symbols:
            self.processors[symbol] = SymbolProcessor(symbol, self.tick_pipeline)

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

        # Start Processors
        for proc in self.processors.values():
            await proc.start()

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
        # Dispatch to the specific symbol processor
        if event.symbol in self.processors:
            self.processors[event.symbol].on_tick()

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
        
        # Stop processors first to stop ingesting new ticks
        for proc in self.processors.values():
            await proc.stop()
            
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
