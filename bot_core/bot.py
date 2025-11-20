import asyncio
import time
from typing import Dict, Any, Optional, Set
from datetime import datetime, timezone
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from bot_core.logger import get_logger, set_correlation_id
from bot_core.exchange_api import ExchangeAPI
from bot_core.position_manager import PositionManager, Position
from bot_core.risk_manager import RiskManager
from bot_core.strategy import TradingStrategy, AIEnsembleStrategy, TradeSignal
from bot_core.config import BotConfig
from bot_core.data_handler import DataHandler
from bot_core.monitoring import HealthChecker, InfluxDBMetrics, AlertSystem
from bot_core.position_monitor import PositionMonitor
from bot_core.trade_executor import TradeExecutor
from bot_core.optimizer import StrategyOptimizer
from bot_core.event_system import EventBus, MarketDataEvent

logger = get_logger(__name__)

class TaskSupervisor:
    """
    Manages the lifecycle of background tasks with robust error handling and restarting capabilities.
    """
    def __init__(self):
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.critical_services: Set[str] = set()
        self._shutdown_event = asyncio.Event()

    def spawn(self, name: str, coroutine, critical: bool = False):
        if name in self.active_tasks and not self.active_tasks[name].done():
            return
        task = asyncio.create_task(coroutine, name=name)
        self.active_tasks[name] = task
        if critical:
            self.critical_services.add(name)
        logger.info(f"Spawned task: {name} (Critical: {critical})")

    async def monitor(self):
        while not self._shutdown_event.is_set():
            for name, task in list(self.active_tasks.items()):
                if task.done():
                    try:
                        exc = task.exception()
                        if exc:
                            logger.error(f"Task {name} failed.", error=str(exc))
                            if name in self.critical_services:
                                logger.critical(f"Critical service {name} failed. Initiating shutdown.")
                                self._shutdown_event.set()
                            else:
                                # Optional: Add restart logic for non-critical tasks here
                                del self.active_tasks[name]
                        else:
                            del self.active_tasks[name]
                    except asyncio.CancelledError:
                        del self.active_tasks[name]
            await asyncio.sleep(1)

    async def stop_all(self):
        self._shutdown_event.set()
        tasks = [t for t in self.active_tasks.values() if not t.done()]
        for t in tasks: t.cancel()
        if tasks: 
            await asyncio.gather(*tasks, return_exceptions=True)
        self.active_tasks.clear()

class SymbolProcessor:
    """
    Manages the processing queue for a single symbol to ensure sequential execution
    of ticks while allowing concurrency across different symbols.
    Handles backpressure by dropping older ticks if the queue is full (LIFO/Freshness priority).
    """
    def __init__(self, symbol: str, bot: 'TradingBot', max_queue_size: int = 1):
        self.symbol = symbol
        self.bot = bot
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self._task: Optional[asyncio.Task] = None
        self._running = False

    def start(self):
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._process_loop(), name=f"Processor-{self.symbol}")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def enqueue_tick(self):
        """
        Enqueues a processing request. If the queue is full, it drops the oldest item
        to ensure we always process the latest market data.
        """
        try:
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except asyncio.QueueEmpty:
                    pass
            
            self.queue.put_nowait(True) # Payload is irrelevant, we always pull latest state
        except Exception as e:
            logger.error(f"Failed to enqueue tick for {self.symbol}", error=str(e))

    async def _process_loop(self):
        while self._running:
            try:
                await self.queue.get()
                await self.bot.process_symbol_tick_logic(self.symbol)
                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing tick for {self.symbol}", error=str(e), exc_info=True)

class TradingBot:
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
        self.supervisor = TaskSupervisor()
        
        self.start_time = datetime.now(timezone.utc)
        self.latest_prices = shared_latest_prices
        self.market_details: Dict[str, Dict[str, Any]] = {}
        self.processed_candles: Dict[str, pd.Timestamp] = {}
        
        self.process_executor = ProcessPoolExecutor(max_workers=2)
        self._symbol_heartbeats: Dict[str, float] = {}
        self._watchdog_threshold = 300
        
        # Per-symbol processing queues
        self.symbol_processors: Dict[str, SymbolProcessor] = {}
        
        self._initialize_shared_state()
        logger.info("TradingBot orchestrator initialized.")

    def _initialize_shared_state(self):
        self.shared_bot_state['status'] = 'initializing'
        self.shared_bot_state['start_time'] = self.start_time
        self.shared_bot_state['position_manager'] = self.position_manager
        self.shared_bot_state['risk_manager'] = self.risk_manager
        self.shared_bot_state['latest_prices'] = self.latest_prices
        self.shared_bot_state['config'] = self.config
        self.shared_bot_state['strategy'] = self.strategy

    async def setup(self):
        logger.info("Setting up TradingBot components...")
        if isinstance(self.strategy, AIEnsembleStrategy):
            self.strategy.data_fetcher = self.data_handler

        self.position_monitor.set_strategy(self.strategy)
        await self._load_market_details()
        await self.reconcile_pending_positions()
        await self.reconcile_open_positions()
        
        logger.info("Warming up strategy...")
        await self.strategy.warmup(self.config.strategy.symbols)
        
        # Initialize Symbol Processors
        for symbol in self.config.strategy.symbols:
            self.symbol_processors[symbol] = SymbolProcessor(symbol, self)
            self.symbol_processors[symbol].start()

        # Subscribe to Event Bus
        self.event_bus.subscribe(MarketDataEvent, self.on_market_data)
        logger.info("Subscribed to MarketDataEvent.")
        
        logger.info("TradingBot setup complete.")

    async def _load_market_details(self):
        for symbol in self.config.strategy.symbols:
            try:
                details = await self.exchange_api.fetch_market_details(symbol)
                if details: self.market_details[symbol] = details
            except Exception as e:
                logger.error("Could not load market details", symbol=symbol, error=str(e))
        self.trade_executor.market_details = self.market_details

    async def reconcile_pending_positions(self):
        pending = await self.position_manager.get_pending_positions()
        for pos in pending:
            await self.position_manager.mark_position_failed(pos.symbol, pos.order_id, "Startup Reconciliation")

    async def reconcile_open_positions(self):
        open_positions = await self.position_manager.get_all_open_positions()
        if not open_positions: return
        try:
            balances = await self.exchange_api.get_balance()
            for pos in open_positions:
                base = pos.symbol.split('/')[0]
                bal = balances.get(base, {}).get('total', 0.0)
                if bal < (pos.quantity * 0.90):
                    logger.warning("Phantom position detected.", symbol=pos.symbol)
                    price = self.latest_prices.get(pos.symbol, pos.entry_price)
                    await self.position_manager.close_position(pos.symbol, price, reason="Startup Reconciliation")
        except Exception:
            pass

    async def run(self):
        self.shared_bot_state['status'] = 'running'
        logger.info("Starting TradingBot...", symbols=self.config.strategy.symbols)
        
        await self.setup()

        # Spawn Critical Services
        self.supervisor.spawn("DataHandler", self.data_handler.run(), critical=True)
        self.supervisor.spawn("Monitoring", self._monitoring_loop(), critical=False)
        self.supervisor.spawn("PositionMonitor", self.position_monitor.run(), critical=True)
        self.supervisor.spawn("Retraining", self._retraining_loop(), critical=False)
        self.supervisor.spawn("Optimizer", self.optimizer.run(), critical=False)
        self.supervisor.spawn("Watchdog", self._watchdog_loop(), critical=False)

        await self.supervisor.monitor()
        logger.info("Supervisor loop ended. Shutting down.")

    async def on_market_data(self, event: MarketDataEvent):
        """
        Event Handler for new market data.
        Delegates processing to the specific symbol's processor queue.
        """
        set_correlation_id()
        self._symbol_heartbeats[event.symbol] = time.time()
        
        processor = self.symbol_processors.get(event.symbol)
        if processor:
            await processor.enqueue_tick()
        else:
            logger.warning(f"Received data for unknown symbol: {event.symbol}")

    async def process_symbol_tick_logic(self, symbol: str):
        """
        The core trading logic for a symbol. Executed sequentially per symbol.
        """
        if self.risk_manager.is_halted: return

        # Fetch latest data (thread-safe copy)
        df = await self.data_handler.get_market_data_safe(symbol)
        if df is None or df.empty: return
        
        last_ts = df.index[-1]
        if self.processed_candles.get(symbol) == last_ts:
            return

        if symbol not in self.latest_prices:
            return

        position = await self.position_manager.get_open_position(symbol)
        try:
            signal = await self.strategy.analyze_market(symbol, df, position)
        except Exception as e:
            logger.error("Strategy analysis failed", symbol=symbol, error=str(e), exc_info=True)
            return

        if signal: 
            await self._handle_signal(signal, df, position)
        
        self.processed_candles[symbol] = last_ts

    async def _watchdog_loop(self):
        logger.info("Watchdog started.")
        while not self.supervisor._shutdown_event.is_set():
            await asyncio.sleep(60)
            now = time.time()
            for symbol in self.config.strategy.symbols:
                last_beat = self._symbol_heartbeats.get(symbol, 0)
                if (now - last_beat) > self._watchdog_threshold:
                    logger.warning("Watchdog Alert: Symbol stalled.", symbol=symbol)
                    if self.alert_system:
                        await self.alert_system.send_alert('warning', f"Watchdog: No data for {symbol} in {int(now-last_beat)}s.")

    async def _monitoring_loop(self):
        reconcile_counter = 0
        while not self.supervisor._shutdown_event.is_set():
            try:
                health = self.health_checker.get_health_status()
                if self.metrics_writer: await self.metrics_writer.write_metric('health', fields=health)
                
                reconcile_counter += 1
                if reconcile_counter >= 2:
                    await self.reconcile_pending_positions()
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

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
            await asyncio.sleep(60)

    async def _liquidate_all_positions(self, reason: str):
        logger.warning("Initiating portfolio liquidation...", reason=reason)
        open_pos = await self.position_manager.get_all_open_positions()
        tasks = [self.trade_executor.close_position(pos, reason) for pos in open_pos]
        await asyncio.gather(*tasks)

    async def _retraining_loop(self):
        logger.info("Starting model retraining loop.")
        await asyncio.sleep(10) 
        while not self.supervisor._shutdown_event.is_set():
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
                await asyncio.sleep(300)

    async def _handle_signal(self, signal: TradeSignal, df: pd.DataFrame, position: Optional[Position]):
        result = await self.trade_executor.execute_trade_signal(signal, df, position)
        if result and self.metrics_writer:
            await self.metrics_writer.write_metric('trade_execution', fields=result.dict())

    async def _close_position(self, position: Position, reason: str):
        await self.trade_executor.close_position(position, reason)

    async def stop(self):
        logger.info("Stopping TradingBot...")
        self.shared_bot_state['status'] = 'stopping'
        
        # Stop symbol processors first
        for processor in self.symbol_processors.values():
            await processor.stop()
            
        await self.supervisor.stop_all()
        await self.data_handler.stop()
        await self.position_monitor.stop()
        await self.optimizer.stop()
        await self.strategy.close()
        if self.exchange_api: await self.exchange_api.close()
        if self.position_manager: self.position_manager.close()
        if self.metrics_writer: await self.metrics_writer.close()
        self.process_executor.shutdown(wait=False)
        logger.info("TradingBot stopped gracefully.")
        self.shared_bot_state['status'] = 'stopped'
