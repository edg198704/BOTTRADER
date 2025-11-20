import asyncio
import time
from typing import Dict, Any, Optional, Set, List
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

class ServiceManager:
    """
    Manages the lifecycle of background services with robust monitoring, 
    restart capabilities, and dependency handling.
    """
    def __init__(self):
        self.services: Dict[str, asyncio.Task] = {}
        self.critical_services: Set[str] = set()
        self._shutdown_event = asyncio.Event()

    def register(self, name: str, coroutine, critical: bool = False):
        """Registers a service task to be managed."""
        if name in self.services and not self.services[name].done():
            logger.warning(f"Service {name} is already running.")
            return
        
        async def wrapped_service():
            try:
                logger.info(f"Service {name} started.")
                await coroutine
                logger.info(f"Service {name} stopped gracefully.")
            except asyncio.CancelledError:
                logger.info(f"Service {name} cancelled.")
            except Exception as e:
                logger.error(f"Service {name} crashed.", error=str(e), exc_info=True)
                raise

        task = asyncio.create_task(wrapped_service(), name=name)
        self.services[name] = task
        if critical:
            self.critical_services.add(name)

    async def monitor(self):
        """Monitors services and handles failures."""
        while not self._shutdown_event.is_set():
            for name, task in list(self.services.items()):
                if task.done():
                    try:
                        exc = task.exception()
                        if exc:
                            logger.error(f"Service {name} failed with exception.", error=str(exc))
                            if name in self.critical_services:
                                logger.critical(f"Critical service {name} failed. Initiating system shutdown.")
                                self._shutdown_event.set()
                            else:
                                logger.warning(f"Non-critical service {name} stopped. It may need manual restart logic here if desired.")
                        else:
                            # Normal exit
                            pass
                    except asyncio.CancelledError:
                        pass
                    
                    # Remove done task
                    if name in self.services:
                        del self.services[name]
            
            await asyncio.sleep(1)

    async def stop_all(self):
        """Stops all registered services."""
        self._shutdown_event.set()
        logger.info("Stopping all services...")
        
        # Cancel all running tasks
        tasks_to_cancel = [t for t in self.services.values() if not t.done()]
        for t in tasks_to_cancel:
            t.cancel()
        
        if tasks_to_cancel:
            # Wait for them to finish cancelling
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        
        self.services.clear()
        logger.info("All services stopped.")

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
        self.service_manager = ServiceManager()
        
        self.start_time = datetime.now(timezone.utc)
        self.latest_prices = shared_latest_prices
        self.market_details: Dict[str, Dict[str, Any]] = {}
        self.processed_candles: Dict[str, pd.Timestamp] = {}
        
        # Concurrency Control
        self._symbol_locks: Dict[str, asyncio.Lock] = {}
        
        # Use ProcessPoolExecutor for CPU-bound AI training
        self.process_executor = ProcessPoolExecutor(max_workers=2)
        self._symbol_heartbeats: Dict[str, float] = {}
        self._watchdog_threshold = 300
        
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
        await self.reconcile_pending_positions()
        await self.reconcile_open_positions()
        
        logger.info("Warming up strategy...")
        await self.strategy.warmup(self.config.strategy.symbols)
        
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

        # Register Services
        self.service_manager.register("DataHandler", self.data_handler.run(), critical=True)
        self.service_manager.register("Monitoring", self._monitoring_loop(), critical=False)
        self.service_manager.register("PositionMonitor", self.position_monitor.run(), critical=True)
        self.service_manager.register("Retraining", self._retraining_loop(), critical=False)
        self.service_manager.register("Optimizer", self.optimizer.run(), critical=False)
        self.service_manager.register("Watchdog", self._watchdog_loop(), critical=False)

        # The main loop just monitors the service manager
        await self.service_manager.monitor()
        logger.info("Service Manager loop ended. Shutting down.")

    async def on_market_data(self, event: MarketDataEvent):
        """Event Handler for new market data."""
        set_correlation_id()
        self._symbol_heartbeats[event.symbol] = time.time()
        
        # Fire and forget processing to avoid blocking the event bus
        asyncio.create_task(self.process_symbol_tick(event.symbol))

    async def process_symbol_tick(self, symbol: str):
        """
        Processes a new tick for a symbol. 
        Uses a per-symbol lock to ensure sequential processing of candles.
        """
        if self.risk_manager.is_halted: return

        lock = self._get_symbol_lock(symbol)
        if lock.locked():
            logger.warning(f"Skipping tick for {symbol} - previous tick still processing.")
            return

        async with lock:
            tick_start = time.perf_counter()
            try:
                # Backpressure check: If we are processing too slowly, skip old data
                last_processed = self.processed_candles.get(symbol)
                
                # Fetch data without forming candle to ensure we act on closed bars
                df = self.data_handler.get_market_data(symbol, include_forming=False)
                if df is None or df.empty: return
                
                last_ts = df.index[-1]
                if last_processed == last_ts:
                    return

                if symbol not in self.latest_prices:
                    return

                # Fast cached read
                position = await self.position_manager.get_open_position(symbol)
                
                # Execute Strategy Analysis
                try:
                    # Add timeout to strategy analysis to prevent hangs
                    signal = await asyncio.wait_for(self.strategy.analyze_market(symbol, df, position), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.error("Strategy analysis timed out", symbol=symbol)
                    return
                except Exception as e:
                    logger.error("Strategy analysis failed", symbol=symbol, error=str(e), exc_info=True)
                    return

                if signal: 
                    await self._handle_signal(signal, df, position)
                
                self.processed_candles[symbol] = last_ts
                
                # Latency Metrics
                duration_ms = (time.perf_counter() - tick_start) * 1000
                if self.metrics_writer:
                    await self.metrics_writer.write_metric('tick_latency', fields={'duration_ms': duration_ms}, tags={'symbol': symbol})
                
            except Exception as e:
                logger.error(f"Critical error processing tick for {symbol}", error=str(e), exc_info=True)

    async def _watchdog_loop(self):
        logger.info("Watchdog started.")
        while not self.service_manager._shutdown_event.is_set():
            await asyncio.sleep(60)
            now = time.time()
            for symbol in self.config.strategy.symbols:
                last_beat = self._symbol_heartbeats.get(symbol, 0)
                if last_beat > 0 and (now - last_beat) > self._watchdog_threshold:
                    logger.warning("Watchdog Alert: Symbol stalled.", symbol=symbol)
                    if self.alert_system:
                        await self.alert_system.send_alert('warning', f"Watchdog: No data for {symbol} in {int(now-last_beat)}s.")

    async def _monitoring_loop(self):
        reconcile_counter = 0
        while not self.service_manager._shutdown_event.is_set():
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
        await self.service_manager.stop_all()
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
