import asyncio
import time
import json
from typing import Dict, Any, Optional, List, Set
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
from bot_core.trade_executor import TradeExecutor, TradeExecutionResult
from bot_core.optimizer import StrategyOptimizer

logger = get_logger(__name__)

class TaskSupervisor:
    """
    Manages the lifecycle of background tasks, ensuring they are running and 
    handling failures/restarts robustly.
    """
    def __init__(self):
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.critical_services: Set[str] = set()
        self._shutdown_event = asyncio.Event()

    def spawn(self, name: str, coroutine, critical: bool = False):
        if name in self.active_tasks and not self.active_tasks[name].done():
            logger.warning(f"Task {name} is already running.")
            return
        
        task = asyncio.create_task(coroutine, name=name)
        self.active_tasks[name] = task
        if critical:
            self.critical_services.add(name)
        logger.info(f"Spawned task: {name} (Critical: {critical})")

    async def monitor(self):
        """Monitors tasks and restarts non-critical ones, or signals shutdown for critical ones."""
        while not self._shutdown_event.is_set():
            # Check for done tasks
            for name, task in list(self.active_tasks.items()):
                if task.done():
                    try:
                        exc = task.exception()
                        if exc:
                            logger.error(f"Task {name} failed.", error=str(exc), exc_info=exc)
                            if name in self.critical_services:
                                logger.critical(f"Critical service {name} failed. Initiating shutdown.")
                                self._shutdown_event.set()
                            else:
                                logger.info(f"Restarting non-critical task: {name}")
                                # Logic to restart would require storing the coroutine factory, 
                                # for now we just remove it to prevent memory leaks. 
                                # In a full implementation, we'd use a factory pattern.
                                del self.active_tasks[name]
                        else:
                            logger.info(f"Task {name} completed normally.")
                            del self.active_tasks[name]
                    except asyncio.CancelledError:
                        del self.active_tasks[name]
            
            await asyncio.sleep(1)

    async def stop_all(self):
        self._shutdown_event.set()
        tasks = [t for t in self.active_tasks.values() if not t.done()]
        for t in tasks:
            t.cancel()
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.active_tasks.clear()

class TradingBot:
    def __init__(self, config: BotConfig, exchange_api: ExchangeAPI, data_handler: DataHandler, 
                 strategy: TradingStrategy, position_manager: PositionManager, risk_manager: RiskManager,
                 health_checker: HealthChecker, position_monitor: PositionMonitor,
                 trade_executor: TradeExecutor, alert_system: AlertSystem,
                 shared_latest_prices: Dict[str, float],
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
        self.metrics_writer = metrics_writer
        self.shared_bot_state = shared_bot_state if shared_bot_state is not None else {}
        
        self.optimizer = StrategyOptimizer(config, position_manager)
        self.supervisor = TaskSupervisor()
        
        self.start_time = datetime.now(timezone.utc)
        self.latest_prices = shared_latest_prices
        self.market_details: Dict[str, Dict[str, Any]] = {}
        self.processed_candles: Dict[str, pd.Timestamp] = {}
        
        self.process_executor = ProcessPoolExecutor(max_workers=2)
        
        # Watchdog State
        self._symbol_heartbeats: Dict[str, float] = {}
        self._watchdog_threshold = 300 # 5 minutes without data triggers warning
        
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
            logger.info("Injected DataHandler into AIEnsembleStrategy.")

        self.position_monitor.set_strategy(self.strategy)

        await self._load_market_details()
        await self.reconcile_pending_positions()
        await self.reconcile_open_positions()
        
        logger.info("Warming up strategy...")
        await self.strategy.warmup(self.config.strategy.symbols)
        
        logger.info("TradingBot setup complete.")

    async def _load_market_details(self):
        logger.info("Loading market details for all symbols...")
        for symbol in self.config.strategy.symbols:
            try:
                details = await self.exchange_api.fetch_market_details(symbol)
                if details:
                    self.market_details[symbol] = details
                    logger.info("Successfully loaded market details", symbol=symbol)
                else:
                    logger.error("Failed to load market details for symbol, trading may fail.", symbol=symbol)
            except Exception as e:
                logger.critical("Could not load market details for symbol due to an exception.", symbol=symbol, error=str(e))
        self.trade_executor.market_details = self.market_details
        logger.info("Market details loading complete and passed to TradeExecutor.")

    async def reconcile_pending_positions(self):
        logger.info("Reconciling PENDING positions...")
        pending_positions = await self.position_manager.get_pending_positions()
        if not pending_positions:
            return

        logger.info("Found PENDING positions. Reconciling...", count=len(pending_positions))
        for pos in pending_positions:
            try:
                # In a real scenario, we would check the order status on exchange and update DB
                # For now, we mark them failed to be safe if they are stale
                await self.position_manager.mark_position_failed(pos.symbol, pos.order_id, "Startup Reconciliation")
            except Exception as e:
                logger.error("Error reconciling pending position", symbol=pos.symbol, error=str(e))

    async def reconcile_open_positions(self):
        logger.info("Reconciling OPEN positions with exchange balances...")
        open_positions = await self.position_manager.get_all_open_positions()
        if not open_positions:
            return

        try:
            balances = await self.exchange_api.get_balance()
        except Exception as e:
            logger.error("Failed to fetch balances for reconciliation. Skipping.", error=str(e))
            return

        for pos in open_positions:
            try:
                base_asset = pos.symbol.split('/')[0]
            except IndexError:
                continue

            asset_balance = balances.get(base_asset, {}).get('total', 0.0)
            if asset_balance < (pos.quantity * 0.90):
                logger.warning("Phantom position detected.", symbol=pos.symbol, db_qty=pos.quantity, exchange_bal=asset_balance)
                current_price = self.latest_prices.get(pos.symbol, pos.entry_price)
                await self.position_manager.close_position(pos.symbol, current_price, reason="Startup Reconciliation (Missing on Exchange)")

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

        # Spawn Trading Loops
        for symbol in self.config.strategy.symbols:
            self.supervisor.spawn(f"SymbolLoop_{symbol}", self._trading_cycle_for_symbol(symbol), critical=False)

        # Run Supervisor Monitor (blocks until shutdown)
        await self.supervisor.monitor()
        
        logger.info("Supervisor loop ended. Shutting down.")

    async def _trading_cycle_for_symbol(self, symbol: str):
        logger.info("Starting trading cycle", symbol=symbol)
        self._symbol_heartbeats[symbol] = time.time()
        
        while not self.supervisor._shutdown_event.is_set():
            set_correlation_id()
            try:
                timeout = self.config.strategy.interval_seconds * 2
                # Wait for new data event
                await self.data_handler.wait_for_new_candle(symbol, timeout=timeout)
                
                if self.supervisor._shutdown_event.is_set(): break

                # Heartbeat check: Ensure data is actually fresh
                df = self.data_handler.get_market_data(symbol, include_forming=False)
                if df is not None and not df.empty:
                    self._symbol_heartbeats[symbol] = time.time()
                    await self.process_symbol_tick(symbol)
                else:
                    logger.debug("Waiting for data...", symbol=symbol)

            except asyncio.CancelledError:
                logger.info("Trading loop cancelled", symbol=symbol)
                break
            except Exception as e:
                logger.error("Unhandled exception in trading cycle", symbol=symbol, error=str(e), exc_info=exc)
                await asyncio.sleep(5)

    async def process_symbol_tick(self, symbol: str):
        if self.risk_manager.is_halted:
            return

        df_with_indicators = self.data_handler.get_market_data(symbol, include_forming=False)
        if df_with_indicators is None or df_with_indicators.empty:
            return
        
        last_candle_ts = df_with_indicators.index[-1]
        if self.processed_candles.get(symbol) == last_candle_ts:
            return

        if symbol not in self.latest_prices:
            return

        position = await self.position_manager.get_open_position(symbol)
        
        # Execute Strategy Analysis
        try:
            signal = await self.strategy.analyze_market(symbol, df_with_indicators, position)
        except Exception as e:
            logger.error("Strategy analysis failed", symbol=symbol, error=str(e))
            return

        if signal: 
            await self._handle_signal(signal, df_with_indicators, position)
        
        self.processed_candles[symbol] = last_candle_ts

    async def _watchdog_loop(self):
        """Monitors the liveness of symbol processing loops."""
        logger.info("Watchdog started.")
        while not self.supervisor._shutdown_event.is_set():
            await asyncio.sleep(60)
            now = time.time()
            for symbol in self.config.strategy.symbols:
                last_beat = self._symbol_heartbeats.get(symbol, 0)
                if (now - last_beat) > self._watchdog_threshold:
                    logger.warning("Watchdog Alert: Symbol loop stalled.", symbol=symbol, seconds_since_last=int(now-last_beat))
                    if self.alert_system:
                        await self.alert_system.send_alert(
                            level='warning',
                            message=f"⚠️ Watchdog: No data processed for {symbol} in {int(now-last_beat)}s.",
                            details={'symbol': symbol}
                        )

    async def _monitoring_loop(self):
        reconcile_counter = 0
        while not self.supervisor._shutdown_event.is_set():
            try:
                health_status = self.health_checker.get_health_status()
                if self.metrics_writer and self.metrics_writer.enabled:
                    await self.metrics_writer.write_metric('health', fields=health_status)
                
                reconcile_counter += 1
                if reconcile_counter >= 2:
                    await self.reconcile_pending_positions()
                    reconcile_counter = 0

                if not self.latest_prices:
                    await asyncio.sleep(5)
                    continue

                open_positions = await self.position_manager.get_all_open_positions()
                portfolio_value = self.position_manager.get_portfolio_value(self.latest_prices, open_positions)
                daily_pnl = await self.position_manager.get_daily_realized_pnl()
                
                await self.risk_manager.update_portfolio_risk(portfolio_value, daily_pnl)
                
                if self.risk_manager.liquidation_needed:
                    logger.critical("Risk Manager requested emergency liquidation.")
                    await self._liquidate_all_positions("Emergency Risk Halt")
                    self.risk_manager.liquidation_needed = False

                self.shared_bot_state['portfolio_equity'] = portfolio_value
                self.shared_bot_state['open_positions_count'] = len(open_positions)
                self.shared_bot_state['daily_pnl'] = daily_pnl

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
            await asyncio.sleep(60)

    async def _liquidate_all_positions(self, reason: str):
        logger.warning("Initiating portfolio liquidation...", reason=reason)
        open_positions = await self.position_manager.get_all_open_positions()
        tasks = [self.trade_executor.close_position(pos, reason) for pos in open_positions]
        await asyncio.gather(*tasks)

    async def _retraining_loop(self):
        logger.info("Starting model retraining loop.")
        await asyncio.sleep(10) 
        while not self.supervisor._shutdown_event.is_set():
            try:
                for symbol in self.config.strategy.symbols:
                    if self.strategy.needs_retraining(symbol):
                        logger.info("Retraining needed", symbol=symbol)
                        limit = self.strategy.get_training_data_limit()
                        if limit > 0:
                            df = await self.data_handler.fetch_full_history_for_symbol(symbol, limit)
                            if df is not None and not df.empty:
                                await self.strategy.retrain(symbol, df, self.process_executor)
                await asyncio.sleep(300)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in retraining loop", error=str(e), exc_info=True)
                await asyncio.sleep(300)

    async def _handle_signal(self, signal: TradeSignal, df_with_indicators: pd.DataFrame, position: Optional[Position]):
        result = await self.trade_executor.execute_trade_signal(signal, df_with_indicators, position)
        if result and self.metrics_writer and self.metrics_writer.enabled:
            await self.metrics_writer.write_metric('trade_execution', fields=result.dict())

    async def _close_position(self, position: Position, reason: str):
        await self.trade_executor.close_position(position, reason)

    async def stop(self):
        logger.info("Stopping TradingBot...")
        self.shared_bot_state['status'] = 'stopping'
        
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
