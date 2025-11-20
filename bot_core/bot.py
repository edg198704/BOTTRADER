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
from bot_core.monitoring import HealthChecker, InfluxDBMetrics, AlertSystem
from bot_core.position_monitor import PositionMonitor
from bot_core.trade_executor import TradeExecutor
from bot_core.optimizer import StrategyOptimizer

logger = get_logger(__name__)

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
        
        # Initialize Optimizer
        self.optimizer = StrategyOptimizer(config, position_manager)
        
        self.running = False
        self.start_time = datetime.now(timezone.utc)
        self.latest_prices = shared_latest_prices
        self.market_details: Dict[str, Dict[str, Any]] = {}
        self.tasks: list[asyncio.Task] = []
        
        # Track processed candles to avoid redundant analysis on the same closed candle
        self.processed_candles: Dict[str, pd.Timestamp] = {}
        
        # Executor for CPU-intensive tasks like AI training
        self.process_executor = ProcessPoolExecutor(max_workers=2)
        
        self._initialize_shared_state()
        logger.info("TradingBot orchestrator initialized.")

    def _initialize_shared_state(self):
        """Initializes the shared state dictionary for external components like Telegram."""
        self.shared_bot_state['status'] = 'initializing'
        self.shared_bot_state['start_time'] = self.start_time
        self.shared_bot_state['position_manager'] = self.position_manager
        self.shared_bot_state['risk_manager'] = self.risk_manager
        self.shared_bot_state['latest_prices'] = self.latest_prices
        self.shared_bot_state['config'] = self.config
        self.shared_bot_state['strategy'] = self.strategy

    async def setup(self):
        """Performs initial setup: loads market details and reconciles state. Separated for backtesting."""
        logger.info("Setting up TradingBot components...")
        
        # Inject DataHandler into Strategy if it's an AI strategy needing external data
        if isinstance(self.strategy, AIEnsembleStrategy):
            self.strategy.data_fetcher = self.data_handler
            logger.info("Injected DataHandler into AIEnsembleStrategy.")

        await self._load_market_details()
        await self.reconcile_pending_positions()
        
        # Warmup strategy (preload models, etc.)
        logger.info("Warming up strategy...")
        await self.strategy.warmup(self.config.strategy.symbols)
        
        logger.info("TradingBot setup complete.")

    async def _load_market_details(self):
        """Fetches and caches exchange trading rules for all configured symbols."""
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
        # Pass the loaded market details to the trade executor
        self.trade_executor.market_details = self.market_details
        logger.info("Market details loading complete and passed to TradeExecutor.")

    async def reconcile_pending_positions(self):
        """Performs safety checks and state reconciliation for PENDING positions."""
        logger.info("Reconciling PENDING positions...")
        
        # 1. Reconcile PENDING positions (Two-Phase Commit Recovery)
        pending_positions = await self.position_manager.get_pending_positions()
        if not pending_positions:
            return

        logger.info("Found PENDING positions. Reconciling...", count=len(pending_positions))
        for pos in pending_positions:
            if not pos.order_id:
                logger.warning("Pending position has no Order ID. Voiding.", symbol=pos.symbol)
                await self.position_manager.void_position(pos.symbol, pos.order_id)
                continue

            try:
                # Try to fetch by ID first
                try:
                    order = await self.exchange_api.fetch_order(pos.order_id, pos.symbol)
                except Exception:
                    order = None
                
                # If not found, it might be a Client Order ID that the exchange doesn't index directly.
                # Scan open orders to find a match.
                if not order:
                    logger.info("Order not found by ID, scanning open orders for Client ID match...", symbol=pos.symbol, id=pos.order_id)
                    open_orders = await self.exchange_api.fetch_open_orders(pos.symbol)
                    for o in open_orders:
                        # Check standard CCXT field 'clientOrderId'
                        if o.get('clientOrderId') == pos.order_id:
                            order = o
                            logger.info("Found match in open orders via Client ID.", symbol=pos.symbol, exchange_id=o['id'])
                            # Update DB with real ID
                            await self.position_manager.update_pending_order_id(pos.symbol, pos.order_id, o['id'])
                            pos.order_id = o['id'] # Update local obj for subsequent logic
                            break

                if not order:
                    logger.warning("Order not found on exchange (ID or ClientID). Voiding pending position.", symbol=pos.symbol, order_id=pos.order_id)
                    await self.position_manager.void_position(pos.symbol, pos.order_id)
                    continue

                status = order.get('status')
                
                # Helper to confirm position from order data
                async def confirm_from_order(order_data):
                    filled_qty = order_data.get('filled', 0.0)
                    avg_price = order_data.get('average', 0.0)
                    # Fallback SL/TP since we might not have market data yet
                    sl_pct = 0.05
                    tp_pct = 0.05
                    if pos.side == 'BUY':
                        sl = avg_price * (1 - sl_pct)
                        tp = avg_price * (1 + tp_pct)
                    else:
                        sl = avg_price * (1 + sl_pct)
                        tp = avg_price * (1 - tp_pct)
                    
                    logger.info("Recovered position from order.", symbol=pos.symbol, order_id=pos.order_id, filled=filled_qty)
                    await self.position_manager.confirm_position_open(pos.symbol, pos.order_id, filled_qty, avg_price, sl, tp)

                if status == 'FILLED':
                    await confirm_from_order(order)
                
                elif status in ['CANCELED', 'REJECTED', 'EXPIRED']:
                    # Check for partial fills even in terminal states
                    filled = order.get('filled', 0.0)
                    if filled > 0:
                        logger.info("Pending position was terminal but partially filled. Recovering.", symbol=pos.symbol, status=status)
                        await confirm_from_order(order)
                    else:
                        # ZOMBIE CHECK: Before voiding, check if there are any OTHER orders (Open OR Closed)
                        # that belong to this trade_id (e.g. chase orders placed before crash)
                        zombie_recovered = False
                        if pos.trade_id:
                            logger.info("Checking for zombie chase orders (Open)...", symbol=pos.symbol, trade_id=pos.trade_id)
                            # 1. Check Open Orders
                            open_orders = await self.exchange_api.fetch_open_orders(pos.symbol)
                            for o in open_orders:
                                client_oid = o.get('clientOrderId', '')
                                if client_oid and client_oid.startswith(pos.trade_id):
                                    logger.info("Found zombie chase order (OPEN)! Recovering.", symbol=pos.symbol, new_order_id=o['id'])
                                    await self.position_manager.update_pending_order_id(pos.symbol, pos.order_id, o['id'])
                                    zombie_recovered = True
                                    # We found it, now we need to handle it (it's OPEN, so we likely want to cancel it or let it run)
                                    # For safety, we update the ID and let the next loop/logic handle it, or handle it here.
                                    # Since we are in the 'CANCELED' block of the *original* order, we must handle the new one.
                                    # We'll cancel it to be safe and confirm any partials.
                                    cancelled_order = await self.exchange_api.cancel_order(o['id'], pos.symbol)
                                    if cancelled_order and cancelled_order.get('filled', 0.0) > 0:
                                        await confirm_from_order(cancelled_order)
                                    else:
                                        await self.position_manager.void_position(pos.symbol, o['id'])
                                    break
                            
                            # 2. Check Recent History (Deep Scan) if not found in Open
                            if not zombie_recovered:
                                logger.info("Deep scan for zombie chase orders (History)...", symbol=pos.symbol, trade_id=pos.trade_id)
                                try:
                                    recent_orders = await self.exchange_api.fetch_recent_orders(pos.symbol, limit=20)
                                    for o in recent_orders:
                                        client_oid = o.get('clientOrderId', '')
                                        if client_oid and client_oid.startswith(pos.trade_id):
                                            # We found a related order in history.
                                            status_hist = o.get('status')
                                            if status_hist in ['FILLED', 'PARTIALLY_FILLED']:
                                                logger.info("Found zombie chase order (FILLED) in history! Recovering.", symbol=pos.symbol, new_order_id=o['id'])
                                                await self.position_manager.update_pending_order_id(pos.symbol, pos.order_id, o['id'])
                                                await confirm_from_order(o)
                                                zombie_recovered = True
                                                break
                                            elif status_hist == 'OPEN':
                                                # Should have been caught by fetch_open_orders, but just in case
                                                logger.info("Found zombie chase order (OPEN) in history! Recovering.", symbol=pos.symbol, new_order_id=o['id'])
                                                await self.position_manager.update_pending_order_id(pos.symbol, pos.order_id, o['id'])
                                                cancelled_order = await self.exchange_api.cancel_order(o['id'], pos.symbol)
                                                if cancelled_order and cancelled_order.get('filled', 0.0) > 0:
                                                    await confirm_from_order(cancelled_order)
                                                else:
                                                    await self.position_manager.void_position(pos.symbol, o['id'])
                                                zombie_recovered = True
                                                break
                                except Exception as e:
                                    logger.error("Failed deep zombie scan", error=str(e))
                        
                        if not zombie_recovered:
                            logger.info("Pending position order was terminal and empty. Voiding.", symbol=pos.symbol, status=status)
                            await self.position_manager.void_position(pos.symbol, pos.order_id)
                
                elif status == 'OPEN':
                    logger.info("Pending position order is still OPEN. Cancelling...", symbol=pos.symbol)
                    # Cancel returns the updated order object (or fetches it)
                    cancelled_order = await self.exchange_api.cancel_order(pos.order_id, pos.symbol)
                    
                    if cancelled_order and cancelled_order.get('filled', 0.0) > 0:
                            logger.info("Cancelled order had partial fills. Confirming position.", symbol=pos.symbol)
                            await confirm_from_order(cancelled_order)
                    else:
                            logger.info("Cancelled order was empty. Voiding position.", symbol=pos.symbol)
                            await self.position_manager.void_position(pos.symbol, pos.order_id)

            except Exception as e:
                logger.error("Error reconciling pending position", symbol=pos.symbol, error=str(e))

        # 2. Cancel all open orders to ensure clean slate (remove orphans)
        # Note: We only do this on startup or explicit reconciliation, not every loop.
        # But for safety, we skip this in the periodic loop if not explicitly requested.
        # For now, we assume this method is safe to run periodically as it only targets PENDING positions.
        # Orphan cleanup is separate.

    async def run(self):
        """Main entry point to start all bot activities."""
        self.running = True
        self.shared_bot_state['status'] = 'running'
        logger.info("Starting TradingBot...", symbols=self.config.strategy.symbols)
        
        await self.setup()

        # Start shared loops
        self.tasks.append(asyncio.create_task(self.data_handler.run()))
        self.tasks.append(asyncio.create_task(self._monitoring_loop()))
        self.tasks.append(asyncio.create_task(self.position_monitor.run()))
        self.tasks.append(asyncio.create_task(self._retraining_loop()))
        self.tasks.append(asyncio.create_task(self.optimizer.run()))

        # Start a trading cycle for each symbol
        for symbol in self.config.strategy.symbols:
            self.tasks.append(asyncio.create_task(self._trading_cycle_for_symbol(symbol)))
        
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def _trading_cycle_for_symbol(self, symbol: str):
        """Runs the trading logic loop for a single symbol to find entry/exit signals."""
        logger.info("Starting trading cycle for symbol", symbol=symbol)
        while self.running:
            set_correlation_id()
            
            # Wait for new data event instead of sleeping
            # We use a timeout slightly larger than the interval to ensure we don't hang forever if data stops
            timeout = self.config.strategy.interval_seconds * 2
            await self.data_handler.wait_for_new_candle(symbol, timeout=timeout)
            
            if not self.running: break

            try:
                await self.process_symbol_tick(symbol)
            except asyncio.CancelledError:
                logger.info("Trading loop cancelled for symbol", symbol=symbol)
                break
            except Exception as e:
                logger.critical("Unhandled exception in trading cycle", symbol=symbol, error=str(e), exc_info=True)
            
            # No sleep needed here, wait_for_new_candle handles the timing

    async def process_symbol_tick(self, symbol: str):
        """Executes a single trading logic iteration for a symbol. Public for backtesting."""
        if self.risk_manager.is_halted:
            logger.warning("Trading is halted by RiskManager.")
            return

        # Request only CLOSED candles to prevent repainting
        df_with_indicators = self.data_handler.get_market_data(symbol, include_forming=False)
        if df_with_indicators is None or df_with_indicators.empty:
            logger.warning("Could not get market data from handler.", symbol=symbol)
            return
        
        # Check if we have already processed this specific candle
        last_candle_ts = df_with_indicators.index[-1]
        if self.processed_candles.get(symbol) == last_candle_ts:
            # We have already analyzed this closed candle. Wait for the next one.
            return

        if symbol not in self.latest_prices:
            logger.warning("Latest price for symbol not available yet.", symbol=symbol)
            return

        # Await the async DB call
        position = await self.position_manager.get_open_position(symbol)
        signal = await self.strategy.analyze_market(symbol, df_with_indicators, position)

        if signal: 
            await self._handle_signal(signal, df_with_indicators, position)
        
        # Mark this candle as processed so we don't re-analyze it in the next tick
        self.processed_candles[symbol] = last_candle_ts

    async def _monitoring_loop(self):
        """Periodically runs health checks, portfolio monitoring, and state reconciliation."""
        reconcile_counter = 0
        while self.running:
            try:
                health_status = self.health_checker.get_health_status()
                logger.info("Health Check", **health_status)
                if self.metrics_writer and self.metrics_writer.enabled:
                    await self.metrics_writer.write_metric('health', fields=health_status)
                
                # --- Periodic Reconciliation (Every 2 minutes) ---
                reconcile_counter += 1
                if reconcile_counter >= 2:
                    await self.reconcile_pending_positions()
                    reconcile_counter = 0

                if not self.latest_prices:
                    await asyncio.sleep(5)
                    continue

                # Await the async DB call
                open_positions = await self.position_manager.get_all_open_positions()
                portfolio_value = self.position_manager.get_portfolio_value(self.latest_prices, open_positions)
                
                daily_pnl = await self.position_manager.get_daily_realized_pnl()
                
                # Update Risk Manager and check for emergency liquidation
                await self.risk_manager.update_portfolio_risk(portfolio_value, daily_pnl)
                
                if self.risk_manager.liquidation_needed:
                    logger.critical("Risk Manager requested emergency liquidation. Closing all positions.")
                    await self._liquidate_all_positions("Emergency Risk Halt")
                    self.risk_manager.liquidation_needed = False # Reset flag after action

                # Update shared state for Telegram bot
                self.shared_bot_state['portfolio_equity'] = portfolio_value
                self.shared_bot_state['open_positions_count'] = len(open_positions)
                self.shared_bot_state['daily_pnl'] = daily_pnl

                if self.metrics_writer and self.metrics_writer.enabled:
                    await self.metrics_writer.write_metric('portfolio', fields={'equity': portfolio_value, 'daily_pnl': daily_pnl})

            except asyncio.CancelledError:
                logger.info("Monitoring loop cancelled.")
                break
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
            await asyncio.sleep(60) # Monitoring interval

    async def _liquidate_all_positions(self, reason: str):
        """Closes all open positions immediately."""
        logger.warning("Initiating portfolio liquidation...", reason=reason)
        open_positions = await self.position_manager.get_all_open_positions()
        
        if not open_positions:
            logger.info("No positions to liquidate.")
            return

        tasks = []
        for pos in open_positions:
            logger.info("Liquidating position", symbol=pos.symbol, qty=pos.quantity)
            tasks.append(self.trade_executor.close_position(pos, reason))
        
        await asyncio.gather(*tasks)
        logger.info("Portfolio liquidation complete.")

    async def _retraining_loop(self):
        """Periodically checks if the strategy models need retraining and triggers it."""
        logger.info("Starting model retraining loop.")
        # Wait a bit on startup to let data load
        await asyncio.sleep(60)

        while self.running:
            try:
                for symbol in self.config.strategy.symbols:
                    if self.strategy.needs_retraining(symbol):
                        logger.info("Retraining needed for symbol, initiating process.", symbol=symbol)
                        
                        training_data_limit = self.strategy.get_training_data_limit()
                        if training_data_limit <= 0:
                            logger.info("Strategy does not require training data, skipping.", symbol=symbol, strategy=self.strategy.__class__.__name__)
                            continue

                        training_df = await self.data_handler.fetch_full_history_for_symbol(
                            symbol, training_data_limit
                        )

                        if training_df is not None and not training_df.empty:
                            # Run the potentially CPU-intensive training in a separate process
                            # We pass the executor to the strategy so it can manage the process submission
                            await self.strategy.retrain(
                                symbol,
                                training_df,
                                self.process_executor
                            )
                        else:
                            logger.error("Could not fetch training data, skipping retraining for now.", symbol=symbol)
                
                # Check again in an hour
                await asyncio.sleep(3600)

            except asyncio.CancelledError:
                logger.info("Retraining loop cancelled.")
                break
            except Exception as e:
                logger.error("Error in retraining loop", error=str(e), exc_info=True)
                await asyncio.sleep(3600) # Wait before retrying on error

    async def _handle_signal(self, signal: Dict, df_with_indicators: pd.DataFrame, position: Optional[Position]):
        """Delegates signal handling to the TradeExecutor."""
        await self.trade_executor.execute_trade_signal(signal, df_with_indicators, position)

    async def _close_position(self, position: Position, reason: str):
        """Delegates position closing to the TradeExecutor."""
        await self.trade_executor.close_position(position, reason)

    async def stop(self):
        logger.info("Stopping TradingBot...")
        self.running = False
        self.shared_bot_state['status'] = 'stopping'
        
        await self.data_handler.stop()
        await self.position_monitor.stop()
        await self.optimizer.stop()
        
        # Clean up strategy resources
        await self.strategy.close()

        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        await asyncio.gather(*[task for task in self.tasks if not task.done()], return_exceptions=True)

        if self.exchange_api: await self.exchange_api.close()
        if self.position_manager: self.position_manager.close()
        if self.metrics_writer: await self.metrics_writer.close()
        
        # Shutdown the process executor
        self.process_executor.shutdown(wait=False)
        
        logger.info("TradingBot stopped gracefully.")
        self.shared_bot_state['status'] = 'stopped'
