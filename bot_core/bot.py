import asyncio
import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import pandas as pd

from bot_core.logger import get_logger, set_correlation_id
from bot_core.exchange_api import ExchangeAPI
from bot_core.position_manager import PositionManager, Position
from bot_core.risk_manager import RiskManager
from bot_core.strategy import TradingStrategy
from bot_core.config import BotConfig
from bot_core.data_handler import DataHandler
from bot_core.monitoring import HealthChecker, InfluxDBMetrics, AlertSystem
from bot_core.position_monitor import PositionMonitor
from bot_core.trade_executor import TradeExecutor

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
        
        self.running = False
        self.start_time = datetime.now(timezone.utc)
        self.latest_prices = shared_latest_prices
        self.market_details: Dict[str, Dict[str, Any]] = {}
        self.tasks: list[asyncio.Task] = []
        
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

    async def _reconcile_start_state(self):
        """Performs safety checks and state reconciliation on startup."""
        logger.info("Performing startup state reconciliation...")
        
        # 1. Reconcile PENDING positions (Two-Phase Commit Recovery)
        pending_positions = await self.position_manager.get_pending_positions()
        if pending_positions:
            logger.info("Found PENDING positions from previous session. Reconciling...", count=len(pending_positions))
            for pos in pending_positions:
                if not pos.order_id:
                    logger.warning("Pending position has no Order ID. Voiding.", symbol=pos.symbol)
                    await self.position_manager.void_position(pos.symbol, pos.order_id)
                    continue

                try:
                    order = await self.exchange_api.fetch_order(pos.order_id, pos.symbol)
                    if not order:
                        logger.warning("Order not found on exchange. Voiding pending position.", symbol=pos.symbol, order_id=pos.order_id)
                        await self.position_manager.void_position(pos.symbol, pos.order_id)
                        continue

                    status = order.get('status')
                    if status == 'FILLED':
                        # Recover the position
                        filled_qty = order.get('filled', 0.0)
                        avg_price = order.get('average', 0.0)
                        
                        # Fallback SL/TP since we might not have market data yet
                        # Default to 5% risk if we can't calculate it
                        sl_pct = 0.05
                        tp_pct = 0.05
                        if pos.side == 'BUY':
                            sl = avg_price * (1 - sl_pct)
                            tp = avg_price * (1 + tp_pct)
                        else:
                            sl = avg_price * (1 + sl_pct)
                            tp = avg_price * (1 - tp_pct)

                        logger.info("Recovered FILLED order for pending position.", symbol=pos.symbol, order_id=pos.order_id)
                        await self.position_manager.confirm_position_open(pos.symbol, pos.order_id, filled_qty, avg_price, sl, tp)
                    
                    elif status in ['CANCELED', 'REJECTED', 'EXPIRED']:
                        logger.info("Pending position order was terminal but not filled. Voiding.", symbol=pos.symbol, status=status)
                        await self.position_manager.void_position(pos.symbol, pos.order_id)
                    
                    elif status == 'OPEN':
                        logger.info("Pending position order is still OPEN. Cancelling and voiding to ensure clean state.", symbol=pos.symbol)
                        await self.exchange_api.cancel_order(pos.order_id, pos.symbol)
                        await self.position_manager.void_position(pos.symbol, pos.order_id)

                except Exception as e:
                    logger.error("Error reconciling pending position", symbol=pos.symbol, error=str(e))

        # 2. Cancel all open orders to ensure clean slate (remove orphans)
        for symbol in self.config.strategy.symbols:
            try:
                cancelled = await self.exchange_api.cancel_all_orders(symbol)
                if cancelled:
                    logger.info("Cancelled orphaned orders on startup", symbol=symbol, count=len(cancelled))
            except Exception as e:
                logger.error("Failed to cancel orders on startup", symbol=symbol, error=str(e))

        # 3. Verify Balances vs Positions
        try:
            balances = await self.exchange_api.get_balance()
            expected_positions = await self.position_manager.get_aggregated_open_positions()
            
            for symbol, expected_qty in expected_positions.items():
                base_asset = symbol.split('/')[0]
                # Check 'total' because we just cancelled orders, so 'used' should be 0
                actual_qty = balances.get(base_asset, {}).get('total', 0.0)
                
                # Allow for small dust difference (e.g. fees)
                if actual_qty < (expected_qty * 0.98): 
                    logger.critical("CRITICAL: Balance mismatch detected!", 
                                    symbol=symbol, 
                                    expected_db=expected_qty, 
                                    actual_wallet=actual_qty)
                    await self.alert_system.send_alert(
                        level='critical',
                        message=f"🚨 Balance Mismatch for {symbol}",
                        details={'expected_db': expected_qty, 'actual_wallet': actual_qty}
                    )
        except Exception as e:
            logger.error("Error during balance reconciliation", error=str(e))

    async def run(self):
        """Main entry point to start all bot activities."""
        self.running = True
        self.shared_bot_state['status'] = 'running'
        logger.info("Starting TradingBot...", symbols=self.config.strategy.symbols)
        
        await self._load_market_details()
        await self._reconcile_start_state()

        # Start shared loops
        self.tasks.append(asyncio.create_task(self.data_handler.run()))
        self.tasks.append(asyncio.create_task(self._monitoring_loop()))
        self.tasks.append(asyncio.create_task(self.position_monitor.run()))
        self.tasks.append(asyncio.create_task(self._retraining_loop()))

        # Start a trading cycle for each symbol
        for symbol in self.config.strategy.symbols:
            self.tasks.append(asyncio.create_task(self._trading_cycle_for_symbol(symbol)))
        
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def _trading_cycle_for_symbol(self, symbol: str):
        """Runs the trading logic loop for a single symbol to find entry/exit signals."""
        logger.info("Starting trading cycle for symbol", symbol=symbol)
        while self.running:
            set_correlation_id()
            try:
                await self._run_single_trade_check(symbol)
            except asyncio.CancelledError:
                logger.info("Trading loop cancelled for symbol", symbol=symbol)
                break
            except Exception as e:
                logger.critical("Unhandled exception in trading cycle", symbol=symbol, error=str(e), exc_info=True)
            
            await asyncio.sleep(self.config.strategy.interval_seconds)

    async def _monitoring_loop(self):
        """Periodically runs health checks and portfolio monitoring."""
        while self.running:
            try:
                health_status = self.health_checker.get_health_status()
                logger.info("Health Check", **health_status)
                if self.metrics_writer and self.metrics_writer.enabled:
                    await self.metrics_writer.write_metric('health', fields=health_status)
                
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
                            # Run the potentially CPU-intensive training in a separate thread
                            # to avoid blocking the main async event loop.
                            loop = asyncio.get_running_loop()
                            await loop.run_in_executor(
                                None, 
                                self.strategy.retrain,
                                symbol,
                                training_df
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

    async def _run_single_trade_check(self, symbol: str):
        logger.debug("Starting new trade check", symbol=symbol)

        if self.risk_manager.is_halted:
            logger.warning("Trading is halted by RiskManager.")
            return

        df_with_indicators = self.data_handler.get_market_data(symbol)
        if df_with_indicators is None:
            logger.warning("Could not get market data from handler.", symbol=symbol)
            return
        
        if symbol not in self.latest_prices:
            logger.warning("Latest price for symbol not available yet.", symbol=symbol)
            return

        # Await the async DB call
        position = await self.position_manager.get_open_position(symbol)
        signal = await self.strategy.analyze_market(symbol, df_with_indicators, position)

        if signal: await self._handle_signal(signal, df_with_indicators, position)

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

        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        await asyncio.gather(*[task for task in self.tasks if not task.done()], return_exceptions=True)

        if self.exchange_api: await self.exchange_api.close()
        if self.position_manager: self.position_manager.close()
        if self.metrics_writer: await self.metrics_writer.close()
        logger.info("TradingBot stopped gracefully.")
        self.shared_bot_state['status'] = 'stopped'
