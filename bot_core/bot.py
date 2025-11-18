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
from bot_core.monitoring import HealthChecker, InfluxDBMetrics
from bot_core.order_sizer import OrderSizer

logger = get_logger(__name__)

class TradingBot:
    def __init__(self, config: BotConfig, exchange_api: ExchangeAPI, data_handler: Optional[DataHandler], 
                 strategy: TradingStrategy, position_manager: PositionManager, risk_manager: RiskManager,
                 order_sizer: OrderSizer, health_checker: HealthChecker, 
                 metrics_writer: Optional[InfluxDBMetrics] = None,
                 shared_bot_state: Optional[Dict[str, Any]] = None):
        self.config = config
        self.exchange_api = exchange_api
        self.data_handler = data_handler
        self.strategy = strategy
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.order_sizer = order_sizer
        self.health_checker = health_checker
        self.metrics_writer = metrics_writer
        self.shared_bot_state = shared_bot_state if shared_bot_state is not None else {}
        
        self.running = False
        self.start_time = datetime.now(timezone.utc)
        self.latest_prices: Dict[str, float] = {}
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
        logger.info("Market details loading complete.")

    async def run(self):
        """Main entry point to start all bot activities."""
        self.running = True
        self.shared_bot_state['status'] = 'running'
        logger.info("Starting TradingBot...", symbols=self.config.strategy.symbols)
        
        await self._load_market_details()

        # Start shared loops
        self.tasks.append(asyncio.create_task(self.data_handler.run()))
        self.tasks.append(asyncio.create_task(self._monitoring_loop()))
        self.tasks.append(asyncio.create_task(self._position_management_loop()))

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

    async def _position_management_loop(self):
        """Monitors open positions for stop-loss, take-profit, and trailing stop updates."""
        logger.info("Starting position management loop.")
        while self.running:
            try:
                open_positions = self.position_manager.get_all_open_positions()
                for pos in open_positions:
                    current_price = self.latest_prices.get(pos.symbol)
                    if not current_price:
                        continue

                    # --- Trailing Stop Logic ---
                    if self.config.risk_management.use_trailing_stop:
                        # PositionManager handles the logic and DB updates.
                        # The returned 'pos' object has the latest state.
                        pos = self.position_manager.manage_trailing_stop(
                            pos, current_price, self.config.risk_management
                        )

                    # --- SL/TP Execution ---
                    if pos.side == 'BUY':
                        if current_price <= pos.stop_loss_price:
                            logger.info("Stop-loss triggered for LONG position", symbol=pos.symbol, price=current_price, sl=pos.stop_loss_price)
                            await self._close_position(pos, "Stop-Loss")
                            continue
                        if current_price >= pos.take_profit_price:
                            logger.info("Take-profit triggered for LONG position", symbol=pos.symbol, price=current_price, tp=pos.take_profit_price)
                            await self._close_position(pos, "Take-Profit")
                            continue
                    elif pos.side == 'SELL':
                        if current_price >= pos.stop_loss_price:
                            logger.info("Stop-loss triggered for SHORT position", symbol=pos.symbol, price=current_price, sl=pos.stop_loss_price)
                            await self._close_position(pos, "Stop-Loss")
                            continue
                        if current_price <= pos.take_profit_price:
                            logger.info("Take-profit triggered for SHORT position", symbol=pos.symbol, price=current_price, tp=pos.take_profit_price)
                            await self._close_position(pos, "Take-Profit")
                            continue

            except asyncio.CancelledError:
                logger.info("Position management loop cancelled.")
                break
            except Exception as e:
                logger.error("Error in position management loop", error=str(e), exc_info=True)
            await asyncio.sleep(5) # Check positions frequently

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

                open_positions = self.position_manager.get_all_open_positions()
                portfolio_value = self.position_manager.get_portfolio_value(self.latest_prices, open_positions)
                self.risk_manager.update_portfolio_risk(portfolio_value)

                # Update shared state for Telegram bot
                self.shared_bot_state['portfolio_equity'] = portfolio_value
                self.shared_bot_state['open_positions_count'] = len(open_positions)

                if self.metrics_writer and self.metrics_writer.enabled:
                    await self.metrics_writer.write_metric('portfolio', fields={'equity': portfolio_value})

            except asyncio.CancelledError:
                logger.info("Monitoring loop cancelled.")
                break
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
            await asyncio.sleep(60) # Monitoring interval

    async def _run_single_trade_check(self, symbol: str):
        logger.debug("Starting new trade check", symbol=symbol)

        if self.risk_manager.is_halted:
            logger.warning("Trading is halted by RiskManager circuit breaker.")
            return

        df_with_indicators = self.data_handler.get_market_data(symbol)
        if df_with_indicators is None:
            logger.warning("Could not get market data from handler.", symbol=symbol)
            return
        
        if symbol not in self.latest_prices:
            logger.warning("Latest price for symbol not available yet.", symbol=symbol)
            return

        position = self.position_manager.get_open_position(symbol)
        signal = await self.strategy.analyze_market(symbol, df_with_indicators, position)

        if signal: await self._handle_signal(signal, df_with_indicators, position)

    async def _manage_order_lifecycle(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Tracks an order until it's filled or a timeout is reached.
        If timeout occurs for an open order, it attempts to cancel it.
        """
        logger.info("Managing lifecycle for order", order_id=order_id, symbol=symbol)
        start_time = time.time()
        timeout = self.config.execution.order_fill_timeout_seconds

        while time.time() - start_time < timeout:
            try:
                order_status = await self.exchange_api.fetch_order(order_id, symbol)
                if order_status:
                    status = order_status.get('status')
                    if status == 'FILLED':
                        logger.info("Order fill confirmed", order_id=order_id, symbol=symbol, fill_price=order_status.get('average'))
                        return order_status
                    if status not in ['OPEN', 'UNKNOWN']:
                        logger.warning("Order reached terminal state without being filled", order_id=order_id, status=status)
                        return order_status
                
                await asyncio.sleep(3) # Polling interval
            except Exception as e:
                logger.error("Error while polling for order fill", order_id=order_id, error=str(e))
                await asyncio.sleep(5)

        # Timeout reached, try to cancel
        logger.warning("Order fill timeout reached. Attempting to cancel.", order_id=order_id, symbol=symbol)
        try:
            await self.exchange_api.cancel_order(order_id, symbol)
            logger.info("Cancellation request sent for order", order_id=order_id)
            # Final check on order status after cancellation attempt
            final_status = await self.exchange_api.fetch_order(order_id, symbol)
            if final_status:
                logger.info("Final order status after cancellation attempt", order_id=order_id, status=final_status.get('status'))
                return final_status
            else:
                logger.warning("Could not fetch final order status after cancellation.", order_id=order_id)
                return {'id': order_id, 'status': 'UNKNOWN'}
        except Exception as e:
            logger.critical("Failed to cancel timed-out order", order_id=order_id, error=str(e))
            return None

    async def _handle_signal(self, signal: Dict, df_with_indicators: pd.DataFrame, position: Optional[Position]):
        action = signal.get('action')
        symbol = signal.get('symbol')
        current_price = self.latest_prices.get(symbol)
        market_regime = signal.get('regime')

        if not all([action, symbol, current_price]):
            logger.warning("Received invalid signal", signal=signal)
            return

        open_positions = self.position_manager.get_all_open_positions()
        portfolio_equity = self.position_manager.get_portfolio_value(self.latest_prices, open_positions)

        # --- Handle Opening Positions ---
        if not position:
            if action not in ['BUY', 'SELL']:
                return # Ignore other signals if no position is open

            if not self.risk_manager.check_trade_allowed(symbol, open_positions):
                return

            stop_loss = self.risk_manager.calculate_stop_loss(action, current_price, df_with_indicators, market_regime=market_regime)
            ideal_quantity = self.risk_manager.calculate_position_size(portfolio_equity, current_price, stop_loss, market_regime=market_regime)
            
            market_details = self.market_details.get(symbol)
            if not market_details:
                logger.error("Cannot place order, market details not available for symbol.", symbol=symbol)
                return

            final_quantity = self.order_sizer.adjust_order_quantity(symbol, ideal_quantity, market_details)

            if final_quantity <= 0:
                logger.warning("Calculated position size is zero or less after adjustments. Aborting trade.", 
                             symbol=symbol, ideal_quantity=ideal_quantity, final_quantity=final_quantity)
                return

            order_type = self.config.execution.default_order_type
            limit_price = None
            if order_type == 'LIMIT':
                offset = self.config.execution.limit_price_offset_pct
                if action == 'BUY':
                    limit_price = current_price * (1 + offset)
                else: # SELL
                    limit_price = current_price * (1 - offset)

            order_result = await self.exchange_api.place_order(symbol, action, order_type, final_quantity, price=limit_price)
            if order_result and order_result.get('orderId'):
                final_order_state = await self._manage_order_lifecycle(order_result['orderId'], symbol)
                if final_order_state and final_order_state.get('status') == 'FILLED':
                    fill_price = final_order_state['average']
                    fill_quantity = final_order_state['filled']
                    # Recalculate SL/TP based on actual fill price for higher accuracy
                    final_stop_loss = self.risk_manager.calculate_stop_loss(action, fill_price, df_with_indicators, market_regime=market_regime)
                    final_take_profit = self.risk_manager.calculate_take_profit(action, fill_price, final_stop_loss, market_regime=market_regime)
                    self.position_manager.open_position(symbol, action, fill_quantity, fill_price, final_stop_loss, final_take_profit)
                else:
                    logger.error("Order to open position did not fill.", order_id=order_result.get('orderId'), final_status=final_order_state.get('status') if final_order_state else 'UNKNOWN')
        
        # --- Handle Closing Positions ---
        elif position:
            is_close_long_signal = (action == 'SELL' or action == 'CLOSE') and position.side == 'BUY'
            is_close_short_signal = (action == 'BUY' or action == 'CLOSE') and position.side == 'SELL'
            
            if is_close_long_signal or is_close_short_signal:
                await self._close_position(position, "Strategy Signal")

    async def _close_position(self, position: Position, reason: str):
        close_side = 'SELL' if position.side == 'BUY' else 'BUY'
        
        market_details = self.market_details.get(position.symbol)
        if not market_details:
            logger.error("Cannot close position, market details not available for symbol.", symbol=position.symbol)
            # Attempt to close anyway, might fail
            close_quantity = position.quantity
        else:
            close_quantity = self.order_sizer.adjust_order_quantity(position.symbol, position.quantity, market_details)

        if close_quantity <= 0:
            logger.error("Cannot close position, adjusted quantity is zero.", symbol=position.symbol, original_qty=position.quantity)
            return

        order_type = self.config.execution.default_order_type
        limit_price = None
        current_price = self.latest_prices.get(position.symbol)
        if order_type == 'LIMIT' and current_price:
            offset = self.config.execution.limit_price_offset_pct
            if close_side == 'BUY':
                limit_price = current_price * (1 + offset)
            else: # SELL
                limit_price = current_price * (1 - offset)
        else:
            order_type = 'MARKET' # Fallback if price is missing

        order_result = await self.exchange_api.place_order(position.symbol, close_side, order_type, close_quantity, price=limit_price)
        if order_result and order_result.get('orderId'):
            final_order_state = await self._manage_order_lifecycle(order_result['orderId'], position.symbol)
            if final_order_state and final_order_state.get('status') == 'FILLED':
                close_price = final_order_state['average']
                self.position_manager.close_position(position.symbol, close_price, reason)
            else:
                logger.error("Failed to confirm close order fill. Position remains open.", order_id=order_result.get('orderId'), symbol=position.symbol, final_status=final_order_state.get('status') if final_order_state else 'UNKNOWN')

    async def stop(self):
        logger.info("Stopping TradingBot...")
        self.running = False
        self.shared_bot_state['status'] = 'stopping'
        
        if self.data_handler:
            await self.data_handler.stop()

        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        await asyncio.gather(*[task for task in self.tasks if not task.done()], return_exceptions=True)

        if self.exchange_api: await self.exchange_api.close()
        if self.position_manager: self.position_manager.close()
        if self.metrics_writer: await self.metrics_writer.close()
        logger.info("TradingBot stopped gracefully.")
        self.shared_bot_state['status'] = 'stopped'
