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
from bot_core.position_monitor import PositionMonitor

logger = get_logger(__name__)

class TradingBot:
    def __init__(self, config: BotConfig, exchange_api: ExchangeAPI, data_handler: DataHandler, 
                 strategy: TradingStrategy, position_manager: PositionManager, risk_manager: RiskManager,
                 order_sizer: OrderSizer, health_checker: HealthChecker, position_monitor: PositionMonitor,
                 shared_latest_prices: Dict[str, float],
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
        self.position_monitor = position_monitor
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
                        
                        training_df = await self.data_handler.fetch_full_history_for_symbol(
                            symbol, self.config.strategy.ai_ensemble.training_data_limit
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

    async def _manage_order_lifecycle(self, initial_order: Dict[str, Any], symbol: str, side: str, quantity: float, initial_price: Optional[float]) -> Optional[Dict[str, Any]]:
        """
        Manages an order's lifecycle, including polling, chasing, and timeout handling.
        """
        exec_config = self.config.execution
        order_id = initial_order.get('orderId')
        if not order_id:
            logger.error("Cannot manage order lifecycle without an order ID.", initial_order=initial_order)
            return None

        logger.info("Managing lifecycle for order", order_id=order_id, symbol=symbol)
        
        is_chaseable = exec_config.default_order_type == 'LIMIT' and exec_config.use_order_chasing
        if not is_chaseable:
            return await self._poll_order_until_filled_or_timeout(order_id, symbol)

        # --- Advanced Order Chasing for LIMIT orders ---
        chase_attempts = 0
        current_order_id = order_id
        current_order_price = initial_price

        while chase_attempts <= exec_config.max_chase_attempts:
            await asyncio.sleep(exec_config.chase_interval_seconds)

            order_status = await self.exchange_api.fetch_order(current_order_id, symbol)
            if order_status and order_status.get('status') == 'FILLED':
                logger.info("Chased order filled", order_id=current_order_id, attempt=chase_attempts)
                return order_status
            
            if order_status and order_status.get('status') not in ['OPEN', 'UNKNOWN']:
                logger.warning("Chased order in terminal state without fill", order_id=current_order_id, status=order_status.get('status'))
                return order_status

            market_price = self.latest_prices.get(symbol)
            if not market_price:
                logger.warning("Cannot check for order chasing, latest price is unavailable.", symbol=symbol)
                continue

            is_behind_market = (side == 'BUY' and market_price > current_order_price) or \
                               (side == 'SELL' and market_price < current_order_price)

            if not is_behind_market:
                logger.debug("Order is still competitive, not chasing.", order_id=current_order_id, order_price=current_order_price, market_price=market_price)
                continue

            # --- Execute Chase ---
            chase_attempts += 1
            if chase_attempts > exec_config.max_chase_attempts:
                break

            logger.info("Market moved away, chasing order.", order_id=current_order_id, attempt=f"{chase_attempts}/{exec_config.max_chase_attempts}")
            
            try:
                await self.exchange_api.cancel_order(current_order_id, symbol)
                logger.info("Successfully cancelled old order for chasing.", old_order_id=current_order_id)

                price_improvement = market_price * exec_config.chase_aggressiveness_pct
                new_price = market_price + price_improvement if side == 'BUY' else market_price - price_improvement

                new_order_result = await self.exchange_api.place_order(symbol, side, 'LIMIT', quantity, price=new_price)
                if new_order_result and new_order_result.get('orderId'):
                    current_order_id = new_order_result['orderId']
                    current_order_price = new_price
                    logger.info("Placed new, more aggressive order.", new_order_id=current_order_id, new_price=new_price)
                else:
                    logger.error("Failed to place new chased order. Aborting chase.", symbol=symbol)
                    return None
            except Exception as e:
                logger.error("Exception during order chase. Aborting.", error=str(e), exc_info=True)
                return None

        # --- Handle chase timeout ---
        logger.warning("Max chase attempts reached for order.", original_order_id=order_id)
        if exec_config.execute_on_timeout:
            logger.info("Executing a MARKET order as a fallback.", symbol=symbol)
            try:
                await self.exchange_api.cancel_order(current_order_id, symbol)
                market_order = await self.exchange_api.place_order(symbol, side, 'MARKET', quantity)
                if market_order and market_order.get('orderId'):
                    return await self._poll_order_until_filled_or_timeout(market_order['orderId'], symbol, timeout_override=15)
                else:
                    logger.critical("Failed to place fallback MARKET order.", symbol=symbol)
                    return None
            except Exception as e:
                logger.critical("Exception during fallback MARKET order execution.", error=str(e))
                return None
        else:
            logger.info("execute_on_timeout is false. Cancelling final order.", order_id=current_order_id)
            return await self._cancel_and_get_final_status(current_order_id, symbol)

    async def _poll_order_until_filled_or_timeout(self, order_id: str, symbol: str, timeout_override: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Tracks an order until it's filled or a timeout is reached.
        If timeout occurs for an open order, it attempts to cancel it.
        """
        start_time = time.time()
        timeout = timeout_override or self.config.execution.order_fill_timeout_seconds

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

        logger.warning("Order fill timeout reached. Attempting to cancel.", order_id=order_id, symbol=symbol)
        return await self._cancel_and_get_final_status(order_id, symbol)

    async def _cancel_and_get_final_status(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Cancels an order and fetches its final status."""
        try:
            await self.exchange_api.cancel_order(order_id, symbol)
            logger.info("Cancellation request sent for order", order_id=order_id)
            final_status = await self.exchange_api.fetch_order(order_id, symbol)
            logger.info("Final order status after cancellation attempt", order_id=order_id, status=final_status.get('status') if final_status else 'UNKNOWN')
            return final_status
        except Exception as e:
            logger.critical("Failed to cancel timed-out order", order_id=order_id, error=str(e))
            return {'id': order_id, 'status': 'UNKNOWN'}

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
                    limit_price = current_price * (1 - offset) # Place below market for buys
                else: # SELL
                    limit_price = current_price * (1 + offset) # Place above market for sells

            order_result = await self.exchange_api.place_order(symbol, action, order_type, final_quantity, price=limit_price)
            if order_result and order_result.get('orderId'):
                final_order_state = await self._manage_order_lifecycle(
                    initial_order=order_result,
                    symbol=symbol,
                    side=action,
                    quantity=final_quantity,
                    initial_price=limit_price
                )
                
                fill_quantity = final_order_state.get('filled', 0.0) if final_order_state else 0.0

                if fill_quantity > 0:
                    fill_price = final_order_state.get('average')
                    if not fill_price or fill_price <= 0:
                        logger.critical("Order filled but average price is invalid. Cannot open position.", order_id=order_result.get('orderId'), final_state=final_order_state)
                        return # Avoid creating a position with bad data

                    logger.info("Order to open position was filled (fully or partially).", order_id=order_result.get('orderId'), filled_qty=fill_quantity, fill_price=fill_price)
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
            final_order_state = await self._manage_order_lifecycle(
                initial_order=order_result,
                symbol=position.symbol,
                side=close_side,
                quantity=close_quantity,
                initial_price=limit_price
            )
            if final_order_state and final_order_state.get('status') == 'FILLED':
                close_price = final_order_state['average']
                self.position_manager.close_position(position.symbol, close_price, reason)
            else:
                logger.error("Failed to confirm close order fill. Position remains open.", order_id=order_result.get('orderId'), symbol=position.symbol, final_status=final_order_state.get('status') if final_order_state else 'UNKNOWN')

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
