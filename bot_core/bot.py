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

    async def _execute_order_with_chasing(self, symbol: str, side: str, order_type: str, quantity: float, price: Optional[float]) -> Optional[Dict[str, Any]]:
        """
        Manages an order's lifecycle, with logic to "chase" unfilled LIMIT orders.
        Returns the final state of the order (filled, canceled, etc.).
        """
        exec_cfg = self.config.execution
        
        # --- Simple path for MARKET orders ---
        if order_type == 'MARKET':
            logger.info("Placing MARKET order", symbol=symbol, side=side, quantity=quantity)
            order_result = await self.exchange_api.place_order(symbol, side, 'MARKET', quantity)
            if not (order_result and order_result.get('orderId')):
                logger.error("Market order placement failed.", symbol=symbol)
                return None
            # For market orders, we poll until filled
            return await self._monitor_order_fill(order_result['orderId'], symbol)

        # --- Advanced path for LIMIT orders with chasing ---
        if not exec_cfg.use_order_chasing:
            logger.info("Placing simple LIMIT order (chasing disabled)", symbol=symbol, side=side, quantity=quantity, price=price)
            order_result = await self.exchange_api.place_order(symbol, side, 'LIMIT', quantity, price)
            if not (order_result and order_result.get('orderId')):
                logger.error("Limit order placement failed.", symbol=symbol)
                return None
            return await self._monitor_order_fill(order_result['orderId'], symbol)

        # --- Chasing Logic ---
        current_price = price
        for attempt in range(exec_cfg.max_chase_attempts + 1):
            logger.info(f"Order chase attempt {attempt+1}/{exec_cfg.max_chase_attempts+1}", symbol=symbol, price=current_price)
            order_result = await self.exchange_api.place_order(symbol, side, 'LIMIT', quantity, current_price)
            if not (order_result and order_result.get('orderId')):
                logger.error("Failed to place limit order during chase.", attempt=attempt+1)
                await asyncio.sleep(2) # Wait before retrying placement
                continue

            order_id = order_result['orderId']
            
            # Wait for the chase interval to see if the order gets filled
            try:
                await asyncio.wait_for(self._poll_for_fill(order_id, symbol), timeout=exec_cfg.chase_interval_seconds)
                # If it completes without timeout, it's filled
                return await self.exchange_api.fetch_order(order_id, symbol)
            except asyncio.TimeoutError:
                logger.info("Chase interval ended, order not filled. Re-evaluating.", order_id=order_id)
                # Not filled, continue to chase logic
            
            # Check if we should continue chasing
            if attempt < exec_cfg.max_chase_attempts:
                try:
                    await self.exchange_api.cancel_order(order_id, symbol)
                    logger.info("Canceled previous limit order to chase.", order_id=order_id)
                except Exception as e:
                    logger.warning("Could not cancel order, it might have been filled.", order_id=order_id, error=str(e))
                    # Check status one last time
                    final_status = await self.exchange_api.fetch_order(order_id, symbol)
                    if final_status and final_status.get('status') == 'FILLED':
                        return final_status
                    else:
                        logger.error("Failed to cancel non-filled order, aborting chase.", order_id=order_id)
                        return final_status

                # Update price to be more aggressive
                market_price_data = await self.exchange_api.get_ticker_data(symbol)
                market_price = float(market_price_data['lastPrice'])
                
                if side == 'BUY':
                    current_price = min(current_price * (1 + exec_cfg.chase_aggressiveness_pct), market_price)
                else: # SELL
                    current_price = max(current_price * (1 - exec_cfg.chase_aggressiveness_pct), market_price)
            else:
                # Last attempt, handle timeout
                logger.warning("Max chase attempts reached.", order_id=order_id)
                await self.exchange_api.cancel_order(order_id, symbol)
                if exec_cfg.execute_on_timeout:
                    logger.info("Executing final MARKET order after timeout.", symbol=symbol)
                    return await self._execute_order_with_chasing(symbol, side, 'MARKET', quantity, None)
                else:
                    logger.warning("Order chasing failed and execute_on_timeout is false. Order not filled.", symbol=symbol)
                    return await self.exchange_api.fetch_order(order_id, symbol) # Return final canceled state
        return None

    async def _poll_for_fill(self, order_id: str, symbol: str):
        """A helper that continuously polls until an order is filled."""
        while self.running:
            order_status = await self.exchange_api.fetch_order(order_id, symbol)
            if order_status and order_status.get('status') == 'FILLED':
                return
            await asyncio.sleep(1) # Poll quickly

    async def _monitor_order_fill(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Monitors an order until it's filled or a timeout is reached, then cancels."""
        logger.info("Monitoring order until filled or timeout", order_id=order_id, symbol=symbol)
        try:
            await asyncio.wait_for(self._poll_for_fill(order_id, symbol), timeout=self.config.execution.order_fill_timeout_seconds)
            final_status = await self.exchange_api.fetch_order(order_id, symbol)
            logger.info("Order fill confirmed", order_id=order_id, fill_price=final_status.get('average'))
            return final_status
        except asyncio.TimeoutError:
            logger.warning("Order fill timeout reached. Attempting to cancel.", order_id=order_id)
            try:
                await self.exchange_api.cancel_order(order_id, symbol)
                logger.info("Cancellation request sent for order", order_id=order_id)
            except Exception as e:
                logger.critical("Failed to cancel timed-out order", order_id=order_id, error=str(e))
            
            final_status = await self.exchange_api.fetch_order(order_id, symbol)
            logger.info("Final order status after timeout", order_id=order_id, status=final_status.get('status'))
            return final_status

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

            final_order_state = await self._execute_order_with_chasing(symbol, action, order_type, final_quantity, price=limit_price)
            
            fill_quantity = final_order_state.get('filled', 0.0) if final_order_state else 0.0

            if fill_quantity > 0:
                fill_price = final_order_state.get('average')
                if not fill_price or fill_price <= 0:
                    logger.critical("Order filled but average price is invalid. Cannot open position.", order_id=final_order_state.get('id'), final_state=final_order_state)
                    return # Avoid creating a position with bad data

                logger.info("Order to open position was filled (fully or partially).", order_id=final_order_state.get('id'), filled_qty=fill_quantity, fill_price=fill_price)
                # Recalculate SL/TP based on actual fill price for higher accuracy
                final_stop_loss = self.risk_manager.calculate_stop_loss(action, fill_price, df_with_indicators, market_regime=market_regime)
                final_take_profit = self.risk_manager.calculate_take_profit(action, fill_price, final_stop_loss, market_regime=market_regime)
                self.position_manager.open_position(symbol, action, fill_quantity, fill_price, final_stop_loss, final_take_profit)
            else:
                logger.error("Order to open position did not fill.", order_id=final_order_state.get('id') if final_order_state else 'N/A', final_status=final_order_state.get('status') if final_order_state else 'UNKNOWN')
        
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

        # For critical exits like SL/TP, consider using MARKET order to ensure execution
        order_type = 'MARKET' if reason in ['Stop-Loss', 'Take-Profit'] else self.config.execution.default_order_type
        
        limit_price = None
        current_price = self.latest_prices.get(position.symbol)
        if order_type == 'LIMIT' and current_price:
            offset = self.config.execution.limit_price_offset_pct
            if close_side == 'BUY':
                limit_price = current_price * (1 - offset) # Place below market to get filled
            else: # SELL
                limit_price = current_price * (1 + offset) # Place above market to get filled
        
        final_order_state = await self._execute_order_with_chasing(position.symbol, close_side, order_type, close_quantity, price=limit_price)
        
        if final_order_state and final_order_state.get('status') == 'FILLED':
            close_price = final_order_state['average']
            self.position_manager.close_position(position.symbol, close_price, reason)
        else:
            logger.error("Failed to confirm close order fill. Position remains open.", order_id=final_order_state.get('id') if final_order_state else 'N/A', symbol=position.symbol, final_status=final_order_state.get('status') if final_order_state else 'UNKNOWN')

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
