import asyncio
import time
from typing import Dict, Any, Optional

from bot_core.logger import get_logger
from bot_core.config import ExecutionConfig
from bot_core.exchange_api import ExchangeAPI

logger = get_logger(__name__)

class OrderLifecycleManager:
    """
    Manages an order's lifecycle, including polling, chasing, and timeout handling.
    This class encapsulates the logic for ensuring an order reaches a terminal state (filled, canceled, etc.).
    """
    def __init__(self, exchange_api: ExchangeAPI, exec_config: ExecutionConfig, shared_latest_prices: Dict[str, float]):
        self.exchange_api = exchange_api
        self.config = exec_config
        self.latest_prices = shared_latest_prices
        logger.info("OrderLifecycleManager initialized.")

    async def manage(self, initial_order: Dict[str, Any], symbol: str, side: str, quantity: float, initial_price: Optional[float]) -> Optional[Dict[str, Any]]:
        """
        Main entry point to manage an order's lifecycle.
        """
        order_id = initial_order.get('orderId')
        if not order_id:
            logger.error("Cannot manage order lifecycle without an order ID.", initial_order=initial_order)
            return None

        logger.info("Managing lifecycle for order", order_id=order_id, symbol=symbol)
        
        is_chaseable = self.config.default_order_type == 'LIMIT' and self.config.use_order_chasing
        if not is_chaseable:
            return await self._poll_order_until_filled_or_timeout(order_id, symbol)

        return await self._chase_order_lifecycle(order_id, symbol, side, quantity, initial_price)

    async def _chase_order_lifecycle(self, initial_order_id: str, symbol: str, side: str, quantity: float, initial_price: Optional[float]) -> Optional[Dict[str, Any]]:
        """Handles the advanced order chasing logic for LIMIT orders."""
        chase_attempts = 0
        current_order_id = initial_order_id
        current_order_price = initial_price

        while chase_attempts <= self.config.max_chase_attempts:
            await asyncio.sleep(self.config.chase_interval_seconds)

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

            is_behind_market = (side.upper() == 'BUY' and market_price > current_order_price) or \
                               (side.upper() == 'SELL' and market_price < current_order_price)

            if not is_behind_market:
                logger.debug("Order is still competitive, not chasing.", order_id=current_order_id, order_price=current_order_price, market_price=market_price)
                continue

            # --- Execute Chase ---
            chase_attempts += 1
            if chase_attempts > self.config.max_chase_attempts:
                break

            logger.info("Market moved away, chasing order.", order_id=current_order_id, attempt=f"{chase_attempts}/{self.config.max_chase_attempts}")
            
            try:
                await self.exchange_api.cancel_order(current_order_id, symbol)
                logger.info("Successfully cancelled old order for chasing.", old_order_id=current_order_id)

                price_improvement = market_price * self.config.chase_aggressiveness_pct
                new_price = market_price + price_improvement if side.upper() == 'BUY' else market_price - price_improvement

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
        logger.warning("Max chase attempts reached for order.", original_order_id=initial_order_id)
        if self.config.execute_on_timeout:
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
        timeout = timeout_override or self.config.order_fill_timeout_seconds

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
