import asyncio
import time
from typing import Dict, Any, Optional, Callable, Awaitable

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

    async def manage(self, 
                     initial_order: Dict[str, Any], 
                     symbol: str, 
                     side: str, 
                     quantity: float, 
                     initial_price: Optional[float], 
                     market_details: Optional[Dict[str, Any]] = None,
                     on_order_replace: Optional[Callable[[str, str], Awaitable[None]]] = None,
                     trade_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
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

        return await self._chase_order_lifecycle(order_id, symbol, side, quantity, initial_price, market_details, on_order_replace, trade_id)

    async def _chase_order_lifecycle(self, 
                                     initial_order_id: str, 
                                     symbol: str, 
                                     side: str, 
                                     total_quantity: float, 
                                     initial_price: Optional[float], 
                                     market_details: Optional[Dict[str, Any]],
                                     on_order_replace: Optional[Callable[[str, str], Awaitable[None]]] = None,
                                     trade_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Handles the advanced order chasing logic for LIMIT orders, accounting for partial fills.
        """
        cumulative_filled = 0.0
        cumulative_cost = 0.0
        current_order_id = initial_order_id
        current_order_price = initial_price
        chase_attempts = 0

        # Track the ID currently stored in the persistence layer to ensure updates are linked correctly
        last_persisted_order_id = initial_order_id

        # Get min amount from market details to avoid dust errors
        min_amount = 0.0
        if market_details and 'limits' in market_details:
            min_amount = market_details['limits'].get('amount', {}).get('min', 0.0)

        while chase_attempts <= self.config.max_chase_attempts:
            await asyncio.sleep(self.config.chase_interval_seconds)

            # 1. Check status of current order
            order_status = await self.exchange_api.fetch_order(current_order_id, symbol)
            if not order_status:
                logger.warning("Could not fetch order status during chase. Retrying.", order_id=current_order_id)
                continue
            
            current_filled = order_status.get('filled', 0.0)
            current_avg = order_status.get('average', 0.0) or current_order_price or 0.0
            status = order_status.get('status')

            if status == 'FILLED':
                # Success: Add this order's contribution to cumulative and return
                final_filled = cumulative_filled + current_filled
                final_cost = cumulative_cost + (current_filled * current_avg)
                avg_price = final_cost / final_filled if final_filled > 0 else 0.0
                
                logger.info("Order fully filled during chase cycle.", total_filled=final_filled, avg_price=avg_price)
                return {
                    'orderId': current_order_id,
                    'status': 'FILLED',
                    'filled': final_filled,
                    'average': avg_price,
                    'symbol': symbol,
                    'side': side
                }
            
            if status not in ['OPEN', 'UNKNOWN']:
                # Order dead (rejected, expired, etc.) without full fill
                # We treat this as a partial fill scenario and try to chase the rest if possible
                logger.warning("Order reached terminal state during chase.", status=status, filled=current_filled)
                # We will fall through to the chase logic to place a new order for the remainder
                # But first we must update cumulative stats from this dead order
                cumulative_filled += current_filled
                cumulative_cost += (current_filled * current_avg)
                # We don't need to cancel it, it's already done.
                # We just need to set current_order_id to None so we don't try to cancel it below
                current_order_id = None 

            market_price = self.latest_prices.get(symbol)
            if not market_price:
                logger.warning("Cannot check for order chasing, latest price is unavailable.", symbol=symbol)
                continue

            is_behind_market = (side.upper() == 'BUY' and market_price > current_order_price) or \
                               (side.upper() == 'SELL' and market_price < current_order_price)

            if not is_behind_market and status == 'OPEN':
                logger.debug("Order is still competitive, not chasing.", order_id=current_order_id)
                continue

            # --- Execute Chase ---
            chase_attempts += 1
            if chase_attempts > self.config.max_chase_attempts:
                break

            logger.info("Market moved or order dead, chasing.", attempt=f"{chase_attempts}/{self.config.max_chase_attempts}")
            
            # 2. Cancel current order if it's still open to get final fill amount
            if current_order_id and status == 'OPEN':
                try:
                    cancel_res = await self.exchange_api.cancel_order(current_order_id, symbol)
                    if cancel_res:
                        final_cancel_filled = cancel_res.get('filled', 0.0)
                        final_cancel_avg = cancel_res.get('average', 0.0) or current_order_price or 0.0
                        cumulative_filled += final_cancel_filled
                        cumulative_cost += (final_cancel_filled * final_cancel_avg)
                        logger.info("Cancelled order for chase.", filled_so_far=cumulative_filled)
                except Exception as e:
                    logger.error("Failed to cancel order during chase. Checking status.", error=str(e))
                    # Critical Safety Check: If we failed to cancel, we must verify if it's still OPEN.
                    # If it is OPEN, we cannot proceed with placing a new order (double exposure risk).
                    try:
                        check_status = await self.exchange_api.fetch_order(current_order_id, symbol)
                        if check_status and check_status.get('status') == 'OPEN':
                            logger.critical("Order is stuck OPEN and cannot be cancelled. Aborting chase to prevent double execution.", order_id=current_order_id)
                            # Return the current state with OPEN status so the executor knows it's stuck
                            return {
                                'orderId': current_order_id,
                                'status': 'OPEN',
                                'filled': cumulative_filled,
                                'average': cumulative_cost / cumulative_filled if cumulative_filled > 0 else 0.0,
                                'symbol': symbol,
                                'side': side
                            }
                    except Exception as ex:
                        logger.critical("Failed to verify order status after cancel failure. Assuming stuck.", error=str(ex))
                        return {
                            'orderId': current_order_id,
                            'status': 'OPEN', # Assume worst case
                            'filled': cumulative_filled,
                            'average': 0.0,
                            'symbol': symbol,
                            'side': side
                        }
                    # If it wasn't OPEN (e.g. filled in between), the loop will handle it in next iteration or fall through
            
            # 3. Calculate remaining quantity
            remaining_qty = total_quantity - cumulative_filled
            
            # Check for dust / min limits
            if remaining_qty <= 0 or (min_amount > 0 and remaining_qty < min_amount):
                logger.info("Remaining quantity is dust or zero. Stopping chase.", remaining=remaining_qty, min_amount=min_amount)
                break

            # 4. Place new order
            price_improvement = market_price * self.config.chase_aggressiveness_pct
            new_price = market_price + price_improvement if side.upper() == 'BUY' else market_price - price_improvement
            
            try:
                # Construct deterministic clientOrderId if trade_id is available
                extra_params = {}
                if trade_id:
                    extra_params['clientOrderId'] = f"{trade_id}_chase_{chase_attempts}"

                new_order_result = await self.exchange_api.place_order(
                    symbol, side, 'LIMIT', remaining_qty, price=new_price, extra_params=extra_params
                )
                
                if new_order_result and new_order_result.get('orderId'):
                    new_order_id = new_order_result['orderId']
                    
                    # Trigger callback to persist the new order ID
                    if on_order_replace:
                        try:
                            await on_order_replace(last_persisted_order_id, new_order_id)
                            last_persisted_order_id = new_order_id
                        except Exception as e:
                            logger.error("Error in on_order_replace callback", error=str(e))

                    current_order_id = new_order_id
                    current_order_price = new_price
                    logger.info("Placed new chase order.", new_order_id=current_order_id, price=new_price, qty=remaining_qty)
                else:
                    logger.error("Failed to place new chased order. Aborting.", symbol=symbol)
                    break
            except Exception as e:
                logger.error("Exception placing chase order.", error=str(e))
                break

        # --- Final Result Construction ---
        # If we exit the loop, we return the aggregated result
        avg_price = cumulative_cost / cumulative_filled if cumulative_filled > 0 else 0.0
        status = 'FILLED' if cumulative_filled >= (total_quantity * 0.99) else 'PARTIALLY_FILLED'
        
        return {
            'orderId': current_order_id or initial_order_id, # Return last active ID
            'status': status,
            'filled': cumulative_filled,
            'average': avg_price,
            'symbol': symbol,
            'side': side
        }

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
            # If we failed to cancel, we must check if it is still OPEN
            try:
                check = await self.exchange_api.fetch_order(order_id, symbol)
                return check
            except Exception:
                return {'id': order_id, 'status': 'UNKNOWN'}
