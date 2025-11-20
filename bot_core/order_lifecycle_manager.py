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
        
        # 1. Run Primary Execution Strategy (Chase or Poll)
        if is_chaseable:
            result = await self._chase_order_lifecycle(order_id, symbol, side, quantity, initial_price, market_details, on_order_replace, trade_id)
        else:
            result = await self._poll_order_until_filled_or_timeout(order_id, symbol)

        if not result:
            return None

        # 2. Handle Execution Fallback (Market Order on Timeout)
        # If enabled, and we still have remaining quantity, execute immediately.
        filled_so_far = result.get('filled', 0.0)
        status = result.get('status')
        
        # Tolerance for float precision (99.9% filled is considered done)
        if self.config.execute_on_timeout and filled_so_far < (quantity * 0.999):
            logger.info("Execution timeout/end reached with partial fill. Triggering Market Order fallback.", 
                        filled=filled_so_far, target=quantity)
            
            remaining_qty = quantity - filled_so_far
            
            # Check min amount constraints before placing market order
            min_amount = 0.0
            if market_details and 'limits' in market_details:
                min_amount = market_details['limits'].get('amount', {}).get('min', 0.0)
            
            if remaining_qty >= min_amount:
                try:
                    # Construct clientOrderId for fallback
                    extra_params = {}
                    if trade_id:
                        extra_params['clientOrderId'] = f"{trade_id}_fallback"

                    market_order = await self.exchange_api.place_order(
                        symbol, side, 'MARKET', remaining_qty, extra_params=extra_params
                    )
                    
                    if market_order and market_order.get('orderId'):
                        # Poll the market order briefly to get fill details
                        fallback_result = await self._poll_order_until_filled_or_timeout(
                            market_order['orderId'], symbol, timeout_override=5
                        )
                        
                        if fallback_result:
                            # Aggregate results
                            fb_filled = fallback_result.get('filled', 0.0)
                            fb_avg = fallback_result.get('average', 0.0)
                            fb_fee = self._extract_fee_cost(fallback_result)
                            
                            # Update cumulative stats
                            prev_cost = filled_so_far * result.get('average', 0.0)
                            new_cost = fb_filled * fb_avg
                            
                            total_filled = filled_so_far + fb_filled
                            total_cost = prev_cost + new_cost
                            avg_price = total_cost / total_filled if total_filled > 0 else 0.0
                            
                            # Update result dict
                            result['filled'] = total_filled
                            result['average'] = avg_price
                            result['status'] = 'FILLED' if total_filled >= (quantity * 0.999) else 'PARTIALLY_FILLED'
                            
                            # Aggregate fees
                            current_fee = result.get('fee', {}).get('cost', 0.0) if isinstance(result.get('fee'), dict) else 0.0
                            total_fee = current_fee + fb_fee
                            # We reconstruct the fee object to match CCXT structure roughly
                            result['fee'] = {'cost': total_fee, 'currency': 'aggregated'}
                            
                            logger.info("Fallback Market Order complete.", total_filled=total_filled, avg_price=avg_price)
                        else:
                            logger.warning("Fallback Market Order placed but status unknown.")
                except Exception as e:
                    logger.error("Failed to execute fallback Market Order.", error=str(e))
            else:
                logger.warning("Remaining quantity too small for fallback Market Order.", remaining=remaining_qty, min=min_amount)

        return result

    def _extract_fee_cost(self, order_res: Dict[str, Any]) -> float:
        """Helper to safely extract fee cost from an order result."""
        if not order_res or 'fee' not in order_res or not order_res['fee']:
            return 0.0
        return float(order_res['fee'].get('cost', 0.0))

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
        cumulative_fees = 0.0
        
        current_order_id = initial_order_id
        current_order_price = initial_price
        chase_attempts = 0

        # Track the ID currently stored in the persistence layer
        last_persisted_order_id = initial_order_id

        # Get min amount from market details
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
            
            # Extract fees from this specific order check
            # Note: CCXT usually reports cumulative fees for the order. 
            # Since we switch orders, we must add the fee from the *finished* (or cancelled) order to our total.
            # However, if we poll an open order multiple times, we shouldn't double count.
            # Strategy: We only add fees when we are done with this specific order_id (Filled or Cancelled).

            if status == 'FILLED':
                # Success: Add this order's contribution
                final_filled = cumulative_filled + current_filled
                final_cost = cumulative_cost + (current_filled * current_avg)
                avg_price = final_cost / final_filled if final_filled > 0 else 0.0
                
                # Add fees
                cumulative_fees += self._extract_fee_cost(order_status)
                
                logger.info("Order fully filled during chase cycle.", total_filled=final_filled, avg_price=avg_price)
                return {
                    'orderId': current_order_id,
                    'status': 'FILLED',
                    'filled': final_filled,
                    'average': avg_price,
                    'symbol': symbol,
                    'side': side,
                    'fee': {'cost': cumulative_fees, 'currency': 'aggregated'}
                }
            
            if status not in ['OPEN', 'UNKNOWN']:
                # Order dead (rejected, expired, etc.) without full fill
                logger.warning("Order reached terminal state during chase.", status=status, filled=current_filled)
                cumulative_filled += current_filled
                cumulative_cost += (current_filled * current_avg)
                cumulative_fees += self._extract_fee_cost(order_status)
                current_order_id = None 

            # Fetch fresh ticker for market check
            try:
                ticker = await self.exchange_api.get_ticker_data(symbol)
                bid = ticker.get('bid')
                ask = ticker.get('ask')
                last = ticker.get('last')
            except Exception as e:
                logger.warning("Failed to fetch ticker for chase check.", error=str(e))
                bid, ask, last = None, None, self.latest_prices.get(symbol)

            # Determine reference market price
            if side.upper() == 'BUY':
                market_ref = bid if bid else last
                is_behind_market = market_ref > current_order_price if market_ref else False
            else:
                market_ref = ask if ask else last
                is_behind_market = market_ref < current_order_price if market_ref else False

            if not is_behind_market and status == 'OPEN':
                logger.debug("Order is still competitive, not chasing.", order_id=current_order_id)
                continue

            # --- Execute Chase ---
            chase_attempts += 1
            if chase_attempts > self.config.max_chase_attempts:
                break

            logger.info("Market moved or order dead, chasing.", attempt=f"{chase_attempts}/{self.config.max_chase_attempts}")
            
            # 2. Cancel current order if it's still open
            if current_order_id and status == 'OPEN':
                try:
                    cancel_res = await self.exchange_api.cancel_order(current_order_id, symbol)
                    if cancel_res:
                        final_cancel_filled = cancel_res.get('filled', 0.0)
                        final_cancel_avg = cancel_res.get('average', 0.0) or current_order_price or 0.0
                        
                        cumulative_filled += final_cancel_filled
                        cumulative_cost += (final_cancel_filled * final_cancel_avg)
                        cumulative_fees += self._extract_fee_cost(cancel_res)
                        
                        logger.info("Cancelled order for chase.", filled_so_far=cumulative_filled)
                except Exception as e:
                    logger.error("Failed to cancel order during chase. Checking status.", error=str(e))
                    # Safety Check
                    try:
                        check_status = await self.exchange_api.fetch_order(current_order_id, symbol)
                        if check_status and check_status.get('status') == 'OPEN':
                            logger.critical("Order is stuck OPEN and cannot be cancelled. Aborting chase.", order_id=current_order_id)
                            return {
                                'orderId': current_order_id,
                                'status': 'OPEN',
                                'filled': cumulative_filled,
                                'average': cumulative_cost / cumulative_filled if cumulative_filled > 0 else 0.0,
                                'symbol': symbol,
                                'side': side,
                                'fee': {'cost': cumulative_fees, 'currency': 'aggregated'}
                            }
                    except Exception as ex:
                        logger.critical("Failed to verify order status after cancel failure.", error=str(ex))
                        return {
                            'orderId': current_order_id,
                            'status': 'OPEN',
                            'filled': cumulative_filled,
                            'average': 0.0,
                            'symbol': symbol,
                            'side': side,
                            'fee': {'cost': cumulative_fees, 'currency': 'aggregated'}
                        }
            
            # 3. Calculate remaining quantity
            remaining_qty = total_quantity - cumulative_filled
            
            if remaining_qty <= 0 or (min_amount > 0 and remaining_qty < min_amount):
                logger.info("Remaining quantity is dust or zero. Stopping chase.", remaining=remaining_qty)
                break

            # 4. Place new order
            price_improvement = (last or 0.0) * self.config.chase_aggressiveness_pct
            
            if side.upper() == 'BUY':
                base_price = bid if bid else (last or current_order_price)
                new_price = base_price + price_improvement
            else:
                base_price = ask if ask else (last or current_order_price)
                new_price = base_price - price_improvement
            
            # Slippage Check
            if initial_price and initial_price > 0:
                slippage = abs(new_price - initial_price) / initial_price
                if slippage > self.config.max_chase_slippage_pct:
                    logger.warning("Chase aborted: Price deviated too far.", slippage=slippage)
                    break

            try:
                extra_params = {}
                if trade_id:
                    extra_params['clientOrderId'] = f"{trade_id}_chase_{chase_attempts}"

                new_order_result = await self.exchange_api.place_order(
                    symbol, side, 'LIMIT', remaining_qty, price=new_price, extra_params=extra_params
                )
                
                if new_order_result and new_order_result.get('orderId'):
                    new_order_id = new_order_result['orderId']
                    
                    if on_order_replace:
                        try:
                            await on_order_replace(last_persisted_order_id, new_order_id)
                            last_persisted_order_id = new_order_id
                        except Exception as e:
                            logger.error("Error in on_order_replace callback", error=str(e))

                    current_order_id = new_order_id
                    current_order_price = new_price
                    logger.info("Placed new chase order.", new_order_id=current_order_id, price=new_price)
                else:
                    logger.error("Failed to place new chased order. Aborting.", symbol=symbol)
                    break
            except Exception as e:
                logger.error("Exception placing chase order.", error=str(e))
                break

        # --- Final Result Construction ---
        avg_price = cumulative_cost / cumulative_filled if cumulative_filled > 0 else 0.0
        status = 'FILLED' if cumulative_filled >= (total_quantity * 0.99) else 'PARTIALLY_FILLED'
        
        return {
            'orderId': current_order_id or initial_order_id,
            'status': status,
            'filled': cumulative_filled,
            'average': avg_price,
            'symbol': symbol,
            'side': side,
            'fee': {'cost': cumulative_fees, 'currency': 'aggregated'}
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
            try:
                check = await self.exchange_api.fetch_order(order_id, symbol)
                return check
            except Exception:
                return {'id': order_id, 'status': 'UNKNOWN'}
