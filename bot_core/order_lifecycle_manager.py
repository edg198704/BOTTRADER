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
    Ensures orders reach a terminal state (filled, canceled, etc.) robustly.
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
        
        order_id = initial_order.get('orderId')
        if not order_id:
            return None

        logger.info("Managing lifecycle", order_id=order_id, symbol=symbol)
        
        is_chaseable = self.config.default_order_type == 'LIMIT' and self.config.use_order_chasing
        
        if is_chaseable:
            result = await self._chase_order_lifecycle(order_id, symbol, side, quantity, initial_price, market_details, on_order_replace, trade_id)
        else:
            result = await self._poll_order_until_filled_or_timeout(order_id, symbol)

        if not result:
            return None

        # Fallback to Market Order if enabled and partially filled
        filled_so_far = result.get('filled', 0.0)
        if self.config.execute_on_timeout and filled_so_far < (quantity * 0.999):
            remaining_qty = quantity - filled_so_far
            min_amount = 0.0
            if market_details and 'limits' in market_details:
                min_amount = market_details['limits'].get('amount', {}).get('min', 0.0)
            
            if remaining_qty >= min_amount:
                logger.info("Triggering Fallback Market Order", remaining=remaining_qty)
                try:
                    extra_params = {}
                    if trade_id:
                        extra_params['clientOrderId'] = f"{trade_id}_fallback"

                    market_order = await self.exchange_api.place_order(
                        symbol, side, 'MARKET', remaining_qty, extra_params=extra_params
                    )
                    
                    if market_order and market_order.get('orderId'):
                        fb_result = await self._poll_order_until_filled_or_timeout(
                            market_order['orderId'], symbol, timeout_override=5
                        )
                        
                        if fb_result:
                            fb_filled = fb_result.get('filled', 0.0)
                            fb_avg = fb_result.get('average', 0.0)
                            fb_fee = self._extract_fee_cost(fb_result)
                            
                            prev_cost = filled_so_far * result.get('average', 0.0)
                            new_cost = fb_filled * fb_avg
                            total_filled = filled_so_far + fb_filled
                            avg_price = (prev_cost + new_cost) / total_filled if total_filled > 0 else 0.0
                            
                            result['filled'] = total_filled
                            result['average'] = avg_price
                            result['status'] = 'FILLED' if total_filled >= (quantity * 0.999) else 'PARTIALLY_FILLED'
                            
                            current_fee = self._extract_fee_cost(result)
                            result['fee'] = {'cost': current_fee + fb_fee, 'currency': 'aggregated'}
                except Exception as e:
                    logger.error("Fallback Market Order failed", error=str(e))

        return result

    def _extract_fee_cost(self, order_res: Dict[str, Any]) -> float:
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
        
        cumulative_filled = 0.0
        cumulative_cost = 0.0
        cumulative_fees = 0.0
        
        current_order_id = initial_order_id
        current_order_price = initial_price
        chase_attempts = 0
        last_persisted_order_id = initial_order_id

        min_amount = 0.0
        if market_details and 'limits' in market_details:
            min_amount = market_details['limits'].get('amount', {}).get('min', 0.0)

        while chase_attempts <= self.config.max_chase_attempts:
            await asyncio.sleep(self.config.chase_interval_seconds)

            order_status = await self.exchange_api.fetch_order(current_order_id, symbol)
            if not order_status:
                continue
            
            current_filled = order_status.get('filled', 0.0)
            current_avg = order_status.get('average', 0.0) or current_order_price or 0.0
            status = order_status.get('status')

            if status == 'FILLED':
                final_filled = cumulative_filled + current_filled
                final_cost = cumulative_cost + (current_filled * current_avg)
                avg_price = final_cost / final_filled if final_filled > 0 else 0.0
                cumulative_fees += self._extract_fee_cost(order_status)
                
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
                cumulative_filled += current_filled
                cumulative_cost += (current_filled * current_avg)
                cumulative_fees += self._extract_fee_cost(order_status)
                current_order_id = None 

            # Check if we need to chase
            try:
                ticker = await self.exchange_api.get_ticker_data(symbol)
                bid, ask, last = ticker.get('bid'), ticker.get('ask'), ticker.get('last')
            except Exception:
                bid, ask, last = None, None, self.latest_prices.get(symbol)

            if side.upper() == 'BUY':
                market_ref = bid if bid else last
                is_behind_market = market_ref > current_order_price if market_ref else False
            else:
                market_ref = ask if ask else last
                is_behind_market = market_ref < current_order_price if market_ref else False

            if not is_behind_market and status == 'OPEN':
                continue

            chase_attempts += 1
            if chase_attempts > self.config.max_chase_attempts:
                break

            # Cancel current order
            if current_order_id and status == 'OPEN':
                try:
                    cancel_res = await self.exchange_api.cancel_order(current_order_id, symbol)
                    if cancel_res:
                        final_cancel_filled = cancel_res.get('filled', 0.0)
                        final_cancel_avg = cancel_res.get('average', 0.0) or current_order_price or 0.0
                        
                        cumulative_filled += final_cancel_filled
                        cumulative_cost += (final_cancel_filled * final_cancel_avg)
                        cumulative_fees += self._extract_fee_cost(cancel_res)
                except Exception:
                    # Verify status if cancel failed
                    check = await self.exchange_api.fetch_order(current_order_id, symbol)
                    if check and check.get('status') == 'OPEN':
                        logger.critical("Order stuck OPEN, cannot cancel.", order_id=current_order_id)
                        return {
                            'orderId': current_order_id,
                            'status': 'OPEN',
                            'filled': cumulative_filled,
                            'average': cumulative_cost / cumulative_filled if cumulative_filled > 0 else 0.0,
                            'symbol': symbol,
                            'side': side,
                            'fee': {'cost': cumulative_fees, 'currency': 'aggregated'}
                        }
            
            remaining_qty = total_quantity - cumulative_filled
            if remaining_qty <= 0 or (min_amount > 0 and remaining_qty < min_amount):
                break

            # Place new order
            price_improvement = (last or 0.0) * self.config.chase_aggressiveness_pct
            if side.upper() == 'BUY':
                base_price = bid if bid else (last or current_order_price)
                new_price = base_price + price_improvement
            else:
                base_price = ask if ask else (last or current_order_price)
                new_price = base_price - price_improvement
            
            if initial_price and initial_price > 0:
                slippage = abs(new_price - initial_price) / initial_price
                if slippage > self.config.max_chase_slippage_pct:
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
                        await on_order_replace(last_persisted_order_id, new_order_id)
                        last_persisted_order_id = new_order_id

                    current_order_id = new_order_id
                    current_order_price = new_price
                else:
                    break
            except Exception:
                break

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
        start_time = time.time()
        timeout = timeout_override or self.config.order_fill_timeout_seconds

        while time.time() - start_time < timeout:
            try:
                order_status = await self.exchange_api.fetch_order(order_id, symbol)
                if order_status:
                    status = order_status.get('status')
                    if status == 'FILLED':
                        return order_status
                    if status not in ['OPEN', 'UNKNOWN']:
                        return order_status
                await asyncio.sleep(3)
            except Exception:
                await asyncio.sleep(5)

        return await self._cancel_and_get_final_status(order_id, symbol)

    async def _cancel_and_get_final_status(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            await self.exchange_api.cancel_order(order_id, symbol)
            return await self.exchange_api.fetch_order(order_id, symbol)
        except Exception:
            try:
                return await self.exchange_api.fetch_order(order_id, symbol)
            except Exception:
                return {'id': order_id, 'status': 'UNKNOWN'}
