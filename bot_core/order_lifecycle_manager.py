import asyncio
import time
import uuid
from typing import Dict, Any, Optional, List, Literal
from dataclasses import dataclass, field
from decimal import Decimal

from bot_core.logger import get_logger
from bot_core.config import ExecutionConfig, ExecutionProfile
from bot_core.exchange_api import ExchangeAPI
from bot_core.position_manager import PositionManager, PositionStatus
from bot_core.event_system import EventBus, TradeCompletedEvent
from bot_core.utils import Clock
from bot_core.common import to_decimal, ZERO, ONE

logger = get_logger(__name__)

@dataclass
class ActiveOrderContext:
    trade_id: str
    symbol: str
    side: str
    total_quantity: Decimal
    initial_price: Decimal
    strategy_metadata: Dict[str, Any]
    
    current_order_id: str
    current_price: Decimal
    profile: ExecutionProfile
    
    intent: Literal['OPEN', 'CLOSE'] = 'OPEN'
    cumulative_filled: Decimal = ZERO
    cumulative_fees: Decimal = ZERO
    start_time: float = field(default_factory=time.time)
    last_poll_time: float = 0.0
    chase_attempts: int = 0
    status: str = 'OPEN'
    market_details: Dict[str, Any] = field(default_factory=dict)

class OrderLifecycleService:
    """
    Asynchronous service that manages the lifecycle of active orders in the background.
    Decouples execution monitoring from the signal generation pipeline.
    Implements Adaptive Polling, Smart Chasing, and Startup Reconciliation.
    """
    def __init__(self, 
                 exchange_api: ExchangeAPI, 
                 position_manager: PositionManager, 
                 event_bus: EventBus,
                 exec_config: ExecutionConfig,
                 shared_latest_prices: Dict[str, float]):
        self.exchange_api = exchange_api
        self.position_manager = position_manager
        self.event_bus = event_bus
        self.config = exec_config
        self.latest_prices = shared_latest_prices
        
        self._active_orders: Dict[str, ActiveOrderContext] = {}
        self._lock = asyncio.Lock()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        logger.info("OrderLifecycleService initialized.")

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop(), name="OrderLifecycleMonitor")
        logger.info("OrderLifecycleService started.")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("OrderLifecycleService stopped.")

    async def reconcile_recovery(self):
        """
        CRITICAL: Rebuilds in-memory state from DB and Exchange on startup.
        Handles 'Zombie' orders that were left active during a crash/restart.
        """
        logger.info("Starting Order Lifecycle Reconciliation...")
        
        # 1. Fetch all PENDING positions (Orders initiated but not confirmed open)
        pending_positions = await self.position_manager.get_pending_positions()
        
        # 2. Fetch all OPEN positions (To check for active closing orders)
        open_positions = await self.position_manager.get_all_open_positions()
        
        recovery_count = 0
        
        # --- Recover Pending Entries ---
        for pos in pending_positions:
            logger.info(f"Reconciling PENDING position {pos.trade_id} for {pos.symbol}...")
            
            # Try to find the order on the exchange using the deterministic trade_id (ClientOrderId)
            order = await self.exchange_api.fetch_order_by_client_id(pos.symbol, pos.trade_id)
            
            if not order:
                # If order not found and it's old (> 5 min), mark as failed.
                # If it's very recent, it might be a lag, but on startup, we assume > 5 min.
                logger.warning(f"Order {pos.trade_id} not found on exchange. Marking FAILED.")
                await self.position_manager.mark_position_failed(pos.symbol, pos.trade_id, "Reconciliation: Order Not Found")
                continue
            
            # Reconstruct Context
            profile = self.config.profiles.get(self.config.default_profile)
            
            ctx = ActiveOrderContext(
                trade_id=pos.trade_id,
                symbol=pos.symbol,
                side=pos.side,
                total_quantity=pos.quantity if pos.quantity > ZERO else to_decimal(order['filled']),
                initial_price=pos.decision_price or to_decimal(order['price']),
                strategy_metadata={},
                current_order_id=order['id'],
                current_price=to_decimal(order['price']),
                profile=profile,
                intent='OPEN',
                cumulative_filled=to_decimal(order['filled']),
                start_time=time.time() # Reset timer
            )
            
            await self.register_order(ctx)
            recovery_count += 1

        # --- Recover Active Closing Orders ---
        for pos in open_positions:
            if pos.closing_order_id:
                logger.info(f"Reconciling CLOSING order {pos.closing_order_id} for {pos.symbol}...")
                order = await self.exchange_api.fetch_order(pos.closing_order_id, pos.symbol)
                
                if not order or order['status'] in ['CANCELED', 'REJECTED', 'EXPIRED']:
                    # If closing order is dead, clear the flag so RiskManager can retry if needed
                    # For now, we just log. PositionMonitor will handle new exits.
                    continue
                
                if order['status'] == 'FILLED':
                    # Handle missed fill event
                    fees = to_decimal(order.get('fee', {}).get('cost', 0))
                    await self.position_manager.confirm_position_close(
                        pos.symbol, to_decimal(order['average']), to_decimal(order['filled']), fees, "Reconciliation: Found Filled Close"
                    )
                    continue

                # If still OPEN, resume monitoring
                profile = self.config.profiles.get('neutral')
                ctx = ActiveOrderContext(
                    trade_id=f"close_{pos.trade_id}_recovered",
                    symbol=pos.symbol,
                    side='SELL' if pos.side == 'BUY' else 'BUY',
                    total_quantity=to_decimal(order['amount'] if 'amount' in order else pos.quantity),
                    initial_price=to_decimal(order['price']),
                    strategy_metadata={},
                    current_order_id=order['id'],
                    current_price=to_decimal(order['price']),
                    profile=profile,
                    intent='CLOSE',
                    cumulative_filled=to_decimal(order['filled']),
                    start_time=time.time()
                )
                await self.register_order(ctx)
                recovery_count += 1

        logger.info(f"Reconciliation Complete. Recovered {recovery_count} active contexts.")

    async def register_order(self, context: ActiveOrderContext):
        """Registers a new order to be managed by the service."""
        async with self._lock:
            self._active_orders[context.trade_id] = context
        logger.info("Registered order for lifecycle management", 
                    trade_id=context.trade_id, 
                    symbol=context.symbol, 
                    intent=context.intent,
                    order_id=context.current_order_id)

    async def _monitor_loop(self):
        while self._running:
            try:
                if not self._active_orders:
                    await asyncio.sleep(0.5)
                    continue

                # Snapshot keys to iterate safely
                async with self._lock:
                    trade_ids = list(self._active_orders.keys())

                now = time.time()
                for trade_id in trade_ids:
                    await self._process_order(trade_id, now)
                
                await asyncio.sleep(0.1) # Fast tick for adaptive polling
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in OrderLifecycleService loop", error=str(e), exc_info=True)
                await asyncio.sleep(5)

    async def _process_order(self, trade_id: str, now: float):
        async with self._lock:
            ctx = self._active_orders.get(trade_id)
        
        if not ctx: return

        # Adaptive Polling: Poll new orders frequently (1s), older orders less frequently (3s)
        age = now - ctx.start_time
        poll_interval = 1.0 if age < 10 else 3.0
        
        if (now - ctx.last_poll_time) < poll_interval:
            return

        ctx.last_poll_time = now

        try:
            # 1. Fetch Status
            order_status = await self.exchange_api.fetch_order(ctx.current_order_id, ctx.symbol)
            if not order_status:
                return

            status = order_status.get('status', 'UNKNOWN')
            
            # 2. Handle Terminal States
            if status == 'FILLED':
                await self._finalize_fill(ctx, order_status)
                return
            
            elif status in ['CANCELED', 'REJECTED', 'EXPIRED']:
                await self._handle_failure(ctx, f"Order {status}")
                return

            # 3. Handle Open State (Chasing & Timeout)
            elif status == 'OPEN':
                # Check Timeout
                if age > self.config.order_fill_timeout_seconds:
                    await self._handle_timeout(ctx)
                    return
                
                # Check Chasing
                if ctx.profile.use_chasing:
                    await self._check_and_chase(ctx, now)

        except Exception as e:
            logger.error(f"Error processing order {trade_id}", error=str(e))

    async def _finalize_fill(self, ctx: ActiveOrderContext, order_status: Dict[str, Any]):
        """Handles a completely filled order (or chain of orders)."""
        filled_qty = to_decimal(order_status.get('filled', 0.0))
        avg_price = to_decimal(order_status.get('average', ctx.current_price))
        
        final_filled = ctx.cumulative_filled + filled_qty
        final_avg_price = avg_price
        total_fees = ctx.cumulative_fees + self._extract_fee_cost(order_status)

        try:
            if ctx.intent == 'OPEN':
                # Adjust qty for fees if BUY and fee in base asset
                confirmed_qty = final_filled
                if ctx.side == 'BUY':
                    base_asset = ctx.symbol.split('/')[0]
                    fee_currency = order_status.get('fee', {}).get('currency')
                    if fee_currency == base_asset:
                        confirmed_qty = max(ZERO, final_filled - total_fees)

                pos = await self.position_manager.confirm_position_open(
                    ctx.symbol, ctx.trade_id, confirmed_qty, final_avg_price, 
                    stop_loss=ZERO, take_profit=ZERO, fees=total_fees
                )
                
                if pos:
                    await self.event_bus.publish(TradeCompletedEvent(position=pos))
                    logger.info("Order lifecycle complete: OPEN FILLED", symbol=ctx.symbol, qty=str(confirmed_qty))
                else:
                    logger.error("Failed to confirm position open in DB", trade_id=ctx.trade_id)
            
            elif ctx.intent == 'CLOSE':
                pos = await self.position_manager.confirm_position_close(
                    ctx.symbol, final_avg_price, final_filled, total_fees, reason="Strategy Signal"
                )
                if pos:
                    logger.info("Order lifecycle complete: CLOSE FILLED", symbol=ctx.symbol, qty=str(final_filled))
                else:
                    logger.error("Failed to confirm position close in DB", trade_id=ctx.trade_id)

        except Exception as e:
            logger.critical("Critical error finalizing fill", error=str(e))
        
        await self._remove_context(ctx.trade_id)

    async def _handle_failure(self, ctx: ActiveOrderContext, reason: str):
        logger.warning("Order lifecycle failed", reason=reason, trade_id=ctx.trade_id)
        if ctx.intent == 'OPEN':
            await self.position_manager.mark_position_failed(ctx.symbol, ctx.trade_id, reason)
        await self._remove_context(ctx.trade_id)

    async def _handle_timeout(self, ctx: ActiveOrderContext):
        logger.info("Order timed out", trade_id=ctx.trade_id)
        
        try:
            await self.exchange_api.cancel_order(ctx.current_order_id, ctx.symbol)
        except Exception:
            pass

        if ctx.profile.execute_on_timeout:
            remaining = ctx.total_quantity - ctx.cumulative_filled
            if remaining > ZERO:
                try:
                    logger.info("Placing fallback market order", symbol=ctx.symbol)
                    market_order = await self.exchange_api.place_order(
                        ctx.symbol, ctx.side, 'MARKET', remaining, 
                        extra_params={'clientOrderId': f"{ctx.trade_id}_fallback"}
                    )
                    if market_order and market_order.get('status') == 'FILLED':
                        await self._finalize_fill(ctx, market_order)
                        return
                except Exception as e:
                    logger.error("Fallback market order failed", error=str(e))

        await self._handle_failure(ctx, "Timeout")

    async def _check_and_chase(self, ctx: ActiveOrderContext, now: float):
        if ctx.chase_attempts >= ctx.profile.max_chase_attempts:
            return
        
        next_chase_time = ctx.start_time + ((ctx.chase_attempts + 1) * ctx.profile.chase_interval_seconds)
        if now < next_chase_time:
            return

        market_price_float = self.latest_prices.get(ctx.symbol)
        if not market_price_float:
            return
        market_price = to_decimal(market_price_float)

        is_buy = ctx.side == 'BUY'
        current_price = ctx.current_price
        
        should_chase = False
        if is_buy and market_price > current_price:
            should_chase = True
        elif not is_buy and market_price < current_price:
            should_chase = True
            
        if should_chase:
            aggro = to_decimal(ctx.profile.chase_aggressiveness_pct)
            new_price = market_price * (ONE + aggro) if is_buy else market_price * (ONE - aggro)
            
            slippage_pct = abs(new_price - ctx.initial_price) / ctx.initial_price
            max_slippage = to_decimal(ctx.profile.max_slippage_pct)
            
            if slippage_pct > max_slippage:
                logger.info("Max chase slippage reached, holding position.", symbol=ctx.symbol)
                return

            logger.info("Chasing order", symbol=ctx.symbol, old_price=str(current_price), new_price=str(new_price))
            
            try:
                await self.exchange_api.cancel_order(ctx.current_order_id, ctx.symbol)
                
                remaining = ctx.total_quantity - ctx.cumulative_filled
                
                ctx.chase_attempts += 1
                new_order = await self.exchange_api.place_order(
                    ctx.symbol, ctx.side, 'LIMIT', remaining, price=new_price,
                    extra_params={'clientOrderId': f"{ctx.trade_id}_chase_{ctx.chase_attempts}"}
                )
                
                ctx.current_order_id = new_order['orderId']
                ctx.current_price = new_price
                
                if ctx.intent == 'OPEN':
                    await self.position_manager.update_pending_order_id(ctx.symbol, ctx.trade_id, new_order['orderId'])
                elif ctx.intent == 'CLOSE':
                    await self.position_manager.register_closing_order(ctx.symbol, new_order['orderId'])
                
            except Exception as e:
                logger.error("Failed to chase order", error=str(e))

    async def _remove_context(self, trade_id: str):
        async with self._lock:
            if trade_id in self._active_orders:
                del self._active_orders[trade_id]

    def _extract_fee_cost(self, order_res: Dict[str, Any]) -> Decimal:
        if not order_res or 'fee' not in order_res or not order_res['fee']:
            return ZERO
        return to_decimal(order_res['fee'].get('cost', 0.0))
