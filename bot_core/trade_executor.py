import uuid
import asyncio
import json
import os
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel

from bot_core.logger import get_logger
from bot_core.config import BotConfig
from bot_core.exchange_api import ExchangeAPI, BotInsufficientFundsError, BotInvalidOrderError
from bot_core.position_manager import PositionManager, Position
from bot_core.risk_manager import RiskManager
from bot_core.order_sizer import OrderSizer
from bot_core.order_lifecycle_manager import OrderLifecycleManager
from bot_core.monitoring import AlertSystem
from bot_core.data_handler import DataHandler
from bot_core.common import TradeSignal
from bot_core.event_system import EventBus, TradeCompletedEvent

logger = get_logger(__name__)

class TradeExecutionResult(BaseModel):
    symbol: str
    action: str
    quantity: float
    price: float
    fees: float
    order_id: str
    status: str
    metadata: Dict[str, Any] = {}

class TradeExecutor:
    """
    Institutional-grade execution engine.
    Enforces strict pre-trade checks, deterministic sizing, and robust lifecycle management.
    """
    def __init__(self,
                 config: BotConfig,
                 exchange_api: ExchangeAPI,
                 position_manager: PositionManager,
                 risk_manager: RiskManager,
                 order_sizer: OrderSizer,
                 order_lifecycle_manager: OrderLifecycleManager,
                 alert_system: AlertSystem,
                 shared_latest_prices: Dict[str, float],
                 market_details: Dict[str, Dict[str, Any]],
                 data_handler: DataHandler,
                 event_bus: EventBus):
        self.config = config
        self.exchange_api = exchange_api
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.order_sizer = order_sizer
        self.order_lifecycle_manager = order_lifecycle_manager
        self.alert_system = alert_system
        self.latest_prices = shared_latest_prices
        self.market_details = market_details
        self.data_handler = data_handler
        self.event_bus = event_bus
        
        self._symbol_locks: Dict[str, asyncio.Lock] = {}
        
        # Ensure recovery directory exists
        os.makedirs("recovery", exist_ok=True)
        
        logger.info("TradeExecutor initialized with strict risk enforcement.")

    def _get_lock(self, symbol: str) -> asyncio.Lock:
        if symbol not in self._symbol_locks:
            self._symbol_locks[symbol] = asyncio.Lock()
        return self._symbol_locks[symbol]

    async def execute_trade_signal(self, signal: TradeSignal, df_with_indicators: Any, position: Optional[Position]) -> Optional[TradeExecutionResult]:
        """
        Router for trade signals. Ensures atomic execution per symbol.
        """
        # 0. Signal TTL Check
        signal_age = datetime.now(timezone.utc) - signal.generated_at
        if signal_age > timedelta(seconds=60):
            logger.warning("Signal rejected: TTL expired.", symbol=signal.symbol, age_seconds=signal_age.total_seconds())
            return None

        symbol = signal.symbol
        async with self._get_lock(symbol):
            current_price = self.latest_prices.get(symbol)
            if not current_price:
                logger.warning("Execution skipped: No price data available.", symbol=symbol)
                return None

            # --- Entry Logic ---
            if not position:
                if signal.action in ['BUY', 'SELL']:
                    return await self._handle_entry(signal, current_price, df_with_indicators)
            
            # --- Exit Logic ---
            elif position:
                is_close_long = (signal.action == 'SELL') and position.side == 'BUY'
                is_close_short = (signal.action == 'BUY') and position.side == 'SELL'
                
                if is_close_long or is_close_short:
                    await self.close_position(position, "Strategy Signal")
                    return None
        return None

    async def _handle_entry(self, signal: TradeSignal, current_price: float, df: Any) -> Optional[TradeExecutionResult]:
        symbol = signal.symbol
        side = signal.action

        # 1. Market Pre-Flight Check
        if not await self._check_liquidity(symbol, current_price):
            return None

        # 2. Risk Gatekeeper
        active_positions = await self.position_manager.get_all_active_positions()
        # Await the async risk check (which might save state)
        allowed, reason = await self.risk_manager.validate_entry(symbol, active_positions)
        if not allowed:
            logger.info("Entry rejected by Risk Manager.", symbol=symbol, reason=reason)
            return None

        # 3. Sizing Calculation
        portfolio_equity = self.position_manager.get_portfolio_value(self.latest_prices, active_positions)
        stop_loss = self.risk_manager.calculate_stop_loss(side, current_price, df, market_regime=signal.regime)
        
        confidence_threshold = signal.metadata.get('effective_threshold')
        ideal_quantity = self.risk_manager.calculate_position_size(
            symbol=symbol,
            portfolio_equity=portfolio_equity, 
            entry_price=current_price, 
            stop_loss_price=stop_loss, 
            open_positions=active_positions,
            market_regime=signal.regime,
            confidence=signal.confidence,
            confidence_threshold=confidence_threshold,
            model_metrics=signal.metadata.get('metrics')
        )

        # 4. Wallet & Exchange Constraints
        market_details = self.market_details.get(symbol)
        if not market_details:
            logger.error("Market details missing, aborting entry.", symbol=symbol)
            return None

        # Cap quantity based on actual available wallet balance (Purchasing Power)
        purchasing_power_qty = await self._get_purchasing_power_qty(symbol, side, current_price)
        capped_quantity = min(ideal_quantity, purchasing_power_qty)

        final_quantity = self.order_sizer.adjust_order_quantity(symbol, capped_quantity, current_price, market_details)
        
        if final_quantity <= 0:
            logger.info("Quantity adjusted to zero (insufficient funds or below min limits).", symbol=symbol)
            return None

        # 5. Order Construction
        order_type = self.config.execution.default_order_type
        limit_price = None
        if order_type == 'LIMIT':
            limit_price = await self._calculate_limit_price(symbol, side, current_price, df)

        # 6. Execution
        trade_id = str(uuid.uuid4())
        await self.position_manager.create_pending_position(
            symbol, side, trade_id, trade_id, 
            decision_price=current_price, 
            strategy_metadata=signal.metadata
        )

        try:
            extra_params = {'clientOrderId': trade_id}
            
            # --- Adaptive Execution Logic ---
            # If confidence is high (> 0.8), we are aggressive and allow Taker orders (disable postOnly)
            # Otherwise, we respect the config's post_only setting.
            is_aggressive = signal.confidence >= 0.8
            if self.config.execution.post_only and order_type == 'LIMIT' and not is_aggressive:
                extra_params['postOnly'] = True
            # --------------------------------

            order_result = await self.exchange_api.place_order(
                symbol, side, order_type, final_quantity, price=limit_price, extra_params=extra_params
            )
        except (BotInsufficientFundsError, BotInvalidOrderError) as e:
            logger.error("Order placement failed.", error=str(e))
            await self.position_manager.mark_position_failed(symbol, trade_id, str(e))
            return None
        except Exception as e:
            logger.error("Unexpected execution error.", error=str(e))
            await self.position_manager.mark_position_failed(symbol, trade_id, f"Execution Error: {str(e)}")
            return None

        # 7. Lifecycle Management
        exchange_order_id = order_result.get('orderId', trade_id)
        if exchange_order_id != trade_id:
            await self.position_manager.update_pending_order_id(symbol, trade_id, exchange_order_id)

        async def _on_replace(old_id, new_id):
            await self.position_manager.update_pending_order_id(symbol, old_id, new_id)

        final_state = await self.order_lifecycle_manager.manage(
            initial_order=order_result, 
            symbol=symbol, 
            side=side, 
            quantity=final_quantity,
            initial_price=limit_price,
            market_details=market_details,
            on_order_replace=_on_replace,
            trade_id=trade_id
        )

        return await self._finalize_entry(symbol, side, exchange_order_id, final_state, df, signal)

    async def _finalize_entry(self, symbol: str, side: str, order_id: str, order_state: Optional[Dict], df: Any, signal: TradeSignal) -> Optional[TradeExecutionResult]:
        if not order_state:
            await self.position_manager.mark_position_failed(symbol, order_id, "Lifecycle Timeout")
            return None

        fill_qty = order_state.get('filled', 0.0)
        status = order_state.get('status', 'UNKNOWN')

        if fill_qty > 0 and status in ['FILLED', 'PARTIALLY_FILLED']:
            avg_price = order_state.get('average')
            if not avg_price:
                return None

            fees = self._extract_fee(order_state, avg_price, fill_qty)
            
            # Adjust quantity if fees were deducted from the base asset (common in crypto buys)
            confirmed_qty = fill_qty
            if side == 'BUY':
                base_asset = symbol.split('/')[0]
                fee_currency = order_state.get('fee', {}).get('currency')
                if fee_currency == base_asset:
                    confirmed_qty = max(0.0, fill_qty - fees)

            stop_loss = self.risk_manager.calculate_stop_loss(side, avg_price, df, market_regime=signal.regime)
            take_profit = self.risk_manager.calculate_take_profit(
                side, avg_price, stop_loss, market_regime=signal.regime,
                confidence=signal.confidence, confidence_threshold=signal.metadata.get('effective_threshold')
            )

            try:
                await self.position_manager.confirm_position_open(
                    symbol, order_id, confirmed_qty, avg_price, stop_loss, take_profit, fees=fees
                )
            except Exception as e:
                # CRITICAL: Order filled on exchange but DB update failed.
                # We must persist this state to disk to avoid a zombie position.
                await self._handle_db_failure(symbol, order_id, confirmed_qty, avg_price, side, str(e))
                return None

            return TradeExecutionResult(
                symbol=symbol, action=side, quantity=confirmed_qty, price=avg_price,
                fees=fees, order_id=order_id, status=status, metadata=signal.metadata
            )
        
        elif status == 'OPEN':
            logger.critical("Order stuck OPEN after lifecycle management.", order_id=order_id)
            await self.alert_system.send_alert('critical', f"Zombie Order {order_id} for {symbol}", details={'order_id': order_id})
            return None
        else:
            await self.position_manager.mark_position_failed(symbol, order_id, f"Failed: {status}")
            return None

    async def _handle_db_failure(self, symbol: str, order_id: str, qty: float, price: float, side: str, error: str):
        """Persists trade details to a recovery file if DB update fails."""
        logger.critical("DB Update Failed for Filled Order! Initiating Recovery Protocol.", order_id=order_id, error=error)
        
        recovery_data = {
            'symbol': symbol,
            'order_id': order_id,
            'side': side,
            'quantity': qty,
            'price': price,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'error': error
        }
        
        filename = f"recovery/recovery_{order_id}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(recovery_data, f, indent=2)
            logger.info(f"Recovery data saved to {filename}")
            await self.alert_system.send_alert(
                'critical', 
                f"DB Failure for {symbol}. Recovery file created.", 
                details={'file': filename, 'order_id': order_id}
            )
        except Exception as e:
            logger.critical("Failed to write recovery file! Position is effectively orphaned.", error=str(e))

    async def close_position(self, position: Position, reason: str):
        async with self._get_lock(position.symbol):
            # Verify position is still open in DB
            fresh_pos = await self.position_manager.get_open_position(position.symbol)
            if not fresh_pos:
                return

            close_side = 'SELL' if position.side == 'BUY' else 'BUY'
            current_price = self.latest_prices.get(position.symbol, position.entry_price)
            market_details = self.market_details.get(position.symbol)

            if not market_details:
                logger.error("Cannot close position: Market details missing.", symbol=position.symbol)
                return

            # Adjust quantity for precision
            close_qty = self.order_sizer.adjust_order_quantity(position.symbol, position.quantity, current_price, market_details)
            
            # Dust handling
            if close_qty <= 0:
                await self.position_manager.close_position(position.symbol, current_price, f"{reason} (Dust)")
                return

            # Order Placement
            order_type = self.config.execution.default_order_type
            limit_price = None
            if order_type == 'LIMIT':
                limit_price = await self._calculate_limit_price(position.symbol, close_side, current_price, None)
            else:
                order_type = 'MARKET'

            try:
                extra_params = {'reduceOnly': True} if order_type == 'LIMIT' else {}
                order_result = await self.exchange_api.place_order(
                    position.symbol, close_side, order_type, close_qty, price=limit_price, extra_params=extra_params
                )
            except Exception as e:
                logger.error("Close order placement failed.", error=str(e))
                # Fallback: Try to close via Market if Limit failed, or handle phantom
                return

            final_state = await self.order_lifecycle_manager.manage(
                initial_order=order_result, 
                symbol=position.symbol, 
                side=close_side, 
                quantity=close_qty,
                initial_price=limit_price,
                market_details=market_details
            )

            if final_state and final_state.get('filled', 0) > 0:
                avg_price = final_state['average']
                fees = self._extract_fee(final_state, avg_price, final_state['filled'])
                
                # Check if fully closed (within tolerance)
                if final_state['filled'] >= (position.quantity * 0.99):
                    closed_pos = await self.position_manager.close_position(position.symbol, avg_price, reason, actual_filled_qty=final_state['filled'], fees=fees)
                    if closed_pos:
                        # Await the async update
                        await self.risk_manager.update_trade_outcome(closed_pos.symbol, closed_pos.pnl)
                        # Emit Event for Online Learning
                        await self.event_bus.publish(TradeCompletedEvent(position=closed_pos))
                else:
                    await self.position_manager.reduce_position(position.symbol, final_state['filled'], avg_price, reason, fees=fees)
            else:
                logger.error("Failed to close position after lifecycle.", symbol=position.symbol)

    # --- Helpers ---

    async def _check_liquidity(self, symbol: str, current_price: float) -> bool:
        # Optimization: Use cached order book from DataHandler instead of blocking network call
        book = self.data_handler.get_latest_order_book(symbol)
        if not book:
            # Fallback to fetch if cache is empty (rare)
            try:
                book = await self.exchange_api.fetch_order_book(symbol, limit=5)
            except Exception:
                return False
        
        if not book or not book.get('bids') or not book.get('asks'):
            return False
            
        best_bid = book['bids'][0][0]
        best_ask = book['asks'][0][0]
        spread = (best_ask - best_bid) / current_price
        
        if spread > self.config.execution.max_entry_spread_pct:
            logger.warning("Spread too high.", symbol=symbol, spread=f"{spread:.4%}")
            return False
        return True

    async def _get_purchasing_power_qty(self, symbol: str, side: str, price: float) -> float:
        """Calculates max quantity affordable with FREE balance."""
        try:
            base, quote = symbol.split('/')
            balances = await self.exchange_api.get_balance()
            
            if side == 'BUY':
                # Check Quote Asset (e.g. USDT)
                free_quote = balances.get(quote, {}).get('free', 0.0)
                if free_quote <= 0: return 0.0
                # Reserve fees (approximate)
                max_cost = free_quote / (1 + self.config.exchange.taker_fee_pct)
                return max_cost / price
            else:
                # Check Base Asset (e.g. BTC)
                free_base = balances.get(base, {}).get('free', 0.0)
                return free_base
        except Exception as e:
            logger.error("Failed to check purchasing power.", error=str(e))
            return 0.0

    async def _calculate_limit_price(self, symbol: str, side: str, current_price: float, df: Any) -> float:
        try:
            ticker = await self.exchange_api.get_ticker_data(symbol)
        except Exception:
            ticker = {'last': current_price}
            
        bid = ticker.get('bid') or current_price
        ask = ticker.get('ask') or current_price
        ref_price = bid if side == 'BUY' else ask

        offset = 0.0
        if self.config.execution.limit_offset_type == 'ATR' and df is not None:
            atr_col = self.config.risk_management.atr_column_name
            if atr_col in df.columns:
                atr = df[atr_col].iloc[-1]
                offset = atr * self.config.execution.limit_offset_atr_multiplier
        
        if offset == 0:
            offset = ref_price * self.config.execution.limit_price_offset_pct

        return ref_price - offset if side == 'BUY' else ref_price + offset

    def _extract_fee(self, order_state: Dict, price: float, qty: float) -> float:
        if order_state.get('fee') and 'cost' in order_state['fee']:
            return float(order_state['fee']['cost'])
        return price * qty * self.config.exchange.taker_fee_pct
