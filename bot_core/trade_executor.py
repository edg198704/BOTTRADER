import pandas as pd
import uuid
import asyncio
from typing import Dict, Any, Optional
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
    Handles the entire lifecycle of executing a trade, from signal to position management.
    Encapsulates risk checks, sizing, order placement, and post-execution state updates.
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
                 data_handler: DataHandler):
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
        
        self._symbol_locks: Dict[str, asyncio.Lock] = {}
        logger.info("TradeExecutor initialized.")

    def _get_lock(self, symbol: str) -> asyncio.Lock:
        if symbol not in self._symbol_locks:
            self._symbol_locks[symbol] = asyncio.Lock()
        return self._symbol_locks[symbol]

    async def execute_trade_signal(self, signal: TradeSignal, df_with_indicators: pd.DataFrame, position: Optional[Position]) -> Optional[TradeExecutionResult]:
        """
        Processes a trading signal to open or close a position.
        Acquires a symbol-specific lock to ensure atomicity.
        Returns a TradeExecutionResult if a trade was executed, else None.
        """
        symbol = signal.symbol
        async with self._get_lock(symbol):
            current_price = self.latest_prices.get(symbol)
            if not current_price:
                logger.warning("Cannot execute signal: No price data.", symbol=symbol)
                return None

            # --- Handle Opening a New Position ---
            if not position:
                if signal.action not in ['BUY', 'SELL']:
                    return None
                
                # 1. Liquidity Check
                if not await self._check_liquidity(symbol, current_price):
                    logger.warning("Trade aborted due to poor liquidity/spread.", symbol=symbol)
                    return None

                return await self._execute_open_position(signal, current_price, df_with_indicators)
            
            # --- Handle Closing an Existing Position ---
            elif position:
                is_close_long = (signal.action == 'SELL') and position.side == 'BUY'
                is_close_short = (signal.action == 'BUY') and position.side == 'SELL'
                
                if is_close_long or is_close_short:
                    await self.close_position(position, "Strategy Signal")
                    return None
        return None

    async def _check_liquidity(self, symbol: str, current_price: float) -> bool:
        try:
            book = await self.exchange_api.fetch_order_book(symbol, limit=5)
            if not book or not book.get('bids') or not book.get('asks'):
                return False
            best_bid = book['bids'][0][0]
            best_ask = book['asks'][0][0]
            if best_bid <= 0 or best_ask <= 0:
                return False
            spread = (best_ask - best_bid) / current_price
            if spread > self.config.execution.max_entry_spread_pct:
                logger.warning("Spread too high for entry", symbol=symbol, spread=f"{spread:.4%}", max_allowed=f"{self.config.execution.max_entry_spread_pct:.4%}")
                return False
            return True
        except Exception as e:
            logger.error("Liquidity check failed", symbol=symbol, error=str(e))
            return False

    def _calculate_or_extract_fee(self, order_state: Dict[str, Any], price: float, quantity: float) -> float:
        if order_state and 'fee' in order_state and order_state['fee']:
            fee_data = order_state['fee']
            if 'cost' in fee_data and fee_data['cost'] is not None:
                return float(fee_data['cost'])
        return price * quantity * self.config.exchange.taker_fee_pct

    def _calculate_limit_price(self, symbol: str, side: str, current_price: float, ticker: Dict[str, float], df: Optional[pd.DataFrame] = None) -> float:
        bid = ticker.get('bid')
        ask = ticker.get('ask')
        last = ticker.get('last', current_price)
        ref_price = (bid if bid else last) if side == 'BUY' else (ask if ask else last)
        
        offset_val = 0.0
        if self.config.execution.limit_offset_type == 'ATR' and df is not None:
            atr_col = self.config.risk_management.atr_column_name
            if atr_col in df.columns:
                atr = df[atr_col].iloc[-1]
                offset_val = atr * self.config.execution.limit_offset_atr_multiplier if atr > 0 else ref_price * self.config.execution.limit_price_offset_pct
            else:
                offset_val = ref_price * self.config.execution.limit_price_offset_pct
        else:
            offset_val = ref_price * self.config.execution.limit_price_offset_pct

        return ref_price - offset_val if side == 'BUY' else ref_price + offset_val

    async def _retry_entry_with_smart_balance(self, symbol: str, side: str, order_type: str, limit_price: float, trade_id: str, market_details: Dict) -> Optional[Dict]:
        try:
            base_asset, quote_asset = symbol.split('/')
            target_asset = quote_asset if side == 'BUY' else base_asset
            balances = await self.exchange_api.get_balance()
            free_balance = balances.get(target_asset, {}).get('free', 0.0)
            if free_balance <= 0: return None

            new_qty = 0.0
            if side == 'BUY':
                fee_rate = self.config.exchange.taker_fee_pct
                max_cost = free_balance / (1 + fee_rate)
                new_qty = max_cost / limit_price
            else:
                new_qty = free_balance

            adjusted_qty = self.order_sizer.adjust_order_quantity(symbol, new_qty, limit_price, market_details)
            if adjusted_qty <= 0: return None

            logger.info("Retrying entry with smart balance.", asset=target_asset, balance=free_balance, new_qty=adjusted_qty)
            extra_params = {'clientOrderId': trade_id}
            if self.config.execution.post_only and order_type == 'LIMIT':
                extra_params['postOnly'] = True

            return await self.exchange_api.place_order(symbol, side, order_type, adjusted_qty, price=limit_price, extra_params=extra_params)
        except Exception as e:
            logger.error("Smart Retry exception", error=str(e))
            return None

    async def _execute_open_position(self, signal: TradeSignal, current_price: float, df: pd.DataFrame) -> Optional[TradeExecutionResult]:
        symbol = signal.symbol
        side = signal.action
        
        # 1. Risk Gatekeeper Check
        active_positions = await self.position_manager.get_all_active_positions()
        allowed, reason = self.risk_manager.validate_entry(symbol, active_positions)
        if not allowed:
            logger.info("Trade rejected by Risk Manager", symbol=symbol, reason=reason)
            return None

        # 2. Calculate Sizing
        portfolio_equity = self.position_manager.get_portfolio_value(self.latest_prices, active_positions)
        stop_loss = self.risk_manager.calculate_stop_loss(side, current_price, df, market_regime=signal.regime)
        
        confidence_threshold = signal.metadata.get('effective_threshold')
        if confidence_threshold is None:
            confidence_threshold = getattr(self.config.strategy.params, 'confidence_threshold', None)

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
        
        market_details = self.market_details.get(symbol)
        if not market_details:
            logger.error("Market details missing.", symbol=symbol)
            return None

        final_quantity = self.order_sizer.adjust_order_quantity(symbol, ideal_quantity, current_price, market_details)
        if final_quantity <= 0:
            return None

        # 3. Prepare Order
        order_type = self.config.execution.default_order_type
        limit_price = None
        if order_type == 'LIMIT':
            try:
                ticker = await self.exchange_api.get_ticker_data(symbol)
            except Exception:
                ticker = {'last': current_price}
            limit_price = self._calculate_limit_price(symbol, side, current_price, ticker, df)

        # 4. Create Pending DB Record (The Intent)
        trade_id = str(uuid.uuid4())
        await self.position_manager.create_pending_position(
            symbol, side, trade_id, trade_id, 
            decision_price=current_price, 
            strategy_metadata=signal.metadata
        )

        # 5. Execute Order
        order_result = None
        try:
            extra_params = {'clientOrderId': trade_id}
            if self.config.execution.post_only and order_type == 'LIMIT':
                extra_params['postOnly'] = True

            order_result = await self.exchange_api.place_order(
                symbol, side, order_type, final_quantity, price=limit_price, extra_params=extra_params
            )
        except BotInsufficientFundsError:
            order_result = await self._retry_entry_with_smart_balance(symbol, side, order_type, limit_price, trade_id, market_details)
            if not order_result:
                await self.position_manager.mark_position_failed(symbol, trade_id, "Insufficient Funds")
                return None
        except BotInvalidOrderError as e:
            await self.position_manager.mark_position_failed(symbol, trade_id, f"Invalid Order: {str(e)}")
            return None
        except Exception as e:
            logger.error("Order placement error", error=str(e))
            await self.position_manager.mark_position_failed(symbol, trade_id, f"Placement Error: {str(e)}")
            return None

        if not order_result:
            return None

        # 6. Update DB with Real Order ID
        exchange_order_id = order_result.get('orderId')
        if exchange_order_id and exchange_order_id != trade_id:
            await self.position_manager.update_pending_order_id(symbol, trade_id, exchange_order_id)
            current_order_id = exchange_order_id
        else:
            current_order_id = trade_id

        async def _on_order_replace(old_id: str, new_id: str):
            await self.position_manager.update_pending_order_id(symbol, old_id, new_id)

        # 7. Manage Lifecycle (Chase/Fill)
        final_order_state = await self.order_lifecycle_manager.manage(
            initial_order=order_result, 
            symbol=symbol, 
            side=side, 
            quantity=final_quantity,
            initial_price=limit_price,
            market_details=market_details,
            on_order_replace=_on_order_replace,
            trade_id=trade_id
        )
        
        # 8. Finalize (Confirm Open or Fail)
        return await self._finalize_entry(symbol, side, current_order_id, final_order_state, df, signal.regime, signal.confidence, confidence_threshold)

    async def _finalize_entry(self, symbol: str, side: str, order_id: str, order_state: Optional[Dict], df: pd.DataFrame, regime: Optional[str], confidence: float, threshold: Optional[float]) -> Optional[TradeExecutionResult]:
        fill_quantity = order_state.get('filled', 0.0) if order_state else 0.0
        final_status = order_state.get('status') if order_state else 'UNKNOWN'
        
        if fill_quantity > 0:
            fill_price = order_state.get('average')
            if not fill_price or fill_price <= 0:
                return None

            fees = self._calculate_or_extract_fee(order_state, fill_price, fill_quantity)
            confirmed_quantity = fill_quantity
            
            # Adjust quantity for fees if paid in base asset
            if side == 'BUY':
                fee_currency = None
                if order_state.get('fee') and 'currency' in order_state['fee']:
                    fee_currency = order_state['fee']['currency']
                base_asset = symbol.split('/')[0]
                if fee_currency == base_asset:
                    confirmed_quantity = max(0.0, fill_quantity - fees)

            final_stop_loss = self.risk_manager.calculate_stop_loss(side, fill_price, df, market_regime=regime)
            final_take_profit = self.risk_manager.calculate_take_profit(
                side, fill_price, final_stop_loss, market_regime=regime,
                confidence=confidence, confidence_threshold=threshold
            )
            
            await self.position_manager.confirm_position_open(
                symbol, order_id, confirmed_quantity, fill_price, 
                final_stop_loss, final_take_profit, fees=fees
            )
            
            return TradeExecutionResult(
                symbol=symbol,
                action=side,
                quantity=confirmed_quantity,
                price=fill_price,
                fees=fees,
                order_id=order_id,
                status='FILLED',
                metadata={'regime': regime, 'confidence': confidence}
            )
            
        elif final_status == 'OPEN':
            logger.critical("Order stuck OPEN. Manual intervention required.", order_id=order_id)
            await self.alert_system.send_alert(
                level='critical',
                message=f"🚨 ZOMBIE ORDER: {order_id} for {symbol} stuck OPEN.",
                details={'symbol': symbol, 'order_id': order_id}
            )
            return None
        else:
            await self.position_manager.mark_position_failed(symbol, order_id, f"Lifecycle Failed: {final_status}")
            return None

    async def close_position(self, position: Position, reason: str):
        async with self._get_lock(position.symbol):
            fresh_pos = await self.position_manager.get_open_position(position.symbol)
            if not fresh_pos:
                return

            close_side = 'SELL' if position.side == 'BUY' else 'BUY'
            market_details = self.market_details.get(position.symbol)
            current_price = self.latest_prices.get(position.symbol)

            if not market_details:
                close_quantity = position.quantity
            else:
                close_quantity = self.order_sizer.adjust_order_quantity(position.symbol, position.quantity, current_price or 0.0, market_details)

            if close_quantity <= 0:
                # Handle dust
                estimated_value = position.quantity * (current_price or 0.0)
                if estimated_value < 1.0:
                    await self.position_manager.close_position(position.symbol, current_price or position.entry_price, f"{reason} (Dust)")
                return

            order_type = self.config.execution.default_order_type
            limit_price = None
            if order_type == 'LIMIT':
                try:
                    ticker = await self.exchange_api.get_ticker_data(position.symbol)
                except Exception:
                    ticker = {'last': current_price}
                df = None
                if self.config.execution.limit_offset_type == 'ATR':
                    df = self.data_handler.get_market_data(position.symbol)
                limit_price = self._calculate_limit_price(position.symbol, close_side, current_price, ticker, df)
            else:
                order_type = 'MARKET'

            order_result = None
            used_quantity = close_quantity
            try:
                extra_params = {}
                if self.config.execution.post_only and order_type == 'LIMIT':
                    extra_params['postOnly'] = True
                order_result = await self.exchange_api.place_order(position.symbol, close_side, order_type, close_quantity, price=limit_price, extra_params=extra_params)
            except BotInsufficientFundsError:
                # Try to close whatever we have available (handling partial external closes)
                try:
                    base_asset = position.symbol.split('/')[0]
                    balances = await self.exchange_api.get_balance()
                    available_balance = balances.get(base_asset, {}).get('total', 0.0)
                    tolerance = position.quantity * 0.98
                    if available_balance >= tolerance and available_balance < position.quantity:
                        used_quantity = self.order_sizer.adjust_order_quantity(position.symbol, available_balance, current_price or 0.0, market_details)
                        if used_quantity > 0:
                            order_result = await self.exchange_api.place_order(position.symbol, close_side, order_type, used_quantity, price=limit_price, extra_params=extra_params)
                    else:
                        await self._handle_phantom_position(position)
                        return
                except Exception:
                    pass
            except Exception as e:
                logger.error("Close order failed", error=str(e))
                return

            if not (order_result and order_result.get('orderId')):
                return

            final_order_state = await self.order_lifecycle_manager.manage(
                initial_order=order_result, 
                symbol=position.symbol, 
                side=close_side, 
                quantity=used_quantity,
                initial_price=limit_price,
                market_details=market_details
            )
            
            fill_quantity = final_order_state.get('filled', 0.0) if final_order_state else 0.0
            if fill_quantity > 0:
                close_price = final_order_state['average']
                fees = self._calculate_or_extract_fee(final_order_state, close_price, fill_quantity)
                is_full_close = fill_quantity >= (position.quantity * 0.999)
                is_smart_retry_close = (used_quantity < position.quantity) and (fill_quantity >= (used_quantity * 0.999))

                if is_full_close or is_smart_retry_close:
                    closed_pos = await self.position_manager.close_position(position.symbol, close_price, reason, actual_filled_qty=fill_quantity, fees=fees)
                    if closed_pos: self.risk_manager.update_trade_outcome(closed_pos.symbol, closed_pos.pnl)
                else:
                    await self.position_manager.reduce_position(position.symbol, fill_quantity, close_price, reason, fees=fees)
            else:
                final_status = final_order_state.get('status') if final_order_state else 'UNKNOWN'
                await self.alert_system.send_alert(
                    level='critical',
                    message=f"🔥 FAILED TO CLOSE {position.symbol}.",
                    details={'order_id': order_result.get('orderId'), 'status': final_status}
                )

    async def _handle_phantom_position(self, position: Position):
        try:
            base_asset = position.symbol.split('/')[0]
            balances = await self.exchange_api.get_balance()
            asset_balance = balances.get(base_asset, {}).get('total', 0.0)
            open_orders = await self.exchange_api.fetch_open_orders(position.symbol)
            if asset_balance < (position.quantity * 0.01) and not open_orders:
                logger.info("Confirmed phantom position. Reconciling.")
                close_price = self.latest_prices.get(position.symbol, position.entry_price)
                await self.position_manager.close_position(position.symbol, close_price, reason="Phantom Reconciliation")
            else:
                await self.alert_system.send_alert(
                    level='critical',
                    message=f"🚨 Ambiguous Phantom State for {position.symbol}.",
                    details={'db_qty': position.quantity, 'wallet': asset_balance}
                )
        except Exception as e:
            logger.error("Phantom check failed", error=str(e))
