import pandas as pd
import uuid
from typing import Dict, Any, Optional

from bot_core.logger import get_logger
from bot_core.config import BotConfig
from bot_core.exchange_api import ExchangeAPI, BotInsufficientFundsError, BotInvalidOrderError, BotExchangeError
from bot_core.position_manager import PositionManager, Position
from bot_core.risk_manager import RiskManager
from bot_core.order_sizer import OrderSizer
from bot_core.order_lifecycle_manager import OrderLifecycleManager
from bot_core.monitoring import AlertSystem

logger = get_logger(__name__)

class TradeExecutor:
    """
    Handles the entire lifecycle of executing a trade, from signal to position management.
    This class encapsulates the logic for risk checks, sizing, order placement,
    and post-execution state updates.
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
                 market_details: Dict[str, Dict[str, Any]]):
        self.config = config
        self.exchange_api = exchange_api
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.order_sizer = order_sizer
        self.order_lifecycle_manager = order_lifecycle_manager
        self.alert_system = alert_system
        self.latest_prices = shared_latest_prices
        self.market_details = market_details
        logger.info("TradeExecutor initialized.")

    async def execute_trade_signal(self, signal: Dict, df_with_indicators: pd.DataFrame, position: Optional[Position]):
        """
        Processes a trading signal to open or close a position.
        """
        action = signal.get('action')
        symbol = signal.get('symbol')
        current_price = self.latest_prices.get(symbol)
        market_regime = signal.get('regime')

        if not all([action, symbol, current_price]):
            logger.warning("Received invalid signal for execution", signal=signal)
            return

        # --- Handle Opening a New Position ---
        if not position:
            if action not in ['BUY', 'SELL']:
                return  # Ignore other signals if no position is open
            await self._execute_open_position(symbol, action, current_price, df_with_indicators, market_regime, signal)
        
        # --- Handle Closing an Existing Position ---
        elif position:
            is_close_long_signal = (action == 'SELL' or action == 'CLOSE') and position.side == 'BUY'
            is_close_short_signal = (action == 'BUY' or action == 'CLOSE') and position.side == 'SELL'
            
            if is_close_long_signal or is_close_short_signal:
                await self.close_position(position, "Strategy Signal")

    async def _execute_open_position(self, symbol: str, side: str, current_price: float, df: pd.DataFrame, regime: Optional[str], signal: Dict):
        # Await async DB call to get ALL active positions (OPEN + PENDING) for accurate risk check
        active_positions = await self.position_manager.get_all_active_positions()
        if not self.risk_manager.check_trade_allowed(symbol, active_positions):
            return

        portfolio_equity = self.position_manager.get_portfolio_value(self.latest_prices, active_positions)
        stop_loss = self.risk_manager.calculate_stop_loss(side, current_price, df, market_regime=regime)
        
        # Extract confidence data for risk scaling
        strategy_metadata = signal.get('strategy_metadata', {})
        confidence = strategy_metadata.get('confidence')
        # Try to get threshold from config if it exists for the current strategy
        confidence_threshold = getattr(self.config.strategy.params, 'confidence_threshold', None)

        # Pass symbol and active_positions for correlation check
        ideal_quantity = self.risk_manager.calculate_position_size(
            symbol=symbol,
            portfolio_equity=portfolio_equity, 
            entry_price=current_price, 
            stop_loss_price=stop_loss, 
            open_positions=active_positions,
            market_regime=regime,
            confidence=confidence,
            confidence_threshold=confidence_threshold
        )
        
        market_details = self.market_details.get(symbol)
        if not market_details:
            logger.error("Cannot place order, market details not available for symbol.", symbol=symbol)
            return

        final_quantity = self.order_sizer.adjust_order_quantity(symbol, ideal_quantity, current_price, market_details)

        if final_quantity <= 0:
            logger.warning("Calculated position size is zero or less after adjustments. Aborting trade.", 
                         symbol=symbol, ideal_quantity=ideal_quantity, final_quantity=final_quantity)
            return

        order_type = self.config.execution.default_order_type
        limit_price = None
        if order_type == 'LIMIT':
            offset = self.config.execution.limit_price_offset_pct
            limit_price = current_price * (1 - offset) if side == 'BUY' else current_price * (1 + offset)

        # 1. Generate Client Order ID (Idempotency Key)
        client_order_id = str(uuid.uuid4())

        # 2. Record PENDING Position (Pre-Commit)
        # We persist the intent BEFORE the network call to handle crashes gracefully.
        await self.position_manager.create_pending_position(symbol, side, client_order_id, strategy_metadata=strategy_metadata)

        # 3. Place Order
        try:
            order_result = await self.exchange_api.place_order(
                symbol, side, order_type, final_quantity, price=limit_price, 
                extra_params={'clientOrderId': client_order_id}
            )
        except (BotInvalidOrderError, BotInsufficientFundsError) as e:
            # These are deterministic errors; the order definitely failed.
            logger.error("Order rejected by exchange logic. Voiding pending position.", error=str(e))
            await self.position_manager.void_position(symbol, client_order_id)
            return
        except Exception as e:
            # Network timeout or unknown error. 
            # The order MIGHT have gone through. We leave the PENDING position.
            # The bot's reconciliation logic on restart (or OrderLifecycleManager if we continued) 
            # would handle it. But here we stop execution flow to be safe.
            logger.error("Critical error during order placement. Leaving PENDING position for reconciliation.", error=str(e))
            return

        # 4. Update Order ID if Exchange returns a different one
        # Some exchanges return their own ID. We update our DB to match.
        exchange_order_id = order_result.get('orderId')
        if exchange_order_id and exchange_order_id != client_order_id:
            logger.info("Updating PENDING position with Exchange Order ID", client_id=client_order_id, exchange_id=exchange_order_id)
            await self.position_manager.update_pending_order_id(symbol, client_order_id, exchange_order_id)
            # Use the exchange ID for lifecycle management
            current_order_id = exchange_order_id
        else:
            current_order_id = client_order_id

        # Define callback to update DB if order is replaced during chase
        async def _on_order_replace(old_id: str, new_id: str):
            await self.position_manager.update_pending_order_id(symbol, old_id, new_id)

        # 5. Manage Order Lifecycle
        final_order_state = await self.order_lifecycle_manager.manage(
            initial_order=order_result, 
            symbol=symbol, 
            side=side, 
            quantity=final_quantity, 
            initial_price=limit_price,
            market_details=market_details,
            on_order_replace=_on_order_replace
        )
        
        fill_quantity = final_order_state.get('filled', 0.0) if final_order_state else 0.0
        final_status = final_order_state.get('status') if final_order_state else 'UNKNOWN'
        
        # 6. Confirm or Void Position (Two-Phase Commit End)
        if fill_quantity > 0:
            fill_price = final_order_state.get('average')
            if not fill_price or fill_price <= 0:
                logger.critical("Order filled but average price is invalid. Cannot confirm position.", order_id=current_order_id, final_state=final_order_state)
                # We leave it as PENDING so it can be reconciled manually or on restart.
                return

            logger.info("Order to open position was filled.", order_id=current_order_id, filled_qty=fill_quantity, fill_price=fill_price)
            final_stop_loss = self.risk_manager.calculate_stop_loss(side, fill_price, df, market_regime=regime)
            final_take_profit = self.risk_manager.calculate_take_profit(side, fill_price, final_stop_loss, market_regime=regime)
            
            await self.position_manager.confirm_position_open(symbol, current_order_id, fill_quantity, fill_price, final_stop_loss, final_take_profit)
        
        elif final_status == 'OPEN':
            # CRITICAL: The order is still OPEN on the exchange (cancellation failed), but we stopped chasing.
            logger.critical("Order is stuck in OPEN state after lifecycle management. Leaving position as PENDING.", 
                            symbol=symbol, order_id=current_order_id)
            await self.alert_system.send_alert(
                level='critical',
                message=f"🚨 ZOMBIE ORDER RISK: Order {current_order_id} for {symbol} is stuck OPEN. Manual intervention required.",
                details={'symbol': symbol, 'order_id': current_order_id, 'status': 'OPEN'}
            )
            
        else:
            # Order is definitively dead (CANCELED, REJECTED, EXPIRED) and empty.
            logger.error("Order to open position did not fill. Voiding pending position.", order_id=current_order_id, final_status=final_status)
            await self.position_manager.void_position(symbol, current_order_id)
            
            await self.alert_system.send_alert(
                level='error',
                message=f"🔴 Failed to open position for {symbol}. Order did not fill.",
                details={'symbol': symbol, 'order_id': current_order_id, 'final_status': final_status}
            )

    async def close_position(self, position: Position, reason: str):
        close_side = 'SELL' if position.side == 'BUY' else 'BUY'
        
        market_details = self.market_details.get(position.symbol)
        current_price = self.latest_prices.get(position.symbol)

        if not market_details:
            logger.error("Cannot close position, market details not available for symbol.", symbol=position.symbol)
            close_quantity = position.quantity
        else:
            if not current_price:
                logger.warning("Latest price not available for sizing close order, cost check will be skipped.", symbol=position.symbol)
            close_quantity = self.order_sizer.adjust_order_quantity(
                position.symbol, 
                position.quantity, 
                current_price or 0.0, # Pass 0 if price is unknown, sizer will skip cost check
                market_details
            )

        if close_quantity <= 0:
            logger.error("Cannot close position, adjusted quantity is zero.", symbol=position.symbol, original_qty=position.quantity)
            return

        order_type = self.config.execution.default_order_type
        limit_price = None
        if order_type == 'LIMIT' and current_price:
            offset = self.config.execution.limit_price_offset_pct
            limit_price = current_price * (1 + offset) if close_side == 'BUY' else current_price * (1 - offset)
        else:
            order_type = 'MARKET'

        try:
            order_result = await self.exchange_api.place_order(position.symbol, close_side, order_type, close_quantity, price=limit_price)
        except BotInsufficientFundsError:
            logger.warning("Insufficient funds to close position. Checking for phantom state.", symbol=position.symbol)
            await self._handle_phantom_position(position)
            return
        except BotInvalidOrderError as e:
            logger.error("Invalid order parameters for close.", error=str(e))
            return
        except Exception as e:
            logger.error("Failed to place close order.", error=str(e))
            return

        if not (order_result and order_result.get('orderId')):
            logger.error("Failed to place close order.", symbol=position.symbol)
            return

        final_order_state = await self.order_lifecycle_manager.manage(
            initial_order=order_result, 
            symbol=position.symbol, 
            side=close_side, 
            quantity=close_quantity, 
            initial_price=limit_price,
            market_details=market_details
        )
        if final_order_state and final_order_state.get('filled', 0.0) > 0:
            # Use the aggregated average price for PnL calculation
            close_price = final_order_state['average']
            # Await async DB call
            await self.position_manager.close_position(position.symbol, close_price, reason)
        else:
            final_status = final_order_state.get('status') if final_order_state else 'UNKNOWN'
            logger.error("Failed to confirm close order fill. Position remains open.", order_id=order_result.get('orderId'), symbol=position.symbol, final_status=final_status)
            await self.alert_system.send_alert(
                level='critical',
                message=f"🔥 FAILED TO CLOSE position for {position.symbol}. Manual intervention may be required.",
                details={'symbol': position.symbol, 'order_id': order_result.get('orderId'), 'reason': reason, 'final_status': final_status}
            )

    async def _handle_phantom_position(self, position: Position):
        """
        Verifies if a position is truly missing from the exchange (phantom) and reconciles the DB.
        """
        try:
            base_asset = position.symbol.split('/')[0]
            balances = await self.exchange_api.get_balance()
            asset_balance = balances.get(base_asset, {}).get('total', 0.0)
            
            # Check open orders that might be locking the balance
            open_orders = await self.exchange_api.fetch_open_orders(position.symbol)
            
            # Logic: If we have significantly less than the position quantity AND no open orders locking it
            # Then the position is likely already gone (sold externally or previous close update failed).
            
            # Tolerance for dust (e.g. 1% of position size)
            if asset_balance < (position.quantity * 0.01) and not open_orders:
                logger.info("Confirmed phantom position (DB open, Exchange closed). Reconciling.", 
                            db_qty=position.quantity, exchange_bal=asset_balance)
                
                # Estimate close price (Current market price) for PnL tracking purposes
                close_price = self.latest_prices.get(position.symbol, position.entry_price)
                
                await self.position_manager.close_position(
                    position.symbol, 
                    close_price, 
                    reason="Phantom Reconciliation"
                )
                
                await self.alert_system.send_alert(
                    level='warning',
                    message=f"⚠️ Auto-Reconciled Phantom Position for {position.symbol}",
                    details={'db_qty': position.quantity, 'wallet_bal': asset_balance}
                )
            else:
                logger.error("Insufficient funds error but wallet has balance or open orders. Manual check needed.",
                             db_qty=position.quantity, exchange_bal=asset_balance, open_orders=len(open_orders))
                await self.alert_system.send_alert(
                    level='critical',
                    message=f"🚨 Insufficient Funds to close {position.symbol} but state is ambiguous.",
                    details={'db_qty': position.quantity, 'wallet_bal': asset_balance, 'open_orders': len(open_orders)}
                )
        except Exception as e:
            logger.error("Error during phantom position reconciliation", error=str(e))
