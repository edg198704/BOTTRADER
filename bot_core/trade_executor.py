import pandas as pd
from typing import Dict, Any, Optional

from bot_core.logger import get_logger
from bot_core.config import BotConfig
from bot_core.exchange_api import ExchangeAPI
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
            await self._execute_open_position(symbol, action, current_price, df_with_indicators, market_regime)
        
        # --- Handle Closing an Existing Position ---
        elif position:
            is_close_long_signal = (action == 'SELL' or action == 'CLOSE') and position.side == 'BUY'
            is_close_short_signal = (action == 'BUY' or action == 'CLOSE') and position.side == 'SELL'
            
            if is_close_long_signal or is_close_short_signal:
                await self.close_position(position, "Strategy Signal")

    async def _execute_open_position(self, symbol: str, side: str, current_price: float, df: pd.DataFrame, regime: Optional[str]):
        # Await async DB call
        open_positions = await self.position_manager.get_all_open_positions()
        if not self.risk_manager.check_trade_allowed(symbol, open_positions):
            return

        portfolio_equity = self.position_manager.get_portfolio_value(self.latest_prices, open_positions)
        stop_loss = self.risk_manager.calculate_stop_loss(side, current_price, df, market_regime=regime)
        ideal_quantity = self.risk_manager.calculate_position_size(portfolio_equity, current_price, stop_loss, market_regime=regime)
        
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

        order_result = await self.exchange_api.place_order(symbol, side, order_type, final_quantity, price=limit_price)
        if not (order_result and order_result.get('orderId')):
            logger.error("Order placement failed, no order ID returned.", symbol=symbol)
            return

        final_order_state = await self.order_lifecycle_manager.manage(
            initial_order=order_result, 
            symbol=symbol, 
            side=side, 
            quantity=final_quantity, 
            initial_price=limit_price,
            market_details=market_details
        )
        
        fill_quantity = final_order_state.get('filled', 0.0) if final_order_state else 0.0
        if fill_quantity > 0:
            fill_price = final_order_state.get('average')
            if not fill_price or fill_price <= 0:
                logger.critical("Order filled but average price is invalid. Cannot open position.", order_id=order_result.get('orderId'), final_state=final_order_state)
                return

            logger.info("Order to open position was filled (fully or partially).", order_id=order_result.get('orderId'), filled_qty=fill_quantity, fill_price=fill_price)
            final_stop_loss = self.risk_manager.calculate_stop_loss(side, fill_price, df, market_regime=regime)
            final_take_profit = self.risk_manager.calculate_take_profit(side, fill_price, final_stop_loss, market_regime=regime)
            
            # Await async DB call
            await self.position_manager.open_position(symbol, side, fill_quantity, fill_price, final_stop_loss, final_take_profit)
        else:
            final_status = final_order_state.get('status') if final_order_state else 'UNKNOWN'
            logger.error("Order to open position did not fill.", order_id=order_result.get('orderId'), final_status=final_status)
            await self.alert_system.send_alert(
                level='error',
                message=f"🔴 Failed to open position for {symbol}. Order did not fill.",
                details={'symbol': symbol, 'order_id': order_result.get('orderId'), 'final_status': final_status}
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

        order_result = await self.exchange_api.place_order(position.symbol, close_side, order_type, close_quantity, price=limit_price)
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
