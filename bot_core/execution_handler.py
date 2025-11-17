from typing import Dict, Any
import pandas as pd

from bot_core.logger import get_logger
from bot_core.order_manager import OrderManager
from bot_core.risk_manager import RiskManager

logger = get_logger(__name__)

class ExecutionHandler:
    """Translates strategy signals into concrete order requests for the OrderManager."""

    def __init__(self, order_manager: OrderManager, risk_manager: RiskManager):
        self.order_manager = order_manager
        self.risk_manager = risk_manager
        logger.info("ExecutionHandler initialized.")

    async def execute_trade_proposal(self, trade_proposal: Dict[str, Any], ohlcv_df: pd.DataFrame):
        """Receives a trade proposal, validates it, and submits it to the OrderManager."""
        action_type = trade_proposal.get('action')
        symbol = trade_proposal.get('symbol')
        confidence = trade_proposal.get('confidence', 0.5)

        if not all([action_type, symbol]) or action_type not in ['BUY', 'SELL']:
            logger.warning("Invalid trade proposal received", proposal=trade_proposal)
            return

        # 1. Get current price from the latest candle for sizing and risk checks
        if ohlcv_df.empty:
            logger.error("Cannot execute trade, OHLCV data is empty.", symbol=symbol)
            return
        current_price = ohlcv_df['close'].iloc[-1]

        # 2. Calculate Stop Loss and Take Profit levels
        stop_loss_price = self.risk_manager.calculate_stop_loss(symbol, current_price, action_type, ohlcv_df)
        take_profit_levels = self.risk_manager.calculate_take_profit_levels(current_price, action_type, confidence)

        # 3. Calculate Position Size based on risk
        quantity = self.risk_manager.calculate_position_size(current_price, stop_loss_price)
        if quantity <= 0:
            logger.warning("Calculated position size is zero or negative. Aborting trade.", symbol=symbol, calculated_size=quantity)
            return

        # 4. Perform final pre-trade checks
        if not self.risk_manager.check_trade_allowed(symbol, action_type, quantity, current_price):
            logger.warning("Trade denied by risk manager.", action=action_type, quantity=quantity, symbol=symbol)
            return

        # 5. Prepare metadata for the order
        metadata = {
            "intent": "OPEN",
            "stop_loss": stop_loss_price,
            "take_profit_levels": take_profit_levels
        }

        # 6. Submit the order to the OrderManager
        try:
            logger.info("Submitting order to OrderManager.", action=action_type, quantity=quantity, symbol=symbol)
            order = await self.order_manager.submit_order(
                symbol=symbol,
                side=action_type,
                order_type='MARKET',
                quantity=quantity,
                metadata=metadata
            )
            
            if order.status in ['REJECTED', 'ERROR']:
                logger.error("Order submission failed.", symbol=symbol, reason=order.error_message)
            else:
                logger.info("Order submitted successfully.", symbol=symbol, client_order_id=order.client_order_id)

        except Exception as e:
            logger.error("Error submitting order to OrderManager", symbol=symbol, error=str(e), exc_info=True)
