import logging
from typing import Dict, Any
import pandas as pd

from bot_core.order_manager import OrderManager
from bot_core.risk_manager import RiskManager

logger = logging.getLogger(__name__)

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
            logger.warning(f"Invalid trade proposal received: {trade_proposal}")
            return

        # 1. Get current price from the latest candle for sizing and risk checks
        if ohlcv_df.empty:
            logger.error(f"Cannot execute trade for {symbol}, OHLCV data is empty.")
            return
        current_price = ohlcv_df['close'].iloc[-1]

        # 2. Calculate Stop Loss first, as it's needed for position sizing
        stop_loss_price = self.risk_manager.calculate_stop_loss(symbol, current_price, action_type, ohlcv_df)

        # 3. Calculate Position Size based on risk
        quantity = self.risk_manager.calculate_position_size(current_price, stop_loss_price)
        if quantity <= 0:
            logger.warning(f"Calculated position size is zero or negative for {symbol}. Aborting trade.")
            return

        # 4. Perform final pre-trade checks
        if not self.risk_manager.check_trade_allowed(symbol, action_type, quantity, current_price):
            logger.warning(f"Trade {action_type} {quantity} {symbol} denied by risk manager.")
            return

        # 5. Submit the order to the OrderManager
        try:
            logger.info(f"Submitting {action_type} order for {quantity} of {symbol} to OrderManager.")
            order = await self.order_manager.submit_order(
                symbol=symbol,
                side=action_type,
                order_type='MARKET',
                quantity=quantity
            )
            
            if order.status in ['REJECTED', 'ERROR']:
                logger.error(f"Order submission failed for {symbol}. Reason: {order.error_message}")
            else:
                logger.info(f"Order for {symbol} submitted with client ID: {order.client_order_id}")

        except Exception as e:
            logger.error(f"Error submitting order for {symbol} to OrderManager: {e}", exc_info=True)
