import logging
from typing import Dict, Any
import pandas as pd

from bot_core.exchange_api import ExchangeAPI
from bot_core.position_manager import PositionManager
from bot_core.risk_manager import RiskManager

logger = logging.getLogger(__name__)

class ExecutionHandler:
    """Handles the entire lifecycle of a trade execution."""

    def __init__(self, exchange_api: ExchangeAPI, position_manager: PositionManager, risk_manager: RiskManager):
        self.exchange_api = exchange_api
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        logger.info("ExecutionHandler initialized.")

    async def execute_trade_proposal(self, trade_proposal: Dict[str, Any], ohlcv_df: pd.DataFrame):
        """Receives a trade proposal and manages its execution."""
        action_type = trade_proposal.get('action')
        symbol = trade_proposal.get('symbol')
        confidence = trade_proposal.get('confidence', 0.5)

        if not all([action_type, symbol]) or action_type not in ['BUY', 'SELL']:
            logger.warning(f"Invalid trade proposal received: {trade_proposal}")
            return

        current_price_data = await self.exchange_api.get_market_data(symbol)
        current_price = float(current_price_data.get('lastPrice', 0))
        if current_price == 0:
            logger.error(f"Could not get current price for {symbol} to execute trade.")
            return

        # 1. Calculate Stop Loss first, as it's needed for position sizing
        stop_loss_price = self.risk_manager.calculate_stop_loss(symbol, current_price, action_type, ohlcv_df)

        # 2. Calculate Position Size based on risk
        quantity = self.risk_manager.calculate_position_size(current_price, stop_loss_price)
        if quantity <= 0:
            logger.warning(f"Calculated position size is zero or negative for {symbol}. Aborting trade.")
            return

        # 3. Perform final pre-trade checks
        if not self.risk_manager.check_trade_allowed(symbol, action_type, quantity, current_price):
            logger.warning(f"Trade {action_type} {quantity} {symbol} denied by risk manager.")
            return

        # 4. Place the order
        try:
            order_response = await self.exchange_api.place_order(symbol, action_type, 'MARKET', quantity)
            logger.info(f"Order placed for {symbol}: {order_response}")

            # 5. Process the fill and update the position ledger
            if order_response and order_response.get('status') == 'FILLED':
                entry_price = float(order_response['cummulativeQuoteQty']) / float(order_response['executedQty'])
                filled_quantity = float(order_response['executedQty'])

                take_profit_levels = self.risk_manager.calculate_take_profit_levels(entry_price, action_type, confidence)

                self.position_manager.add_position(
                    symbol=symbol,
                    side=action_type,
                    quantity=filled_quantity,
                    entry_price=entry_price,
                    stop_loss=stop_loss_price,
                    take_profit_levels=take_profit_levels
                )
            else:
                logger.error(f"Order for {symbol} was not filled. Status: {order_response.get('status')}. Not opening position.")

        except Exception as e:
            logger.error(f"Error during order execution for {symbol}: {e}", exc_info=True)
