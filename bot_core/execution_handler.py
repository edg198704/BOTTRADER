import asyncio
from typing import Dict, Any
import pandas as pd

from bot_core.logger import get_logger
from bot_core.order_manager import OrderManager
from bot_core.risk_manager import RiskManager
from bot_core.data_handler import SignalEvent, OrderEvent

logger = get_logger(__name__)

class ExecutionHandler:
    """Translates strategy signals into concrete order events for the OrderManager."""

    def __init__(self, event_queue: asyncio.Queue, order_manager: OrderManager, risk_manager: RiskManager):
        self.event_queue = event_queue
        self.order_manager = order_manager
        self.risk_manager = risk_manager
        logger.info("ExecutionHandler initialized.")

    async def on_signal_event(self, event: SignalEvent, ohlcv_df: pd.DataFrame):
        """Receives a signal event, validates it, and creates an order event."""
        action_type = event.action
        symbol = event.symbol
        confidence = event.confidence

        if not all([action_type, symbol]) or action_type not in ['BUY', 'SELL']:
            logger.warning("Invalid signal event received", event=event)
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

        # 6. Create and queue the OrderEvent
        try:
            logger.info("Queuing OrderEvent.", action=action_type, quantity=quantity, symbol=symbol)
            order_event = OrderEvent(
                symbol=symbol,
                side=action_type,
                order_type='MARKET',
                quantity=quantity,
                metadata=metadata
            )
            await self.event_queue.put(order_event)

        except Exception as e:
            logger.error("Error queuing OrderEvent", symbol=symbol, error=str(e), exc_info=True)
