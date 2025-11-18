from typing import Dict, Any
import ccxt

from bot_core.logger import get_logger

logger = get_logger(__name__)

class OrderSizer:
    """
    Adjusts order quantities to conform to exchange-specific trading rules (precision, limits).
    """
    def __init__(self):
        logger.info("OrderSizer initialized.")

    def adjust_order_quantity(self, symbol: str, quantity: float, market_details: Dict[str, Any]) -> float:
        """
        Adjusts the desired quantity to meet the exchange's precision and minimum lot size rules.

        Args:
            symbol: The trading symbol.
            quantity: The desired quantity calculated by the risk manager.
            market_details: The market details fetched from the exchange API.

        Returns:
            The adjusted quantity, or 0.0 if the desired quantity is below the minimum.
        """
        if not market_details:
            logger.warning("No market details available for sizing, returning original quantity.", symbol=symbol)
            return quantity

        precision_amount = market_details.get('precision', {}).get('amount')
        limits_amount_min = market_details.get('limits', {}).get('amount', {}).get('min')

        if precision_amount is None:
            logger.warning("Amount precision not found in market details, cannot adjust.", symbol=symbol)
            return quantity

        # 1. Adjust for precision (step size)
        adjusted_quantity = float(ccxt.Exchange.amount_to_precision(symbol, quantity, precision_amount))
        logger.debug("Quantity adjusted for precision.", 
                     original=quantity, 
                     adjusted=adjusted_quantity, 
                     precision=precision_amount,
                     symbol=symbol)

        # 2. Check against minimum order size
        if limits_amount_min is not None and adjusted_quantity < limits_amount_min:
            logger.warning("Desired quantity is below the exchange's minimum order size.",
                         symbol=symbol,
                         desired_quantity=adjusted_quantity,
                         min_quantity=limits_amount_min)
            return 0.0

        return adjusted_quantity
