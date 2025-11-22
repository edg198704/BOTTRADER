from typing import Dict, Any
from decimal import Decimal, ROUND_DOWN
from bot_core.logger import get_logger
from bot_core.common import to_decimal, ZERO

logger = get_logger(__name__)

class OrderSizer:
    """
    Adjusts order quantities to conform to exchange-specific trading rules (precision, limits).
    Uses Decimal for all calculations to ensure precision.
    """
    def __init__(self):
        logger.info("OrderSizer initialized.")

    def adjust_order_quantity(self, symbol: str, quantity: Decimal, price: Decimal, market_details: Dict[str, Any]) -> Decimal:
        """
        Adjusts the desired quantity to meet the exchange's precision and limit rules.

        Args:
            symbol: The trading symbol.
            quantity: The desired quantity calculated by the risk manager (Decimal).
            price: The current price of the asset (Decimal).
            market_details: The market details fetched from the exchange API.

        Returns:
            The adjusted quantity (Decimal), or ZERO if the desired quantity is below minimums.
        """
        if not market_details:
            logger.warning("No market details available for sizing, returning original quantity.", symbol=symbol)
            return quantity

        precision_amount = market_details.get('precision', {}).get('amount')
        limits_amount_min = market_details.get('limits', {}).get('amount', {}).get('min')
        limits_cost_min = market_details.get('limits', {}).get('cost', {}).get('min')

        if precision_amount is None:
            logger.warning("Amount precision not found in market details, cannot adjust.", symbol=symbol)
            return quantity

        # 1. Adjust for precision (step size)
        # CCXT precision is often a float (e.g. 0.0001 or 1e-5). Convert to Decimal.
        prec_dec = to_decimal(precision_amount)
        
        # Quantize down to ensure we don't round up into insufficient funds
        # If precision is 1e-5, we quantize to 5 decimal places
        try:
            adjusted_quantity = quantity.quantize(prec_dec, rounding=ROUND_DOWN)
        except Exception:
            # Fallback if precision is not a standard exponent (rare)
            adjusted_quantity = quantity
        
        logger.debug("Quantity adjusted for precision.", 
                     original=str(quantity), 
                     adjusted=str(adjusted_quantity), 
                     precision=str(prec_dec),
                     symbol=symbol)

        # 2. Check against minimum order size (amount)
        if limits_amount_min is not None:
            min_qty = to_decimal(limits_amount_min)
            if adjusted_quantity < min_qty:
                logger.warning("Desired quantity is below the exchange's minimum order size.",
                             symbol=symbol,
                             desired_quantity=str(adjusted_quantity),
                             min_quantity=str(min_qty))
                return ZERO

        # 3. Check against minimum cost (notional value)
        if limits_cost_min is not None and price > ZERO:
            min_cost = to_decimal(limits_cost_min)
            cost = adjusted_quantity * price
            if cost < min_cost:
                logger.warning("Desired position value is below the exchange's minimum cost.",
                             symbol=symbol,
                             cost=str(cost),
                             min_cost=str(min_cost))
                return ZERO

        return adjusted_quantity
