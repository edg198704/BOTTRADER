# bot_core/exchange_api.py
import abc
import time
import random
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class ExchangeAPI(abc.ABC):
    """Abstract Base Class for interacting with a cryptocurrency exchange."""

    @abc.abstractmethod
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Fetches current market data for a given symbol."""
        pass

    @abc.abstractmethod
    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        """Places an order on the exchange."""
        pass

    @abc.abstractmethod
    async def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Retrieves the status of an order."""
        pass

    @abc.abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Cancels an open order."""
        pass

    @abc.abstractmethod
    async def get_balance(self, asset: str) -> Dict[str, Any]:
        """Retrieves the balance for a specific asset."""
        pass

    @abc.abstractmethod
    async def get_all_balances(self) -> Dict[str, Any]:
        """Retrieves all asset balances."""
        pass

    @abc.abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves all open orders or open orders for a specific symbol."""
        pass

class MockExchangeAPI(ExchangeAPI):
    """A mock implementation of ExchangeAPI for testing and development."""

    def __init__(self, initial_balances: Optional[Dict[str, float]] = None):
        self.balances = initial_balances if initial_balances is not None else {"USDT": 10000.0, "BTC": 0.0}
        self.open_orders: Dict[str, Dict[str, Any]] = {}
        self.order_id_counter = 0
        logger.info(f"MockExchangeAPI initialized with balances: {self.balances}")

    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        logger.debug(f"Mock: Fetching market data for {symbol}")
        # Simulate price movement
        if symbol == "BTCUSDT":
            price = 30000 + random.uniform(-100, 100)
            return {"symbol": symbol, "bidPrice": str(price - 0.5), "askPrice": str(price + 0.5), "lastPrice": str(price)}
        return {"symbol": symbol, "bidPrice": "1.0", "askPrice": "1.0", "lastPrice": "1.0"}

    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        self.order_id_counter += 1
        order_id = f"mock_order_{self.order_id_counter}"
        status = "NEW"

        market_data = await self.get_market_data(symbol)
        current_price = float(market_data["lastPrice"])

        if order_type == "MARKET":
            fill_price = current_price
            status = "FILLED"
            base_asset = symbol[:-4] # e.g., BTC from BTCUSDT
            quote_asset = symbol[-4:] # e.g., USDT from BTCUSDT

            if side == "BUY":
                cost = quantity * fill_price
                if self.balances.get(quote_asset, 0) >= cost:
                    self.balances[quote_asset] -= cost
                    self.balances[base_asset] = self.balances.get(base_asset, 0) + quantity
                    logger.info(f"Mock: Market BUY {quantity} {base_asset} at {fill_price}. New balances: {self.balances}")
                else:
                    status = "REJECTED"
                    logger.warning(f"Mock: Insufficient {quote_asset} balance for market BUY.")
            elif side == "SELL":
                if self.balances.get(base_asset, 0) >= quantity:
                    self.balances[base_asset] -= quantity
                    self.balances[quote_asset] = self.balances.get(quote_asset, 0) + (quantity * fill_price)
                    logger.info(f"Mock: Market SELL {quantity} {base_asset} at {fill_price}. New balances: {self.balances}")
                else:
                    status = "REJECTED"
                    logger.warning(f"Mock: Insufficient {base_asset} balance for market SELL.")
        elif order_type == "LIMIT":
            if price is None:
                raise ValueError("Limit orders require a price.")
            # For simplicity, mock limit orders are immediately filled if price condition met
            # In a real system, they would stay open until matched
            fill_price = price
            if side == "BUY" and current_price <= price: # If current price is at or below limit buy price
                status = "FILLED"
                base_asset = symbol[:-4]
                quote_asset = symbol[-4:]
                cost = quantity * fill_price
                if self.balances.get(quote_asset, 0) >= cost:
                    self.balances[quote_asset] -= cost
                    self.balances[base_asset] = self.balances.get(base_asset, 0) + quantity
                    logger.info(f"Mock: Limit BUY {quantity} {base_asset} at {fill_price} (filled). New balances: {self.balances}")
                else:
                    status = "REJECTED"
                    logger.warning(f"Mock: Insufficient {quote_asset} balance for limit BUY.")
            elif side == "SELL" and current_price >= price: # If current price is at or above limit sell price
                status = "FILLED"
                base_asset = symbol[:-4]
                quote_asset = symbol[-4:]
                if self.balances.get(base_asset, 0) >= quantity:
                    self.balances[base_asset] -= quantity
                    self.balances[quote_asset] = self.balances.get(quote_asset, 0) + (quantity * fill_price)
                    logger.info(f"Mock: Limit SELL {quantity} {base_asset} at {fill_price} (filled). New balances: {self.balances}")
                else:
                    status = "REJECTED"
                    logger.warning(f"Mock: Insufficient {base_asset} balance for limit SELL.")
            else:
                self.open_orders[order_id] = {
                    "orderId": order_id, "symbol": symbol, "side": side, "type": order_type,
                    "quantity": quantity, "price": price, "status": "NEW", "timestamp": int(time.time() * 1000)
                }
                logger.info(f"Mock: Limit order {order_id} placed (NEW).")
                return self.open_orders[order_id]

        order_response = {
            "orderId": order_id,
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "price": str(price) if price else str(fill_price),
            "status": status,
            "executedQty": quantity if status == "FILLED" else 0.0,
            "cummulativeQuoteQty": quantity * fill_price if status == "FILLED" else 0.0,
            "timestamp": int(time.time() * 1000)
        }
        if status == "FILLED" and order_id in self.open_orders:
            del self.open_orders[order_id]
        return order_response

    async def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        logger.debug(f"Mock: Getting status for order {order_id} on {symbol}")
        order = self.open_orders.get(order_id)
        if order:
            return order
        # Simulate a filled order if not found in open orders
        return {"orderId": order_id, "symbol": symbol, "status": "FILLED", "executedQty": 0.0, "cummulativeQuoteQty": 0.0}

    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        logger.info(f"Mock: Cancelling order {order_id} on {symbol}")
        if order_id in self.open_orders:
            order = self.open_orders.pop(order_id)
            order["status"] = "CANCELED"
            return order
        return {"orderId": order_id, "symbol": symbol, "status": "NOT_FOUND"}

    async def get_balance(self, asset: str) -> Dict[str, Any]:
        logger.debug(f"Mock: Getting balance for {asset}")
        return {"asset": asset, "free": str(self.balances.get(asset, 0.0)), "locked": "0.0"}

    async def get_all_balances(self) -> Dict[str, Any]:
        logger.debug("Mock: Getting all balances")
        return {asset: {"free": str(amount), "locked": "0.0"} for asset, amount in self.balances.items()}

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        logger.debug(f"Mock: Getting open orders for {symbol if symbol else 'all symbols'}")
        if symbol:
            return [order for order in self.open_orders.values() if order["symbol"] == symbol]
        return list(self.open_orders.values())
