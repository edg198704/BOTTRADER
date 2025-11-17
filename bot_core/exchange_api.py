import abc
import time
import random
import logging
from typing import Dict, Any, List, Optional
import ccxt.async_support as ccxt

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
    async def get_balance(self) -> Dict[str, Any]:
        """Retrieves all asset balances."""
        pass

    @abc.abstractmethod
    async def close(self):
        """Closes the exchange connection."""
        pass

class MockExchangeAPI(ExchangeAPI):
    """A mock implementation of ExchangeAPI for testing and development."""

    def __init__(self, initial_balances: Optional[Dict[str, float]] = None):
        self.balances = initial_balances if initial_balances is not None else {"USDT": 10000.0, "BTC": 0.0}
        self.order_id_counter = 0
        self.last_price = 30000.0
        logger.info(f"MockExchangeAPI initialized with balances: {self.balances}")

    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        logger.debug(f"Mock: Fetching market data for {symbol}")
        self.last_price += random.uniform(-100, 100)
        return {"symbol": symbol, "lastPrice": str(self.last_price)}

    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        self.order_id_counter += 1
        order_id = f"mock_order_{self.order_id_counter}"
        fill_price = self.last_price
        base_asset, quote_asset = symbol.split('/')

        if side.upper() == "BUY":
            cost = quantity * fill_price
            if self.balances.get(quote_asset, 0) >= cost:
                self.balances[quote_asset] -= cost
                self.balances[base_asset] = self.balances.get(base_asset, 0) + quantity
                status = "FILLED"
                logger.info(f"Mock: Market BUY {quantity} {base_asset} at {fill_price}. New balances: {self.balances}")
            else:
                status = "REJECTED"
                logger.warning(f"Mock: Insufficient {quote_asset} balance for market BUY.")
        elif side.upper() == "SELL":
            if self.balances.get(base_asset, 0) >= quantity:
                self.balances[base_asset] -= quantity
                self.balances[quote_asset] = self.balances.get(quote_asset, 0) + (quantity * fill_price)
                status = "FILLED"
                logger.info(f"Mock: Market SELL {quantity} {base_asset} at {fill_price}. New balances: {self.balances}")
            else:
                status = "REJECTED"
                logger.warning(f"Mock: Insufficient {base_asset} balance for market SELL.")
        
        return {
            "orderId": order_id,
            "status": status,
            "executedQty": quantity if status == "FILLED" else 0.0,
            "cummulativeQuoteQty": quantity * fill_price if status == "FILLED" else 0.0
        }

    async def get_balance(self) -> Dict[str, Any]:
        logger.debug("Mock: Getting all balances")
        return {asset: {"free": amount, "total": amount} for asset, amount in self.balances.items()}

    async def close(self):
        logger.info("MockExchangeAPI connection closed.")

class CCXTExchangeAPI(ExchangeAPI):
    """Concrete implementation for a real exchange using ccxt."""
    def __init__(self, name: str, api_key: Optional[str], api_secret: Optional[str], testnet: bool):
        exchange_class = getattr(ccxt, name.lower(), None)
        if not exchange_class:
            raise ValueError(f"Exchange '{name}' is not supported by ccxt.")
        
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })

        if testnet and self.exchange.has['test']:
            self.exchange.set_sandbox_mode(True)
            logger.info(f"CCXTExchangeAPI for {name} initialized in TESTNET mode.")
        else:
            logger.info(f"CCXTExchangeAPI for {name} initialized in LIVE mode.")

    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return {"symbol": symbol, "lastPrice": str(ticker['last'])}
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching market data for {symbol}: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching market data for {symbol}: {e}")
            raise

    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        try:
            if order_type.upper() == 'MARKET':
                order = await self.exchange.create_market_order(symbol, side.lower(), quantity)
            elif order_type.upper() == 'LIMIT':
                if price is None:
                    raise ValueError("Price is required for a LIMIT order.")
                order = await self.exchange.create_limit_order(symbol, side.lower(), quantity, price)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            # ccxt returns a dictionary that we can adapt
            return {
                "orderId": order.get('id'),
                "status": order.get('status', 'unknown').upper(),
                "executedQty": order.get('filled', 0.0),
                "cummulativeQuoteQty": order.get('cost', 0.0)
            }
        except ccxt.InsufficientFunds as e:
            logger.error(f"Insufficient funds to place order for {symbol}: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error placing order for {symbol}: {e}")
            raise

    async def get_balance(self) -> Dict[str, Any]:
        try:
            balance = await self.exchange.fetch_balance()
            return balance.get('total', {})
        except ccxt.ExchangeError as e:
            logger.error(f"Error fetching balance: {e}")
            raise

    async def close(self):
        logger.info(f"Closing connection to {self.exchange.name}...")
        await self.exchange.close()
