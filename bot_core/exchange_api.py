import abc
import time
import random
from typing import Dict, Any, List, Optional
import ccxt.async_support as ccxt

from bot_core.logger import get_logger

logger = get_logger(__name__)

class ExchangeAPI(abc.ABC):
    """Abstract Base Class for interacting with a cryptocurrency exchange."""

    @abc.abstractmethod
    async def get_market_data(self, symbol: str, timeframe: str, limit: int) -> List[List[Any]]:
        """Fetches OHLCV market data for a given symbol."""
        pass

    @abc.abstractmethod
    async def get_ticker_data(self, symbol: str) -> Dict[str, Any]:
        """Fetches current ticker data for a given symbol."""
        pass

    @abc.abstractmethod
    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        """Places an order on the exchange."""
        pass
    
    @abc.abstractmethod
    async def fetch_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetches the status of a specific order."""
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
        self.open_orders = {}
        logger.info("MockExchangeAPI initialized", initial_balances=self.balances)

    async def get_market_data(self, symbol: str, timeframe: str, limit: int) -> List[List[Any]]:
        logger.debug("Mock: Fetching market data", symbol=symbol, limit=limit)
        now = int(time.time() * 1000)
        interval_ms = 3600 * 1000 # 1h
        ohlcv = []
        price = self.last_price
        for i in range(limit):
            ts = now - (limit - i - 1) * interval_ms
            open_price = price + random.uniform(-50, 50)
            high = max(open_price, price) + random.uniform(0, 20)
            low = min(open_price, price) - random.uniform(0, 20)
            close_price = price
            volume = random.uniform(10, 100)
            ohlcv.append([ts, open_price, high, low, close_price, volume])
            price += random.uniform(-100, 100)
        self.last_price = price
        return ohlcv

    async def get_ticker_data(self, symbol: str) -> Dict[str, Any]:
        logger.debug("Mock: Fetching ticker data", symbol=symbol)
        self.last_price += random.uniform(-100, 100)
        return {"symbol": symbol, "lastPrice": str(self.last_price)}

    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        self.order_id_counter += 1
        order_id = f"mock_order_{self.order_id_counter}"
        fill_price = self.last_price
        base_asset, quote_asset = symbol.split('/')

        cost = quantity * fill_price
        can_fill = (side.upper() == "BUY" and self.balances.get(quote_asset, 0) >= cost) or \
                   (side.upper() == "SELL" and self.balances.get(base_asset, 0) >= quantity)

        if can_fill:
            if side.upper() == "BUY":
                self.balances[quote_asset] -= cost
                self.balances[base_asset] = self.balances.get(base_asset, 0) + quantity
            else: # SELL
                self.balances[base_asset] -= quantity
                self.balances[quote_asset] = self.balances.get(quote_asset, 0) + cost
            
            status = "FILLED"
            logger.info("Mock: Order filled", side=side, quantity=quantity, symbol=symbol, fill_price=fill_price, new_balances=self.balances)
            self.open_orders[order_id] = {'id': order_id, 'status': 'closed', 'filled': quantity, 'average': fill_price, 'symbol': symbol}
        else:
            status = "REJECTED"
            logger.warning("Mock: Insufficient balance for order", side=side, symbol=symbol)
            self.open_orders[order_id] = {'id': order_id, 'status': 'rejected', 'symbol': symbol}

        return {
            "orderId": order_id,
            "status": status,
            "executedQty": quantity if status == "FILLED" else 0.0,
            "cummulativeQuoteQty": cost if status == "FILLED" else 0.0
        }

    async def fetch_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        order = self.open_orders.get(order_id)
        if not order:
            return None
        return {
            'id': order['id'],
            'status': order['status'],
            'filled': order.get('filled', 0.0),
            'average': order.get('average', 0.0),
            'symbol': order['symbol']
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
            logger.info("CCXTExchangeAPI initialized in TESTNET mode", exchange=name)
        else:
            logger.info("CCXTExchangeAPI initialized in LIVE mode", exchange=name)

    async def get_market_data(self, symbol: str, timeframe: str, limit: int) -> List[List[Any]]:
        try:
            return await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except ccxt.NetworkError as e:
            logger.error("Network error fetching OHLCV", symbol=symbol, error=str(e))
            raise
        except ccxt.ExchangeError as e:
            logger.error("Exchange error fetching OHLCV", symbol=symbol, error=str(e))
            raise

    async def get_ticker_data(self, symbol: str) -> Dict[str, Any]:
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return {"symbol": symbol, "lastPrice": str(ticker['last'])}
        except ccxt.NetworkError as e:
            logger.error("Network error fetching ticker", symbol=symbol, error=str(e))
            raise
        except ccxt.ExchangeError as e:
            logger.error("Exchange error fetching ticker", symbol=symbol, error=str(e))
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
            
            return {
                "orderId": order.get('id'),
                "status": order.get('status', 'unknown').upper(),
                "executedQty": order.get('filled', 0.0),
                "cummulativeQuoteQty": order.get('cost', 0.0)
            }
        except ccxt.InsufficientFunds as e:
            logger.error("Insufficient funds to place order", symbol=symbol, error=str(e))
            raise
        except ccxt.ExchangeError as e:
            logger.error("Exchange error placing order", symbol=symbol, error=str(e))
            raise

    async def fetch_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            order = await self.exchange.fetch_order(order_id, symbol)
            return {
                'id': order.get('id'),
                'status': order.get('status', 'unknown').upper(),
                'filled': order.get('filled', 0.0),
                'average': order.get('average', 0.0),
                'symbol': order.get('symbol')
            }
        except ccxt.OrderNotFound:
            logger.warning("Order not found on exchange", order_id=order_id, symbol=symbol)
            return None
        except ccxt.ExchangeError as e:
            logger.error("Error fetching order", order_id=order_id, error=str(e))
            raise

    async def get_balance(self) -> Dict[str, Any]:
        try:
            balance = await self.exchange.fetch_balance()
            return balance.get('total', {})
        except ccxt.ExchangeError as e:
            logger.error("Error fetching balance", error=str(e))
            raise

    async def close(self):
        logger.info("Closing connection to exchange", exchange=self.exchange.name)
        await self.exchange.close()
