import abc
import time
import random
from typing import Dict, Any, List, Optional
import ccxt.async_support as ccxt
from ccxt.base.errors import NetworkError, ExchangeError, InsufficientFunds, OrderNotFound

from bot_core.logger import get_logger
from bot_core.utils import async_retry
from bot_core.config import ExchangeConfig

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
    async def cancel_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Cancels an open order."""
        pass

    @abc.abstractmethod
    async def get_balance(self) -> Dict[str, Any]:
        """Retrieves all asset balances."""
        pass

    @abc.abstractmethod
    async def fetch_market_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetches exchange-specific trading rules for a symbol (precision, limits)."""
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
        self.open_orders: Dict[str, Dict[str, Any]] = {}
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
        
        order = {
            'id': order_id,
            'symbol': symbol,
            'side': side.upper(),
            'type': order_type.upper(),
            'price': price,
            'quantity': quantity,
            'status': 'OPEN',
            'filled': 0.0,
            'average': 0.0
        }
        self.open_orders[order_id] = order
        logger.info("Mock: Order placed", **order)

        return {"orderId": order_id, "status": "OPEN"}

    async def fetch_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        order = self.open_orders.get(order_id)
        if not order:
            return None

        # Simulate order fill
        if order['status'] == 'OPEN':
            fill_price = self.last_price
            
            can_fill_price = False
            if order['type'] == 'MARKET':
                can_fill_price = True
            elif order['type'] == 'LIMIT':
                if order['side'] == 'BUY' and self.last_price <= order['price']:
                    can_fill_price = True
                    fill_price = order['price'] # Assume no slippage for mock limit
                elif order['side'] == 'SELL' and self.last_price >= order['price']:
                    can_fill_price = True
                    fill_price = order['price']

            if can_fill_price:
                base_asset, quote_asset = symbol.split('/')
                cost = order['quantity'] * fill_price
                can_fill_balance = (order['side'] == "BUY" and self.balances.get(quote_asset, 0) >= cost) or \
                                   (order['side'] == "SELL" and self.balances.get(base_asset, 0) >= order['quantity'])

                if can_fill_balance:
                    if order['side'] == "BUY":
                        self.balances[quote_asset] -= cost
                        self.balances[base_asset] = self.balances.get(base_asset, 0) + order['quantity']
                    else: # SELL
                        self.balances[base_asset] -= order['quantity']
                        self.balances[quote_asset] = self.balances.get(quote_asset, 0) + cost
                    
                    order['status'] = 'FILLED'
                    order['filled'] = order['quantity']
                    order['average'] = fill_price
                    logger.info("Mock: Order filled on fetch", order_id=order_id, fill_price=fill_price)
                else:
                    order['status'] = 'REJECTED'
                    logger.warning("Mock: Insufficient balance for order", order_id=order_id)
        
        return order

    async def cancel_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        order = self.open_orders.get(order_id)
        if order and order['status'] == 'OPEN':
            order['status'] = 'CANCELED'
            logger.info("Mock: Order canceled", order_id=order_id)
            return order
        logger.warning("Mock: Order not found or not open for cancellation", order_id=order_id)
        return order

    async def get_balance(self) -> Dict[str, Any]:
        logger.debug("Mock: Getting all balances")
        return {asset: {"free": amount, "total": amount} for asset, amount in self.balances.items()}

    async def fetch_market_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        logger.debug("Mock: Fetching market details", symbol=symbol)
        # Return mock details that are permissive but allow for testing
        return {
            'symbol': symbol,
            'precision': {'amount': 1e-5, 'price': 1e-2},
            'limits': {
                'amount': {'min': 1e-5, 'max': 1000.0},
                'price': {'min': 0.01, 'max': 1000000.0},
                'cost': {'min': 10.0, 'max': None}
            }
        }

    async def close(self):
        logger.info("MockExchangeAPI connection closed.")

class CCXTExchangeAPI(ExchangeAPI):
    """Concrete implementation for a real exchange using ccxt."""
    def __init__(self, config: ExchangeConfig):
        exchange_class = getattr(ccxt, config.name.lower(), None)
        if not exchange_class:
            raise ValueError(f"Exchange '{config.name}' is not supported by ccxt.")
        
        if not config.api_key or not config.api_secret:
            logger.critical("API key and secret are required for a live exchange but were not found in environment variables (BOT_EXCHANGE_API_KEY, BOT_EXCHANGE_API_SECRET).")
            raise ValueError("Missing API key/secret for CCXT exchange.")

        self.exchange = exchange_class({
            'apiKey': config.api_key,
            'secret': config.api_secret,
            'enableRateLimit': True,
        })
        self._markets_cache: Optional[Dict[str, Any]] = None

        if config.testnet and self.exchange.has['test']:
            self.exchange.set_sandbox_mode(True)
            logger.info("CCXTExchangeAPI initialized in TESTNET mode", exchange=config.name)
        else:
            logger.info("CCXTExchangeAPI initialized in LIVE mode", exchange=config.name)

        # Dynamically apply the configured retry decorator to instance methods
        retry_decorator = async_retry(
            max_attempts=config.retry.max_attempts,
            delay_seconds=config.retry.delay_seconds,
            backoff_factor=config.retry.backoff_factor,
            exceptions=(NetworkError, ExchangeError)
        )
        self.get_market_data = retry_decorator(self.get_market_data)
        self.get_ticker_data = retry_decorator(self.get_ticker_data)
        self.place_order = retry_decorator(self.place_order)
        self.fetch_order = retry_decorator(self.fetch_order)
        self.cancel_order = retry_decorator(self.cancel_order)
        self.get_balance = retry_decorator(self.get_balance)
        self.fetch_market_details = retry_decorator(self.fetch_market_details)

    async def _load_markets_if_needed(self):
        if self._markets_cache is None:
            logger.info("Loading exchange markets for the first time...")
            self._markets_cache = await self.exchange.load_markets()
            logger.info("Exchange markets loaded and cached.")

    async def fetch_market_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            await self._load_markets_if_needed()
            return self.exchange.markets.get(symbol)
        except (NetworkError, ExchangeError) as e:
            logger.error("Final attempt failed for fetch_market_details", symbol=symbol, error=str(e))
            raise

    async def get_market_data(self, symbol: str, timeframe: str, limit: int) -> List[List[Any]]:
        try:
            return await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except (NetworkError, ExchangeError) as e:
            logger.error("Final attempt failed for get_market_data", symbol=symbol, error=str(e))
            raise

    async def get_ticker_data(self, symbol: str) -> Dict[str, Any]:
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return {"symbol": symbol, "lastPrice": str(ticker['last'])}
        except (NetworkError, ExchangeError) as e:
            logger.error("Final attempt failed for get_ticker_data", symbol=symbol, error=str(e))
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
                "filled": order.get('filled', 0.0),
                "cost": order.get('cost', 0.0)
            }
        except InsufficientFunds as e:
            logger.error("Insufficient funds to place order, not retrying.", symbol=symbol, error=str(e))
            raise
        except (NetworkError, ExchangeError) as e:
            logger.error("Final attempt failed for place_order", symbol=symbol, error=str(e))
            raise

    async def fetch_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            order = await self.exchange.fetch_order(order_id, symbol)
            status_map = {
                'open': 'OPEN',
                'closed': 'FILLED',  # 'closed' in ccxt means fully filled
                'canceled': 'CANCELED',
                'rejected': 'REJECTED',
                'expired': 'EXPIRED'
            }
            return {
                'id': order.get('id'),
                'status': status_map.get(order.get('status'), 'UNKNOWN'),
                'filled': order.get('filled', 0.0),
                'average': order.get('average', 0.0),
                'symbol': order.get('symbol'),
                'price': order.get('price', 0.0),
                'side': order.get('side'),
                'type': order.get('type')
            }
        except OrderNotFound:
            logger.warning("Order not found on exchange, not retrying.", order_id=order_id, symbol=symbol)
            return None
        except (NetworkError, ExchangeError) as e:
            logger.error("Final attempt failed for fetch_order", order_id=order_id, error=str(e))
            raise

    async def cancel_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            # ccxt's cancel_order returns the order structure
            await self.exchange.cancel_order(order_id, symbol)
            # Fetch to get standardized status after cancellation
            return await self.fetch_order(order_id, symbol)
        except OrderNotFound:
            logger.warning("Attempted to cancel an order that was not found (might be already filled or canceled).", order_id=order_id)
            # Try to fetch it anyway to see its final state
            return await self.fetch_order(order_id, symbol)
        except (NetworkError, ExchangeError) as e:
            logger.error("Final attempt failed for cancel_order", order_id=order_id, error=str(e))
            raise

    async def get_balance(self) -> Dict[str, Any]:
        try:
            balance = await self.exchange.fetch_balance()
            return balance.get('total', {})
        except (NetworkError, ExchangeError) as e:
            logger.error("Final attempt failed for get_balance", error=str(e))
            raise

    async def close(self):
        logger.info("Closing connection to exchange", exchange=self.exchange.name)
        await self.exchange.close()
