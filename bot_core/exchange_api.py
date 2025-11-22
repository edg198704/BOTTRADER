import abc
import time
import random
import asyncio
from typing import Dict, Any, List, Optional, Deque
from collections import deque
from decimal import Decimal
import ccxt.async_support as ccxt
from ccxt.base.errors import NetworkError, ExchangeError, InsufficientFunds, OrderNotFound, NotSupported, InvalidOrder, RequestTimeout, RateLimitExceeded

from bot_core.logger import get_logger
from bot_core.utils import async_retry, Clock
from bot_core.config import ExchangeConfig, BacktestConfig
from bot_core.common import Arith

logger = get_logger(__name__)

# --- Custom Exchange Exceptions ---

class BotExchangeError(Exception):
    pass

class BotInsufficientFundsError(BotExchangeError):
    pass

class BotInvalidOrderError(BotExchangeError):
    pass

# --- Caching Utilities ---

class BalanceCache:
    """Simple TTL cache for balance data to prevent rate limit exhaustion."""
    def __init__(self, ttl_seconds: float = 2.0):
        self.ttl = ttl_seconds
        self.last_update = 0.0
        self.data: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def get(self) -> Optional[Dict[str, Any]]:
        async with self._lock:
            if time.time() - self.last_update < self.ttl:
                return self.data
            return None

    async def update(self, data: Dict[str, Any]):
        async with self._lock:
            self.data = data
            self.last_update = time.time()

# ----------------------------------

class ExchangeAPI(abc.ABC):
    """Abstract Base Class for interacting with a cryptocurrency exchange."""

    @abc.abstractmethod
    async def get_market_data(self, symbol: str, timeframe: str, limit: int) -> List[List[float]]:
        """Returns OHLCV data. Kept as floats for Pandas compatibility."""
        pass

    @abc.abstractmethod
    async def get_ticker_data(self, symbol: str) -> Dict[str, float]:
        """Returns ticker data. Kept as floats for Strategy analysis."""
        pass

    @abc.abstractmethod
    async def get_tickers(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        pass

    @abc.abstractmethod
    async def fetch_order_book(self, symbol: str, limit: int = 5) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    async def place_order(self, symbol: str, side: str, order_type: str, quantity: Decimal, price: Optional[Decimal] = None, extra_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Places an order. Quantity and Price must be Decimals."""
        pass
    
    @abc.abstractmethod
    async def fetch_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    async def fetch_order_by_client_id(self, symbol: str, client_order_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    async def fetch_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    async def get_balance(self) -> Dict[str, Dict[str, Decimal]]:
        """Returns balances as Decimals."""
        pass

    @abc.abstractmethod
    async def fetch_market_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    async def close(self):
        pass

class MockExchangeAPI(ExchangeAPI):
    def __init__(self, initial_balances: Optional[Dict[str, float]] = None):
        # Convert initial balances to Decimal
        raw_balances = initial_balances if initial_balances is not None else {"USDT": 10000.0, "BTC": 0.0}
        self.balances = {k: Arith.decimal(v) for k, v in raw_balances.items()}
        self.order_id_counter = 0
        self.last_price = 30000.0
        self.open_orders: Dict[str, Dict[str, Any]] = {}

    async def get_market_data(self, symbol: str, timeframe: str, limit: int) -> List[List[float]]:
        # Mock data generation (floats are fine here for pandas)
        return [[time.time()*1000, 30000.0, 30100.0, 29900.0, 30050.0, 10.0] for _ in range(limit)]

    async def get_ticker_data(self, symbol: str) -> Dict[str, float]:
        return {"symbol": symbol, "last": self.last_price, "bid": self.last_price - 1, "ask": self.last_price + 1}

    async def get_tickers(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        return {s: await self.get_ticker_data(s) for s in symbols}

    async def fetch_order_book(self, symbol: str, limit: int = 5) -> Dict[str, Any]:
        return {'symbol': symbol, 'bids': [], 'asks': [], 'timestamp': int(time.time()*1000)}

    async def place_order(self, symbol: str, side: str, order_type: str, quantity: Decimal, price: Optional[Decimal] = None, extra_params: Dict[str, Any] = None) -> Dict[str, Any]:
        self.order_id_counter += 1
        order_id = f"mock_order_{self.order_id_counter}"
        order = {
            'id': order_id, 'symbol': symbol, 'side': side.upper(), 'type': order_type.upper(),
            'price': float(price) if price else None, 'quantity': float(quantity),
            'status': 'OPEN', 'filled': 0.0, 'average': 0.0, 'timestamp': int(time.time() * 1000)
        }
        if extra_params and 'clientOrderId' in extra_params:
            order['clientOrderId'] = extra_params['clientOrderId']
        self.open_orders[order_id] = order
        return {"orderId": order_id, "status": "OPEN"}

    async def fetch_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        return self.open_orders.get(order_id)

    async def fetch_order_by_client_id(self, symbol: str, client_order_id: str) -> Optional[Dict[str, Any]]:
        for order in self.open_orders.values():
            if order.get('clientOrderId') == client_order_id: return order
        return None

    async def fetch_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        return list(self.open_orders.values())

    async def cancel_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        if order_id in self.open_orders:
            self.open_orders[order_id]['status'] = 'CANCELED'
            return self.open_orders[order_id]
        return None

    async def get_balance(self) -> Dict[str, Dict[str, Decimal]]:
        return {k: {'free': v, 'total': v} for k, v in self.balances.items()}

    async def fetch_market_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        return {'symbol': symbol, 'precision': {'amount': 1e-5, 'price': 1e-2}, 'limits': {'amount': {'min': 1e-5}}}

    async def close(self): pass

class CCXTExchangeAPI(ExchangeAPI):
    def __init__(self, config: ExchangeConfig):
        exchange_class = getattr(ccxt, config.name.lower(), None)
        if not exchange_class: raise ValueError(f"Exchange '{config.name}' not supported.")
        
        conf = {'enableRateLimit': True}
        if config.api_key: conf['apiKey'] = config.api_key.get_secret_value()
        if config.api_secret: conf['secret'] = config.api_secret.get_secret_value()
        
        self.exchange = exchange_class(conf)
        self._balance_cache = BalanceCache(ttl_seconds=2.0)
        self._order_id_cache: Dict[str, str] = {}
        
        if config.testnet and self.exchange.has['test']:
            self.exchange.set_sandbox_mode(True)

    async def get_market_data(self, symbol: str, timeframe: str, limit: int) -> List[List[float]]:
        return await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    async def get_ticker_data(self, symbol: str) -> Dict[str, float]:
        t = await self.exchange.fetch_ticker(symbol)
        return {"symbol": symbol, "last": t.get('last'), "bid": t.get('bid'), "ask": t.get('ask')}

    async def get_tickers(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        if self.exchange.has['fetchTickers']:
            raw = await self.exchange.fetch_tickers(symbols)
            return {s: {"symbol": s, "last": t.get('last'), "bid": t.get('bid'), "ask": t.get('ask')} for s, t in raw.items() if s in symbols}
        return {s: await self.get_ticker_data(s) for s in symbols}

    async def fetch_order_book(self, symbol: str, limit: int = 5) -> Dict[str, Any]:
        return await self.exchange.fetch_order_book(symbol, limit=limit)

    async def place_order(self, symbol: str, side: str, order_type: str, quantity: Decimal, price: Optional[Decimal] = None, extra_params: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            # Convert Decimals to floats for CCXT
            qty_float = float(quantity)
            price_float = float(price) if price else None
            params = extra_params or {}
            
            if order_type.upper() == 'MARKET':
                order = await self.exchange.create_market_order(symbol, side.lower(), qty_float, params=params)
            else:
                if price_float is None: raise ValueError("Price required for LIMIT order")
                order = await self.exchange.create_limit_order(symbol, side.lower(), qty_float, price_float, params=params)
            
            if 'clientOrderId' in params and order.get('id'):
                self._order_id_cache[params['clientOrderId']] = order['id']
                
            return order
        except InsufficientFunds as e: raise BotInsufficientFundsError(str(e)) from e
        except InvalidOrder as e: raise BotInvalidOrderError(str(e)) from e

    async def fetch_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        try: return await self.exchange.fetch_order(order_id, symbol)
        except OrderNotFound: return None

    async def fetch_order_by_client_id(self, symbol: str, client_order_id: str) -> Optional[Dict[str, Any]]:
        if client_order_id in self._order_id_cache:
            return await self.fetch_order(self._order_id_cache[client_order_id], symbol)
        
        open_orders = await self.fetch_open_orders(symbol)
        for order in open_orders:
            if order.get('clientOrderId') == client_order_id:
                self._order_id_cache[client_order_id] = order['id']
                return order
        return None

    async def fetch_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        return await self.exchange.fetch_open_orders(symbol)

    async def cancel_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            await self.exchange.cancel_order(order_id, symbol)
            return await self.fetch_order(order_id, symbol)
        except OrderNotFound: return None

    async def get_balance(self) -> Dict[str, Dict[str, Decimal]]:
        cached = await self._balance_cache.get()
        if cached: return cached
        
        raw = await self.exchange.fetch_balance()
        total = raw.get('total', {})
        # Convert to Decimal
        decimal_total = {k: {'free': Arith.decimal(v), 'total': Arith.decimal(v)} for k, v in total.items()}
        
        await self._balance_cache.update(decimal_total)
        return decimal_total

    async def fetch_market_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        await self.exchange.load_markets()
        return self.exchange.markets.get(symbol)

    async def close(self):
        await self.exchange.close()
