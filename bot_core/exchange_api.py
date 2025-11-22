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
from bot_core.utils import async_retry
from bot_core.config import ExchangeConfig
from bot_core.common import to_decimal, ZERO, Dec

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
    def __init__(self, ttl_seconds: float = 2.0):
        self.ttl = ttl_seconds
        self.last_update = 0.0
        self.data: Dict[str, Dict[str, Decimal]] = {}
        self._lock = asyncio.Lock()

    async def get(self) -> Optional[Dict[str, Dict[str, Decimal]]]:
        async with self._lock:
            if time.time() - self.last_update < self.ttl:
                return self.data
            return None

    async def update(self, data: Dict[str, Dict[str, Decimal]]):
        async with self._lock:
            self.data = data
            self.last_update = time.time()

# ----------------------------------

class ExchangeAPI(abc.ABC):
    """Abstract Base Class for interacting with a cryptocurrency exchange."""

    @abc.abstractmethod
    async def get_market_data(self, symbol: str, timeframe: str, limit: int) -> List[List[float]]:
        # Returns OHLCV as floats (Analysis usually prefers floats for numpy speed)
        pass

    @abc.abstractmethod
    async def get_ticker_data(self, symbol: str) -> Dict[str, Any]:
        # Returns floats for analysis, but TradeExecutor should convert to Decimal for pricing
        pass

    @abc.abstractmethod
    async def get_tickers(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        pass

    @abc.abstractmethod
    async def fetch_order_book(self, symbol: str, limit: int = 5) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    async def place_order(self, symbol: str, side: str, order_type: str, quantity: Decimal, price: Optional[Decimal] = None, extra_params: Dict[str, Any] = None) -> Dict[str, Any]:
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
    async def fetch_recent_orders(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    async def cancel_all_orders(self, symbol: str) -> List[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    async def get_balance(self) -> Dict[str, Dict[str, Decimal]]:
        pass

    @abc.abstractmethod
    async def fetch_market_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    async def close(self):
        pass

class MockExchangeAPI(ExchangeAPI):
    """A mock implementation of ExchangeAPI for testing and development."""

    def __init__(self, initial_balances: Optional[Dict[str, float]] = None):
        raw_balances = initial_balances if initial_balances is not None else {"USDT": 10000.0, "BTC": 0.0}
        self.balances = {k: to_decimal(v) for k, v in raw_balances.items()}
        self.order_id_counter = 0
        self.last_price = 30000.0
        self.open_orders: Dict[str, Dict[str, Any]] = {}
        logger.info("MockExchangeAPI initialized", initial_balances=self.balances)

    async def get_market_data(self, symbol: str, timeframe: str, limit: int) -> List[List[float]]:
        now = int(time.time() * 1000)
        interval_ms = 60000 
        ohlcv = []
        price = self.last_price
        for i in range(limit):
            ts = now - (limit - i - 1) * interval_ms
            open_price = price + random.uniform(-50, 50)
            high = max(open_price, price) + random.uniform(0, 20)
            low = min(open_price, price) - random.uniform(0, 20)
            close_price = price
            volume = random.uniform(10, 100)
            ohlcv.append([float(ts), open_price, high, low, close_price, volume])
            price += random.uniform(-100, 100)
        self.last_price = price
        return ohlcv

    async def get_ticker_data(self, symbol: str) -> Dict[str, Any]:
        self.last_price += random.uniform(-100, 100)
        spread = self.last_price * 0.0005
        return {
            "symbol": symbol, 
            "last": self.last_price,
            "bid": self.last_price - (spread / 2),
            "ask": self.last_price + (spread / 2)
        }

    async def get_tickers(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        results = {}
        for sym in symbols:
            results[sym] = await self.get_ticker_data(sym)
        return results

    async def fetch_order_book(self, symbol: str, limit: int = 5) -> Dict[str, Any]:
        price = self.last_price
        spread = price * 0.0005
        bids = [[price - spread/2 - i*0.1, 1.0] for i in range(limit)]
        asks = [[price + spread/2 + i*0.1, 1.0] for i in range(limit)]
        return {'symbol': symbol, 'bids': bids, 'asks': asks, 'timestamp': int(time.time()*1000)}

    async def place_order(self, symbol: str, side: str, order_type: str, quantity: Decimal, price: Optional[Decimal] = None, extra_params: Dict[str, Any] = None) -> Dict[str, Any]:
        self.order_id_counter += 1
        order_id = f"mock_order_{self.order_id_counter}"
        
        order = {
            'id': order_id,
            'symbol': symbol,
            'side': side.upper(),
            'type': order_type.upper(),
            'price': price if price else to_decimal(self.last_price),
            'quantity': quantity,
            'status': 'OPEN',
            'filled': ZERO,
            'average': ZERO,
            'timestamp': int(time.time() * 1000)
        }
        
        if extra_params and 'clientOrderId' in extra_params:
            order['clientOrderId'] = extra_params['clientOrderId']

        self.open_orders[order_id] = order
        return {"orderId": order_id, "status": "OPEN"}

    async def fetch_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        order = self.open_orders.get(order_id)
        if not order: return None

        if order['status'] == 'OPEN':
            fill_price = to_decimal(self.last_price)
            can_fill = False
            order_price = order['price']
            
            if order['type'] == 'MARKET': can_fill = True
            elif order['type'] == 'LIMIT':
                if order['side'] == 'BUY' and fill_price <= order_price: can_fill = True; fill_price = order_price
                elif order['side'] == 'SELL' and fill_price >= order_price: can_fill = True; fill_price = order_price

            if can_fill:
                base, quote = symbol.split('/')
                cost = order['quantity'] * fill_price
                fee = cost * to_decimal("0.001")
                if order['side'] == "BUY":
                    self.balances[quote] -= (cost + fee)
                    self.balances[base] = self.balances.get(base, ZERO) + order['quantity']
                else:
                    self.balances[base] -= order['quantity']
                    self.balances[quote] = self.balances.get(quote, ZERO) + (cost - fee)
                
                order['status'] = 'FILLED'
                order['filled'] = order['quantity']
                order['average'] = fill_price
                order['fee'] = {'cost': fee, 'currency': quote}
        return order

    async def fetch_order_by_client_id(self, symbol: str, client_order_id: str) -> Optional[Dict[str, Any]]:
        for order in self.open_orders.values():
            if order.get('clientOrderId') == client_order_id:
                return order
        return None

    async def fetch_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        return [order for order in self.open_orders.values() if order['symbol'] == symbol and order['status'] == 'OPEN']

    async def fetch_recent_orders(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        orders = [o for o in self.open_orders.values() if o['symbol'] == symbol]
        orders.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        return orders[:limit]

    async def cancel_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        order = self.open_orders.get(order_id)
        if order and order['status'] == 'OPEN':
            order['status'] = 'CANCELED'
            return order
        return order

    async def cancel_all_orders(self, symbol: str) -> List[Dict[str, Any]]:
        canceled = []
        for order in self.open_orders.values():
            if order['symbol'] == symbol and order['status'] == 'OPEN':
                order['status'] = 'CANCELED'
                canceled.append(order)
        return canceled

    async def get_balance(self) -> Dict[str, Dict[str, Decimal]]:
        return {asset: {"free": amount, "total": amount} for asset, amount in self.balances.items()}

    async def fetch_market_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        return {
            'symbol': symbol,
            'precision': {'amount': 1e-5, 'price': 1e-2},
            'limits': {'amount': {'min': 1e-5, 'max': 1000.0}, 'cost': {'min': 10.0, 'max': None}}
        }

    async def close(self): pass

class CCXTExchangeAPI(ExchangeAPI):
    """Concrete implementation for a real exchange using ccxt with caching and circuit breaker."""
    def __init__(self, config: ExchangeConfig):
        exchange_class = getattr(ccxt, config.name.lower(), None)
        if not exchange_class:
            raise ValueError(f"Exchange '{config.name}' is not supported by ccxt.")
        
        exchange_config = {'enableRateLimit': True}
        if config.api_key: exchange_config['apiKey'] = config.api_key.get_secret_value()
        if config.api_secret: exchange_config['secret'] = config.api_secret.get_secret_value()

        self.exchange = exchange_class(exchange_config)
        self._markets_cache: Optional[Dict[str, Any]] = None
        self._order_id_cache: Dict[str, str] = {}
        self._cache_max_size = 1000
        self._cache_keys: Deque[str] = deque()
        self._balance_cache = BalanceCache(ttl_seconds=2.0)

        # Circuit Breaker State
        self._circuit_open = False
        self._circuit_open_until = 0.0
        self._failure_count = 0
        self._failure_threshold = 5
        self._reset_timeout = 60.0

        if config.testnet and self.exchange.has['test']:
            self.exchange.set_sandbox_mode(True)
            logger.info("CCXTExchangeAPI initialized in TESTNET mode", exchange=config.name)
        else:
            logger.info("CCXTExchangeAPI initialized in LIVE mode", exchange=config.name)

        retry_decorator = async_retry(
            max_attempts=config.retry.max_attempts,
            delay_seconds=config.retry.delay_seconds,
            backoff_factor=config.retry.backoff_factor,
            exceptions=(NetworkError, ExchangeError, RequestTimeout, RateLimitExceeded)
        )
        
        methods = [
            'get_market_data', 'get_ticker_data', 'get_tickers', 'fetch_order_book',
            'place_order', 'fetch_order', 'fetch_open_orders', 'fetch_recent_orders',
            'cancel_order', 'cancel_all_orders', 'fetch_market_details'
        ]
        for method in methods:
            setattr(self, method, self._circuit_breaker_guard(retry_decorator(getattr(self, method))))

    def _circuit_breaker_guard(self, func):
        async def wrapper(*args, **kwargs):
            if self._circuit_open:
                if time.time() < self._circuit_open_until:
                    raise BotExchangeError("Exchange Circuit Breaker Open")
                else:
                    self._circuit_open = False
                    self._failure_count = 0
                    logger.info("Exchange Circuit Breaker Reset")
            
            try:
                result = await func(*args, **kwargs)
                self._failure_count = 0
                return result
            except (NetworkError, RequestTimeout, RateLimitExceeded) as e:
                self._failure_count += 1
                if self._failure_count >= self._failure_threshold:
                    self._circuit_open = True
                    self._circuit_open_until = time.time() + self._reset_timeout
                    logger.critical("Exchange Circuit Breaker TRIPPED", error=str(e))
                raise
        return wrapper

    def _cache_order_id(self, client_id: str, exchange_id: str):
        if client_id in self._order_id_cache:
            return
        if len(self._order_id_cache) >= self._cache_max_size:
            oldest = self._cache_keys.popleft()
            del self._order_id_cache[oldest]
        self._order_id_cache[client_id] = exchange_id
        self._cache_keys.append(client_id)

    async def _load_markets_if_needed(self):
        if self._markets_cache is None:
            self._markets_cache = await self.exchange.load_markets()

    async def fetch_market_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        await self._load_markets_if_needed()
        return self.exchange.markets.get(symbol)

    async def get_market_data(self, symbol: str, timeframe: str, limit: int) -> List[List[float]]:
        if limit <= 1000:
            return await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        duration = self.exchange.parse_timeframe(timeframe) * 1000
        now = self.exchange.milliseconds()
        since = now - int(limit * duration * 1.1)
        all_ohlcv = []
        while True:
            remaining = limit - len(all_ohlcv)
            if remaining <= 0: break
            batch = await self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=remaining + 50)
            if not batch: break
            if all_ohlcv:
                last_ts = all_ohlcv[-1][0]
                batch = [c for c in batch if c[0] > last_ts]
                if not batch: break
            all_ohlcv.extend(batch)
            since = batch[-1][0] + 1
            if since >= now: break
        return all_ohlcv[-limit:] if len(all_ohlcv) > limit else all_ohlcv

    async def get_ticker_data(self, symbol: str) -> Dict[str, Any]:
        t = await self.exchange.fetch_ticker(symbol)
        return {"symbol": symbol, "last": t.get('last'), "bid": t.get('bid'), "ask": t.get('ask')}

    async def get_tickers(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        if self.exchange.has['fetchTickers']:
            raw = await self.exchange.fetch_tickers(symbols)
            return {s: {"symbol": s, "last": t.get('last'), "bid": t.get('bid'), "ask": t.get('ask')} for s, t in raw.items() if s in symbols}
        else:
            return {s: await self.get_ticker_data(s) for s in symbols}

    async def fetch_order_book(self, symbol: str, limit: int = 5) -> Dict[str, Any]:
        return await self.exchange.fetch_order_book(symbol, limit=limit)

    async def place_order(self, symbol: str, side: str, order_type: str, quantity: Decimal, price: Optional[Decimal] = None, extra_params: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            params = extra_params or {}
            # Convert Decimal to float for CCXT
            qty_float = float(quantity)
            price_float = float(price) if price else None

            if order_type.upper() == 'MARKET':
                order = await self.exchange.create_market_order(symbol, side.lower(), qty_float, params=params)
            elif order_type.upper() == 'LIMIT':
                if price_float is None: raise ValueError("Price required for LIMIT order.")
                order = await self.exchange.create_limit_order(symbol, side.lower(), qty_float, price_float, params=params)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            if 'clientOrderId' in params and order.get('id'):
                self._cache_order_id(params['clientOrderId'], order['id'])
            
            return self._normalize_order(order)
        except InsufficientFunds as e:
            raise BotInsufficientFundsError(str(e)) from e
        except InvalidOrder as e:
            raise BotInvalidOrderError(str(e)) from e

    def _normalize_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Converts CCXT order response to Decimal-based dictionary."""
        status_map = {'open': 'OPEN', 'closed': 'FILLED', 'canceled': 'CANCELED', 'rejected': 'REJECTED', 'expired': 'EXPIRED'}
        return {
            'id': order.get('id'),
            'status': status_map.get(order.get('status'), 'UNKNOWN'),
            'filled': to_decimal(order.get('filled', 0.0)),
            'average': to_decimal(order.get('average', 0.0)),
            'symbol': order.get('symbol'),
            'price': to_decimal(order.get('price', 0.0)),
            'side': order.get('side'),
            'type': order.get('type'),
            'clientOrderId': order.get('clientOrderId'),
            'fee': order.get('fee')
        }

    async def fetch_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            order = await self.exchange.fetch_order(order_id, symbol)
            return self._normalize_order(order)
        except OrderNotFound: return None

    async def fetch_order_by_client_id(self, symbol: str, client_order_id: str) -> Optional[Dict[str, Any]]:
        if client_order_id in self._order_id_cache:
            exchange_id = self._order_id_cache[client_order_id]
            return await self.fetch_order(exchange_id, symbol)

        open_orders = await self.fetch_open_orders(symbol)
        for order in open_orders:
            if order.get('clientOrderId') == client_order_id:
                self._cache_order_id(client_order_id, order['id'])
                return order
        
        if self.exchange.has.get('fetchOrders'):
            recent = await self.exchange.fetch_orders(symbol, limit=50)
            for order in recent:
                if order.get('clientOrderId') == client_order_id:
                    self._cache_order_id(client_order_id, order['id'])
                    return self._normalize_order(order)
        return None

    async def fetch_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        orders = await self.exchange.fetch_open_orders(symbol)
        return [self._normalize_order(o) for o in orders]

    async def fetch_recent_orders(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        if self.exchange.has.get('fetchOrders'):
            orders = await self.exchange.fetch_orders(symbol, limit=limit)
            return [self._normalize_order(o) for o in orders]
        return []

    async def cancel_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            await self.exchange.cancel_order(order_id, symbol)
            return await self.fetch_order(order_id, symbol)
        except OrderNotFound:
            return await self.fetch_order(order_id, symbol)

    async def cancel_all_orders(self, symbol: str) -> List[Dict[str, Any]]:
        if self.exchange.has.get('cancelAllOrders'):
            await self.exchange.cancel_all_orders(symbol)
            return [] 
        open_orders = await self.fetch_open_orders(symbol)
        results = []
        for order in open_orders:
            results.append(await self.cancel_order(order['id'], symbol))
        return results

    async def get_balance(self) -> Dict[str, Dict[str, Decimal]]:
        cached = await self._balance_cache.get()
        if cached:
            return cached
        
        balance = await self.exchange.fetch_balance()
        total = balance.get('total', {})
        decimal_total = {k: {"free": to_decimal(balance[k]['free']), "total": to_decimal(v)} for k, v in total.items()}
        await self._balance_cache.update(decimal_total)
        return decimal_total

    async def close(self):
        await self.exchange.close()
