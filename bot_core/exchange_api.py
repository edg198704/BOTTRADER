import abc
import time
import random
import asyncio
from typing import Dict, Any, List, Optional, Deque
from collections import deque
import pandas as pd
import ccxt.async_support as ccxt
from ccxt.base.errors import NetworkError, ExchangeError, InsufficientFunds, OrderNotFound, NotSupported, InvalidOrder, RequestTimeout, RateLimitExceeded

from bot_core.logger import get_logger
from bot_core.utils import async_retry, Clock
from bot_core.config import ExchangeConfig, BacktestConfig

logger = get_logger(__name__)

# --- Custom Exchange Exceptions ---

class BotExchangeError(Exception):
    """Base exception for exchange interactions."""
    pass

class BotInsufficientFundsError(BotExchangeError):
    """Raised when the exchange reports insufficient funds."""
    pass

class BotInvalidOrderError(BotExchangeError):
    """Raised when the order parameters are invalid (e.g. size too small/large)."""
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
    async def get_market_data(self, symbol: str, timeframe: str, limit: int) -> List[List[Any]]:
        pass

    @abc.abstractmethod
    async def get_ticker_data(self, symbol: str) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    async def get_tickers(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        pass

    @abc.abstractmethod
    async def fetch_order_book(self, symbol: str, limit: int = 5) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: Optional[float] = None, extra_params: Dict[str, Any] = None) -> Dict[str, Any]:
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
    async def get_balance(self) -> Dict[str, Any]:
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
        self.balances = initial_balances if initial_balances is not None else {"USDT": 10000.0, "BTC": 0.0}
        self.order_id_counter = 0
        self.last_price = 30000.0
        self.open_orders: Dict[str, Dict[str, Any]] = {}
        logger.info("MockExchangeAPI initialized", initial_balances=self.balances)

    async def get_market_data(self, symbol: str, timeframe: str, limit: int) -> List[List[Any]]:
        now = int(time.time() * 1000)
        try:
            unit = timeframe[-1]
            value = int(timeframe[:-1])
            multipliers = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800}
            interval_ms = value * multipliers.get(unit, 60) * 1000
        except (ValueError, IndexError):
            interval_ms = 3600 * 1000

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

    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: Optional[float] = None, extra_params: Dict[str, Any] = None) -> Dict[str, Any]:
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
            'average': 0.0,
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
            fill_price = self.last_price
            can_fill = False
            if order['type'] == 'MARKET': can_fill = True
            elif order['type'] == 'LIMIT':
                if order['side'] == 'BUY' and self.last_price <= order['price']: can_fill = True; fill_price = order['price']
                elif order['side'] == 'SELL' and self.last_price >= order['price']: can_fill = True; fill_price = order['price']

            if can_fill:
                base, quote = symbol.split('/')
                cost = order['quantity'] * fill_price
                fee = cost * 0.001
                if order['side'] == "BUY":
                    self.balances[quote] -= (cost + fee)
                    self.balances[base] = self.balances.get(base, 0) + order['quantity']
                else:
                    self.balances[base] -= order['quantity']
                    self.balances[quote] = self.balances.get(quote, 0) + (cost - fee)
                
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

    async def get_balance(self) -> Dict[str, Any]:
        return {asset: {"free": amount, "total": amount} for asset, amount in self.balances.items()}

    async def fetch_market_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        return {
            'symbol': symbol,
            'precision': {'amount': 1e-5, 'price': 1e-2},
            'limits': {'amount': {'min': 1e-5, 'max': 1000.0}, 'cost': {'min': 10.0, 'max': None}}
        }

    async def close(self): pass

class BacktestExchangeAPI(ExchangeAPI):
    """Simulated exchange for backtesting."""
    def __init__(self, data_source: Dict[str, pd.DataFrame], initial_balances: Dict[str, float], config: BacktestConfig):
        self.data = data_source
        self.balances = initial_balances
        self.config = config
        self.open_orders: Dict[str, Dict[str, Any]] = {}
        self.order_id_counter = 0

    def _get_current_candle(self, symbol: str) -> Optional[pd.Series]:
        df = self.data.get(symbol)
        if df is None or df.empty: return None
        current_time = pd.Timestamp(Clock.now()).tz_localize(None)
        try:
            idx = df.index.get_indexer([current_time], method='pad')[0]
            if idx == -1: return None
            return df.iloc[idx]
        except Exception: return None

    async def get_market_data(self, symbol: str, timeframe: str, limit: int) -> List[List[Any]]:
        df = self.data.get(symbol)
        if df is None: return []
        current_time = pd.Timestamp(Clock.now()).tz_localize(None)
        mask = df.index <= current_time
        sliced = df.loc[mask].tail(limit)
        ohlcv = []
        for ts, row in sliced.iterrows():
            ts_ms = int(ts.timestamp() * 1000)
            ohlcv.append([ts_ms, row['open'], row['high'], row['low'], row['close'], row['volume']])
        return ohlcv

    async def get_ticker_data(self, symbol: str) -> Dict[str, Any]:
        candle = self._get_current_candle(symbol)
        price = candle['close'] if candle is not None else 0.0
        spread = price * 0.0005
        return {"symbol": symbol, "last": price, "bid": price - (spread/2), "ask": price + (spread/2)}

    async def get_tickers(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        return {sym: await self.get_ticker_data(sym) for sym in symbols}

    async def fetch_order_book(self, symbol: str, limit: int = 5) -> Dict[str, Any]:
        candle = self._get_current_candle(symbol)
        price = candle['close'] if candle is not None else 0.0
        spread = price * 0.0005
        bids = [[price - spread/2 - i*0.1, 1.0] for i in range(limit)]
        asks = [[price + spread/2 + i*0.1, 1.0] for i in range(limit)]
        return {'symbol': symbol, 'bids': bids, 'asks': asks, 'timestamp': int(Clock.now().timestamp()*1000)}

    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: Optional[float] = None, extra_params: Dict[str, Any] = None) -> Dict[str, Any]:
        self.order_id_counter += 1
        order_id = f"bt_order_{self.order_id_counter}"
        order = {
            'id': order_id, 'symbol': symbol, 'side': side.upper(), 'type': order_type.upper(),
            'price': price, 'quantity': quantity, 'status': 'OPEN', 'filled': 0.0, 'average': 0.0,
            'timestamp': Clock.now()
        }
        if extra_params and 'clientOrderId' in extra_params:
            order['clientOrderId'] = extra_params['clientOrderId']
        self.open_orders[order_id] = order
        return {"orderId": order_id, "status": "OPEN"}

    async def fetch_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        order = self.open_orders.get(order_id)
        if not order: return None
        if order['status'] == 'OPEN':
            candle = self._get_current_candle(symbol)
            if candle is not None:
                should_fill = False
                fill_price = candle['close']
                if order['type'] == 'MARKET':
                    should_fill = True
                    slippage = self.config.slippage_pct
                    fill_price = fill_price * (1 + slippage) if order['side'] == 'BUY' else fill_price * (1 - slippage)
                elif order['type'] == 'LIMIT':
                    if order['side'] == 'BUY' and candle['low'] <= order['price']: should_fill = True; fill_price = order['price']
                    elif order['side'] == 'SELL' and candle['high'] >= order['price']: should_fill = True; fill_price = order['price']
                
                if should_fill:
                    base, quote = symbol.split('/')
                    cost = order['quantity'] * fill_price
                    fee_rate = self.config.taker_fee_pct if order['type'] == 'MARKET' else self.config.maker_fee_pct
                    fee = cost * fee_rate
                    if order['side'] == 'BUY':
                        if self.balances.get(quote, 0) >= (cost + fee):
                            self.balances[quote] -= (cost + fee)
                            self.balances[base] = self.balances.get(base, 0) + order['quantity']
                            order['status'] = 'FILLED'; order['filled'] = order['quantity']; order['average'] = fill_price; order['fee'] = {'cost': fee, 'currency': quote}
                    else:
                        if self.balances.get(base, 0) >= order['quantity']:
                            self.balances[base] -= order['quantity']
                            self.balances[quote] = self.balances.get(quote, 0) + (cost - fee)
                            order['status'] = 'FILLED'; order['filled'] = order['quantity']; order['average'] = fill_price; order['fee'] = {'cost': fee, 'currency': quote}
        return order

    async def fetch_order_by_client_id(self, symbol: str, client_order_id: str) -> Optional[Dict[str, Any]]:
        for order in self.open_orders.values():
            if order.get('clientOrderId') == client_order_id: return order
        return None

    async def fetch_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        return [o for o in self.open_orders.values() if o['symbol'] == symbol and o['status'] == 'OPEN']

    async def fetch_recent_orders(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        orders = [o for o in self.open_orders.values() if o['symbol'] == symbol]
        orders.sort(key=lambda x: x.get('timestamp'), reverse=True)
        return orders[:limit]

    async def cancel_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        order = self.open_orders.get(order_id)
        if order and order['status'] == 'OPEN': order['status'] = 'CANCELED'
        return order

    async def cancel_all_orders(self, symbol: str) -> List[Dict[str, Any]]:
        cancelled = []
        for order in self.open_orders.values():
            if order['symbol'] == symbol and order['status'] == 'OPEN':
                order['status'] = 'CANCELED'
                cancelled.append(order)
        return cancelled

    async def get_balance(self) -> Dict[str, Any]:
        return {k: {'free': v, 'total': v} for k, v in self.balances.items()}

    async def fetch_market_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        return {'symbol': symbol, 'precision': {'amount': 1e-5, 'price': 1e-2}, 'limits': {'amount': {'min': 1e-5, 'max': 10000}, 'cost': {'min': 1, 'max': None}}}

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
        
        # --- Client Order ID Cache ---
        # Maps client_order_id -> exchange_order_id
        # Critical for O(1) reconciliation without scanning open orders
        self._order_id_cache: Dict[str, str] = {}
        self._cache_max_size = 1000
        self._cache_keys: Deque[str] = deque()

        # --- Balance Cache ---
        self._balance_cache = BalanceCache(ttl_seconds=2.0)

        # --- Circuit Breaker ---
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
        
        # Apply retry and circuit breaker to all methods
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
                self._failure_count = 0 # Reset on success
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

    async def get_market_data(self, symbol: str, timeframe: str, limit: int) -> List[List[Any]]:
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

    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: Optional[float] = None, extra_params: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            params = extra_params or {}
            if order_type.upper() == 'MARKET':
                order = await self.exchange.create_market_order(symbol, side.lower(), quantity, params=params)
            elif order_type.upper() == 'LIMIT':
                if price is None: raise ValueError("Price required for LIMIT order.")
                order = await self.exchange.create_limit_order(symbol, side.lower(), quantity, price, params=params)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            # --- CACHE UPDATE ---
            if 'clientOrderId' in params and order.get('id'):
                self._cache_order_id(params['clientOrderId'], order['id'])
            
            return {"orderId": order.get('id'), "status": order.get('status', 'unknown').upper(), "filled": order.get('filled', 0.0), "cost": order.get('cost', 0.0)}
        except InsufficientFunds as e:
            raise BotInsufficientFundsError(str(e)) from e
        except InvalidOrder as e:
            raise BotInvalidOrderError(str(e)) from e

    async def fetch_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            order = await self.exchange.fetch_order(order_id, symbol)
            status_map = {'open': 'OPEN', 'closed': 'FILLED', 'canceled': 'CANCELED', 'rejected': 'REJECTED', 'expired': 'EXPIRED'}
            return {
                'id': order.get('id'), 'status': status_map.get(order.get('status'), 'UNKNOWN'),
                'filled': order.get('filled', 0.0), 'average': order.get('average', 0.0),
                'symbol': order.get('symbol'), 'price': order.get('price', 0.0),
                'side': order.get('side'), 'type': order.get('type'),
                'clientOrderId': order.get('clientOrderId'), 'fee': order.get('fee')
            }
        except OrderNotFound: return None

    async def fetch_order_by_client_id(self, symbol: str, client_order_id: str) -> Optional[Dict[str, Any]]:
        # 1. Fast Path: Check Cache
        if client_order_id in self._order_id_cache:
            exchange_id = self._order_id_cache[client_order_id]
            return await self.fetch_order(exchange_id, symbol)

        # 2. Slow Path: Scan Open Orders
        open_orders = await self.fetch_open_orders(symbol)
        for order in open_orders:
            if order.get('clientOrderId') == client_order_id:
                self._cache_order_id(client_order_id, order['id']) # Update cache
                return order
        
        # 3. Deep Scan: Recent Orders (if supported)
        if self.exchange.has.get('fetchOrders'):
            recent = await self.exchange.fetch_orders(symbol, limit=50)
            for order in recent:
                if order.get('clientOrderId') == client_order_id:
                    # Map status
                    status_map = {'open': 'OPEN', 'closed': 'FILLED', 'canceled': 'CANCELED', 'rejected': 'REJECTED', 'expired': 'EXPIRED'}
                    mapped = {
                        'id': order.get('id'), 'status': status_map.get(order.get('status'), 'UNKNOWN'),
                        'filled': order.get('filled', 0.0), 'average': order.get('average', 0.0),
                        'symbol': order.get('symbol'), 'price': order.get('price', 0.0),
                        'side': order.get('side'), 'type': order.get('type'),
                        'clientOrderId': order.get('clientOrderId'), 'fee': order.get('fee')
                    }
                    self._cache_order_id(client_order_id, order['id']) # Update cache
                    return mapped
        return None

    async def fetch_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        return await self.exchange.fetch_open_orders(symbol)

    async def fetch_recent_orders(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        if self.exchange.has.get('fetchOrders'):
            return await self.exchange.fetch_orders(symbol, limit=limit)
        return []

    async def cancel_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            await self.exchange.cancel_order(order_id, symbol)
            return await self.fetch_order(order_id, symbol)
        except OrderNotFound:
            return await self.fetch_order(order_id, symbol)

    async def cancel_all_orders(self, symbol: str) -> List[Dict[str, Any]]:
        if self.exchange.has.get('cancelAllOrders'):
            return await self.exchange.cancel_all_orders(symbol)
        open_orders = await self.fetch_open_orders(symbol)
        results = []
        for order in open_orders:
            results.append(await self.cancel_order(order['id'], symbol))
        return results

    async def get_balance(self) -> Dict[str, Any]:
        # Use Cache
        cached = await self._balance_cache.get()
        if cached:
            return cached
        
        # Fetch and Update Cache
        balance = await self.exchange.fetch_balance()
        total = balance.get('total', {})
        await self._balance_cache.update(total)
        return total

    async def close(self):
        await self.exchange.close()
