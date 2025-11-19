import abc
import time
import random
from typing import Dict, Any, List, Optional
import pandas as pd
import ccxt.async_support as ccxt
from ccxt.base.errors import NetworkError, ExchangeError, InsufficientFunds, OrderNotFound, NotSupported, InvalidOrder

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

# ----------------------------------

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
    async def fetch_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetches all currently open orders for a symbol."""
        pass

    @abc.abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Cancels an open order."""
        pass

    @abc.abstractmethod
    async def cancel_all_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """Cancels all open orders for a symbol."""
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
        
        try:
            unit = timeframe[-1]
            value = int(timeframe[:-1])
            multipliers = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800}
            interval_ms = value * multipliers.get(unit, 60) * 1000
        except (ValueError, IndexError):
            logger.warning("Mock: Could not parse timeframe, defaulting to 1h", timeframe=timeframe)
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
                    fill_price = order['price']
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

    async def fetch_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        return [order for order in self.open_orders.values() if order['symbol'] == symbol and order['status'] == 'OPEN']

    async def cancel_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        order = self.open_orders.get(order_id)
        if order and order['status'] == 'OPEN':
            order['status'] = 'CANCELED'
            logger.info("Mock: Order canceled", order_id=order_id)
            return order
        logger.warning("Mock: Order not found or not open for cancellation", order_id=order_id)
        return order

    async def cancel_all_orders(self, symbol: str) -> List[Dict[str, Any]]:
        canceled_orders = []
        for order_id, order in list(self.open_orders.items()):
            if order['symbol'] == symbol and order['status'] == 'OPEN':
                order['status'] = 'CANCELED'
                canceled_orders.append(order)
        logger.info("Mock: All orders canceled", symbol=symbol, count=len(canceled_orders))
        return canceled_orders

    async def get_balance(self) -> Dict[str, Any]:
        logger.debug("Mock: Getting all balances")
        return {asset: {"free": amount, "total": amount} for asset, amount in self.balances.items()}

    async def fetch_market_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        logger.debug("Mock: Fetching market details", symbol=symbol)
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

class BacktestExchangeAPI(ExchangeAPI):
    """
    A simulated exchange that replays historical data for backtesting.
    It uses the centralized Clock to determine the 'current' time and data availability.
    Supports fee deduction and slippage simulation.
    """
    def __init__(self, data_source: Dict[str, pd.DataFrame], initial_balances: Dict[str, float], config: BacktestConfig):
        self.data = data_source
        self.balances = initial_balances
        self.config = config
        self.open_orders: Dict[str, Dict[str, Any]] = {}
        self.order_id_counter = 0
        logger.info("BacktestExchangeAPI initialized.", 
                    maker_fee=config.maker_fee_pct, 
                    taker_fee=config.taker_fee_pct, 
                    slippage=config.slippage_pct)

    def _get_current_candle(self, symbol: str) -> Optional[pd.Series]:
        df = self.data.get(symbol)
        if df is None or df.empty:
            return None
        
        current_time = pd.Timestamp(Clock.now()).tz_localize(None)
        try:
            idx = df.index.get_indexer([current_time], method='pad')[0]
            if idx == -1:
                return None
            return df.iloc[idx]
        except Exception:
            return None

    async def get_market_data(self, symbol: str, timeframe: str, limit: int) -> List[List[Any]]:
        df = self.data.get(symbol)
        if df is None:
            return []
        
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
        return {"symbol": symbol, "lastPrice": str(price)}

    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        self.order_id_counter += 1
        order_id = f"bt_order_{self.order_id_counter}"
        
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
            'timestamp': Clock.now()
        }
        self.open_orders[order_id] = order
        return {"orderId": order_id, "status": "OPEN"}

    async def fetch_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        order = self.open_orders.get(order_id)
        if not order:
            return None
        
        if order['status'] == 'OPEN':
            candle = self._get_current_candle(symbol)
            if candle is not None:
                should_fill = False
                fill_price = candle['close']
                
                if order['type'] == 'MARKET':
                    should_fill = True
                    # Apply Slippage for MARKET orders
                    slippage = self.config.slippage_pct
                    if order['side'] == 'BUY':
                        fill_price = fill_price * (1 + slippage)
                    else:
                        fill_price = fill_price * (1 - slippage)

                elif order['type'] == 'LIMIT':
                    limit_price = order['price']
                    if order['side'] == 'BUY':
                        if candle['low'] <= limit_price:
                            should_fill = True
                            fill_price = limit_price
                    else: # SELL
                        if candle['high'] >= limit_price:
                            should_fill = True
                            fill_price = limit_price
                
                if should_fill:
                    base, quote = symbol.split('/')
                    cost = order['quantity'] * fill_price
                    
                    # Calculate Fee
                    fee_rate = self.config.taker_fee_pct if order['type'] == 'MARKET' else self.config.maker_fee_pct
                    fee = cost * fee_rate
                    
                    # Execute Balance Updates with Fees
                    # We simulate fees being paid in the QUOTE asset for simplicity and consistency
                    if order['side'] == 'BUY':
                        total_cost = cost + fee
                        if self.balances.get(quote, 0) >= total_cost:
                            self.balances[quote] -= total_cost
                            self.balances[base] = self.balances.get(base, 0) + order['quantity']
                            order['status'] = 'FILLED'
                            order['filled'] = order['quantity']
                            order['average'] = fill_price
                    else: # SELL
                        if self.balances.get(base, 0) >= order['quantity']:
                            self.balances[base] -= order['quantity']
                            net_proceeds = cost - fee
                            self.balances[quote] = self.balances.get(quote, 0) + net_proceeds
                            order['status'] = 'FILLED'
                            order['filled'] = order['quantity']
                            order['average'] = fill_price

        return order

    async def fetch_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        return [o for o in self.open_orders.values() if o['symbol'] == symbol and o['status'] == 'OPEN']

    async def cancel_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        order = self.open_orders.get(order_id)
        if order and order['status'] == 'OPEN':
            order['status'] = 'CANCELED'
            return order
        return order

    async def cancel_all_orders(self, symbol: str) -> List[Dict[str, Any]]:
        cancelled = []
        for oid, order in self.open_orders.items():
            if order['symbol'] == symbol and order['status'] == 'OPEN':
                order['status'] = 'CANCELED'
                cancelled.append(order)
        return cancelled

    async def get_balance(self) -> Dict[str, Any]:
        return {k: {'free': v, 'total': v} for k, v in self.balances.items()}

    async def fetch_market_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        return {
            'symbol': symbol,
            'precision': {'amount': 1e-5, 'price': 1e-2},
            'limits': {'amount': {'min': 1e-5, 'max': 10000}, 'cost': {'min': 1, 'max': None}}
        }

    async def close(self):
        pass

class CCXTExchangeAPI(ExchangeAPI):
    """Concrete implementation for a real exchange using ccxt."""
    def __init__(self, config: ExchangeConfig):
        exchange_class = getattr(ccxt, config.name.lower(), None)
        if not exchange_class:
            raise ValueError(f"Exchange '{config.name}' is not supported by ccxt.")
        
        exchange_config = {
            'enableRateLimit': True,
        }
        if config.api_key:
            exchange_config['apiKey'] = config.api_key.get_secret_value()
        if config.api_secret:
            exchange_config['secret'] = config.api_secret.get_secret_value()

        self.exchange = exchange_class(exchange_config)
        self._markets_cache: Optional[Dict[str, Any]] = None

        if config.testnet and self.exchange.has['test']:
            self.exchange.set_sandbox_mode(True)
            logger.info("CCXTExchangeAPI initialized in TESTNET mode", exchange=config.name)
        else:
            logger.info("CCXTExchangeAPI initialized in LIVE mode", exchange=config.name)

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
        self.fetch_open_orders = retry_decorator(self.fetch_open_orders)
        self.cancel_order = retry_decorator(self.cancel_order)
        self.cancel_all_orders = retry_decorator(self.cancel_all_orders)
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
            if limit <= 1000:
                return await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            duration = self.exchange.parse_timeframe(timeframe) * 1000
            now = self.exchange.milliseconds()
            since = now - int(limit * duration * 1.1)
            
            all_ohlcv = []
            while True:
                remaining = limit - len(all_ohlcv)
                if remaining <= 0: break
                
                request_limit = remaining + 50
                batch = await self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=request_limit)
                if not batch: break
                
                if all_ohlcv:
                    last_ts = all_ohlcv[-1][0]
                    batch = [candle for candle in batch if candle[0] > last_ts]
                    if not batch: break

                all_ohlcv.extend(batch)
                since = batch[-1][0] + 1
                if since >= now: break

            if len(all_ohlcv) > limit:
                return all_ohlcv[-limit:]
            return all_ohlcv

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
            logger.error("Insufficient funds to place order.", symbol=symbol, error=str(e))
            raise BotInsufficientFundsError(str(e)) from e
        except InvalidOrder as e:
            logger.error("Invalid order parameters.", symbol=symbol, error=str(e))
            raise BotInvalidOrderError(str(e)) from e
        except (NetworkError, ExchangeError) as e:
            logger.error("Final attempt failed for place_order", symbol=symbol, error=str(e))
            raise

    async def fetch_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            order = await self.exchange.fetch_order(order_id, symbol)
            status_map = {
                'open': 'OPEN', 'closed': 'FILLED', 'canceled': 'CANCELED',
                'rejected': 'REJECTED', 'expired': 'EXPIRED'
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

    async def fetch_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        try:
            orders = await self.exchange.fetch_open_orders(symbol)
            return orders
        except (NetworkError, ExchangeError) as e:
            logger.error("Final attempt failed for fetch_open_orders", symbol=symbol, error=str(e))
            raise

    async def cancel_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            await self.exchange.cancel_order(order_id, symbol)
            return await self.fetch_order(order_id, symbol)
        except OrderNotFound:
            logger.warning("Attempted to cancel an order that was not found.", order_id=order_id)
            return await self.fetch_order(order_id, symbol)
        except (NetworkError, ExchangeError) as e:
            logger.error("Final attempt failed for cancel_order", order_id=order_id, error=str(e))
            raise

    async def cancel_all_orders(self, symbol: str) -> List[Dict[str, Any]]:
        try:
            # Try native method first if available
            if self.exchange.has.get('cancelAllOrders'):
                return await self.exchange.cancel_all_orders(symbol)
            
            # Fallback: fetch and cancel individually
            open_orders = await self.fetch_open_orders(symbol)
            results = []
            for order in open_orders:
                res = await self.cancel_order(order['id'], symbol)
                results.append(res)
            return results
        except (NetworkError, ExchangeError) as e:
            logger.error("Final attempt failed for cancel_all_orders", symbol=symbol, error=str(e))
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
