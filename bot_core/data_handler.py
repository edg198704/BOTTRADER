import asyncio
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from bot_core.logger import get_logger
from bot_core.exchange_api import ExchangeAPI

logger = get_logger(__name__)

# --- Event Definitions ---

@dataclass
class Event:
    """Base class for all events."""
    pass

@dataclass
class MarketEvent(Event):
    """Handles the event of receiving new market data for a symbol."""
    symbol: str
    ohlcv_df: pd.DataFrame
    last_price: float

@dataclass
class SignalEvent(Event):
    """Handles the event of sending a signal from a Strategy object."""
    symbol: str
    action: str  # 'BUY', 'SELL'
    confidence: float

@dataclass
class OrderEvent(Event):
    """Handles the event of sending an Order to an execution system."""
    symbol: str
    side: str  # 'BUY', 'SELL'
    order_type: str  # 'MARKET', 'LIMIT'
    quantity: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FillEvent(Event):
    """Encapsulates the notion of a filled order, as returned from a brokerage."""
    symbol: str
    side: str
    quantity: float
    price: float
    order_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

# --- Data Handler Interface ---

class DataHandler(ABC):
    """Abstract base class for data handlers."""
    def __init__(self, event_queue: asyncio.Queue):
        self.event_queue = event_queue

    @abstractmethod
    async def start_streaming(self):
        """Starts the data streaming process."""
        pass

    @abstractmethod
    async def stop_streaming(self):
        """Stops the data streaming process."""
        pass

class LiveAPIDataHandler(DataHandler):
    """Handles fetching live market data from the exchange API."""
    def __init__(self, event_queue: asyncio.Queue, exchange_api: ExchangeAPI, symbols: List[str], interval_seconds: int):
        super().__init__(event_queue)
        self.exchange_api = exchange_api
        self.symbols = symbols
        self.interval_seconds = interval_seconds
        self.running = False
        self._stream_task: Optional[asyncio.Task] = None

    async def start_streaming(self):
        self.running = True
        self._stream_task = asyncio.create_task(self._stream_data())
        logger.info("Live data streaming started.", symbols=self.symbols)

    async def stop_streaming(self):
        self.running = False
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        logger.info("Live data streaming stopped.")

    async def _stream_data(self):
        while self.running:
            for symbol in self.symbols:
                try:
                    ohlcv_data = await self.exchange_api.get_market_data(symbol, '1h', 200)
                    ticker_data = await self.exchange_api.get_ticker_data(symbol)

                    if not ohlcv_data or not ticker_data or not ticker_data.get('lastPrice'):
                        logger.warning("Invalid market data received for symbol", symbol=symbol)
                        continue

                    df = create_dataframe(ohlcv_data)
                    if df is None: continue

                    last_price = float(ticker_data['lastPrice'])

                    market_event = MarketEvent(
                        symbol=symbol,
                        ohlcv_df=df,
                        last_price=last_price
                    )
                    await self.event_queue.put(market_event)
                    logger.debug("MarketEvent queued", symbol=symbol)

                except Exception as e:
                    logger.error("Error fetching data for symbol", symbol=symbol, error=str(e))
            
            await asyncio.sleep(self.interval_seconds)

# --- Utility Functions (preserved) ---

def create_dataframe(ohlcv_data: list) -> pd.DataFrame | None:
    """Create DataFrame from OHLCV data with validation."""
    try:
        if not ohlcv_data or len(ohlcv_data) < 20:
            logger.warning("Insufficient OHLCV data to create DataFrame.", data_length=len(ohlcv_data) if ohlcv_data else 0)
            return None

        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        if len(df) < 20:
            logger.warning("Insufficient data rows after cleaning.", row_count=len(df))
            return None

        return df
    except Exception as e:
        logger.error("Error creating DataFrame", error=str(e), exc_info=True)
        return None
