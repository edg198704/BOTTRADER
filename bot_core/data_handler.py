import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import pandas_ta as ta

from bot_core.logger import get_logger
from bot_core.config import BotConfig
from bot_core.exchange_api import ExchangeAPI

logger = get_logger(__name__)

class DataHandler:
    """
    Manages the lifecycle of market data for all symbols.
    Fetches historical data, calculates technical indicators, and efficiently updates it.
    """
    def __init__(self, exchange_api: ExchangeAPI, config: BotConfig, shared_latest_prices: Dict[str, float]):
        self.exchange_api = exchange_api
        self.config = config
        self.symbols = config.strategy.symbols
        self.timeframe = config.strategy.timeframe
        self.history_limit = config.data_handler.history_limit
        self.update_interval = config.strategy.interval_seconds * config.data_handler.update_interval_multiplier
        
        self._dataframes: Dict[str, pd.DataFrame] = {}
        self._shared_latest_prices = shared_latest_prices
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        logger.info("DataHandler initialized.")

    async def initialize_data(self):
        """Fetches the initial batch of historical data for all symbols."""
        logger.info("Initializing historical data...", symbols=self.symbols)
        tasks = [self._fetch_initial_history(symbol) for symbol in self.symbols]
        await asyncio.gather(*tasks)
        logger.info("Historical data initialization complete.")

    async def _fetch_initial_history(self, symbol: str):
        try:
            ohlcv_data = await self.exchange_api.get_market_data(symbol, self.timeframe, self.history_limit)
            if ohlcv_data:
                df = create_dataframe(ohlcv_data)
                if df is not None and not df.empty:
                    # Pre-calculate indicators on initialization
                    self._dataframes[symbol] = calculate_technical_indicators(df)
                    self._update_latest_price(symbol, self._dataframes[symbol])
                    logger.info("Loaded and processed initial historical data", symbol=symbol, records=len(df))
            else:
                logger.warning("Could not fetch initial OHLCV data.", symbol=symbol)
        except Exception as e:
            logger.error("Failed to fetch initial market data", symbol=symbol, error=str(e))

    async def run(self):
        """Starts the background data update loop."""
        if self._running:
            return
        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("DataHandler update loop started.")

    async def stop(self):
        """Stops the background data update loop."""
        self._running = False
        if self._update_task and not self._update_task.done():
            self._update_task.cancel()
        logger.info("DataHandler update loop stopped.")

    async def _update_loop(self):
        while self._running:
            try:
                tasks = [self._update_symbol_data(symbol) for symbol in self.symbols]
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                logger.info("Data update loop cancelled.")
                break
            except Exception as e:
                logger.error("Error in data update loop", error=str(e), exc_info=True)
            
            await asyncio.sleep(self.update_interval)

    async def _update_symbol_data(self, symbol: str):
        try:
            # Fetch last 2 candles to handle incomplete current candle
            latest_ohlcv = await self.exchange_api.get_market_data(symbol, self.timeframe, 2)
            if not latest_ohlcv:
                return

            latest_df = create_dataframe(latest_ohlcv)
            if latest_df is None or latest_df.empty:
                return

            if symbol in self._dataframes:
                # Get the raw data by dropping indicator columns before merging
                current_raw_df = self._dataframes[symbol][['open', 'high', 'low', 'close', 'volume']]
                combined_df = pd.concat([current_raw_df, latest_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                combined_df = combined_df.tail(self.history_limit + 50) # Keep a larger buffer for indicator calculations
            else:
                combined_df = latest_df

            # Re-calculate indicators on the updated, raw dataframe
            self._dataframes[symbol] = calculate_technical_indicators(combined_df)
            self._update_latest_price(symbol, self._dataframes[symbol])
            logger.debug("Market data updated and indicators recalculated", symbol=symbol)

        except Exception as e:
            logger.warning("Failed to update market data for symbol", symbol=symbol, error=str(e))

    def _update_latest_price(self, symbol: str, df: pd.DataFrame):
        if df is not None and not df.empty:
            self._shared_latest_prices[symbol] = df['close'].iloc[-1]

    def get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Returns the latest DataFrame for a symbol with pre-calculated technical indicators.
        Returns None if data is not available.
        """
        df = self._dataframes.get(symbol)
        if df is None or df.empty:
            logger.warning("No market data available for symbol", symbol=symbol)
            return None
        
        return df.copy() # Return a copy to prevent mutation

    async def fetch_full_history_for_symbol(self, symbol: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetches a large batch of historical data for a symbol, intended for model training."""
        logger.info("Fetching full historical data for training", symbol=symbol, limit=limit)
        try:
            ohlcv_data = await self.exchange_api.get_market_data(symbol, self.timeframe, limit)
            if ohlcv_data:
                df = create_dataframe(ohlcv_data)
                if df is not None and not df.empty:
                    # Calculate indicators on the full dataset
                    full_df = calculate_technical_indicators(df)
                    logger.info("Successfully fetched and processed training data", symbol=symbol, records=len(full_df))
                    return full_df
            logger.warning("Could not fetch sufficient training data.", symbol=symbol)
            return None
        except Exception as e:
            logger.error("Failed to fetch full historical data", symbol=symbol, error=str(e))
            return None

def create_dataframe(ohlcv_data: list) -> pd.DataFrame | None:
    """Create DataFrame from OHLCV data with complete validation."""
    try:
        if not ohlcv_data or len(ohlcv_data) == 0:
            logger.warning("OHLCV data is empty")
            return None
        
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if set(required_cols) - set(df.columns):
            logger.error("Missing required columns in OHLCV data")
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=['timestamp', 'close', 'open', 'high', 'low'], inplace=True)
        df['volume'].fillna(0, inplace=True)

        if len(df) < 20:
            logger.warning("Insufficient data after cleaning", final_rows=len(df))
            return None

        df.set_index('timestamp', inplace=True)
        return df
        
    except Exception as e:
        logger.error("Failed to create DataFrame", error=str(e), exc_info=True)
        return None

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate a comprehensive set of technical indicators using pandas-ta."""
    if df is None or len(df) < 50:
        logger.debug("DataFrame has insufficient data for all indicators.", data_length=len(df) if df is not None else 0)
        return df

    df_out = df.copy()

    # Define the strategy for pandas-ta
    ta_strategy = ta.Strategy(
        name="BotTrader Indicators",
        description="A collection of standard indicators for the trading bot.",
        ta=[
            {"kind": "rsi", "length": 14},
            {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
            {"kind": "bbands", "length": 20, "std": 2},
            {"kind": "atr", "length": 14},
            {"kind": "adx", "length": 14},
            {"kind": "sma", "length": 10},
            {"kind": "sma", "length": 20},
        ]
    )

    # Apply the strategy
    df_out.ta.strategy(ta_strategy)

    # Rename columns to match application's expectations
    rename_map = {
        'RSI_14': 'rsi',
        'MACD_12_26_9': 'macd',
        'MACDh_12_26_9': 'macd_hist',
        'MACDs_12_26_9': 'macd_signal',
        'BBL_20_2.0': 'bb_lower',
        'BBM_20_2.0': 'bb_middle',
        'BBU_20_2.0': 'bb_upper',
        'ATRr_14': 'atr',
        'ADX_14': 'adx',
        'SMA_10': 'sma_fast',
        'SMA_20': 'sma_slow'
    }
    df_out.rename(columns=rename_map, inplace=True)

    df_out.dropna(inplace=True)
    logger.debug("Technical indicators calculated using pandas-ta", row_count=len(df_out))
    return df_out
