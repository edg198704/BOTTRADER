import asyncio
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import pandas_ta as ta
from concurrent.futures import ThreadPoolExecutor

from bot_core.logger import get_logger
from bot_core.config import BotConfig
from bot_core.exchange_api import ExchangeAPI
from bot_core.utils import generate_indicator_rename_map

logger = get_logger(__name__)

class DataHandler:
    """
    Manages the lifecycle of market data for all symbols.
    Fetches historical data, calculates technical indicators, and efficiently updates it.
    Includes caching and non-blocking processing to protect the async event loop.
    """
    def __init__(self, exchange_api: ExchangeAPI, config: BotConfig, shared_latest_prices: Dict[str, float]):
        self.exchange_api = exchange_api
        self.config = config
        self.symbols = config.strategy.symbols
        self.timeframe = config.strategy.timeframe
        self.history_limit = config.data_handler.history_limit
        self.update_interval = config.strategy.interval_seconds * config.data_handler.update_interval_multiplier
        self.indicators_config = config.strategy.indicators
        
        self._dataframes: Dict[str, pd.DataFrame] = {}
        self._shared_latest_prices = shared_latest_prices
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        
        # Persistence settings
        self.cache_dir = "market_data_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Executor for CPU-bound tasks (indicators) and Disk I/O
        # Using ThreadPoolExecutor as pandas releases GIL for many ops, and it's lighter than ProcessPool
        self._executor = ThreadPoolExecutor(max_workers=min(len(self.symbols) + 2, 8))
        
        logger.info("DataHandler initialized.", cache_dir=self.cache_dir)

    async def initialize_data(self):
        """Fetches the initial batch of historical data for all symbols, utilizing local cache."""
        logger.info("Initializing historical data...", symbols=self.symbols)
        tasks = [self._initialize_symbol(symbol) for symbol in self.symbols]
        await asyncio.gather(*tasks)
        logger.info("Historical data initialization complete.")

    async def _initialize_symbol(self, symbol: str):
        """Loads cache and fetches fresh data for a single symbol."""
        try:
            # 1. Load cached data (Non-blocking I/O)
            cached_df = await self._load_from_cache(symbol)
            
            # 2. Fetch fresh data from API (history_limit)
            # We always fetch the recent history to ensure we cover the gap since last run
            ohlcv_data = await self.exchange_api.get_market_data(symbol, self.timeframe, self.history_limit)
            fresh_df = create_dataframe(ohlcv_data)
            
            # 3. Merge
            if cached_df is not None and not cached_df.empty:
                if fresh_df is not None and not fresh_df.empty:
                    # Combine and drop duplicates based on index (timestamp)
                    combined = pd.concat([cached_df, fresh_df])
                    combined = combined[~combined.index.duplicated(keep='last')]
                    combined.sort_index(inplace=True)
                    df = combined
                else:
                    df = cached_df
            else:
                df = fresh_df

            if df is not None and not df.empty:
                # 4. Calculate indicators (Non-blocking CPU)
                self._dataframes[symbol] = await self._calculate_indicators_async(df)
                self._update_latest_price(symbol, self._dataframes[symbol])
                logger.info("Loaded initial data", symbol=symbol, total_records=len(df))
            else:
                logger.warning("No data available for symbol after initialization.", symbol=symbol)

        except Exception as e:
            logger.error("Failed to initialize data for symbol", symbol=symbol, error=str(e))

    async def run(self):
        """Starts the background data update loop."""
        if self._running:
            return
        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("DataHandler update loop started.")

    async def stop(self):
        """Stops the background data update loop and saves cache."""
        self._running = False
        if self._update_task and not self._update_task.done():
            self._update_task.cancel()
        
        # Save cache on shutdown
        logger.info("Saving market data cache...")
        save_tasks = [self._save_to_cache(symbol, df) for symbol, df in self._dataframes.items()]
        await asyncio.gather(*save_tasks)
        
        self._executor.shutdown(wait=True)
        logger.info("DataHandler stopped and executor shutdown.")

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
            # Fetch last few candles to handle current candle updates
            latest_ohlcv = await self.exchange_api.get_market_data(symbol, self.timeframe, 5)
            if not latest_ohlcv:
                return

            latest_df = create_dataframe(latest_ohlcv)
            if latest_df is None or latest_df.empty:
                return

            if symbol in self._dataframes:
                # Get the raw data (columns 0-4 are OHLCV)
                current_df = self._dataframes[symbol]
                raw_cols = ['open', 'high', 'low', 'close', 'volume']
                # Ensure we have the raw columns. If indicators renamed them, we might need to be careful.
                # But create_dataframe ensures these exist.
                current_raw = current_df[raw_cols]
                
                combined_df = pd.concat([current_raw, latest_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                
                # Keep a buffer larger than history_limit to build training data over time,
                # but cap it to prevent memory explosion (e.g., 10,000 candles).
                max_buffer = 10000 
                if len(combined_df) > max_buffer:
                    combined_df = combined_df.iloc[-max_buffer:]
            else:
                combined_df = latest_df

            # Re-calculate indicators (Non-blocking CPU)
            self._dataframes[symbol] = await self._calculate_indicators_async(combined_df)
            self._update_latest_price(symbol, self._dataframes[symbol])
            logger.debug("Market data updated", symbol=symbol)

        except Exception as e:
            logger.warning("Failed to update market data for symbol", symbol=symbol, error=str(e))

    async def _calculate_indicators_async(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper to run indicator calculation in thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.calculate_technical_indicators, df)

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
        
        # 1. Check current in-memory data first
        current_df = self._dataframes.get(symbol)
        if current_df is not None and len(current_df) >= limit:
            logger.info("Using in-memory data for training", symbol=symbol, records=len(current_df))
            return current_df.copy()

        # 2. If not enough, try to fetch from API
        try:
            ohlcv_data = await self.exchange_api.get_market_data(symbol, self.timeframe, limit)
            if ohlcv_data:
                df = create_dataframe(ohlcv_data)
                if df is not None and not df.empty:
                    # Calculate indicators on the full dataset (Non-blocking)
                    full_df = await self._calculate_indicators_async(df)
                    logger.info("Successfully fetched and processed training data", symbol=symbol, records=len(full_df))
                    return full_df
            
            logger.warning("Could not fetch sufficient training data from API, returning available data.", symbol=symbol)
            return current_df.copy() if current_df is not None else None
            
        except Exception as e:
            logger.error("Failed to fetch full historical data", symbol=symbol, error=str(e))
            return None

    async def _save_to_cache(self, symbol: str, df: pd.DataFrame):
        """Saves dataframe to CSV in executor."""
        if df is None or df.empty:
            return
        
        safe_symbol = symbol.replace('/', '_')
        path = os.path.join(self.cache_dir, f"{safe_symbol}.csv")
        
        loop = asyncio.get_running_loop()
        try:
            # Save only OHLCV to save space/time, indicators are recalculated on load
            raw_cols = ['open', 'high', 'low', 'close', 'volume']
            to_save = df[raw_cols]
            await loop.run_in_executor(self._executor, to_save.to_csv, path)
            logger.debug("Saved cache for symbol", symbol=symbol)
        except Exception as e:
            logger.error("Failed to save cache", symbol=symbol, error=str(e))

    async def _load_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """Loads dataframe from CSV in executor."""
        safe_symbol = symbol.replace('/', '_')
        path = os.path.join(self.cache_dir, f"{safe_symbol}.csv")
        
        if not os.path.exists(path):
            return None
            
        loop = asyncio.get_running_loop()
        try:
            df = await loop.run_in_executor(
                self._executor, 
                pd.read_csv, 
                path, 
                {'index_col': 'timestamp', 'parse_dates': True}
            )
            logger.debug("Loaded cache for symbol", symbol=symbol, records=len(df))
            return df
        except Exception as e:
            logger.error("Failed to load cache", symbol=symbol, error=str(e))
            return None

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators dynamically based on the strategy configuration."""
        if df is None or len(df) < 50:
            logger.debug("DataFrame has insufficient data for indicators.", data_length=len(df) if df is not None else 0)
            return df

        df_out = df.copy()

        # Create a pandas-ta strategy from the configuration
        ta_strategy = ta.Strategy(
            name="BotTrader Dynamic Indicators",
            description="Indicators dynamically loaded from config.",
            ta=self.indicators_config
        )

        # Apply the strategy
        # This is the CPU intensive part
        df_out.ta.strategy(ta_strategy)

        # Rename columns to a consistent, simplified format using the utility function
        try:
            rename_map = generate_indicator_rename_map(self.indicators_config)
            df_out.rename(columns=rename_map, inplace=True, errors='ignore')
        except ValueError as e:
            logger.error("Failed to generate indicator rename map. Check config.", error=str(e))
            return df_out

        df_out.dropna(inplace=True)
        logger.debug("Technical indicators calculated dynamically from config", row_count=len(df_out))
        return df_out

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
