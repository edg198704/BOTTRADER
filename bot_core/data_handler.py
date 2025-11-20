import asyncio
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import pandas_ta as ta
from concurrent.futures import ThreadPoolExecutor

from bot_core.logger import get_logger

from bot_core.config import BotConfig, AIEnsembleStrategyParams
from bot_core.exchange_api import ExchangeAPI
from bot_core.utils import generate_indicator_rename_map, parse_timeframe_to_seconds, Clock, calculate_min_history_depth

logger = get_logger(__name__)

class DataHandler:
    """
    Manages the lifecycle of market data for all symbols.
    Maintains two buffers:
    1. _raw_buffers: Long-term storage of raw OHLCV data (for AI training).
    2. _dataframes: Short-term window with calculated indicators (for Strategy analysis).
    """
    def __init__(self, exchange_api: ExchangeAPI, config: BotConfig, shared_latest_prices: Dict[str, float]):
        self.exchange_api = exchange_api
        self.config = config
        
        # Initialize symbols list from strategy config
        self.symbols = list(config.strategy.symbols)
        
        # Automatically track Market Leader if configured in AI Strategy
        if isinstance(config.strategy.params, AIEnsembleStrategyParams):
            leader = config.strategy.params.market_leader_symbol
            if leader and leader not in self.symbols:
                self.symbols.append(leader)
                logger.info(f"Added market leader symbol {leader} to DataHandler tracking.")

        self.timeframe = config.strategy.timeframe
        self.secondary_timeframes = config.strategy.secondary_timeframes
        
        # Dynamic History Limit Calculation
        min_required = calculate_min_history_depth(config.strategy.indicators)
        
        # Scale history limit if secondary timeframes are used
        scale_factor = 1.0
        if self.secondary_timeframes:
            base_seconds = parse_timeframe_to_seconds(self.timeframe)
            max_tf_seconds = max([parse_timeframe_to_seconds(tf) for tf in self.secondary_timeframes])
            if base_seconds > 0:
                scale_factor = max(1.0, max_tf_seconds / base_seconds)
                scale_factor = min(scale_factor, 24.0) 
        
        adjusted_min_required = int(min_required * scale_factor)
        self.history_limit = max(config.data_handler.history_limit, adjusted_min_required)
        
        if self.history_limit > config.data_handler.history_limit:
            logger.info("Adjusted history limit for MTF/Indicators.", 
                        configured=config.data_handler.history_limit, 
                        required=min_required,
                        scale_factor=scale_factor,
                        final=self.history_limit)
        
        self.update_interval = config.strategy.interval_seconds * config.data_handler.update_interval_multiplier
        self.indicators_config = config.strategy.indicators
        
        # _dataframes holds the processed data (with indicators) for the strategy
        self._dataframes: Dict[str, pd.DataFrame] = {}
        # _raw_buffers holds the raw OHLCV data for history accumulation
        self._raw_buffers: Dict[str, pd.DataFrame] = {}
        
        self._shared_latest_prices = shared_latest_prices
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        self._save_task: Optional[asyncio.Task] = None
        
        # Event system for new candle notification
        self._new_candle_events: Dict[str, asyncio.Event] = {s: asyncio.Event() for s in self.symbols}
        self._last_emitted_candle_ts: Dict[str, pd.Timestamp] = {}
        
        # Persistence settings
        self.cache_dir = "market_data_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.auto_save_interval = 900 # Save every 15 minutes
        
        # Dynamic Buffer Sizing for AI
        self.max_training_buffer = 10000
        if hasattr(config.strategy.params, 'training_data_limit'):
             self.max_training_buffer = int(config.strategy.params.training_data_limit * 1.2)
             logger.info(f"Adjusted max_training_buffer to {self.max_training_buffer} based on strategy config.")

        # Analysis window: Enough for indicators to converge (e.g. SMA 200 + warmup)
        self.analysis_window = max(self.history_limit * 2, 500)
        
        # Executor for CPU-bound tasks (indicators) and Disk I/O
        self._executor = ThreadPoolExecutor(max_workers=min(len(self.symbols) + 2, 8))
        
        logger.info("DataHandler initialized.", 
                    cache_dir=self.cache_dir, 
                    analysis_window=self.analysis_window,
                    max_training_buffer=self.max_training_buffer,
                    history_limit=self.history_limit)

    async def initialize_data(self):
        """Fetches the initial batch of historical data for all symbols, utilizing local cache."""
        logger.info("Initializing historical data...", symbols=self.symbols)
        tasks = [self._initialize_symbol(symbol) for symbol in self.symbols]
        await asyncio.gather(*tasks)
        logger.info("Historical data initialization complete.")

    async def _initialize_symbol(self, symbol: str):
        """Loads cache and fetches fresh data for a single symbol."""
        try:
            # 1. Load cached raw data (Non-blocking I/O)
            cached_df = await self._load_from_cache(symbol)
            
            # 2. Fetch fresh data from API (history_limit)
            ohlcv_data = await self.exchange_api.get_market_data(symbol, self.timeframe, self.history_limit)
            fresh_df = create_dataframe(ohlcv_data)
            
            # 3. Merge into Raw Buffer
            if cached_df is not None and not cached_df.empty:
                if fresh_df is not None and not fresh_df.empty:
                    combined = pd.concat([cached_df, fresh_df])
                    combined = combined[~combined.index.duplicated(keep='last')]
                    combined.sort_index(inplace=True)
                    self._raw_buffers[symbol] = combined
                else:
                    self._raw_buffers[symbol] = cached_df
            else:
                self._raw_buffers[symbol] = fresh_df

            # 4. Process Indicators on Analysis Window
            await self._process_analysis_window(symbol)
            
            if symbol in self._raw_buffers:
                logger.info("Loaded initial data", symbol=symbol, total_records=len(self._raw_buffers[symbol]))
            else:
                logger.warning("No data available for symbol after initialization.", symbol=symbol)

        except Exception as e:
            logger.error("Failed to initialize data for symbol", symbol=symbol, error=str(e))

    async def run(self):
        """Starts the background data update and auto-save loops."""
        if self._running:
            return
        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        self._save_task = asyncio.create_task(self._auto_save_loop())
        logger.info("DataHandler update and auto-save loops started.")

    async def stop(self):
        """Stops the background loops and saves cache."""
        self._running = False
        if self._update_task and not self._update_task.done():
            self._update_task.cancel()
        if self._save_task and not self._save_task.done():
            self._save_task.cancel()
        
        # Save cache on shutdown (saving the RAW buffers)
        logger.info("Saving market data cache...")
        save_tasks = [self._save_to_cache(symbol, df) for symbol, df in self._raw_buffers.items()]
        await asyncio.gather(*save_tasks)
        
        self._executor.shutdown(wait=True)
        logger.info("DataHandler stopped and executor shutdown.")

    async def _update_loop(self):
        while self._running:
            try:
                tasks = [self.update_symbol_data(symbol) for symbol in self.symbols]
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                logger.info("Data update loop cancelled.")
                break
            except Exception as e:
                logger.error("Error in data update loop", error=str(e), exc_info=True)
            
            await asyncio.sleep(self.update_interval)

    async def _auto_save_loop(self):
        """Periodically saves the raw data buffers to disk."""
        while self._running:
            try:
                await asyncio.sleep(self.auto_save_interval)
                logger.debug("Auto-saving market data cache...")
                save_tasks = [self._save_to_cache(symbol, df) for symbol, df in self._raw_buffers.items()]
                await asyncio.gather(*save_tasks)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in auto-save loop", error=str(e))

    async def update_symbol_data(self, symbol: str):
        """Fetches latest data for a symbol and updates internal buffers. Public for backtesting."""
        try:
            # Self-Healing Logic: Check if we have enough data and if it's up to date
            current_buffer = self._raw_buffers.get(symbol)
            required_history = self.history_limit
            
            fetch_limit = 5 # Default small update
            is_recovery = False

            # 1. Check for empty or insufficient buffer
            if current_buffer is None or len(current_buffer) < (required_history * 0.9):
                logger.warning("Insufficient data buffer detected. Attempting recovery fetch.", 
                               symbol=symbol, 
                               current_len=len(current_buffer) if current_buffer is not None else 0,
                               target_len=required_history)
                fetch_limit = required_history
                is_recovery = True
            
            # 2. Check for stale buffer (Gap Detection)
            elif current_buffer is not None and not current_buffer.empty:
                last_ts = current_buffer.index[-1]
                # Ensure we compare apples to apples (UTC)
                # Use Clock.now() for time abstraction
                now = pd.Timestamp(Clock.now()).tz_localize(None)
                if last_ts.tzinfo is not None:
                    last_ts = last_ts.tz_convert(None)
                
                time_diff = (now - last_ts).total_seconds()
                tf_seconds = parse_timeframe_to_seconds(self.timeframe)
                
                if tf_seconds > 0:
                    missed_candles = int(time_diff / tf_seconds)
                    # If we missed more than the default fetch, expand the limit
                    if missed_candles > 5:
                        fetch_limit = missed_candles + 5 # Add buffer
                        logger.info("Data gap detected.", symbol=symbol, missed_candles=missed_candles, fetch_limit=fetch_limit)

                # Safety cap: if gap is huge, treat as full recovery
                if fetch_limit > required_history:
                    fetch_limit = required_history
                    is_recovery = True

            # Fetch data
            latest_ohlcv = await self.exchange_api.get_market_data(symbol, self.timeframe, fetch_limit)
            if not latest_ohlcv:
                return

            latest_df = create_dataframe(latest_ohlcv)
            if latest_df is None or latest_df.empty:
                return

            # Update Raw Buffer
            if is_recovery:
                # In recovery mode, we prioritize the new block but merge to be safe
                if current_buffer is not None:
                    combined_df = pd.concat([current_buffer, latest_df])
                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                    combined_df.sort_index(inplace=True)
                    self._raw_buffers[symbol] = combined_df
                else:
                    self._raw_buffers[symbol] = latest_df
                logger.info("Data recovery successful.", symbol=symbol, total_records=len(self._raw_buffers[symbol]))
            else:
                # Standard incremental update
                if symbol in self._raw_buffers:
                    current_raw = self._raw_buffers[symbol]
                    combined_df = pd.concat([current_raw, latest_df])
                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                    
                    if len(combined_df) > self.max_training_buffer:
                        combined_df = combined_df.iloc[-self.max_training_buffer:]
                    
                    self._raw_buffers[symbol] = combined_df
                else:
                    self._raw_buffers[symbol] = latest_df

            # Process Indicators on Analysis Window
            await self._process_analysis_window(symbol)
            
            logger.debug("Market data updated", symbol=symbol, mode="recovery" if is_recovery else "incremental")

        except Exception as e:
            logger.warning("Failed to update market data for symbol", symbol=symbol, error=str(e))

    async def _process_analysis_window(self, symbol: str):
        """
        Slices the raw buffer to the analysis window size, calculates indicators,
        and updates the shared _dataframes dictionary.
        Triggers an event if a new closed candle is detected.
        """
        raw_df = self._raw_buffers.get(symbol)
        if raw_df is None or raw_df.empty:
            return

        # Slice the last N rows for efficient indicator calculation
        if len(raw_df) > self.analysis_window:
            analysis_slice = raw_df.iloc[-self.analysis_window:].copy()
        else:
            analysis_slice = raw_df.copy()

        # Calculate indicators (Non-blocking CPU)
        processed_df = await self._calculate_indicators_async(analysis_slice)
        
        # Atomic update of the consumption dataframe
        self._dataframes[symbol] = processed_df
        self._update_latest_price(symbol, processed_df)

        # Check for new closed candle to trigger event
        # We use the same logic as get_market_data(include_forming=False) to determine the closed candle
        closed_df = self.get_market_data(symbol, include_forming=False)
        if closed_df is not None and not closed_df.empty:
            last_closed_ts = closed_df.index[-1]
            
            last_emitted = self._last_emitted_candle_ts.get(symbol)
            if last_emitted is None or last_closed_ts > last_emitted:
                self._last_emitted_candle_ts[symbol] = last_closed_ts
                if symbol in self._new_candle_events:
                    self._new_candle_events[symbol].set()

    async def wait_for_new_candle(self, symbol: str, timeout: float = 60.0):
        """
        Waits until a new closed candle is available for the symbol.
        Uses asyncio.Event to avoid busy waiting.
        """
        event = self._new_candle_events.get(symbol)
        if not event:
            # Fallback if symbol not initialized correctly
            await asyncio.sleep(timeout)
            return
        
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass # Just return to allow loop to check status/stop flags
        finally:
            event.clear() # Reset for next time

    async def _calculate_indicators_async(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper to run indicator calculation in thread pool."""
        loop = asyncio.get_running_loop() 
        return await loop.run_in_executor(self._executor, self.calculate_technical_indicators, df)

    def _update_latest_price(self, symbol: str, df: pd.DataFrame):
        if df is not None and not df.empty:
            self._shared_latest_prices[symbol] = df['close'].iloc[-1]

    def get_market_data(self, symbol: str, include_forming: bool = True) -> Optional[pd.DataFrame]:
        """
        Returns the latest DataFrame for a symbol with pre-calculated technical indicators.
        
        Args:
            symbol: The trading symbol.
            include_forming: If True, returns the latest candle even if it's still forming (incomplete).
                             If False, strips the last candle if it hasn't closed yet.
        """
        df = self._dataframes.get(symbol)
        if df is None or df.empty:
            logger.warning("No market data available for symbol", symbol=symbol)
            return None
        
        df_copy = df.copy() # Return a copy to prevent mutation

        if not include_forming:
            # Logic to drop last row if it is still forming
            last_ts = df_copy.index[-1]
            tf_seconds = parse_timeframe_to_seconds(self.timeframe)
            
            # Handle timezone naivety/awareness consistency
            now = pd.Timestamp(Clock.now()).tz_localize(None)
            if last_ts.tzinfo is not None:
                last_ts = last_ts.tz_convert(None)
                
            # Candle close time is start_time + timeframe
            candle_end_time = last_ts + pd.Timedelta(seconds=tf_seconds)
            
            if now < candle_end_time:
                # It is forming, drop it
                df_copy = df_copy.iloc[:-1]
        
        if df_copy.empty:
            return None
            
        return df_copy

    def get_correlation(self, symbol_a: str, symbol_b: str, lookback: int = 50) -> float:
        """
        Calculates the correlation between the returns of two symbols.
        Uses the 'close' price from the analysis dataframes.
        """
        df_a = self._dataframes.get(symbol_a)
        df_b = self._dataframes.get(symbol_b)
        
        if df_a is None or df_b is None or df_a.empty or df_b.empty:
            return 0.0
        
        # Align data by index (timestamp) and calculate returns
        # We use the tail(lookback) to focus on recent correlation
        try:
            # Ensure we have enough data
            if len(df_a) < lookback or len(df_b) < lookback:
                return 0.0

            series_a = df_a['close'].pct_change().tail(lookback)
            series_b = df_b['close'].pct_change().tail(lookback)
            
            # Pandas corr() automatically aligns indices
            correlation = series_a.corr(series_b)
            
            if np.isnan(correlation):
                return 0.0
                
            return float(correlation)
        except Exception as e:
            logger.error("Error calculating correlation", symbol_a=symbol_a, symbol_b=symbol_b, error=str(e))
            return 0.0

    async def fetch_full_history_for_symbol(self, symbol: str, limit: int) -> Optional[pd.DataFrame]:
        """
        Fetches a large batch of historical data for a symbol, intended for model training.
        Uses the internal raw buffer if sufficient, or fetches from API.
        Calculates indicators on the FULL dataset on demand.
        """
        logger.info("Fetching full historical data for training", symbol=symbol, limit=limit)
        
        # 1. Check current in-memory raw buffer first
        raw_df = self._raw_buffers.get(symbol)
        
        target_df = None
        
        if raw_df is not None and len(raw_df) >= limit:
            logger.info("Using in-memory raw data for training", symbol=symbol, records=len(raw_df))
            target_df = raw_df.copy()
        else:
            # 2. If not enough, try to fetch from API
            try:
                ohlcv_data = await self.exchange_api.get_market_data(symbol, self.timeframe, limit)
                if ohlcv_data:
                    df = create_dataframe(ohlcv_data)
                    if df is not None and not df.empty:
                        target_df = df
            except Exception as e:
                logger.error("Failed to fetch full historical data from API", symbol=symbol, error=str(e))

        # 3. If we have data (either from buffer or API), calculate indicators on the full set
        if target_df is not None and not target_df.empty:
            logger.info("Calculating indicators on full training set...", symbol=symbol, rows=len(target_df))
            full_df = await self._calculate_indicators_async(target_df)
            return full_df
        
        # Fallback: return whatever we have in the analysis buffer if everything else fails
        logger.warning("Could not fetch sufficient training data, returning available analysis data.", symbol=symbol)
        return self.get_market_data(symbol)

    async def _save_to_cache(self, symbol: str, df: pd.DataFrame):
        """Saves dataframe to CSV in executor. Saves RAW data."""
        if df is None or df.empty:
            return
        
        safe_symbol = symbol.replace('/', '_')
        path = os.path.join(self.cache_dir, f"{safe_symbol}.csv")
        
        loop = asyncio.get_running_loop()
        try:
            # Save only OHLCV to save space/time, indicators are recalculated on load
            raw_cols = ['open', 'high', 'low', 'close', 'volume']
            # Ensure we only save columns that exist
            valid_cols = [c for c in raw_cols if c in df.columns]
            to_save = df[valid_cols]
            
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

    def _enforce_data_continuity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures the DataFrame has a continuous DatetimeIndex based on the timeframe.
        Gaps are filled: Prices -> Forward Fill, Volume -> 0.
        """
        if df is None or df.empty:
            return df

        try:
            # Ensure sorted
            if not df.index.is_monotonic_increasing:
                df = df.sort_index()

            # Determine expected frequency in seconds
            tf_seconds = parse_timeframe_to_seconds(self.timeframe)
            if tf_seconds <= 0:
                return df

            # Create a complete index from start to end
            start = df.index[0]
            end = df.index[-1]
            
            # Generate expected timestamps using 's' (seconds) as frequency
            full_index = pd.date_range(start=start, end=end, freq=f"{tf_seconds}s")
            
            if len(full_index) == len(df.index):
                return df # No gaps

            # Reindex
            df_continuous = df.reindex(full_index)
            
            # Forward fill prices (Open, High, Low, Close)
            # If a gap exists, price stays at the last known close
            cols_to_ffill = ['open', 'high', 'low', 'close']
            # Only fill columns that exist
            cols_to_ffill = [c for c in cols_to_ffill if c in df_continuous.columns]
            df_continuous[cols_to_ffill] = df_continuous[cols_to_ffill].ffill()
            
            # Fill Volume with 0 (no activity during gap)
            if 'volume' in df_continuous.columns:
                df_continuous['volume'] = df_continuous['volume'].fillna(0)
            
            # Handle any remaining NaNs (e.g. at the very start if not covered)
            df_continuous.dropna(inplace=True)
            
            # Restore index name
            df_continuous.index.name = 'timestamp'
            
            gap_count = len(full_index) - len(df)
            if gap_count > 0:
                # Log only if significant to avoid spam
                if gap_count > 5:
                    logger.warning(f"Enforced data continuity. Filled {gap_count} gaps.", timeframe=self.timeframe)
            
            return df_continuous

        except Exception as e:
            logger.error("Error enforcing data continuity", error=str(e))
            return df

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators dynamically based on the strategy configuration, including Multi-Timeframe."""
        if df is None or len(df) < 50:
            logger.debug("DataFrame has insufficient data for indicators.", data_length=len(df) if df is not None else 0)
            return df

        # Enforce continuity before calculation to ensure rolling windows are valid
        df_continuous = self._enforce_data_continuity(df)
        
        df_out = df_continuous.copy()

        # Create a pandas-ta strategy from the configuration
        ta_strategy = ta.Strategy(
            name="BotTrader Dynamic Indicators",
            description="Indicators dynamically loaded from config.",
            ta=self.indicators_config
        )

        # 1. Calculate Base Indicators (Primary Timeframe)
        try:
            df_out.ta.strategy(ta_strategy)
        except Exception as e:
            logger.error("Pandas-TA strategy execution failed", error=str(e))
            return df_out

        # 2. Calculate Secondary Timeframe Indicators (MTF)
        if self.secondary_timeframes:
            rename_map = generate_indicator_rename_map(self.indicators_config)
            
            for tf in self.secondary_timeframes:
                try:
                    # Resample to HTF
                    tf_seconds = parse_timeframe_to_seconds(tf)
                    rule = f"{tf_seconds}s"
                    
                    resampled = df_continuous.resample(rule).agg({
                        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                    }).dropna()
                    
                    if len(resampled) < 20:
                        continue

                    # Run Strategy on HTF
                    resampled.ta.strategy(ta_strategy)
                    
                    # Rename columns to avoid collision (e.g. 'rsi' -> '1h_rsi')
                    # We only care about the indicator columns, not the raw OHLCV
                    htf_cols = []
                    for col in resampled.columns:
                        # Check if this is an indicator column (in our rename map or generated by ta)
                        # Simple heuristic: if it's not in raw OHLCV
                        if col not in ['open', 'high', 'low', 'close', 'volume']:
                            # Apply rename map if applicable to get clean base name
                            base_name = rename_map.get(col, col)
                            new_name = f"{tf}_{base_name}"
                            resampled.rename(columns={col: new_name}, inplace=True)
                            htf_cols.append(new_name)
                    
                    # CRITICAL: Shift HTF data by 1 to prevent lookahead bias.
                    # The value at 10:00 represents the candle 10:00-11:00.
                    # At 10:05, we should only see the value from 09:00-10:00.
                    resampled_shifted = resampled[htf_cols].shift(1)
                    
                    # Merge back to LTF
                    # Reindex to match LTF index and forward fill
                    # This propagates the closed HTF value forward until a new one closes
                    resampled_aligned = resampled_shifted.reindex(df_out.index).ffill()
                    
                    # Join
                    df_out = df_out.join(resampled_aligned)
                    
                except Exception as e:
                    logger.error(f"Failed to calculate MTF indicators for {tf}", error=str(e))

        # Rename base columns to a consistent, simplified format using the utility function
        try:
            rename_map = generate_indicator_rename_map(self.indicators_config)
            df_out.rename(columns=rename_map, inplace=True, errors='ignore')
        except ValueError as e:
            logger.error("Failed to generate indicator rename map. Check config.", error=str(e))
            return df_out

        df_out.dropna(inplace=True)
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
