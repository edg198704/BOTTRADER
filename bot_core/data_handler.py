import asyncio
import os
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import pandas_ta as ta
from concurrent.futures import ThreadPoolExecutor

from bot_core.logger import get_logger
from bot_core.config import BotConfig, AIEnsembleStrategyParams
from bot_core.exchange_api import ExchangeAPI
from bot_core.utils import generate_indicator_rename_map, parse_timeframe_to_seconds, Clock, calculate_min_history_depth
from bot_core.event_system import EventBus, MarketDataEvent

logger = get_logger(__name__)

# --- Pure Function for Indicator Calculation (Picklable) ---

def calculate_indicators_pure(df: pd.DataFrame, indicators_config: List[Dict[str, Any]], secondary_timeframes: List[str], timeframe: str) -> pd.DataFrame:
    """
    Pure function to calculate technical indicators.
    Can be run in a separate thread/process without object state.
    """
    if df is None or len(df) < 20: return df
    
    # Enforce Continuity
    try:
        if not df.index.is_monotonic_increasing: 
            df = df.sort_index()
        
        # Ensure we have a valid DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Basic data validation
        if df['close'].isnull().any():
            df = df.dropna(subset=['close'])

    except Exception as e:
        # If preprocessing fails, return original to avoid crashing worker
        return df

    df_out = df.copy()
    ta_strategy = ta.Strategy(name="BotTrader", ta=indicators_config)
    
    try:
        df_out.ta.strategy(ta_strategy)
    except Exception:
        return df_out

    if secondary_timeframes:
        rename_map = generate_indicator_rename_map(indicators_config)
        for tf in secondary_timeframes:
            try:
                tf_seconds = parse_timeframe_to_seconds(tf)
                if tf_seconds == 0: continue
                
                # Resample logic
                resampled = df.resample(f"{tf_seconds}s").agg(
                    {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
                ).dropna()
                
                if len(resampled) < 10: continue
                
                resampled.ta.strategy(ta_strategy)
                
                htf_cols = []
                for col in resampled.columns:
                    if col not in ['open', 'high', 'low', 'close', 'volume']:
                        base_name = rename_map.get(col, col)
                        new_name = f"{tf}_{base_name}"
                        resampled.rename(columns={col: new_name}, inplace=True)
                        htf_cols.append(new_name)
                
                # Merge back to lower timeframe
                # Shift by 1 to avoid lookahead bias (using closed HTF candle)
                resampled_shifted = resampled[htf_cols].shift(1)
                
                # Reindex to match original DF and forward fill
                resampled_aligned = resampled_shifted.reindex(df_out.index).ffill()
                df_out = df_out.join(resampled_aligned)
            except Exception:
                pass

    try:
        rename_map = generate_indicator_rename_map(indicators_config)
        df_out.rename(columns=rename_map, inplace=True, errors='ignore')
    except ValueError:
        pass
    
    # Drop NaN values created by indicators (warmup period)
    df_out.dropna(inplace=True)
    return df_out

# ---------------------------------------------------------

class DataHandler:
    """
    Manages market data lifecycle and acts as the primary Event Producer for the system.
    Implements a drift-correcting polling loop and Optimized Incremental Updates.
    """
    def __init__(self, exchange_api: ExchangeAPI, config: BotConfig, shared_latest_prices: Dict[str, float], event_bus: Optional[EventBus] = None):
        self.exchange_api = exchange_api
        self.config = config
        self.event_bus = event_bus
        
        self.symbols = list(config.strategy.symbols)
        
        if isinstance(config.strategy.params, AIEnsembleStrategyParams):
            leader = config.strategy.params.market_leader_symbol
            if leader and leader not in self.symbols:
                self.symbols.append(leader)
                logger.info(f"Added market leader symbol {leader} to DataHandler tracking.")

        self.timeframe = config.strategy.timeframe
        self.secondary_timeframes = config.strategy.secondary_timeframes
        
        # Calculate history requirements
        min_required = calculate_min_history_depth(config.strategy.indicators)
        scale_factor = 1.0
        if self.secondary_timeframes:
            base_seconds = parse_timeframe_to_seconds(self.timeframe)
            max_tf_seconds = max([parse_timeframe_to_seconds(tf) for tf in self.secondary_timeframes])
            if base_seconds > 0:
                scale_factor = max(1.0, max_tf_seconds / base_seconds)
                scale_factor = min(scale_factor, 24.0) 
        
        adjusted_min_required = int(min_required * scale_factor)
        self.history_limit = max(config.data_handler.history_limit, adjusted_min_required)
        
        self.update_interval = config.strategy.interval_seconds * config.data_handler.update_interval_multiplier
        self.indicators_config = config.strategy.indicators
        
        # Buffers
        self._dataframes: Dict[str, pd.DataFrame] = {} # Processed DFs (with indicators)
        self._raw_buffers: Dict[str, pd.DataFrame] = {} # Raw OHLCV DFs
        self._latest_order_books: Dict[str, Dict[str, Any]] = {}
        self._latencies: Dict[str, float] = {}
        
        self._buffer_lock = asyncio.Lock()
        self._shared_latest_prices = shared_latest_prices
        self._running = False
        
        self._symbol_tasks: Dict[str, asyncio.Task] = {}
        self._ticker_task: Optional[asyncio.Task] = None
        self._save_task: Optional[asyncio.Task] = None
        
        self._last_emitted_candle_ts: Dict[str, pd.Timestamp] = {}
        
        self.cache_dir = "market_data_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.auto_save_interval = 900
        
        self.max_training_buffer = 10000
        if hasattr(config.strategy.params, 'training_data_limit'):
             self.max_training_buffer = int(config.strategy.params.training_data_limit * 1.2)

        # Analysis window is the subset sent to indicator calculation
        self.analysis_window = max(self.history_limit * 2, 500)
        
        # Thread pool for heavy indicator math
        self._executor = ThreadPoolExecutor(max_workers=min(len(self.symbols) + 2, 8))
        
        self.use_order_book = False
        if isinstance(config.strategy.params, AIEnsembleStrategyParams):
            self.use_order_book = config.strategy.params.features.use_order_book_features

        logger.info("DataHandler initialized.", 
                    symbols=self.symbols, 
                    history_limit=self.history_limit, 
                    analysis_window=self.analysis_window)

    async def initialize_data(self):
        logger.info("Initializing historical data...", symbols=self.symbols)
        tasks = [self._initialize_symbol(symbol) for symbol in self.symbols]
        await asyncio.gather(*tasks)
        logger.info("Historical data initialization complete.")

    async def _initialize_symbol(self, symbol: str):
        try:
            cached_df = await self._load_from_cache(symbol)
            
            # Fetch fresh data to fill gaps or start fresh
            ohlcv_data = await self.exchange_api.get_market_data(symbol, self.timeframe, self.history_limit)
            fresh_df = create_dataframe(ohlcv_data)
            
            async with self._buffer_lock:
                if cached_df is not None and not cached_df.empty:
                    if fresh_df is not None and not fresh_df.empty:
                        # Merge and deduplicate
                        combined = pd.concat([cached_df, fresh_df])
                        combined = combined[~combined.index.duplicated(keep='last')]
                        combined.sort_index(inplace=True)
                        self._raw_buffers[symbol] = combined
                    else:
                        self._raw_buffers[symbol] = cached_df
                else:
                    self._raw_buffers[symbol] = fresh_df

            await self._process_analysis_window(symbol)
            
        except Exception as e:
            logger.error("Failed to initialize data for symbol", symbol=symbol, error=str(e))

    async def run(self):
        if self._running:
            return
        self._running = True
        
        for symbol in self.symbols:
            self._symbol_tasks[symbol] = asyncio.create_task(self._maintain_symbol_data(symbol))
            
        self._ticker_task = asyncio.create_task(self._ticker_loop())
        self._save_task = asyncio.create_task(self._auto_save_loop())
        logger.info("DataHandler started.")

    async def stop(self):
        self._running = False
        for task in self._symbol_tasks.values():
            if not task.done():
                task.cancel()
        if self._ticker_task and not self._ticker_task.done():
            self._ticker_task.cancel()
        if self._save_task and not self._save_task.done():
            self._save_task.cancel()
        
        logger.info("Saving market data cache...")
        save_tasks = [self._save_to_cache(symbol, df) for symbol, df in self._raw_buffers.items()]
        await asyncio.gather(*save_tasks)
        
        self._executor.shutdown(wait=True)
        logger.info("DataHandler stopped.")

    async def _maintain_symbol_data(self, symbol: str):
        """Drift-correcting loop for fetching data."""
        next_tick = time.time()
        while self._running:
            try:
                await self.update_symbol_data(symbol)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in data loop for {symbol}", error=str(e))
            
            now = time.time()
            next_tick += self.update_interval
            if next_tick < now:
                # We are lagging, skip ticks to catch up
                next_tick = now + self.update_interval
            
            sleep_duration = next_tick - now
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)
            else:
                await asyncio.sleep(0.01)

    async def _ticker_loop(self):
        while self._running:
            try:
                tickers = await self.exchange_api.get_tickers(self.symbols)
                for symbol, ticker_data in tickers.items():
                    if ticker_data and 'last' in ticker_data:
                        self._shared_latest_prices[symbol] = ticker_data['last']
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in ticker loop", error=str(e))
            await asyncio.sleep(self.config.data_handler.ticker_update_interval_seconds)

    async def _auto_save_loop(self):
        while self._running:
            try:
                await asyncio.sleep(self.auto_save_interval)
                async with self._buffer_lock:
                    # Snapshot for saving
                    save_tasks = [self._save_to_cache(symbol, df.copy()) for symbol, df in self._raw_buffers.items()]
                await asyncio.gather(*save_tasks)
            except asyncio.CancelledError:
                break

    async def update_symbol_data(self, symbol: str):
        try:
            # 1. Determine fetch requirements (Read-only check)
            current_len = 0
            last_ts = None
            
            # Optimization: Access without lock first (eventual consistency is fine here)
            raw_buf = self._raw_buffers.get(symbol)
            if raw_buf is not None:
                current_len = len(raw_buf)
                if current_len > 0:
                    last_ts = raw_buf.index[-1]

            fetch_limit = 5
            is_recovery = False

            # Recovery Logic
            if current_len < (self.history_limit * 0.9):
                fetch_limit = self.history_limit
                is_recovery = True
            elif last_ts is not None:
                now = pd.Timestamp(Clock.now()).tz_localize(None)
                if last_ts.tzinfo is not None: last_ts = last_ts.tz_convert(None)
                
                time_diff = (now - last_ts).total_seconds()
                tf_seconds = parse_timeframe_to_seconds(self.timeframe)
                if tf_seconds > 0:
                    missed_candles = int(time_diff / tf_seconds)
                    if missed_candles > 5:
                        fetch_limit = min(missed_candles + 5, self.history_limit)
                        is_recovery = True

            # 2. Network I/O (No Lock)
            latest_ohlcv = await self.exchange_api.get_market_data(symbol, self.timeframe, fetch_limit)
            if not latest_ohlcv: return

            latest_df = create_dataframe(latest_ohlcv)
            if latest_df is None or latest_df.empty: return

            # 3. Fetch Order Book (Optional)
            obi_val = 0.0
            if self.use_order_book:
                try:
                    ob = await self.exchange_api.fetch_order_book(symbol, limit=self.config.data_handler.order_book_depth)
                    if ob and 'bids' in ob and 'asks' in ob:
                        self._latest_order_books[symbol] = ob
                        bids_vol = sum(b[1] for b in ob['bids'])
                        asks_vol = sum(a[1] for a in ob['asks'])
                        if (bids_vol + asks_vol) > 0:
                            obi_val = (bids_vol - asks_vol) / (bids_vol + asks_vol)
                except Exception as e:
                    logger.warning("Failed to fetch order book", symbol=symbol, error=str(e))

            # 4. Latency Calculation
            last_candle_ts = latest_df.index[-1]
            if last_candle_ts.tzinfo is None: 
                last_candle_ts = last_candle_ts.tz_localize(timezone.utc)
            
            now_utc = Clock.now()
            latency = (now_utc - last_candle_ts).total_seconds()
            self._latencies[symbol] = latency

            # 5. Incremental Merge (Critical Section)
            async with self._buffer_lock:
                current_buffer = self._raw_buffers.get(symbol)
                
                if is_recovery or current_buffer is None or current_buffer.empty:
                    # Full replacement/initialization
                    if current_buffer is not None and not current_buffer.empty:
                        # Merge carefully if we have some data
                        combined_df = pd.concat([current_buffer, latest_df])
                        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                        combined_df.sort_index(inplace=True)
                        self._raw_buffers[symbol] = combined_df
                    else:
                        self._raw_buffers[symbol] = latest_df
                else:
                    # OPTIMIZED INCREMENTAL UPDATE
                    last_buffer_ts = current_buffer.index[-1]
                    
                    # 1. New candles (strictly greater timestamp)
                    new_rows = latest_df[latest_df.index > last_buffer_ts]
                    
                    # 2. Update forming candle (timestamp matches last buffer ts)
                    if last_buffer_ts in latest_df.index:
                        forming_row = latest_df.loc[last_buffer_ts]
                        # Update values directly using iloc for speed
                        idx_loc = len(current_buffer) - 1
                        current_buffer.iloc[idx_loc] = forming_row
                    
                    # 3. Append new rows if any
                    if not new_rows.empty:
                        self._raw_buffers[symbol] = pd.concat([current_buffer, new_rows])
                    
                    # 4. Prune if too large (Ring Buffer logic)
                    if len(self._raw_buffers[symbol]) > self.max_training_buffer:
                        self._raw_buffers[symbol] = self._raw_buffers[symbol].iloc[-self.max_training_buffer:]

                # Inject OBI
                if self.use_order_book:
                    if 'obi' not in self._raw_buffers[symbol].columns:
                        self._raw_buffers[symbol]['obi'] = 0.0
                    self._raw_buffers[symbol].iloc[-1, self._raw_buffers[symbol].columns.get_loc('obi')] = obi_val

            await self._process_analysis_window(symbol)

        except Exception as e:
            logger.warning("Failed to update market data", symbol=symbol, error=str(e))

    async def _process_analysis_window(self, symbol: str):
        # Snapshot the buffer for processing to avoid holding lock during heavy calculation
        async with self._buffer_lock:
            raw_df = self._raw_buffers.get(symbol)
            if raw_df is None or raw_df.empty: return
            
            # Only take what we need for indicators
            if len(raw_df) > self.analysis_window:
                analysis_slice = raw_df.iloc[-self.analysis_window:].copy()
            else:
                analysis_slice = raw_df.copy()

        # Heavy calculation runs in thread pool
        processed_df = await self._calculate_indicators_async(analysis_slice)
        
        # Atomic update of the processed dataframe
        self._dataframes[symbol] = processed_df
        self._update_latest_price(symbol, processed_df)

        # Emit event if new candle closed
        closed_df = self.get_market_data(symbol, include_forming=False)
        if closed_df is not None and not closed_df.empty:
            last_closed_ts = closed_df.index[-1]
            last_emitted = self._last_emitted_candle_ts.get(symbol)
            
            if last_emitted is None or last_closed_ts > last_emitted:
                self._last_emitted_candle_ts[symbol] = last_closed_ts
                if self.event_bus:
                    # Fire and forget event
                    await self.event_bus.publish(MarketDataEvent(symbol=symbol, data=closed_df))

    async def _calculate_indicators_async(self, df: pd.DataFrame) -> pd.DataFrame:
        loop = asyncio.get_running_loop() 
        return await loop.run_in_executor(
            self._executor, 
            calculate_indicators_pure, 
            df, 
            self.indicators_config, 
            self.secondary_timeframes, 
            self.timeframe
        )

    def _update_latest_price(self, symbol: str, df: pd.DataFrame):
        if df is not None and not df.empty:
            self._shared_latest_prices[symbol] = df['close'].iloc[-1]

    def get_market_data(self, symbol: str, include_forming: bool = True) -> Optional[pd.DataFrame]:
        df = self._dataframes.get(symbol)
        if df is None or df.empty: return None
        
        # Return a copy to prevent external mutation
        df_copy = df.copy()
        if not include_forming:
            last_ts = df_copy.index[-1]
            tf_seconds = parse_timeframe_to_seconds(self.timeframe)
            now = pd.Timestamp(Clock.now()).tz_localize(None)
            if last_ts.tzinfo is not None: last_ts = last_ts.tz_convert(None)
            
            candle_end_time = last_ts + pd.Timedelta(seconds=tf_seconds)
            if now < candle_end_time:
                df_copy = df_copy.iloc[:-1]
        
        return df_copy if not df_copy.empty else None

    def get_latest_order_book(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self._latest_order_books.get(symbol)

    def get_latency(self, symbol: str) -> float:
        return self._latencies.get(symbol, 0.0)

    def get_correlation(self, symbol_a: str, symbol_b: str, lookback: int = 50) -> float:
        df_a = self._dataframes.get(symbol_a)
        df_b = self._dataframes.get(symbol_b)
        if df_a is None or df_b is None or df_a.empty or df_b.empty: return 0.0
        try:
            if len(df_a) < lookback or len(df_b) < lookback: return 0.0
            # Use tail for correlation
            series_a = df_a['close'].pct_change().tail(lookback)
            series_b = df_b['close'].pct_change().tail(lookback)
            correlation = series_a.corr(series_b)
            return float(correlation) if not np.isnan(correlation) else 0.0
        except Exception:
            return 0.0

    async def fetch_full_history_for_symbol(self, symbol: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetches deep history for training, bypassing the analysis window."""
        async with self._buffer_lock:
            raw_df = self._raw_buffers.get(symbol)
            target_df = None
            if raw_df is not None and len(raw_df) >= limit:
                target_df = raw_df.copy()
        
        if target_df is None:
            try:
                ohlcv_data = await self.exchange_api.get_market_data(symbol, self.timeframe, limit)
                if ohlcv_data:
                    df = create_dataframe(ohlcv_data)
                    if df is not None and not df.empty: target_df = df
            except Exception as e:
                logger.error("Failed to fetch full history", symbol=symbol, error=str(e))

        if target_df is not None and not target_df.empty:
            return await self._calculate_indicators_async(target_df)
        return self.get_market_data(symbol)

    async def _save_to_cache(self, symbol: str, df: pd.DataFrame):
        if df is None or df.empty: return
        safe_symbol = symbol.replace('/', '_')
        path = os.path.join(self.cache_dir, f"{safe_symbol}.csv")
        loop = asyncio.get_running_loop()
        try:
            raw_cols = ['open', 'high', 'low', 'close', 'volume', 'obi']
            valid_cols = [c for c in raw_cols if c in df.columns]
            to_save = df[valid_cols]
            await loop.run_in_executor(self._executor, to_save.to_csv, path)
        except Exception as e:
            logger.error("Failed to save cache", symbol=symbol, error=str(e))

    async def _load_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        safe_symbol = symbol.replace('/', '_')
        path = os.path.join(self.cache_dir, f"{safe_symbol}.csv")
        if not os.path.exists(path): return None
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self._executor, pd.read_csv, path, {'index_col': 'timestamp', 'parse_dates': True})
        except Exception:
            return None

def create_dataframe(ohlcv_data: list) -> pd.DataFrame | None:
    try:
        if not ohlcv_data: return None
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        for col in ['open', 'high', 'low', 'close', 'volume']: 
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=['timestamp', 'close'], inplace=True)
        df['volume'].fillna(0, inplace=True)
        
        if len(df) < 5: return None
        
        df.set_index('timestamp', inplace=True)
        return df
    except Exception:
        return None
