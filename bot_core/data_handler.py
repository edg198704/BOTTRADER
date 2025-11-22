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

def calculate_indicators_pure(df: pd.DataFrame, indicators_config: List[Dict[str, Any]], secondary_timeframes: List[str], timeframe: str) -> pd.DataFrame:
    """
    Pure function to calculate technical indicators.
    Can be run in a separate thread/process without object state.
    """
    if df is None or len(df) < 20: return df
    
    try:
        # Ensure we are working with a copy to avoid SettingWithCopy warnings on the original buffer
        df_out = df.copy()
        
        if not df_out.index.is_monotonic_increasing: 
            df_out = df_out.sort_index()
        
        if not isinstance(df_out.index, pd.DatetimeIndex):
            df_out.index = pd.to_datetime(df_out.index)

        if df_out['close'].isnull().any():
            df_out = df_out.dropna(subset=['close'])

    except Exception:
        return df

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
                
                resampled = df_out.resample(f"{tf_seconds}s").agg(
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
                
                resampled_shifted = resampled[htf_cols].shift(1)
                resampled_aligned = resampled_shifted.reindex(df_out.index).ffill()
                df_out = df_out.join(resampled_aligned)
            except Exception:
                pass

    try:
        rename_map = generate_indicator_rename_map(indicators_config)
        df_out.rename(columns=rename_map, inplace=True, errors='ignore')
    except ValueError:
        pass
    
    df_out.dropna(inplace=True)
    return df_out

def create_dataframe(ohlcv_data: list) -> Optional[pd.DataFrame]:
    try:
        if not ohlcv_data: return None
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        for col in ['open', 'high', 'low', 'close', 'volume']: 
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=['timestamp', 'close'], inplace=True)
        df['volume'].fillna(0, inplace=True)
        
        if len(df) < 1: return None
        
        df.set_index('timestamp', inplace=True)
        return df
    except Exception:
        return None

class DataHandler:
    """
    Manages market data lifecycle using an Atomic Swap pattern for lock-free reads.
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

        self.timeframe = config.strategy.timeframe
        self.secondary_timeframes = config.strategy.secondary_timeframes
        
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
        
        # Atomic State Containers
        self._dataframes: Dict[str, pd.DataFrame] = {} 
        self._raw_buffers: Dict[str, pd.DataFrame] = {}
        self._latest_order_books: Dict[str, Dict[str, Any]] = {}
        self._latencies: Dict[str, float] = {}
        
        # Lock only for writing to the reference map, not for the data itself
        self._write_lock = asyncio.Lock()
        
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

        self.analysis_window = max(self.history_limit * 2, 500)
        self._executor = ThreadPoolExecutor(max_workers=min(len(self.symbols) + 2, 8))
        
        self.use_order_book = False
        if isinstance(config.strategy.params, AIEnsembleStrategyParams):
            self.use_order_book = config.strategy.params.features.use_order_book_features

        self._shared_latest_prices = shared_latest_prices
        logger.info("DataHandler initialized.", symbols=self.symbols)

    async def initialize_data(self):
        logger.info("Initializing historical data...")
        tasks = [self._initialize_symbol(symbol) for symbol in self.symbols]
        await asyncio.gather(*tasks)
        logger.info("Historical data initialization complete.")

    async def _initialize_symbol(self, symbol: str):
        try:
            cached_df = await self._load_from_cache(symbol)
            ohlcv_data = await self.exchange_api.get_market_data(symbol, self.timeframe, self.history_limit)
            fresh_df = create_dataframe(ohlcv_data)
            
            combined_df = None
            if cached_df is not None and not cached_df.empty:
                if fresh_df is not None and not fresh_df.empty:
                    combined = pd.concat([cached_df, fresh_df])
                    combined = combined[~combined.index.duplicated(keep='last')]
                    combined.sort_index(inplace=True)
                    combined_df = combined
                else:
                    combined_df = cached_df
            else:
                combined_df = fresh_df

            if combined_df is not None:
                async with self._write_lock:
                    self._raw_buffers[symbol] = combined_df
                await self._process_analysis_window(symbol)
            
        except Exception as e:
            logger.error("Failed to initialize data for symbol", symbol=symbol, error=str(e))

    async def run(self):
        if self._running: return
        self._running = True
        for symbol in self.symbols:
            self._symbol_tasks[symbol] = asyncio.create_task(self._maintain_symbol_data(symbol))
        self._ticker_task = asyncio.create_task(self._ticker_loop())
        self._save_task = asyncio.create_task(self._auto_save_loop())
        logger.info("DataHandler started.")

    async def stop(self):
        self._running = False
        for task in self._symbol_tasks.values(): task.cancel()
        if self._ticker_task: self._ticker_task.cancel()
        if self._save_task: self._save_task.cancel()
        
        logger.info("Saving market data cache...")
        save_tasks = [self._save_to_cache(symbol, df) for symbol, df in self._raw_buffers.items()]
        await asyncio.gather(*save_tasks)
        self._executor.shutdown(wait=True)
        logger.info("DataHandler stopped.")

    async def _maintain_symbol_data(self, symbol: str):
        next_tick = time.time()
        while self._running:
            try:
                await self.update_symbol_data(symbol)
            except asyncio.CancelledError: break
            except Exception as e:
                logger.error(f"Error in data loop for {symbol}", error=str(e))
            
            now = time.time()
            next_tick += self.update_interval
            if next_tick < now: next_tick = now + self.update_interval
            await asyncio.sleep(max(0.01, next_tick - now))

    async def _ticker_loop(self):
        while self._running:
            try:
                tickers = await self.exchange_api.get_tickers(self.symbols)
                for symbol, ticker_data in tickers.items():
                    if ticker_data and 'last' in ticker_data:
                        self._shared_latest_prices[symbol] = ticker_data['last']
            except asyncio.CancelledError: break
            except Exception: pass
            await asyncio.sleep(self.config.data_handler.ticker_update_interval_seconds)

    async def _auto_save_loop(self):
        while self._running:
            try:
                await asyncio.sleep(self.auto_save_interval)
                # Snapshot references safely
                snapshot = {s: df.copy() for s, df in self._raw_buffers.items() if df is not None}
                save_tasks = [self._save_to_cache(s, df) for s, df in snapshot.items()]
                await asyncio.gather(*save_tasks)
            except asyncio.CancelledError: break

    async def update_symbol_data(self, symbol: str):
        try:
            # 1. Read current state (Atomic read)
            current_buffer = self._raw_buffers.get(symbol)
            
            # 2. Determine fetch logic
            fetch_limit = 5
            is_recovery = False
            last_ts = current_buffer.index[-1] if current_buffer is not None and not current_buffer.empty else None

            if current_buffer is None or len(current_buffer) < (self.history_limit * 0.9):
                fetch_limit = self.history_limit
                is_recovery = True
            elif last_ts is not None:
                now = pd.Timestamp(Clock.now()).tz_localize(None)
                if last_ts.tzinfo is not None: last_ts = last_ts.tz_convert(None)
                time_diff = (now - last_ts).total_seconds()
                tf_seconds = parse_timeframe_to_seconds(self.timeframe)
                if tf_seconds > 0 and time_diff > (tf_seconds * 5):
                    fetch_limit = min(int(time_diff / tf_seconds) + 5, self.history_limit)
                    is_recovery = True

            # 3. Network I/O (No Lock)
            latest_ohlcv = await self.exchange_api.get_market_data(symbol, self.timeframe, fetch_limit)
            if not latest_ohlcv: return
            latest_df = create_dataframe(latest_ohlcv)
            if latest_df is None or latest_df.empty: return

            # 4. Order Book (Optional)
            obi_val = 0.0
            if self.use_order_book:
                try:
                    ob = await self.exchange_api.fetch_order_book(symbol, limit=self.config.data_handler.order_book_depth)
                    if ob: 
                        self._latest_order_books[symbol] = ob
                        bids = sum(b[1] for b in ob.get('bids', []))
                        asks = sum(a[1] for a in ob.get('asks', []))
                        if (bids + asks) > 0: obi_val = (bids - asks) / (bids + asks)
                except Exception: pass

            # 5. Latency
            last_candle_ts = latest_df.index[-1]
            if last_candle_ts.tzinfo is None: last_candle_ts = last_candle_ts.tz_localize(timezone.utc)
            self._latencies[symbol] = (Clock.now() - last_candle_ts).total_seconds()

            # 6. Prepare New State (Heavy Lifting OUTSIDE Lock)
            new_buffer = None
            if is_recovery or current_buffer is None or current_buffer.empty:
                if current_buffer is not None and not current_buffer.empty:
                    new_buffer = pd.concat([current_buffer, latest_df])
                    new_buffer = new_buffer[~new_buffer.index.duplicated(keep='last')].sort_index()
                else:
                    new_buffer = latest_df
            else:
                # Incremental Update Logic
                last_buffer_ts = current_buffer.index[-1]
                new_rows = latest_df[latest_df.index > last_buffer_ts]
                
                # Check if we need to update the forming candle
                forming_update = False
                if last_buffer_ts in latest_df.index:
                    forming_row = latest_df.loc[last_buffer_ts]
                    # Check if values actually changed to avoid unnecessary copies
                    if not current_buffer.iloc[-1].equals(forming_row):
                        forming_update = True
                
                if not new_rows.empty or forming_update:
                    # Copy is necessary to ensure immutability for readers
                    new_buffer = current_buffer.copy()
                    if forming_update:
                        new_buffer.iloc[-1] = latest_df.loc[last_buffer_ts]
                    if not new_rows.empty:
                        new_buffer = pd.concat([new_buffer, new_rows])
                    
                    if len(new_buffer) > self.max_training_buffer:
                        new_buffer = new_buffer.iloc[-self.max_training_buffer:]

            # 7. Atomic Swap (Critical Section - Microseconds)
            if new_buffer is not None:
                if self.use_order_book:
                    if 'obi' not in new_buffer.columns: new_buffer['obi'] = 0.0
                    new_buffer.iloc[-1, new_buffer.columns.get_loc('obi')] = obi_val
                
                async with self._write_lock:
                    self._raw_buffers[symbol] = new_buffer
                
                await self._process_analysis_window(symbol, new_buffer)

        except Exception as e:
            logger.warning("Failed to update market data", symbol=symbol, error=str(e))

    async def _process_analysis_window(self, symbol: str, raw_df: pd.DataFrame):
        # Slice for analysis
        if len(raw_df) > self.analysis_window:
            analysis_slice = raw_df.iloc[-self.analysis_window:].copy()
        else:
            analysis_slice = raw_df.copy()

        # Offload heavy math
        processed_df = await self._calculate_indicators_async(analysis_slice)
        
        # Atomic Swap of Processed Data
        async with self._write_lock:
            self._dataframes[symbol] = processed_df
            if not processed_df.empty:
                self._shared_latest_prices[symbol] = processed_df['close'].iloc[-1]

        # Event Emission
        closed_df = self.get_market_data(symbol, include_forming=False)
        if closed_df is not None and not closed_df.empty:
            last_closed_ts = closed_df.index[-1]
            last_emitted = self._last_emitted_candle_ts.get(symbol)
            if last_emitted is None or last_closed_ts > last_emitted:
                self._last_emitted_candle_ts[symbol] = last_closed_ts
                if self.event_bus:
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

    def get_market_data(self, symbol: str, include_forming: bool = True) -> Optional[pd.DataFrame]:
        # Lock-free read (dictionary lookup is atomic in Python)
        df = self._dataframes.get(symbol)
        if df is None or df.empty: return None
        
        # Return a copy to prevent external mutation affecting other readers
        # (Though we swap references on write, so this is mostly for safety against user code)
        if not include_forming:
            last_ts = df.index[-1]
            tf_seconds = parse_timeframe_to_seconds(self.timeframe)
            now = pd.Timestamp(Clock.now()).tz_localize(None)
            if last_ts.tzinfo is not None: last_ts = last_ts.tz_convert(None)
            if now < (last_ts + pd.Timedelta(seconds=tf_seconds)):
                return df.iloc[:-1].copy()
        
        return df.copy()

    def get_latest_order_book(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self._latest_order_books.get(symbol)

    def get_latency(self, symbol: str) -> float:
        return self._latencies.get(symbol, 0.0)

    def get_correlation(self, symbol_a: str, symbol_b: str, lookback: int = 50) -> float:
        df_a = self._dataframes.get(symbol_a)
        df_b = self._dataframes.get(symbol_b)
        if df_a is None or df_b is None or df_a.empty or df_b.empty: return 0.0
        try:
            min_len = min(len(df_a), len(df_b), lookback)
            if min_len < 10: return 0.0
            series_a = df_a['close'].pct_change().tail(min_len)
            series_b = df_b['close'].pct_change().tail(min_len)
            return float(series_a.corr(series_b))
        except Exception:
            return 0.0

    async def fetch_full_history_for_symbol(self, symbol: str, limit: int) -> Optional[pd.DataFrame]:
        # Snapshot raw buffer
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
