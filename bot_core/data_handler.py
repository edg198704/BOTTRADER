import asyncio
import os
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import pandas_ta as ta
from concurrent.futures import ThreadPoolExecutor

from bot_core.logger import get_logger
from bot_core.config import BotConfig, AIEnsembleStrategyParams
from bot_core.exchange_api import ExchangeAPI
from bot_core.utils import generate_indicator_rename_map, parse_timeframe_to_seconds, Clock, calculate_min_history_depth
from bot_core.event_system import EventBus, MarketDataEvent

logger = get_logger(__name__)

class DataHandler:
    """
    Manages market data lifecycle and acts as the primary Event Producer for the system.
    Implements a drift-correcting polling loop for precise data fetching.
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
        
        self._dataframes: Dict[str, pd.DataFrame] = {}
        self._raw_buffers: Dict[str, pd.DataFrame] = {}
        
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

        self.analysis_window = max(self.history_limit * 2, 500)
        self._executor = ThreadPoolExecutor(max_workers=min(len(self.symbols) + 2, 8))
        
        logger.info("DataHandler initialized.", event_bus_connected=self.event_bus is not None)

    async def initialize_data(self):
        logger.info("Initializing historical data...", symbols=self.symbols)
        tasks = [self._initialize_symbol(symbol) for symbol in self.symbols]
        await asyncio.gather(*tasks)
        logger.info("Historical data initialization complete.")

    async def _initialize_symbol(self, symbol: str):
        try:
            cached_df = await self._load_from_cache(symbol)
            ohlcv_data = await self.exchange_api.get_market_data(symbol, self.timeframe, self.history_limit)
            fresh_df = create_dataframe(ohlcv_data)
            
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
        """
        Continuously updates data for a symbol using a drift-correcting loop.
        """
        next_tick = time.time()
        
        while self._running:
            try:
                await self.update_symbol_data(symbol)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in data loop for {symbol}", error=str(e))
            
            # Drift correction logic
            now = time.time()
            next_tick += self.update_interval
            
            # If we are lagging significantly (more than one interval), reset the clock
            if next_tick < now:
                next_tick = now + self.update_interval
            
            sleep_duration = next_tick - now
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)
            else:
                # Yield control briefly if we are running hot
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
                save_tasks = [self._save_to_cache(symbol, df) for symbol, df in self._raw_buffers.items()]
                await asyncio.gather(*save_tasks)
            except asyncio.CancelledError:
                break

    async def update_symbol_data(self, symbol: str):
        try:
            current_buffer = self._raw_buffers.get(symbol)
            required_history = self.history_limit
            fetch_limit = 5
            is_recovery = False

            if current_buffer is None or len(current_buffer) < (required_history * 0.9):
                fetch_limit = required_history
                is_recovery = True
            elif current_buffer is not None and not current_buffer.empty:
                last_ts = current_buffer.index[-1]
                now = pd.Timestamp(Clock.now()).tz_localize(None)
                if last_ts.tzinfo is not None: last_ts = last_ts.tz_convert(None)
                
                time_diff = (now - last_ts).total_seconds()
                tf_seconds = parse_timeframe_to_seconds(self.timeframe)
                if tf_seconds > 0:
                    missed_candles = int(time_diff / tf_seconds)
                    if missed_candles > 5:
                        fetch_limit = missed_candles + 5
                if fetch_limit > required_history:
                    fetch_limit = required_history
                    is_recovery = True

            latest_ohlcv = await self.exchange_api.get_market_data(symbol, self.timeframe, fetch_limit)
            if not latest_ohlcv: return

            latest_df = create_dataframe(latest_ohlcv)
            if latest_df is None or latest_df.empty: return

            if is_recovery:
                if current_buffer is not None:
                    combined_df = pd.concat([current_buffer, latest_df])
                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                    combined_df.sort_index(inplace=True)
                    self._raw_buffers[symbol] = combined_df
                else:
                    self._raw_buffers[symbol] = latest_df
            else:
                if symbol in self._raw_buffers:
                    current_raw = self._raw_buffers[symbol]
                    combined_df = pd.concat([current_raw, latest_df])
                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                    if len(combined_df) > self.max_training_buffer:
                        combined_df = combined_df.iloc[-self.max_training_buffer:]
                    self._raw_buffers[symbol] = combined_df
                else:
                    self._raw_buffers[symbol] = latest_df

            await self._process_analysis_window(symbol)

        except Exception as e:
            logger.warning("Failed to update market data", symbol=symbol, error=str(e))

    async def _process_analysis_window(self, symbol: str):
        raw_df = self._raw_buffers.get(symbol)
        if raw_df is None or raw_df.empty: return

        # Optimization: Only process what we need for indicators
        if len(raw_df) > self.analysis_window:
            analysis_slice = raw_df.iloc[-self.analysis_window:].copy()
        else:
            analysis_slice = raw_df.copy()

        processed_df = await self._calculate_indicators_async(analysis_slice)
        self._dataframes[symbol] = processed_df
        self._update_latest_price(symbol, processed_df)

        # Event Emission Logic
        # We only emit if we have a new closed candle
        closed_df = self.get_market_data(symbol, include_forming=False)
        if closed_df is not None and not closed_df.empty:
            last_closed_ts = closed_df.index[-1]
            last_emitted = self._last_emitted_candle_ts.get(symbol)
            
            if last_emitted is None or last_closed_ts > last_emitted:
                self._last_emitted_candle_ts[symbol] = last_closed_ts
                if self.event_bus:
                    # Publish event to the bus (non-blocking by default in bus implementation)
                    # We pass the closed_df explicitly to ensure the consumer sees exactly what triggered the event
                    await self.event_bus.publish(MarketDataEvent(symbol=symbol, data=closed_df))

    async def _calculate_indicators_async(self, df: pd.DataFrame) -> pd.DataFrame:
        loop = asyncio.get_running_loop() 
        return await loop.run_in_executor(self._executor, self.calculate_technical_indicators, df)

    def _update_latest_price(self, symbol: str, df: pd.DataFrame):
        if df is not None and not df.empty:
            self._shared_latest_prices[symbol] = df['close'].iloc[-1]

    def get_market_data(self, symbol: str, include_forming: bool = True) -> Optional[pd.DataFrame]:
        df = self._dataframes.get(symbol)
        if df is None or df.empty: return None
        
        # Optimization: Avoid copy if not needed, but safety first for consumers
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

    def get_correlation(self, symbol_a: str, symbol_b: str, lookback: int = 50) -> float:
        df_a = self._dataframes.get(symbol_a)
        df_b = self._dataframes.get(symbol_b)
        if df_a is None or df_b is None or df_a.empty or df_b.empty: return 0.0
        try:
            if len(df_a) < lookback or len(df_b) < lookback: return 0.0
            series_a = df_a['close'].pct_change().tail(lookback)
            series_b = df_b['close'].pct_change().tail(lookback)
            correlation = series_a.corr(series_b)
            return float(correlation) if not np.isnan(correlation) else 0.0
        except Exception:
            return 0.0

    async def fetch_full_history_for_symbol(self, symbol: str, limit: int) -> Optional[pd.DataFrame]:
        raw_df = self._raw_buffers.get(symbol)
        target_df = None
        if raw_df is not None and len(raw_df) >= limit:
            target_df = raw_df.copy()
        else:
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
            raw_cols = ['open', 'high', 'low', 'close', 'volume']
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

    def _enforce_data_continuity(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty: return df
        try:
            if not df.index.is_monotonic_increasing: df = df.sort_index()
            tf_seconds = parse_timeframe_to_seconds(self.timeframe)
            if tf_seconds <= 0: return df
            start, end = df.index[0], df.index[-1]
            full_index = pd.date_range(start=start, end=end, freq=f"{tf_seconds}s")
            if len(full_index) == len(df.index): return df
            df_continuous = df.reindex(full_index)
            cols_to_ffill = [c for c in ['open', 'high', 'low', 'close'] if c in df_continuous.columns]
            df_continuous[cols_to_ffill] = df_continuous[cols_to_ffill].ffill()
            if 'volume' in df_continuous.columns: df_continuous['volume'] = df_continuous['volume'].fillna(0)
            df_continuous.dropna(inplace=True)
            df_continuous.index.name = 'timestamp'
            return df_continuous
        except Exception:
            return df

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) < 50: return df
        df_continuous = self._enforce_data_continuity(df)
        df_out = df_continuous.copy()
        ta_strategy = ta.Strategy(name="BotTrader", ta=self.indicators_config)
        try:
            df_out.ta.strategy(ta_strategy)
        except Exception:
            return df_out

        if self.secondary_timeframes:
            rename_map = generate_indicator_rename_map(self.indicators_config)
            for tf in self.secondary_timeframes:
                try:
                    tf_seconds = parse_timeframe_to_seconds(tf)
                    resampled = df_continuous.resample(f"{tf_seconds}s").agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
                    if len(resampled) < 20: continue
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
            rename_map = generate_indicator_rename_map(self.indicators_config)
            df_out.rename(columns=rename_map, inplace=True, errors='ignore')
        except ValueError:
            pass
        df_out.dropna(inplace=True)
        return df_out

def create_dataframe(ohlcv_data: list) -> pd.DataFrame | None:
    try:
        if not ohlcv_data: return None
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = pd.to_numeric(df[col], errors='coerce')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=['timestamp', 'close'], inplace=True)
        df['volume'].fillna(0, inplace=True)
        if len(df) < 20: return None
        df.set_index('timestamp', inplace=True)
        return df
    except Exception:
        return None
