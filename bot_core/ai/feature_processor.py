import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict
from numpy.lib.stride_tricks import sliding_window_view
from functools import lru_cache

from bot_core.logger import get_logger
from bot_core.config import AIEnsembleStrategyParams

logger = get_logger(__name__)

class FeatureProcessor:
    """
    Encapsulates all feature engineering, normalization, and labeling logic.
    Implements institutional-grade labeling (Triple Barrier) and stationarity checks.
    Optimized with LRU caching for expensive weight calculations.
    """

    @staticmethod
    def get_feature_names(config: AIEnsembleStrategyParams) -> List[str]:
        cols = list(config.feature_columns)
        if config.features.use_leader_features:
            cols.extend(["leader_log_return", "leader_rsi"])
        if config.features.use_order_book_features:
            cols.append("obi")

        lags = config.features.lag_features
        depth = config.features.lag_depth
        if depth > 0 and lags:
            for col in lags:
                if col in config.feature_columns or col == "obi":
                    for i in range(1, depth + 1):
                        cols.append(f"{col}_lag_{i}")
        return cols

    @staticmethod
    def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        dt_series = df['timestamp'] if 'timestamp' in df.columns else df.index.to_series()
        
        df['time_hour_sin'] = np.sin(2 * np.pi * dt_series.dt.hour / 24)
        df['time_hour_cos'] = np.cos(2 * np.pi * dt_series.dt.hour / 24)
        df['time_dow_sin'] = np.sin(2 * np.pi * dt_series.dt.dayofweek / 7)
        df['time_dow_cos'] = np.cos(2 * np.pi * dt_series.dt.dayofweek / 7)
        return df

    @staticmethod
    def _add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close_safe = df['close'].replace(0, 1.0)
        df['pa_body_size'] = (df['close'] - df['open']).abs() / close_safe
        df['pa_upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / close_safe
        df['pa_lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / close_safe
        return df

    @staticmethod
    def _add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        log_hl = np.log(df['high'] / df['low'].replace(0, 1e-8))
        log_co = np.log(df['close'] / df['open'].replace(0, 1e-8))
        df['volatility_gk'] = np.sqrt(0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2)
        return df

    @staticmethod
    def _add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        const = 1.0 / (4.0 * np.log(2.0))
        log_hl = np.log(df['high'] / df['low'].replace(0, 1e-8))
        df['volatility_parkinson'] = np.sqrt(const * log_hl**2)

        abs_return = df['close'].pct_change().abs()
        dollar_volume = (df['volume'] * df['close']).replace(0, 1.0)
        df['amihud_illiquidity'] = (abs_return / dollar_volume).rolling(window=10).mean()
        return df

    @staticmethod
    def _add_order_book_features(df: pd.DataFrame) -> pd.DataFrame:
        if 'obi' not in df.columns:
            df = df.copy()
            df['obi'] = 0.0
        return df

    @staticmethod
    @lru_cache(maxsize=32)
    def _get_frac_diff_weights(d: float, thres: float, max_len: int = 2000) -> np.ndarray:
        """
        Calculates weights for fractional differentiation.
        Cached to prevent re-computation on every tick.
        """
        w = [1.0]
        k = 1
        while True:
            w_k = -w[-1] / k * (d - k + 1)
            if abs(w_k) < thres or k >= max_len:
                break
            w.append(w_k)
            k += 1
        return np.array(w[::-1]) # Reverse for convolution

    @staticmethod
    def _apply_frac_diff(df: pd.DataFrame, d: float, thres: float) -> pd.DataFrame:
        """
        Applies Fractional Differentiation to the 'close' column.
        Uses cached weights for performance.
        """
        df = df.copy()
        if 'close' not in df.columns: return df
        
        weights = FeatureProcessor._get_frac_diff_weights(d, thres)

        # 2. Apply Convolution
        close_vals = df['close'].values
        if len(close_vals) < len(weights):
            df['close_frac'] = np.nan
            return df
            
        # 'valid' mode returns only points where signals completely overlap
        frac_diff = np.convolve(close_vals, weights, mode='valid')
        
        # 3. Pad with NaNs to match original length
        pad_len = len(close_vals) - len(frac_diff)
        df['close_frac'] = np.concatenate([np.full(pad_len, np.nan), frac_diff])
        return df

    @staticmethod
    def _stationarize_data(df: pd.DataFrame, cols_to_process: List[str]) -> pd.DataFrame:
        """
        Converts non-stationary price series to stationary returns.
        Uses explicit column name matching instead of heuristic value checking.
        """
        df_trans = df.copy()
        
        # Vectorized Volume Normalization
        if 'volume' in df_trans.columns and 'volume' in cols_to_process:
            vol_ma = df_trans['volume'].rolling(window=50, min_periods=1).mean().replace(0, 1.0)
            df_trans['volume'] = df_trans['volume'] / vol_ma

        # Explicit Price Column Keywords
        price_keywords = ['open', 'high', 'low', 'close', 'sma', 'ema', 'wma', 'kama', 'mid', 'top', 'bot', 'upper', 'lower']
        
        if 'close' in df_trans.columns:
            close_price = df_trans['close'].replace(0, 1e-8)
            
            # Identify columns that are likely prices based on name
            price_cols = [c for c in cols_to_process 
                          if c in df_trans.columns 
                          and c != 'close' 
                          and c != 'obi'
                          and any(k in c.lower() for k in price_keywords)]
            
            # Vectorized Stationarization: (Price / Close) - 1.0
            # This normalizes all price indicators relative to the current close price
            if price_cols:
                df_trans[price_cols] = df_trans[price_cols].div(close_price, axis=0) - 1.0

        return df_trans

    @staticmethod
    def process_data(df: pd.DataFrame, config: AIEnsembleStrategyParams, leader_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        df_proc = df.copy()
        
        if config.features.use_time_features: df_proc = FeatureProcessor._add_time_features(df_proc)
        if config.features.use_price_action_features: df_proc = FeatureProcessor._add_price_action_features(df_proc)
        if config.features.use_volatility_estimators: df_proc = FeatureProcessor._add_volatility_features(df_proc)
        if config.features.use_microstructure_features: df_proc = FeatureProcessor._add_microstructure_features(df_proc)
        if config.features.use_order_book_features: df_proc = FeatureProcessor._add_order_book_features(df_proc)
        if config.features.use_frac_diff: df_proc = FeatureProcessor._apply_frac_diff(df_proc, config.features.frac_diff_d, config.features.frac_diff_thres)

        if config.features.use_leader_features and leader_df is not None and not leader_df.empty:
            leader_feats = pd.DataFrame(index=leader_df.index)
            leader_feats['leader_log_return'] = np.log(leader_df['close'] / leader_df['close'].shift(1).replace(0, np.nan))
            if 'rsi' in leader_df.columns: leader_feats['leader_rsi'] = leader_df['rsi']
            # Align leader features to current df index
            df_proc = df_proc.join(leader_feats, how='left').ffill(limit=5)

        df_stat = FeatureProcessor._stationarize_data(df_proc, config.feature_columns)
        
        # Select and Lag
        cols = list(config.feature_columns)
        if config.features.use_leader_features: cols.extend(["leader_log_return", "leader_rsi"])
        if config.features.use_order_book_features: cols.append("obi")
        
        valid_cols = [c for c in cols if c in df_stat.columns]
        subset = df_stat[valid_cols].copy()

        lags = config.features.lag_features
        depth = config.features.lag_depth
        if depth > 0 and lags:
            for col in lags:
                if col in subset.columns:
                    for i in range(1, depth + 1):
                        subset[f"{col}_lag_{i}"] = subset[col].shift(i)

        subset.dropna(inplace=True)
        
        # Normalize
        window = config.features.normalization_window
        if config.features.scaling_method == 'robust':
            median = subset.rolling(window=window).median()
            iqr = subset.rolling(window=window).quantile(0.75) - subset.rolling(window=window).quantile(0.25)
            normalized = (subset - median) / iqr.replace(0, 1e-8)
        else:
            mean = subset.rolling(window=window).mean()
            std = subset.rolling(window=window).std()
            normalized = (subset - mean) / std.replace(0, 1e-8)
            
        normalized.dropna(inplace=True)
        
        # Enforce Order
        expected = FeatureProcessor.get_feature_names(config)
        final_cols = [c for c in expected if c in normalized.columns]
        return normalized[final_cols]

    @staticmethod
    def create_labels(df: pd.DataFrame, config: AIEnsembleStrategyParams) -> Tuple[pd.Series, pd.Series]:
        if config.features.use_triple_barrier:
            return FeatureProcessor._create_triple_barrier_labels(df, config)

        horizon = config.features.labeling_horizon
        future_price = df['close'].shift(-horizon)
        pct_change = (future_price - df['close']) / df['close']
        
        labels = pd.Series(1, index=df.index) # Hold
        
        threshold = config.features.labeling_threshold
        if config.features.use_dynamic_labeling and config.market_regime.volatility_col in df.columns:
            threshold = (df[config.market_regime.volatility_col] * config.features.labeling_atr_multiplier) / df['close']
        
        labels[pct_change > threshold] = 2 # Buy
        labels[pct_change < -threshold] = 0 # Sell
        
        t1 = pd.Series(df.index, index=df.index).shift(-horizon)
        
        return labels.iloc[:-horizon], t1.iloc[:-horizon]

    @staticmethod
    def _create_triple_barrier_labels(df: pd.DataFrame, config: AIEnsembleStrategyParams) -> Tuple[pd.Series, pd.Series]:
        horizon = config.features.labeling_horizon
        tp_mult = config.features.triple_barrier_tp_multiplier
        sl_mult = config.features.triple_barrier_sl_multiplier
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        timestamps = df.index.values
        
        atr_col = config.market_regime.volatility_col
        vol = df[atr_col].values if atr_col in df.columns else close * 0.01
        
        upper_barriers = close + (vol * tp_mult)
        lower_barriers = close - (vol * sl_mult)
        
        n_samples = len(df)
        if n_samples <= horizon:
            return pd.Series(1, index=df.index), pd.Series(df.index, index=df.index)

        # Efficient sliding window view (memory efficient view, not copy)
        high_windows = sliding_window_view(high, window_shape=horizon)
        low_windows = sliding_window_view(low, window_shape=horizon)
        time_windows = sliding_window_view(timestamps, window_shape=horizon)
        
        # Align barriers to the start of each window
        # We only need to check the first N-horizon windows
        valid_len = high_windows.shape[0]
        upper_barriers = upper_barriers[:valid_len].reshape(-1, 1)
        lower_barriers = lower_barriers[:valid_len].reshape(-1, 1)
        
        # Boolean masks for barrier hits
        hit_upper = high_windows >= upper_barriers
        hit_lower = low_windows <= lower_barriers
        
        # Find first index of hit (argmax returns 0 if no True, so we check any())
        upper_indices = np.argmax(hit_upper, axis=1)
        lower_indices = np.argmax(hit_lower, axis=1)
        
        any_upper = np.any(hit_upper, axis=1)
        any_lower = np.any(hit_lower, axis=1)
        
        labels = np.ones(valid_len, dtype=int) # Default Hold (1)
        t1_vals = time_windows[:, -1].copy() # Default to horizon end
        
        # Logic: 
        # If Upper hit first -> Buy (2)
        # If Lower hit first -> Sell (0)
        # If Both hit same candle -> Sell (Conservative)
        
        # Buy: Hit Upper AND (Not Hit Lower OR Upper < Lower)
        buy_mask = any_upper & (~any_lower | (upper_indices < lower_indices))
        
        # Sell: Hit Lower AND (Not Hit Upper OR Lower < Upper)
        sell_mask = any_lower & (~any_upper | (lower_indices < upper_indices))
        
        labels[buy_mask] = 2
        labels[sell_mask] = 0
        
        # Set t1 (exit time) to the time of the barrier hit
        row_indices = np.arange(valid_len)
        t1_vals[buy_mask] = time_windows[row_indices[buy_mask], upper_indices[buy_mask]]
        t1_vals[sell_mask] = time_windows[row_indices[sell_mask], lower_indices[sell_mask]]
        
        result_index = df.index[:valid_len]
        
        return pd.Series(labels, index=result_index), pd.Series(t1_vals, index=result_index)

    @staticmethod
    def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        if len(X) <= seq_length:
            return np.array([]), np.array([])
        shape = (len(X) - seq_length + 1, seq_length, X.shape[1])
        strides = (X.strides[0], X.strides[0], X.strides[1])
        X_seq = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
        y_seq = y[seq_length-1:]
        return X_seq, y_seq
