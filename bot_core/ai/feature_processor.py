import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict

from bot_core.logger import get_logger
from bot_core.config import AIEnsembleStrategyParams

logger = get_logger(__name__)

class FeatureProcessor:
    """
    Encapsulates all feature engineering, normalization, and labeling logic.
    Implements institutional-grade labeling (Triple Barrier) and stationarity checks.
    """

    @staticmethod
    def get_feature_names(config: AIEnsembleStrategyParams) -> List[str]:
        cols = list(config.feature_columns)
        if config.features.use_leader_features:
            cols.extend(["leader_log_return", "leader_rsi"])

        lags = config.features.lag_features
        depth = config.features.lag_depth
        if depth > 0 and lags:
            for col in lags:
                if col in config.feature_columns:
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
    def _apply_frac_diff(df: pd.DataFrame, d: float, thres: float) -> pd.DataFrame:
        df = df.copy()
        if 'close' not in df.columns: return df
        
        # Weights calculation
        w, k = [1.], 1
        while True:
            w_k = -w[-1] / k * (d - k + 1)
            if abs(w_k) < thres or len(w) >= 2000:
                break
            w.append(w_k)
            k += 1
        weights = np.array(w[::-1])

        # Apply convolution
        close_vals = df['close'].values
        if len(close_vals) < len(weights):
            df['close_frac'] = np.nan
            return df
            
        frac_diff = np.convolve(close_vals, weights, mode='valid')
        pad_len = len(close_vals) - len(frac_diff)
        df['close_frac'] = np.concatenate([np.full(pad_len, np.nan), frac_diff])
        return df

    @staticmethod
    def _stationarize_data(df: pd.DataFrame, cols_to_process: List[str]) -> pd.DataFrame:
        df_trans = df.copy()
        if 'volume' in df_trans.columns and 'volume' in cols_to_process:
            vol_ma = df_trans['volume'].rolling(window=50, min_periods=1).mean().replace(0, 1.0)
            df_trans['volume'] = df_trans['volume'] / vol_ma

        price_keywords = ['upper', 'lower', 'sma', 'ema', 'wma', 'kama', 'mid', 'top', 'bot', 'open', 'high', 'low']
        if 'close' in df_trans.columns:
            close_price = df_trans['close'].replace(0, 1e-8)
            for col in cols_to_process:
                if col == 'close' or col not in df_trans.columns: continue
                if any(k in col.lower() for k in price_keywords):
                    # Check magnitude similarity to avoid transforming oscillators
                    if 0.5 < (df_trans[col].iloc[-1] / close_price.iloc[-1]) < 1.5:
                        df_trans[col] = (df_trans[col] - close_price) / close_price
        return df_trans

    @staticmethod
    def process_data(df: pd.DataFrame, config: AIEnsembleStrategyParams, leader_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        df_proc = df.copy()
        
        if config.features.use_time_features: df_proc = FeatureProcessor._add_time_features(df_proc)
        if config.features.use_price_action_features: df_proc = FeatureProcessor._add_price_action_features(df_proc)
        if config.features.use_volatility_estimators: df_proc = FeatureProcessor._add_volatility_features(df_proc)
        if config.features.use_microstructure_features: df_proc = FeatureProcessor._add_microstructure_features(df_proc)
        if config.features.use_frac_diff: df_proc = FeatureProcessor._apply_frac_diff(df_proc, config.features.frac_diff_d, config.features.frac_diff_thres)

        if config.features.use_leader_features and leader_df is not None and not leader_df.empty:
            leader_feats = pd.DataFrame(index=leader_df.index)
            leader_feats['leader_log_return'] = np.log(leader_df['close'] / leader_df['close'].shift(1).replace(0, np.nan))
            if 'rsi' in leader_df.columns: leader_feats['leader_rsi'] = leader_df['rsi']
            df_proc = df_proc.join(leader_feats, how='left').ffill(limit=5)

        df_stat = FeatureProcessor._stationarize_data(df_proc, config.feature_columns)
        
        # Select and Lag
        cols = list(config.feature_columns)
        if config.features.use_leader_features: cols.extend(["leader_log_return", "leader_rsi"])
        
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
        """
        Returns:
            labels: Series of target classes (0, 1, 2)
            t1: Series of event end timestamps (for Purged CV)
        """
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
        
        # t1 is simply index + horizon for fixed horizon labeling
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
        
        # Pre-calculate barriers
        upper_barriers = close + (vol * tp_mult)
        lower_barriers = close - (vol * sl_mult)
        
        labels = np.ones(len(df), dtype=int)
        t1_arr = np.full(len(df), np.nan, dtype='object')
        
        # Optimized Window Search
        # We iterate through the array, but we limit the inner loop scope
        # For very large datasets, this is still O(N*Horizon), but much faster than pandas iterrows
        
        limit = len(df) - horizon
        
        for i in range(limit):
            if np.isnan(vol[i]) or vol[i] == 0: continue
            
            # Define window
            end_idx = i + horizon
            window_high = high[i+1 : end_idx+1]
            window_low = low[i+1 : end_idx+1]
            window_times = timestamps[i+1 : end_idx+1]
            
            # Check touches
            # np.argmax returns the index of the first True. If no True, returns 0.
            # We must check if any are True.
            
            hit_upper_mask = window_high >= upper_barriers[i]
            hit_lower_mask = window_low <= lower_barriers[i]
            
            has_upper = hit_upper_mask.any()
            has_lower = hit_lower_mask.any()
            
            if not has_upper and not has_lower:
                # Vertical Barrier
                labels[i] = 1
                t1_arr[i] = window_times[-1]
                continue
                
            first_upper = np.argmax(hit_upper_mask) if has_upper else 999999
            first_lower = np.argmax(hit_lower_mask) if has_lower else 999999
            
            if first_upper < first_lower:
                labels[i] = 2 # Buy Signal (Hit Upper)
                t1_arr[i] = window_times[first_upper]
            else:
                labels[i] = 0 # Sell Signal (Hit Lower)
                t1_arr[i] = window_times[first_lower]
                
        return pd.Series(labels, index=df.index).iloc[:-horizon], pd.Series(t1_arr, index=df.index).iloc[:-horizon]

    @staticmethod
    def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        if len(X) <= seq_length:
            return np.array([]), np.array([])
        # Sliding window view for memory efficiency
        shape = (len(X) - seq_length + 1, seq_length, X.shape[1])
        strides = (X.strides[0], X.strides[0], X.strides[1])
        X_seq = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
        y_seq = y[seq_length-1:]
        return X_seq, y_seq
