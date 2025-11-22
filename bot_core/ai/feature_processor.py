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
    Optimized for performance and numerical stability.
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
    def validate_inputs(df: pd.DataFrame) -> bool:
        if df is None or df.empty:
            return False
        if len(df) < 50:
            return False
        return True

    @staticmethod
    def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        # Use .dt accessor efficiently
        dt = df.index
        hour = dt.hour
        dow = dt.dayofweek
        
        # Vectorized operations
        df['time_hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['time_hour_cos'] = np.cos(2 * np.pi * hour / 24)
        df['time_dow_sin'] = np.sin(2 * np.pi * dow / 7)
        df['time_dow_cos'] = np.cos(2 * np.pi * dow / 7)
        return df

    @staticmethod
    def _add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
        close = df['close'].replace(0, 1e-8)
        df['pa_body_size'] = (df['close'] - df['open']).abs() / close
        df['pa_upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / close
        df['pa_lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / close
        return df

    @staticmethod
    def _add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        # Garman-Klass Volatility
        # 0.5 * ln(High/Low)^2 - (2*ln(2)-1) * ln(Close/Open)^2
        high_low = np.log(df['high'] / df['low'].replace(0, 1e-8))
        close_open = np.log(df['close'] / df['open'].replace(0, 1e-8))
        df['volatility_gk'] = np.sqrt(0.5 * high_low**2 - (2 * np.log(2) - 1) * close_open**2)
        return df

    @staticmethod
    def _add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
        # Parkinson Volatility
        const = 1.0 / (4.0 * np.log(2.0))
        log_hl = np.log(df['high'] / df['low'].replace(0, 1e-8))
        df['volatility_parkinson'] = np.sqrt(const * log_hl**2)

        # Amihud Illiquidity: |Return| / (Price * Volume)
        abs_return = df['close'].pct_change().abs()
        dollar_volume = (df['volume'] * df['close']).replace(0, 1.0)
        df['amihud_illiquidity'] = (abs_return / dollar_volume).rolling(window=10).mean()
        return df

    @staticmethod
    def _add_order_book_features(df: pd.DataFrame) -> pd.DataFrame:
        if 'obi' not in df.columns:
            df['obi'] = 0.0
        return df

    @staticmethod
    @lru_cache(maxsize=32)
    def _get_frac_diff_weights(d: float, thres: float, max_len: int = 2000) -> np.ndarray:
        w = [1.0]
        k = 1
        while True:
            w_k = -w[-1] / k * (d - k + 1)
            if abs(w_k) < thres or k >= max_len:
                break
            w.append(w_k)
            k += 1
        return np.array(w[::-1])

    @staticmethod
    def _apply_frac_diff(df: pd.DataFrame, d: float, thres: float) -> pd.DataFrame:
        if 'close' not in df.columns: return df
        weights = FeatureProcessor._get_frac_diff_weights(d, thres)
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
        """
        Converts non-stationary price series to stationary returns relative to Close.
        """
        # Avoid deep copy if possible, but we are modifying columns so copy is safer
        df_trans = df.copy()
        
        if 'volume' in df_trans.columns and 'volume' in cols_to_process:
            vol_ma = df_trans['volume'].rolling(window=50, min_periods=1).mean().replace(0, 1.0)
            df_trans['volume'] = df_trans['volume'] / vol_ma

        if 'close' in df_trans.columns:
            close_price = df_trans['close'].replace(0, 1e-8)
            
            # Identify price columns strictly
            price_keywords = {'open', 'high', 'low', 'close', 'sma', 'ema', 'wma', 'kama', 'mid', 'top', 'bot', 'upper', 'lower', 'bb_upper', 'bb_lower'}
            
            price_cols = []
            for c in cols_to_process:
                if c in df_trans.columns and c != 'close' and c != 'obi':
                    # Check if any keyword is a substring of the column name
                    if any(k in c.lower() for k in price_keywords):
                        price_cols.append(c)
            
            if price_cols:
                # Vectorized normalization: (Price / Close) - 1.0
                df_trans[price_cols] = df_trans[price_cols].div(close_price, axis=0) - 1.0

        return df_trans

    @staticmethod
    def process_data(df: pd.DataFrame, config: AIEnsembleStrategyParams, leader_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not FeatureProcessor.validate_inputs(df):
            return pd.DataFrame()

        # Work on a copy to preserve original data integrity
        df_proc = df.copy()
        
        # 1. Generate Base Features
        if config.features.use_time_features: df_proc = FeatureProcessor._add_time_features(df_proc)
        if config.features.use_price_action_features: df_proc = FeatureProcessor._add_price_action_features(df_proc)
        if config.features.use_volatility_estimators: df_proc = FeatureProcessor._add_volatility_features(df_proc)
        if config.features.use_microstructure_features: df_proc = FeatureProcessor._add_microstructure_features(df_proc)
        if config.features.use_order_book_features: df_proc = FeatureProcessor._add_order_book_features(df_proc)
        if config.features.use_frac_diff: df_proc = FeatureProcessor._apply_frac_diff(df_proc, config.features.frac_diff_d, config.features.frac_diff_thres)

        # 2. Inject Leader Features
        if config.features.use_leader_features and leader_df is not None and not leader_df.empty:
            # Align leader data to current index
            leader_aligned = leader_df.reindex(df_proc.index, method='ffill', limit=5)
            df_proc['leader_log_return'] = np.log(leader_aligned['close'] / leader_aligned['close'].shift(1).replace(0, np.nan))
            if 'rsi' in leader_aligned.columns: 
                df_proc['leader_rsi'] = leader_aligned['rsi']

        # 3. Stationarize
        df_stat = FeatureProcessor._stationarize_data(df_proc, config.feature_columns)
        
        # 4. Select and Lag
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

        # 5. Clean Infinite/NaN values
        # Replace inf with nan, then dropna. This is safer than fillna(0) which poisons models.
        subset.replace([np.inf, -np.inf], np.nan, inplace=True)
        subset.dropna(inplace=True)
        
        if subset.empty:
            return subset

        # 6. Normalize
        window = config.features.normalization_window
        if config.features.scaling_method == 'robust':
            median = subset.rolling(window=window).median()
            iqr = subset.rolling(window=window).quantile(0.75) - subset.rolling(window=window).quantile(0.25)
            normalized = (subset - median) / iqr.replace(0, 1e-8)
        else:
            mean = subset.rolling(window=window).mean()
            std = subset.rolling(window=window).std()
            normalized = (subset - mean) / std.replace(0, 1e-8)
            
        normalized.replace([np.inf, -np.inf], np.nan, inplace=True)
        normalized.dropna(inplace=True)
        
        # 7. Enforce Column Order
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

        # Efficient sliding window view
        high_windows = sliding_window_view(high, window_shape=horizon)
        low_windows = sliding_window_view(low, window_shape=horizon)
        time_windows = sliding_window_view(timestamps, window_shape=horizon)
        
        valid_len = high_windows.shape[0]
        upper_barriers = upper_barriers[:valid_len].reshape(-1, 1)
        lower_barriers = lower_barriers[:valid_len].reshape(-1, 1)
        
        hit_upper = high_windows >= upper_barriers
        hit_lower = low_windows <= lower_barriers
        
        upper_indices = np.argmax(hit_upper, axis=1)
        lower_indices = np.argmax(hit_lower, axis=1)
        
        any_upper = np.any(hit_upper, axis=1)
        any_lower = np.any(hit_lower, axis=1)
        
        labels = np.ones(valid_len, dtype=int) # Hold
        t1_vals = time_windows[:, -1].copy()
        
        # Logic: Buy if Upper hit first. Sell if Lower hit first.
        buy_mask = any_upper & (~any_lower | (upper_indices < lower_indices))
        sell_mask = any_lower & (~any_upper | (lower_indices < upper_indices))
        
        labels[buy_mask] = 2
        labels[sell_mask] = 0
        
        row_indices = np.arange(valid_len)
        t1_vals[buy_mask] = time_windows[row_indices[buy_mask], upper_indices[buy_mask]]
        t1_vals[sell_mask] = time_windows[row_indices[sell_mask], lower_indices[sell_mask]]
        
        result_index = df.index[:valid_len]
        return pd.Series(labels, index=result_index), pd.Series(t1_vals, index=result_index)
