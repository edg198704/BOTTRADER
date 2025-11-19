import pandas as pd
import numpy as np
from typing import Tuple, Optional, List

from bot_core.logger import get_logger
from bot_core.config import AIEnsembleStrategyParams

logger = get_logger(__name__)

class FeatureProcessor:
    """
    Encapsulates all feature engineering, normalization, and labeling logic
    to ensure consistency between training and inference phases.
    """

    @staticmethod
    def get_expected_feature_count(config: AIEnsembleStrategyParams) -> int:
        """
        Calculates the total number of features expected after processing,
        including base features and generated lags.
        """
        return len(FeatureProcessor.get_feature_names(config))

    @staticmethod
    def get_feature_names(config: AIEnsembleStrategyParams) -> List[str]:
        """
        Returns the ordered list of feature names including generated lags.
        This serves as the single source of truth for feature alignment.
        """
        # Start with base feature columns in the order defined in config
        cols = list(config.feature_columns)
        
        # Add lag features in the exact order they are generated
        lags = config.features.lag_features
        depth = config.features.lag_depth
        
        if depth > 0 and lags:
            for col in lags:
                # Only generate lags for columns that are actually in the feature set
                if col in config.feature_columns:
                    for i in range(1, depth + 1):
                        cols.append(f"{col}_lag_{i}")
        return cols

    @staticmethod
    def _stationarize_data(df: pd.DataFrame, cols_to_process: List[str]) -> pd.DataFrame:
        """
        Transforms non-stationary features (absolute prices, raw volume) into stationary ratios.
        This improves model generalization across different price regimes.
        """
        df_trans = df.copy()
        
        # 1. Handle Volume -> Relative Volume
        if 'volume' in df_trans.columns and 'volume' in cols_to_process:
            # Use a 50-period rolling average for baseline
            vol_ma = df_trans['volume'].rolling(window=50, min_periods=1).mean()
            # Avoid division by zero
            vol_ma = vol_ma.replace(0, 1.0)
            df_trans['volume'] = df_trans['volume'] / vol_ma

        # 2. Handle Price Levels -> % Distance from Close
        # Heuristic: If a column name suggests it's a price level (e.g. 'upper', 'sma')
        # and its value is close to the current price, transform it to a relative distance.
        price_keywords = ['upper', 'lower', 'sma', 'ema', 'wma', 'kama', 'mid', 'top', 'bot', 'open', 'high', 'low']
        
        if 'close' in df_trans.columns:
            close_price = df_trans['close']
            for col in cols_to_process:
                if col == 'close': continue # 'close' itself is usually not a feature, or handled via log_return
                if col not in df_trans.columns: continue
                
                # Check naming convention
                is_price_like = any(k in col.lower() for k in price_keywords)
                
                if is_price_like:
                    try:
                        # Safety check: Magnitude should be comparable to close price (e.g. within 50%)
                        # This prevents transforming oscillators like RSI that might accidentally match a keyword
                        # We check the last valid value for efficiency
                        last_val = df_trans[col].iloc[-1]
                        last_close = close_price.iloc[-1]
                        
                        if last_close > 0 and 0.5 < (last_val / last_close) < 1.5:
                            # Transform to percentage distance: (Indicator - Close) / Close
                            df_trans[col] = (df_trans[col] - close_price) / close_price
                    except Exception:
                        # If check fails (e.g. NaNs), skip transformation
                        pass

        return df_trans

    @staticmethod
    def process_data(df: pd.DataFrame, config: AIEnsembleStrategyParams) -> pd.DataFrame:
        """
        Applies the full feature engineering pipeline:
        1. Stationarize features (Absolute -> Relative).
        2. Select base feature columns.
        3. Generates lagged features (if configured).
        4. Drops NaNs introduced by lags.
        5. Applies Z-score normalization.
        6. Enforces column order matching get_feature_names.
        """
        # 1. Stationarize Data (Transform absolute prices/volume to ratios)
        # We pass the list of intended features so we only transform what's needed
        df_stationary = FeatureProcessor._stationarize_data(df, config.feature_columns)

        # 2. Select Base Columns
        cols = config.feature_columns
        valid_cols = [c for c in cols if c in df_stationary.columns]
        
        if len(valid_cols) != len(cols):
            missing = set(cols) - set(valid_cols)
            logger.warning("Missing columns during processing", missing=missing)
        
        subset = df_stationary[valid_cols].copy()
        
        # 3. Generate Lags
        lags = config.features.lag_features
        depth = config.features.lag_depth
        
        if depth > 0 and lags:
            for col in lags:
                if col in subset.columns:
                    for i in range(1, depth + 1):
                        subset[f"{col}_lag_{i}"] = subset[col].shift(i)
        
        # 4. Drop NaNs (from lags and missing data)
        subset.dropna(inplace=True)
        
        # 5. Normalize
        window = config.features.normalization_window
        rolling_mean = subset.rolling(window=window).mean()
        rolling_std = subset.rolling(window=window).std()
        
        epsilon = 1e-8
        normalized = (subset - rolling_mean) / (rolling_std + epsilon)
        
        # Rolling normalization introduces NaNs at the start of the window
        normalized.dropna(inplace=True)
        
        # 6. Enforce Column Order
        # Ensure the output DataFrame has columns in the exact order expected by the model
        expected_cols = FeatureProcessor.get_feature_names(config)
        # Filter expected cols to those present (handling potential missing base cols gracefully)
        final_cols = [c for c in expected_cols if c in normalized.columns]
        
        return normalized[final_cols]

    @staticmethod
    def create_labels(df: pd.DataFrame, config: AIEnsembleStrategyParams) -> pd.Series:
        """Generates target labels based on configuration (Triple Barrier or Simple)."""
        if config.features.use_triple_barrier:
            return FeatureProcessor._create_triple_barrier_labels(df, config)

        horizon = config.features.labeling_horizon
        future_price = df['close'].shift(-horizon)
        price_change_pct = (future_price - df['close']) / df['close']

        labels = pd.Series(1, index=df.index)  # Default to 'hold' (1)

        if config.features.use_dynamic_labeling:
            atr_col = config.market_regime.volatility_col
            if atr_col in df.columns:
                multiplier = config.features.labeling_atr_multiplier
                dynamic_threshold = (df[atr_col] * multiplier) / df['close']
                labels[price_change_pct > dynamic_threshold] = 2  # 'buy'
                labels[price_change_pct < -dynamic_threshold] = 0 # 'sell'
            else:
                threshold = config.features.labeling_threshold
                labels[price_change_pct > threshold] = 2
                labels[price_change_pct < -threshold] = 0
        else:
            threshold = config.features.labeling_threshold
            labels[price_change_pct > threshold] = 2
            labels[price_change_pct < -threshold] = 0
        
        return labels.iloc[:-horizon]

    @staticmethod
    def _create_triple_barrier_labels(df: pd.DataFrame, config: AIEnsembleStrategyParams) -> pd.Series:
        """
        Implements the Triple Barrier Method for labeling.
        Labels: 0 (SELL), 1 (HOLD), 2 (BUY)
        """
        horizon = config.features.labeling_horizon
        tp_mult = config.features.triple_barrier_tp_multiplier
        sl_mult = config.features.triple_barrier_sl_multiplier
        
        # Use ATR for volatility if available, else fallback to percentage
        atr_col = config.market_regime.volatility_col
        
        # Prepare arrays for speed
        close_arr = df['close'].values
        high_arr = df['high'].values
        low_arr = df['low'].values
        
        if atr_col in df.columns:
            vol_arr = df[atr_col].values
        else:
            # Fallback volatility: 1% of price if ATR is missing
            vol_arr = close_arr * 0.01
        
        labels = np.ones(len(df), dtype=int) # Default 1 (HOLD)
        
        # Iterate up to len - horizon
        limit = len(df) - horizon
        
        for i in range(limit):
            current_close = close_arr[i]
            vol = vol_arr[i]
            
            if np.isnan(vol) or vol == 0:
                continue
                
            upper_barrier = current_close + (vol * tp_mult)
            lower_barrier = current_close - (vol * sl_mult)
            
            # Slice the future window (path dependency)
            window_high = high_arr[i+1 : i+1+horizon]
            window_low = low_arr[i+1 : i+1+horizon]
            
            # Check hits
            hit_upper_mask = window_high >= upper_barrier
            hit_lower_mask = window_low <= lower_barrier
            
            has_upper = hit_upper_mask.any()
            has_lower = hit_lower_mask.any()
            
            if has_upper and not has_lower:
                labels[i] = 2 # BUY
            elif has_lower and not has_upper:
                labels[i] = 0 # SELL
            elif has_upper and has_lower:
                # Both hit. Which was first?
                first_upper_idx = np.argmax(hit_upper_mask)
                first_lower_idx = np.argmax(hit_lower_mask)
                
                if first_upper_idx < first_lower_idx:
                    labels[i] = 2 # BUY
                elif first_lower_idx < first_upper_idx:
                    labels[i] = 0 # SELL
                else:
                    # Same bar hit both barriers (extreme volatility). Default to HOLD/Avoid.
                    labels[i] = 1
            else:
                labels[i] = 1 # HOLD
                
        return pd.Series(labels, index=df.index).iloc[:-horizon]

    @staticmethod
    def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Creates sequences for LSTM/Attention models."""
        X_seq, y_seq = [], []
        if len(X) <= seq_length:
            return np.array([]), np.array([])
            
        for i in range(len(X) - seq_length + 1):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length-1])
            
        return np.array(X_seq), np.array(y_seq)
