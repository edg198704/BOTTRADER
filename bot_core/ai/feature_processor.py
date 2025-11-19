import pandas as pd
import numpy as np
from typing import Tuple, Optional

from bot_core.logger import get_logger
from bot_core.config import AIEnsembleStrategyParams

logger = get_logger(__name__)

class FeatureProcessor:
    """
    Encapsulates all feature engineering, normalization, and labeling logic
    to ensure consistency between training and inference phases.
    """

    @staticmethod
    def normalize(df: pd.DataFrame, config: AIEnsembleStrategyParams) -> pd.DataFrame:
        """Applies rolling Z-score normalization."""
        window = config.features.normalization_window
        cols = config.feature_columns
        
        # Ensure we only select columns that exist
        valid_cols = [c for c in cols if c in df.columns]
        if len(valid_cols) != len(cols):
            missing = set(cols) - set(valid_cols)
            logger.warning("Missing columns during normalization", missing=missing)
        
        subset = df[valid_cols].copy()
        rolling_mean = subset.rolling(window=window).mean()
        rolling_std = subset.rolling(window=window).std()
        
        epsilon = 1e-8
        normalized = (subset - rolling_mean) / (rolling_std + epsilon)
        return normalized

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
