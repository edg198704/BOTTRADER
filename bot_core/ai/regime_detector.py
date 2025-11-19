from enum import Enum
import pandas as pd
import numpy as np
from typing import Dict, Any

from bot_core.logger import get_logger
from bot_core.config import AIEnsembleStrategyParams

logger = get_logger(__name__)

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"

class MarketRegimeDetector:
    """Detects the current market regime based on configured technical indicators."""
    def __init__(self, config: AIEnsembleStrategyParams):
        self.config = config
        logger.info("MarketRegimeDetector initialized.", 
                    trend_fast=config.market_regime.trend_fast_ma_col,
                    trend_slow=config.market_regime.trend_slow_ma_col,
                    use_adx=config.market_regime.use_adx_filter,
                    efficiency_period=config.market_regime.efficiency_period)

    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates continuous regime metrics and adds them as columns to the DataFrame.
        These features (regime_trend, regime_volatility, regime_efficiency) are injected 
        into the AI model's feature set.
        """
        if df is None or df.empty:
            return df
            
        df = df.copy()
        mr_config = self.config.market_regime
        
        fast_ma_col = mr_config.trend_fast_ma_col
        slow_ma_col = mr_config.trend_slow_ma_col
        vol_col = mr_config.volatility_col
        
        # 1. Trend Feature: (Fast - Slow) / Slow
        if fast_ma_col in df.columns and slow_ma_col in df.columns:
            slow_ma = df[slow_ma_col].replace(0, np.nan)
            df['regime_trend'] = (df[fast_ma_col] - slow_ma) / slow_ma
        else:
            df['regime_trend'] = 0.0
            
        # 2. Volatility Feature: Current / Avg(50)
        if vol_col in df.columns:
            avg_vol = df[vol_col].rolling(window=50, min_periods=1).mean()
            avg_vol = avg_vol.replace(0, np.nan)
            df['regime_volatility'] = df[vol_col] / avg_vol
        else:
            df['regime_volatility'] = 1.0

        # 3. Efficiency Feature (Kaufman Efficiency Ratio)
        # KER = Abs(Change) / Sum(Abs(Diff))
        # Measures signal-to-noise. 1.0 = Perfect Trend, ~0.0 = Pure Noise.
        period = mr_config.efficiency_period
        if 'close' in df.columns:
            change = df['close'].diff(period).abs()
            volatility = df['close'].diff(1).abs().rolling(window=period).sum()
            
            # Avoid division by zero
            volatility = volatility.replace(0, np.nan)
            df['regime_efficiency'] = change / volatility
        else:
            df['regime_efficiency'] = 0.5 # Neutral fallback
            
        # Fill NaNs
        values = {'regime_trend': 0.0, 'regime_volatility': 1.0, 'regime_efficiency': 0.5}
        return df.fillna(value=values)

    def get_regime_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Vectorized calculation of market regimes for the entire DataFrame.
        Returns a Series of regime strings.
        """
        if df is None or df.empty:
            return pd.Series()

        mr_config = self.config.market_regime
        
        # Ensure features exist
        if not {'regime_trend', 'regime_volatility', 'regime_efficiency'}.issubset(df.columns):
            df = self.add_regime_features(df)

        # --- Thresholds ---
        trend_thresh = mr_config.trend_strength_threshold
        vol_thresh = mr_config.volatility_multiplier
        eff_thresh = mr_config.efficiency_threshold

        # Dynamic Thresholds
        if mr_config.use_dynamic_thresholds:
            window = mr_config.dynamic_window
            # Calculate rolling quantiles
            # Note: This is computationally expensive for very large DFs, but necessary for adaptive logic
            dynamic_trend_thresh = df['regime_trend'].abs().rolling(window=window, min_periods=100).quantile(mr_config.trend_percentile)
            dynamic_vol_thresh = df['regime_volatility'].rolling(window=window, min_periods=100).quantile(mr_config.volatility_percentile)
            
            # Use dynamic if available, else static fallback (handled by max/fillna logic implicitly via series ops)
            # We'll use the series directly in the conditions below
            trend_thresh_series = dynamic_trend_thresh.fillna(trend_thresh)
            vol_thresh_series = dynamic_vol_thresh.fillna(vol_thresh)
        else:
            trend_thresh_series = pd.Series(trend_thresh, index=df.index)
            vol_thresh_series = pd.Series(vol_thresh, index=df.index)

        # --- Conditions ---
        # Initialize as SIDEWAYS
        regimes = pd.Series(MarketRegime.SIDEWAYS.value, index=df.index)

        # 1. ADX Filter (Priority 1 - Forces Sideways)
        adx_mask = pd.Series(False, index=df.index)
        if mr_config.use_adx_filter and mr_config.adx_col in df.columns:
            adx_mask = df[mr_config.adx_col] < mr_config.adx_threshold
            # We don't need to set it here, as default is SIDEWAYS, but we use mask to prevent other assignments

        # 2. Efficiency Filter (Priority 2)
        # If efficiency < threshold, it's either SIDEWAYS or VOLATILE
        inefficient_mask = df['regime_efficiency'] < eff_thresh
        volatile_mask = df['regime_volatility'] > vol_thresh_series
        
        # 3. Trend/Vol Logic
        bull_mask = df['regime_trend'] > trend_thresh_series
        bear_mask = df['regime_trend'] < -trend_thresh_series
        
        # Apply Logic Hierarchy (Vectorized)
        # Start with Trend
        regimes[bull_mask] = MarketRegime.BULL.value
        regimes[bear_mask] = MarketRegime.BEAR.value
        
        # Overwrite with Volatile
        regimes[volatile_mask] = MarketRegime.VOLATILE.value
        
        # Overwrite with Inefficient Logic
        # If inefficient, it's Volatile if vol is high, else Sideways
        regimes[inefficient_mask & volatile_mask] = MarketRegime.VOLATILE.value
        regimes[inefficient_mask & ~volatile_mask] = MarketRegime.SIDEWAYS.value
        
        # Overwrite with ADX (Forces Sideways)
        regimes[adx_mask] = MarketRegime.SIDEWAYS.value
        
        return regimes

    async def detect_regime(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes the provided DataFrame to determine the market regime.
        Uses pre-calculated regime features if available, otherwise calculates on the fly.
        Supports adaptive thresholds based on historical distribution.
        """
        if len(df) < 50:
            return {'regime': MarketRegime.UNKNOWN.value, 'confidence': 0.0}
        
        mr_config = self.config.market_regime
        rsi_col = mr_config.rsi_col

        try:
            # Use pre-calculated features if they exist
            if {'regime_trend', 'regime_volatility', 'regime_efficiency'}.issubset(df.columns):
                trend_strength = df['regime_trend'].iloc[-1]
                vol_ratio = df['regime_volatility'].iloc[-1]
                efficiency = df['regime_efficiency'].iloc[-1]
            else:
                # Fallback calculation (simplified)
                fast_ma = df[mr_config.trend_fast_ma_col].iloc[-1] if mr_config.trend_fast_ma_col in df.columns else 0
                slow_ma = df[mr_config.trend_slow_ma_col].iloc[-1] if mr_config.trend_slow_ma_col in df.columns else 1
                trend_strength = (fast_ma - slow_ma) / slow_ma if slow_ma > 0 else 0
                vol_ratio = 1.0 # Simplified
                efficiency = 0.5 # Simplified

            # --- Determine Thresholds (Adaptive vs Static) ---
            trend_thresh = mr_config.trend_strength_threshold
            vol_thresh = mr_config.volatility_multiplier
            eff_thresh = mr_config.efficiency_threshold

            if mr_config.use_dynamic_thresholds and len(df) >= mr_config.dynamic_window:
                window = df.iloc[-mr_config.dynamic_window:]
                
                if 'regime_trend' in window.columns:
                    trend_series = window['regime_trend'].abs()
                    calc_trend_thresh = trend_series.quantile(mr_config.trend_percentile)
                    trend_thresh = max(calc_trend_thresh, 0.001)
                
                if 'regime_volatility' in window.columns:
                    vol_series = window['regime_volatility']
                    vol_thresh = vol_series.quantile(mr_config.volatility_percentile)

            # --- Regime Classification ---
            regime = MarketRegime.SIDEWAYS
            confidence = 0.0
            
            # 1. ADX Filter (Priority 1)
            forced_sideways = False
            if mr_config.use_adx_filter and mr_config.adx_col in df.columns:
                adx_val = df[mr_config.adx_col].iloc[-1]
                if adx_val < mr_config.adx_threshold:
                    regime = MarketRegime.SIDEWAYS
                    forced_sideways = True
                    confidence = min(1.0, (mr_config.adx_threshold - adx_val) / mr_config.adx_threshold)
                    confidence = 0.5 + (confidence * 0.5)

            # 2. Efficiency Filter (Priority 2)
            # If efficiency is low, it's choppy/noisy regardless of trend strength
            if not forced_sideways:
                if efficiency < eff_thresh:
                    # Low efficiency = Chop or Volatile
                    if vol_ratio > vol_thresh:
                        regime = MarketRegime.VOLATILE
                    else:
                        regime = MarketRegime.SIDEWAYS
                    
                    forced_sideways = True
                    # Confidence based on how inefficient it is
                    confidence = min(1.0, (eff_thresh - efficiency) / eff_thresh)

            # 3. Standard Trend/Vol Logic (Priority 3)
            if not forced_sideways:
                is_volatile = vol_ratio > vol_thresh
                
                if is_volatile:
                    regime = MarketRegime.VOLATILE
                elif trend_strength > trend_thresh:
                    regime = MarketRegime.BULL
                elif trend_strength < -trend_thresh:
                    regime = MarketRegime.BEAR
                else:
                    regime = MarketRegime.SIDEWAYS

                # --- Confidence Calculation ---
                rsi_val = df[rsi_col].iloc[-1] if rsi_col in df.columns else 50
                trend_conf = min(1.0, abs(trend_strength) / (trend_thresh * 2)) if trend_thresh > 0 else 0.0
                rsi_conf = abs(rsi_val - 50) / 50
                
                confidence = (trend_conf * 0.6) + (rsi_conf * 0.2) + (efficiency * 0.2)
            
            result = {
                'regime': regime.value, 
                'confidence': round(confidence, 2),
                'thresholds': {'trend': round(trend_thresh, 5), 'vol': round(vol_thresh, 3), 'eff': round(eff_thresh, 2)}
            }
            logger.debug("Market regime detected", symbol=symbol, result=result)
            return result
            
        except Exception as e:
            logger.error("Error in regime detection", symbol=symbol, error=str(e))
            return {'regime': MarketRegime.UNKNOWN.value, 'confidence': 0.0}
