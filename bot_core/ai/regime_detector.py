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
                    use_adx=config.market_regime.use_adx_filter)

    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates continuous regime metrics and adds them as columns to the DataFrame.
        These features (regime_trend, regime_volatility) are injected into the AI model's feature set.
        """
        if df is None or df.empty:
            return df
            
        df = df.copy()
        mr_config = self.config.market_regime
        
        fast_ma_col = mr_config.trend_fast_ma_col
        slow_ma_col = mr_config.trend_slow_ma_col
        vol_col = mr_config.volatility_col
        
        # 1. Trend Feature: (Fast - Slow) / Slow
        # Positive = Bullish, Negative = Bearish, Magnitude = Strength
        if fast_ma_col in df.columns and slow_ma_col in df.columns:
            slow_ma = df[slow_ma_col].replace(0, np.nan)
            df['regime_trend'] = (df[fast_ma_col] - slow_ma) / slow_ma
        else:
            # Fallback if columns missing (e.g. during warmup)
            df['regime_trend'] = 0.0
            
        # 2. Volatility Feature: Current / Avg(50)
        # > 1.0 means volatility is expanding, < 1.0 means contracting
        if vol_col in df.columns:
            # Use a rolling mean for average volatility baseline
            avg_vol = df[vol_col].rolling(window=50, min_periods=1).mean()
            avg_vol = avg_vol.replace(0, np.nan)
            df['regime_volatility'] = df[vol_col] / avg_vol
        else:
            df['regime_volatility'] = 1.0
            
        # Fill NaNs that might result from rolling windows or division
        # We use 0.0 for trend (neutral) and 1.0 for volatility (average)
        values = {'regime_trend': 0.0, 'regime_volatility': 1.0}
        return df.fillna(value=values)

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
            # Use pre-calculated features if they exist (optimization)
            if 'regime_trend' in df.columns and 'regime_volatility' in df.columns:
                trend_strength = df['regime_trend'].iloc[-1]
                vol_ratio = df['regime_volatility'].iloc[-1]
            else:
                # Fallback to manual calculation for the last row
                fast_ma_col = mr_config.trend_fast_ma_col
                slow_ma_col = mr_config.trend_slow_ma_col
                vol_col = mr_config.volatility_col
                
                fast_ma = df[fast_ma_col].iloc[-1] if fast_ma_col in df.columns else 0
                slow_ma = df[slow_ma_col].iloc[-1] if slow_ma_col in df.columns else 1
                trend_strength = (fast_ma - slow_ma) / slow_ma if slow_ma > 0 else 0
                
                vol_ratio = 1.0
                if vol_col in df.columns:
                    current_vol = df[vol_col].iloc[-1]
                    avg_vol = df[vol_col].rolling(50).mean().iloc[-1]
                    if avg_vol > 0:
                        vol_ratio = current_vol / avg_vol

            # --- Determine Thresholds (Adaptive vs Static) ---
            trend_thresh = mr_config.trend_strength_threshold
            vol_thresh = mr_config.volatility_multiplier

            if mr_config.use_dynamic_thresholds and len(df) >= mr_config.dynamic_window:
                # Calculate dynamic thresholds based on historical distribution
                window = df.iloc[-mr_config.dynamic_window:]
                
                # Trend Strength: We care about magnitude (abs value)
                if 'regime_trend' in window.columns:
                    trend_series = window['regime_trend'].abs()
                    # e.g., 75th percentile of absolute trend strength
                    calc_trend_thresh = trend_series.quantile(mr_config.trend_percentile)
                    # Ensure we don't get a near-zero threshold in flat markets
                    trend_thresh = max(calc_trend_thresh, 0.001)
                
                # Volatility Ratio
                if 'regime_volatility' in window.columns:
                    vol_series = window['regime_volatility']
                    # e.g., 80th percentile of volatility ratio
                    vol_thresh = vol_series.quantile(mr_config.volatility_percentile)

            # --- Regime Classification ---
            regime = MarketRegime.SIDEWAYS
            confidence = 0.0
            
            # 1. ADX Check (Priority)
            adx_forced_sideways = False
            if mr_config.use_adx_filter and mr_config.adx_col in df.columns:
                adx_val = df[mr_config.adx_col].iloc[-1]
                if adx_val < mr_config.adx_threshold:
                    regime = MarketRegime.SIDEWAYS
                    adx_forced_sideways = True
                    # Confidence is higher if ADX is significantly below threshold
                    # e.g. Thresh 25, ADX 10 -> (25-10)/25 = 0.6 base confidence
                    confidence = min(1.0, (mr_config.adx_threshold - adx_val) / mr_config.adx_threshold)
                    # Boost confidence for sideways
                    confidence = 0.5 + (confidence * 0.5)

            # 2. Standard Logic (if not forced by ADX)
            if not adx_forced_sideways:
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
                # RSI (Momentum) Check
                rsi_val = df[rsi_col].iloc[-1] if rsi_col in df.columns else 50
                
                # Normalize trend confidence relative to the threshold used
                trend_conf = min(1.0, abs(trend_strength) / (trend_thresh * 2)) if trend_thresh > 0 else 0.0
                rsi_conf = abs(rsi_val - 50) / 50
                
                confidence = (trend_conf * 0.7) + (rsi_conf * 0.3)
            
            result = {
                'regime': regime.value, 
                'confidence': round(confidence, 2),
                'thresholds': {'trend': round(trend_thresh, 5), 'vol': round(vol_thresh, 3)}
            }
            logger.debug("Market regime detected", symbol=symbol, result=result)
            return result
            
        except Exception as e:
            logger.error("Error in regime detection", symbol=symbol, error=str(e))
            return {'regime': MarketRegime.UNKNOWN.value, 'confidence': 0.0}
