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
                    trend_slow=config.market_regime.trend_slow_ma_col)

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

            # 3. RSI (Momentum) Check
            rsi_val = df[rsi_col].iloc[-1] if rsi_col in df.columns else 50

            # Determine Regime
            trend_thresh = mr_config.trend_strength_threshold
            is_volatile = vol_ratio > mr_config.volatility_multiplier
            
            if is_volatile:
                regime = MarketRegime.VOLATILE
            elif trend_strength > trend_thresh:
                regime = MarketRegime.BULL
            elif trend_strength < -trend_thresh:
                regime = MarketRegime.BEAR
            else:
                regime = MarketRegime.SIDEWAYS

            # Calculate Confidence
            trend_conf = min(1.0, abs(trend_strength) * 20)
            rsi_conf = abs(rsi_val - 50) / 50
            confidence = (trend_conf * 0.7) + (rsi_conf * 0.3)
            
            result = {'regime': regime.value, 'confidence': round(confidence, 2)}
            logger.debug("Market regime detected", symbol=symbol, result=result)
            return result
            
        except Exception as e:
            logger.error("Error in regime detection", symbol=symbol, error=str(e))
            return {'regime': MarketRegime.UNKNOWN.value, 'confidence': 0.0}
