from enum import Enum
import pandas as pd
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

    async def detect_regime(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes the provided DataFrame to determine the market regime.
        Uses columns defined in config.market_regime to avoid hardcoded assumptions.
        """
        if len(df) < 50:
            return {'regime': MarketRegime.UNKNOWN.value, 'confidence': 0.0}
        
        mr_config = self.config.market_regime
        
        # Resolve column names from config
        fast_ma_col = mr_config.trend_fast_ma_col
        slow_ma_col = mr_config.trend_slow_ma_col
        vol_col = mr_config.volatility_col
        rsi_col = mr_config.rsi_col

        # Check if required columns exist
        missing_cols = [col for col in [fast_ma_col, slow_ma_col] if col not in df.columns]
        if missing_cols:
            logger.warning("Missing required columns for regime detection", symbol=symbol, missing=missing_cols)
            return {'regime': MarketRegime.UNKNOWN.value, 'confidence': 0.0}

        try:
            # 1. Trend Detection
            fast_ma = df[fast_ma_col].iloc[-1]
            slow_ma = df[slow_ma_col].iloc[-1]
            
            # Calculate trend strength: (Fast - Slow) / Slow
            trend_strength = (fast_ma - slow_ma) / slow_ma if slow_ma > 0 else 0
            
            # 2. Volatility Detection
            # Compare current volatility (e.g., ATR) to its long-term average
            is_volatile = False
            if vol_col in df.columns:
                current_vol = df[vol_col].iloc[-1]
                # Calculate rolling average of volatility on the fly if not present
                # This is fast enough for a single column
                avg_vol = df[vol_col].rolling(50).mean().iloc[-1]
                
                if avg_vol > 0 and current_vol > (avg_vol * mr_config.volatility_multiplier):
                    is_volatile = True
            
            # 3. RSI (Momentum) Check
            rsi_val = df[rsi_col].iloc[-1] if rsi_col in df.columns else 50

            # Determine Regime
            trend_thresh = mr_config.trend_strength_threshold
            
            if is_volatile:
                regime = MarketRegime.VOLATILE
            elif trend_strength > trend_thresh:
                regime = MarketRegime.BULL
            elif trend_strength < -trend_thresh:
                regime = MarketRegime.BEAR
            else:
                regime = MarketRegime.SIDEWAYS

            # Calculate Confidence
            # Base confidence on trend strength magnitude and RSI confirmation
            trend_conf = min(1.0, abs(trend_strength) * 20)
            rsi_conf = abs(rsi_val - 50) / 50
            confidence = (trend_conf * 0.7) + (rsi_conf * 0.3)
            
            result = {'regime': regime.value, 'confidence': round(confidence, 2)}
            logger.debug("Market regime detected", symbol=symbol, result=result)
            return result
            
        except Exception as e:
            logger.error("Error in regime detection", symbol=symbol, error=str(e))
            return {'regime': MarketRegime.UNKNOWN.value, 'confidence': 0.0}
