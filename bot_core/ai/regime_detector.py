from enum import Enum
import pandas as pd
import numpy as np
import asyncio
from typing import Dict, Any, Optional

from bot_core.logger import get_logger
from bot_core.config import AIEnsembleStrategyParams
from bot_core.data_handler import DataHandler

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
                    use_hurst=config.market_regime.use_hurst_filter,
                    confirmation_window=config.market_regime.regime_confirmation_window)

    def _calculate_rolling_hurst(self, series: pd.Series, window: int) -> pd.Series:
        """
        Calculates the rolling Hurst Exponent using a simplified Rescaled Range (R/S) analysis.
        H < 0.5: Mean Reverting
        H = 0.5: Random Walk
        H > 0.5: Trending
        """
        # Calculate Log Returns
        returns = np.log(series / series.shift(1))
        
        def get_hurst(chunk):
            if len(chunk) < 8: return 0.5
            # Simplified R/S calculation for a single window
            # 1. Mean of returns
            mean_ret = np.mean(chunk)
            # 2. Deviations
            deviations = chunk - mean_ret
            # 3. Cumulative Deviations
            cum_dev = np.cumsum(deviations)
            # 4. Range
            R = np.max(cum_dev) - np.min(cum_dev)
            # 5. Standard Deviation
            S = np.std(chunk, ddof=1)
            
            if S == 0 or R == 0:
                return 0.5
            
            # 6. RS
            RS = R / S
            # 7. Hurst
            # H = log(RS) / log(N)
            return np.log(RS) / np.log(len(chunk))

        # Apply rolling window
        return returns.rolling(window=window).apply(get_hurst, raw=True)

    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates continuous regime metrics and adds them as columns to the DataFrame.
        These features (regime_trend, regime_volatility, regime_efficiency, regime_hurst) 
        are injected into the AI model's feature set.
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
        period = mr_config.efficiency_period
        if 'close' in df.columns:
            change = df['close'].diff(period).abs()
            volatility = df['close'].diff(1).abs().rolling(window=period).sum()
            volatility = volatility.replace(0, np.nan)
            df['regime_efficiency'] = change / volatility
        else:
            df['regime_efficiency'] = 0.5

        # 4. Hurst Exponent Feature
        if mr_config.use_hurst_filter and 'close' in df.columns:
            df['regime_hurst'] = self._calculate_rolling_hurst(df['close'], mr_config.hurst_window)
        else:
            df['regime_hurst'] = 0.5
            
        # Fill NaNs
        values = {'regime_trend': 0.0, 'regime_volatility': 1.0, 'regime_efficiency': 0.5, 'regime_hurst': 0.5}
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
        if not {'regime_trend', 'regime_volatility', 'regime_efficiency', 'regime_hurst'}.issubset(df.columns):
            df = self.add_regime_features(df)

        # --- Thresholds ---
        trend_thresh = mr_config.trend_strength_threshold
        vol_thresh = mr_config.volatility_multiplier
        eff_thresh = mr_config.efficiency_threshold
        hurst_mr_thresh = mr_config.hurst_mean_reversion_threshold
        hurst_trend_thresh = mr_config.hurst_trending_threshold

        # Dynamic Thresholds
        if mr_config.use_dynamic_thresholds:
            window = mr_config.dynamic_window
            dynamic_trend_thresh = df['regime_trend'].abs().rolling(window=window, min_periods=100).quantile(mr_config.trend_percentile)
            dynamic_vol_thresh = df['regime_volatility'].rolling(window=window, min_periods=100).quantile(mr_config.volatility_percentile)
            trend_thresh_series = dynamic_trend_thresh.fillna(trend_thresh)
            vol_thresh_series = dynamic_vol_thresh.fillna(vol_thresh)
        else:
            trend_thresh_series = pd.Series(trend_thresh, index=df.index)
            vol_thresh_series = pd.Series(vol_thresh, index=df.index)

        # --- Conditions ---
        regimes = pd.Series(MarketRegime.SIDEWAYS.value, index=df.index)

        # 1. ADX Filter (Priority 1)
        adx_mask = pd.Series(False, index=df.index)
        if mr_config.use_adx_filter and mr_config.adx_col in df.columns:
            adx_mask = df[mr_config.adx_col] < mr_config.adx_threshold

        # 2. Efficiency & Hurst Filter (Priority 2)
        inefficient_mask = df['regime_efficiency'] < eff_thresh
        volatile_mask = df['regime_volatility'] > vol_thresh_series
        
        # Hurst Logic: If H < 0.45, it's Mean Reverting (Sideways)
        mean_reverting_mask = df['regime_hurst'] < hurst_mr_thresh
        
        # 3. Trend Logic
        bull_mask = df['regime_trend'] > trend_thresh_series
        bear_mask = df['regime_trend'] < -trend_thresh_series
        
        # Apply Logic Hierarchy
        regimes[bull_mask] = MarketRegime.BULL.value
        regimes[bear_mask] = MarketRegime.BEAR.value
        
        # Overwrite with Volatile
        regimes[volatile_mask] = MarketRegime.VOLATILE.value
        
        # Overwrite with Inefficient/Mean Reverting Logic
        chop_mask = inefficient_mask | mean_reverting_mask
        regimes[chop_mask & ~volatile_mask] = MarketRegime.SIDEWAYS.value
        regimes[chop_mask & volatile_mask] = MarketRegime.VOLATILE.value
        
        # Overwrite with ADX
        regimes[adx_mask] = MarketRegime.SIDEWAYS.value
        
        return regimes

    async def detect_regime(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes the provided DataFrame to determine the market regime.
        Uses a sliding window voting mechanism (Hysteresis) to prevent signal flickering.
        Offloads heavy calculation to an executor.
        """
        if len(df) < 50:
            return {'regime': MarketRegime.UNKNOWN.value, 'confidence': 0.0}
        
        loop = asyncio.get_running_loop()
        # Offload the heavy lifting (feature calc + regime logic) to a thread
        return await loop.run_in_executor(None, self._detect_regime_sync, symbol, df)

    def _detect_regime_sync(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        mr_config = self.config.market_regime
        rsi_col = mr_config.rsi_col

        try:
            # 1. Ensure features exist
            if not {'regime_trend', 'regime_volatility', 'regime_efficiency', 'regime_hurst'}.issubset(df.columns):
                df = self.add_regime_features(df)

            # 2. Determine Dominant Regime (Hysteresis)
            window_size = mr_config.regime_confirmation_window
            if len(df) >= window_size:
                tail_df = df.iloc[-window_size:]
            else:
                tail_df = df
            
            regime_series = self.get_regime_series(tail_df)
            
            if not regime_series.empty:
                dominant_regime_str = regime_series.mode()[0]
                regime = MarketRegime(dominant_regime_str)
            else:
                regime = MarketRegime.SIDEWAYS

            # 3. Calculate Confidence Metrics based on the LATEST candle
            last_row = df.iloc[-1]
            trend_strength = last_row['regime_trend']
            efficiency = last_row['regime_efficiency']
            hurst = last_row['regime_hurst']
            rsi_val = last_row[rsi_col] if rsi_col in df.columns else 50

            # Determine Thresholds
            trend_thresh = mr_config.trend_strength_threshold
            vol_thresh = mr_config.volatility_multiplier
            eff_thresh = mr_config.efficiency_threshold

            if mr_config.use_dynamic_thresholds and len(df) >= mr_config.dynamic_window:
                window = df.iloc[-mr_config.dynamic_window:]
                if 'regime_trend' in window.columns:
                    trend_thresh = max(window['regime_trend'].abs().quantile(mr_config.trend_percentile), 0.001)
                if 'regime_volatility' in window.columns:
                    vol_thresh = window['regime_volatility'].quantile(mr_config.volatility_percentile)

            # Calculate Base Confidence
            trend_conf = min(1.0, abs(trend_strength) / (trend_thresh * 2)) if trend_thresh > 0 else 0.0
            rsi_conf = abs(rsi_val - 50) / 50
            
            hurst_boost = 0.0
            if hurst > mr_config.hurst_trending_threshold:
                hurst_boost = (hurst - mr_config.hurst_trending_threshold) * 2.0
            
            confidence = (trend_conf * 0.5) + (rsi_conf * 0.2) + (efficiency * 0.1) + (hurst_boost * 0.2)

            # 4. Apply Hysteresis Penalty
            immediate_regime_series = self.get_regime_series(df.iloc[-1:])
            if not immediate_regime_series.empty:
                immediate_regime_str = immediate_regime_series.iloc[0]
                if immediate_regime_str != regime.value:
                    confidence *= 0.8

            result = {
                'regime': regime.value, 
                'confidence': round(confidence, 2),
                'thresholds': {'trend': round(trend_thresh, 5), 'vol': round(vol_thresh, 3), 'eff': round(eff_thresh, 2), 'hurst': round(hurst, 2)},
                'enriched_df': df
            }
            return result
            
        except Exception as e:
            logger.error("Error in regime detection", symbol=symbol, error=str(e))
            return {'regime': MarketRegime.UNKNOWN.value, 'confidence': 0.0, 'enriched_df': df}

class RegimeTracker:
    """
    Background service that continuously updates market regime state.
    Allows strategies to query regime instantly without blocking.
    """
    def __init__(self, config: AIEnsembleStrategyParams, data_handler: DataHandler):
        self.config = config
        self.data_handler = data_handler
        self.detector = MarketRegimeDetector(config)
        self._current_regimes: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._update_loop(), name="RegimeTracker")
        logger.info("RegimeTracker service started.")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try: await self._task
            except asyncio.CancelledError: pass
        logger.info("RegimeTracker service stopped.")

    async def get_regime(self, symbol: str) -> Dict[str, Any]:
        async with self._lock:
            return self._current_regimes.get(symbol, {'regime': MarketRegime.UNKNOWN.value, 'confidence': 0.0})

    async def _update_loop(self):
        interval = self.config.market_regime.update_interval_seconds
        while self._running:
            try:
                for symbol in self.config.symbols:
                    df = self.data_handler.get_market_data(symbol)
                    if df is not None and not df.empty:
                        res = await self.detector.detect_regime(symbol, df)
                        async with self._lock:
                            self._current_regimes[symbol] = res
                
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in RegimeTracker loop", error=str(e))
                await asyncio.sleep(5)
