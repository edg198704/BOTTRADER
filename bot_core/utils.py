import asyncio
import logging
import functools
from datetime import datetime, timezone
from typing import Callable, Dict, List, Any

logger = logging.getLogger(__name__)

class Clock:
    """
    Centralized clock for time abstraction.
    Allows backtesting to override 'now' without patching datetime.now().
    """
    _mock_time = None

    @classmethod
    def now(cls) -> datetime:
        if cls._mock_time:
            return cls._mock_time
        return datetime.now(timezone.utc)

    @classmethod
    def set_time(cls, dt: datetime):
        cls._mock_time = dt

    @classmethod
    def reset(cls):
        cls._mock_time = None

def async_retry(max_attempts: int = 3, delay_seconds: float = 1.0, backoff_factor: float = 2.0, exceptions: tuple = (Exception,)):
    """Decorator for retrying async functions with exponential backoff."""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay_seconds
            while attempt < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise
                    logger.warning(f"Retrying {func.__name__} due to {e}. Attempt {attempt}/{max_attempts}")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor
        return wrapper
    return decorator

def parse_timeframe_to_seconds(timeframe: str) -> int:
    """Converts a timeframe string (e.g., '5m', '1h') to seconds."""
    unit = timeframe[-1]
    if unit not in ['s', 'm', 'h', 'd', 'w']:
        return 0
    try:
        value = int(timeframe[:-1])
    except ValueError:
        return 0
        
    multipliers = {
        's': 1,
        'm': 60,
        'h': 3600,
        'd': 86400,
        'w': 604800
    }
    return value * multipliers[unit]

def calculate_min_history_depth(indicators: List[Dict[str, Any]]) -> int:
    """Calculates the minimum historical data needed based on indicator lengths."""
    max_depth = 0
    for ind in indicators:
        kind = ind.get('kind', '').lower()
        depth = 0
        
        if 'length' in ind:
            depth = int(ind['length'])
        elif kind == 'macd':
            slow = ind.get('slow', 26)
            signal = ind.get('signal', 9)
            depth = slow + signal
        
        # Add buffer for convergence (EMA, RSI, etc. need more history)
        if kind in ['ema', 'rsi', 'adx', 'atr']:
            depth = depth * 3
        
        if depth > max_depth:
            max_depth = depth
            
    return max(max_depth, 100) # Return at least 100 candles

def generate_indicator_rename_map(indicators: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Generates a mapping from pandas-ta default column names to configured aliases.
    Handles specific logic for multi-column indicators like MACD and BBANDS to match
    the feature columns expected by the AI strategy.
    """
    rename_map = {}
    for ind in indicators:
        kind = ind.get('kind', '').lower()
        alias = ind.get('alias')
        
        # Helper to format pandas-ta default names
        length = ind.get('length')
        
        if kind == 'rsi':
            l = length if length else 14
            rename_map[f"RSI_{l}"] = alias if alias else "rsi"
            
        elif kind == 'sma':
            l = length if length else 10
            rename_map[f"SMA_{l}"] = alias if alias else f"sma_{l}"
            
        elif kind == 'ema':
            l = length if length else 10
            rename_map[f"EMA_{l}"] = alias if alias else f"ema_{l}"
            
        elif kind == 'atr':
            l = length if length else 14
            # pandas-ta often uses ATRr or ATR depending on version/config
            rename_map[f"ATRr_{l}"] = alias if alias else "atr"
            
        elif kind == 'adx':
            l = length if length else 14
            # ADX produces ADX, DMP, DMN. We usually just want ADX line.
            rename_map[f"ADX_{l}"] = alias if alias else "adx"
            
        elif kind == 'log_return':
            l = length if length else 1
            rename_map[f"LOGRET_{l}"] = alias if alias else "log_return"
            
        elif kind == 'macd':
            fast = ind.get('fast', 12)
            slow = ind.get('slow', 26)
            signal = ind.get('signal', 9)
            # Default pandas-ta output: MACD_12_26_9
            col_name = f"MACD_{fast}_{slow}_{signal}"
            rename_map[col_name] = alias if alias else "macd"
            
        elif kind == 'bbands':
            l = length if length else 20
            std = ind.get('std', 2.0)
            # Default pandas-ta outputs: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
            # We map them to standard names expected by config if no alias provided
            rename_map[f"BBU_{l}_{std}"] = "bb_upper"
            rename_map[f"BBL_{l}_{std}"] = "bb_lower"
            rename_map[f"BBM_{l}_{std}"] = "bb_mid"
            
    return rename_map
