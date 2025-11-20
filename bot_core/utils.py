import asyncio
import functools
import logging
from datetime import datetime, timezone
from typing import Callable, Type, Tuple, Dict, Any, List, Optional

# Use a module-level logger
logger = logging.getLogger(__name__)

class Clock:
    """
    A centralized clock to handle time. 
    Allows mocking time for backtesting without changing business logic.
    """
    _mock_time: Optional[datetime] = None

    @classmethod
    def now(cls) -> datetime:
        """Returns the current time in UTC. If mock time is set, returns that."""
        if cls._mock_time:
            return cls._mock_time
        return datetime.now(timezone.utc)

    @classmethod
    def set_time(cls, t: datetime):
        """Sets a mock time for backtesting."""
        if t.tzinfo is None:
            # Assume UTC if naive
            t = t.replace(tzinfo=timezone.utc)
        cls._mock_time = t

    @classmethod
    def reset(cls):
        """Resets the clock to system time."""
        cls._mock_time = None

def async_retry(max_attempts: int = 3, delay_seconds: float = 1.0, backoff_factor: float = 2.0, exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """
    Decorator for retrying async functions with exponential backoff.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay_seconds
            last_exception = None

            while attempt < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    last_exception = e
                    if attempt >= max_attempts:
                        break
                    
                    logger.warning(f"Retrying {func.__name__} due to {e.__class__.__name__}: {e}. Attempt {attempt}/{max_attempts}. Waiting {current_delay}s.")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor
            
            logger.error(f"Function {func.__name__} failed after {max_attempts} attempts.")
            raise last_exception
        return wrapper
    return decorator

def parse_timeframe_to_seconds(timeframe: str) -> int:
    """
    Converts a timeframe string (e.g., '1m', '5m', '1h', '1d') to seconds.
    """
    if not timeframe:
        return 0
        
    unit = timeframe[-1].lower()
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
    
    return value * multipliers.get(unit, 0)

def calculate_min_history_depth(indicators_config: List[Dict[str, Any]]) -> int:
    """
    Calculates the minimum number of historical candles required to calculate all configured indicators.
    """
    max_lookback = 0
    
    for ind in indicators_config:
        # Check common lookback parameters
        lookback = 0
        if 'length' in ind:
            lookback = int(ind['length'])
        elif 'slow' in ind: # MACD
            lookback = int(ind['slow'])
        elif 'timeperiod' in ind:
            lookback = int(ind['timeperiod'])
            
        # Add a safety buffer (e.g. for EMA convergence)
        if ind.get('kind') in ['ema', 'rsi', 'adx', 'atr']:
            lookback = lookback * 3
            
        if lookback > max_lookback:
            max_lookback = lookback
            
    # Default minimum if no indicators or very short ones
    return max(max_lookback + 50, 200)

def generate_indicator_rename_map(indicators_config: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Generates a mapping from pandas_ta default column names to the aliases defined in the config.
    This allows the strategy to refer to 'rsi' instead of 'RSI_14'.
    """
    rename_map = {}
    
    for ind in indicators_config:
        kind = ind.get('kind')
        alias = ind.get('alias')
        
        if not kind or not alias:
            continue
            
        # Predict pandas_ta column name
        # This is a heuristic based on pandas_ta conventions
        
        predicted_names = []
        length = ind.get('length')
        
        if kind == 'rsi':
            length = length or 14
            predicted_names.append(f"RSI_{length}")
            
        elif kind == 'sma':
            length = length or 10
            predicted_names.append(f"SMA_{length}")
            
        elif kind == 'ema':
            length = length or 10
            predicted_names.append(f"EMA_{length}")
            
        elif kind == 'atr':
            length = length or 14
            # pandas_ta can output ATRr_ or ATR_
            predicted_names.append(f"ATRr_{length}")
            predicted_names.append(f"ATR_{length}")
            
        elif kind == 'log_return':
            length = length or 1
            predicted_names.append(f"LOGRET_{length}")

        # If we predicted names and have an alias, add to map
        for name in predicted_names:
            rename_map[name] = alias

    return rename_map
