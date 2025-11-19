import asyncio
import functools
from typing import Tuple, Type, List, Dict, Any

from bot_core.logger import get_logger

# Note: The logger is fetched at the module level.
# When used inside the decorator, it will correctly reference the decorated function's module.
logger = get_logger(__name__)

def async_retry(
    max_attempts: int = 3,
    delay_seconds: int = 1,
    backoff_factor: int = 2,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    A decorator for retrying asynchronous functions with exponential backoff.

    Args:
        max_attempts: The maximum number of attempts.
        delay_seconds: The initial delay between retries in seconds.
        backoff_factor: The factor by which the delay is multiplied after each retry.
        exceptions: A tuple of exception types to catch and trigger a retry.
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            _delay = delay_seconds
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    # Use the logger from the module where the decorated function is defined
                    func_logger = get_logger(func.__module__)
                    func_logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}",
                        error=str(e),
                        attempt=attempt,
                        max_attempts=max_attempts,
                        func_name=func.__name__
                    )
                    if attempt < max_attempts:
                        func_logger.info(
                            f"Retrying {func.__name__} in {_delay} seconds...",
                            delay=_delay,
                            func_name=func.__name__
                        )
                        await asyncio.sleep(_delay)
                        _delay *= backoff_factor
                    else:
                        func_logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}.",
                            func_name=func.__name__
                        )
                        raise
        return wrapper
    return decorator

def generate_indicator_rename_map(indicators_config: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Generates a mapping from verbose pandas-ta column names to simplified, consistent names.
    This function is the single source of truth for renaming indicators, ensuring consistency
    between startup validation and runtime data processing.
    """
    rename_map = {}
    for conf in indicators_config:
        kind = conf.get("kind")
        if not kind:
            continue

        alias = conf.get("alias")

        if kind == "rsi":
            length = conf.get("length", 14)
            rename_map[f"RSI_{length}"] = alias or "rsi"
        elif kind == "macd":
            fast = conf.get("fast", 12)
            slow = conf.get("slow", 26)
            signal = conf.get("signal", 9)
            base = f"MACD_{fast}_{slow}_{signal}"
            base_alias = alias or "macd"
            rename_map[base] = base_alias
            rename_map[f"{base}h"] = f"{base_alias}_hist"
            rename_map[f"{base}s"] = f"{base_alias}_signal"
        elif kind == "bbands":
            length = conf.get("length", 20)
            std = float(conf.get("std", 2.0))
            base = f"_{length}_{std:.1f}"
            base_alias = alias or "bb"
            rename_map[f"BBL{base}"] = f"{base_alias}_lower"
            rename_map[f"BBM{base}"] = f"{base_alias}_middle"
            rename_map[f"BBU{base}"] = f"{base_alias}_upper"
        elif kind == "atr":
            length = conf.get("length", 14)
            rename_map[f"ATRr_{length}"] = alias or "atr"
        elif kind == "adx":
            length = conf.get("length", 14)
            rename_map[f"ADX_{length}"] = alias or "adx"
        elif kind == "sma":
            if not alias:
                raise ValueError(f"Indicator 'sma' with length {conf.get('length')} must have an 'alias' in the configuration to avoid ambiguity.")
            length = conf.get("length")
            rename_map[f"SMA_{length}"] = alias
        else:
            # Handle generic case with alias
            if alias:
                params = "_".join(str(v) for k, v in conf.items() if k not in ['kind', 'alias'])
                ta_name = f"{kind.upper()}_{params}"
                rename_map[ta_name] = alias

    return rename_map

def parse_timeframe_to_seconds(timeframe: str) -> int:
    """
    Parses a timeframe string (e.g., '1m', '1h', '1d') into seconds.
    """
    if not timeframe:
        return 60
    
    unit = timeframe[-1]
    try:
        value = int(timeframe[:-1])
    except ValueError:
        return 60

    multipliers = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800}
    return value * multipliers.get(unit, 60)
