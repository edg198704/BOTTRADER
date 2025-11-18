import asyncio
import functools
from typing import Tuple, Type

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
