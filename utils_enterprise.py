import logging
import asyncio
import functools
import time

def setup_logging(config):
    """
    Sets up logging for the application based on the provided configuration.
    """
    log_level_str = config.get('logging.level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    log_file = config.get('logging.file', 'bot_enterprise.log')
    max_bytes = config.get('logging.max_bytes', 10 * 1024 * 1024) # Default 10 MB
    backup_count = config.get('logging.backup_count', 5)

    # Clear existing handlers to prevent duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(), # Console output
            logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
        ]
    )
    logging.getLogger('aiohttp').setLevel(logging.WARNING) # Suppress verbose aiohttp logs
    logging.getLogger('asyncio').setLevel(logging.WARNING) # Suppress verbose asyncio logs
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized with level: {log_level_str}, file: {log_file}")


def async_retry(max_attempts=3, delay=1, backoff=2, exceptions=(Exception,)): 
    """
    A decorator for retrying asynchronous functions.
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            _delay = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    logger = logging.getLogger(func.__module__)
                    logger.warning(f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}")
                    if attempt < max_attempts:
                        logger.info(f"Retrying {func.__name__} in {_delay} seconds...")
                        await asyncio.sleep(_delay)
                        _delay *= backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}.")
                        raise
        return wrapper
    return decorator
