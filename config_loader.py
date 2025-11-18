import yaml
import os
from typing import Dict, Any
from pydantic import ValidationError

from bot_core.config import BotConfig
from bot_core.logger import get_logger

logger = get_logger(__name__)

def load_config(path: str = 'config_enterprise.yaml') -> BotConfig:
    """Loads configuration from YAML file and environment variables."""
    config_data = _load_yaml_config(path)
    config_data = _override_with_env_vars(config_data)

    try:
        config = BotConfig(**config_data)
        logger.info("Configuration loaded and validated successfully.")
        return config
    except ValidationError as e:
        logger.critical("Configuration validation failed!", errors=e.errors())
        raise

def _load_yaml_config(path: str) -> Dict[str, Any]:
    """Loads the base configuration from a YAML file."""
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.critical(f"Configuration file not found at '{path}'. Exiting.")
        raise
    except yaml.YAMLError as e:
        logger.critical(f"Error parsing YAML configuration file: {e}")
        raise

def _override_with_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """Overrides configuration with environment variables."""
    if 'BOT_EXCHANGE_API_KEY' in os.environ:
        config['exchange']['api_key'] = os.environ['BOT_EXCHANGE_API_KEY']
    if 'BOT_EXCHANGE_API_SECRET' in os.environ:
        config['exchange']['api_secret'] = os.environ['BOT_EXCHANGE_API_SECRET']
    if 'BOT_TELEGRAM_BOT_TOKEN' in os.environ:
        config['telegram']['bot_token'] = os.environ['BOT_TELEGRAM_BOT_TOKEN']
    
    # Example for nested value override
    if 'BOT_STRATEGY_SYMBOL' in os.environ:
        config['strategy']['symbol'] = os.environ['BOT_STRATEGY_SYMBOL']

    logger.info("Configuration updated with environment variables.")
    return config
