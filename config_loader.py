import yaml
import os
from bot_core.config import BotConfig
from bot_core.logger import get_logger

logger = get_logger(__name__)

def load_config(path: str = 'config_enterprise.yaml') -> BotConfig:
    """
    Loads configuration from a YAML file, merges with environment variables,
    and validates it using the BotConfig Pydantic model.
    """
    if not os.path.exists(path):
        logger.critical("Configuration file not found", path=path)
        raise FileNotFoundError(f"Configuration file not found at {path}")

    with open(path, 'r') as f:
        config_data = yaml.safe_load(f)

    # Allow environment variables to override YAML values
    # Example: BOT_EXCHANGE_API_KEY=your_key python start_bot.py
    config_data['exchange']['api_key'] = os.getenv('BOT_EXCHANGE_API_KEY', config_data.get('exchange', {}).get('api_key'))
    config_data['exchange']['api_secret'] = os.getenv('BOT_EXCHANGE_API_SECRET', config_data.get('exchange', {}).get('api_secret'))
    config_data['telegram']['bot_token'] = os.getenv('BOT_TELEGRAM_BOT_TOKEN', config_data.get('telegram', {}).get('bot_token'))

    try:
        config = BotConfig(**config_data)
        logger.info("Configuration loaded and validated successfully.")
        return config
    except Exception as e:
        logger.critical("Configuration validation failed", error=str(e))
        raise ValueError(f"Configuration validation failed: {e}")
