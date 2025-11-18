import yaml
from typing import Dict, Any
from pydantic import ValidationError
from dotenv import load_dotenv

from bot_core.config import BotConfig
from bot_core.logger import get_logger

logger = get_logger(__name__)

def load_config(path: str = 'config_enterprise.yaml') -> BotConfig:
    """
    Loads configuration from a YAML file and environment variables.
    Environment variables (and .env files) have precedence.
    """
    # Load .env file if it exists, for local development
    load_dotenv()

    config_data = _load_yaml_config(path)

    try:
        # Pydantic will automatically override with environment variables
        # where the 'env' attribute is set in the model fields.
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
