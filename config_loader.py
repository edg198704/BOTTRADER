import yaml
import logging
from pydantic import ValidationError
from bot_core.config import BotConfig

logger = logging.getLogger(__name__)

def load_config(path: str = 'config_enterprise.yaml') -> BotConfig:
    """Loads, validates, and returns the bot configuration."""
    try:
        with open(path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        if not config_data:
            raise ValueError("Configuration file is empty or invalid.")

        config = BotConfig(**config_data)
        logger.info(f"Configuration loaded and validated successfully from {path}")
        return config

    except FileNotFoundError:
        logger.critical(f"Configuration file not found at '{path}'. Please create it.")
        raise
    except yaml.YAMLError as e:
        logger.critical(f"Error parsing YAML configuration file: {e}")
        raise
    except ValidationError as e:
        logger.critical(f"Configuration validation error: \n{e}")
        raise
    except Exception as e:
        logger.critical(f"An unexpected error occurred while loading the configuration: {e}")
        raise
