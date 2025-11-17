import yaml
import os
import re
from typing import Any, Dict

from bot_core.config import BotConfig
from bot_core.logger import get_logger

logger = get_logger(__name__)

# Regex to find ${VAR_NAME:default_value}
ENV_VAR_MATCHER = re.compile(r"\$\{([^}]+)}")

def _replace_env_vars(config_part: Any) -> Any:
    """Recursively replaces environment variable placeholders in config values."""
    if isinstance(config_part, dict):
        return {k: _replace_env_vars(v) for k, v in config_part.items()}
    if isinstance(config_part, list):
        return [_replace_env_vars(i) for i in config_part]
    if isinstance(config_part, str):
        match = ENV_VAR_MATCHER.match(config_part)
        if match:
            var_name, _, default_val = match.group(1).partition(':')
            return os.environ.get(var_name, default_val)
    return config_part

def load_config(path: str = "config_enterprise.yaml") -> BotConfig:
    """
    Loads, validates, and returns the bot's configuration.

    Args:
        path: The path to the YAML configuration file.

    Returns:
        A validated BotConfig object.
    """
    logger.info("Loading configuration...", path=path)
    try:
        with open(path, 'r') as f:
            raw_config = yaml.safe_load(f)

        # Substitute environment variables
        processed_config = _replace_env_vars(raw_config)

        # Validate with Pydantic
        config = BotConfig(**processed_config)
        logger.info("Configuration loaded and validated successfully.")
        return config

    except FileNotFoundError:
        logger.critical("Configuration file not found.", path=path)
        raise
    except (yaml.YAMLError, ValueError) as e:
        logger.critical("Error parsing configuration file.", error=str(e))
        raise
    except Exception as e:
        logger.critical("Failed to load configuration.", error=str(e))
        raise
