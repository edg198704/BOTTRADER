# config_loader.py
import yaml
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Loads configuration from a YAML file.
    Provides methods to access configuration values with defaults and validation.
    """
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self):
        """Loads the YAML configuration file."""
        if not os.path.exists(self.config_path):
            logger.error(f"Configuration file not found: {self.config_path}")
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Configuration loaded successfully from {self.config_path}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration from {self.config_path}: {e}")
            raise ValueError(f"Invalid YAML configuration: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading config from {self.config_path}: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value.
        Supports dot notation for nested keys (e.g., "exchange.api_key").
        """
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            if default is not None:
                logger.debug(f"Configuration key '{key}' not found, using default value.")
                return default
            logger.error(f"Configuration key '{key}' not found and no default value provided.")
            raise KeyError(f"Missing configuration key: {key}")

    def get_section(self, section: str) -> Dict[str, Any]:
        """Retrieves an entire section of the configuration."""
        section_data = self.config.get(section)
        if section_data is None:
            logger.warning(f"Configuration section '{section}' not found. Returning empty dictionary.")
            return {}
        if not isinstance(section_data, dict):
            logger.error(f"Configuration section '{section}' is not a dictionary.")
            raise TypeError(f"Configuration section '{section}' must be a dictionary.")
        return section_data

    def reload_config(self):
        """Reloads the configuration from the file."""
        logger.info(f"Reloading configuration from {self.config_path}")
        self._load_config()
