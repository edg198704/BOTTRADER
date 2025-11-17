import yaml
import os
import logging
from bot_core.config import BotConfig

logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Loads and validates configuration from a YAML file using Pydantic models.
    """
    def __init__(self, config_path: str):
        self.config_path = config_path

    def load_and_validate(self) -> BotConfig:
        """Loads the YAML configuration file and validates it against the BotConfig model."""
        if not os.path.exists(self.config_path):
            logger.error(f"Configuration file not found: {self.config_path}")
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            if not config_dict:
                raise ValueError("Configuration file is empty or invalid.")

            validated_config = BotConfig(**config_dict)
            logger.info(f"Configuration loaded and validated successfully from {self.config_path}")
            return validated_config
        
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration from {self.config_path}: {e}")
            raise ValueError(f"Invalid YAML configuration: {e}")
        except Exception as e:
            # This will catch Pydantic validation errors as well
            logger.error(f"An unexpected error occurred while loading or validating config: {e}")
            raise
