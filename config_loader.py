import yaml
import os
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            logger.error(f"Configuration file not found: {self.config_path}")
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            self._validate_config(config_data)
            return config_data
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise ValueError(f"Invalid YAML configuration: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading config: {e}")
            raise

    def _validate_config(self, config_data):
        required_sections = ['api', 'market_data', 'strategy', 'risk_management']
        for section in required_sections:
            if section not in config_data:
                raise ValueError(f"Missing required configuration section: '{section}'")

        # Basic validation for API keys
        if 'api' in config_data:
            if not all(k in config_data['api'] for k in ['api_key', 'api_secret', 'base_url']):
                raise ValueError("Missing required API parameters (api_key, api_secret, base_url) in 'api' section.")
            # Placeholder for environment variable check - recommended for production
            # if not os.getenv('API_KEY') and not config_data['api'].get('api_key'):
            #     raise ValueError("API_KEY not found in config or environment variables.")

        # Basic validation for strategy parameters
        if 'strategy' in config_data:
            if not all(k in config_data['strategy'] for k in ['symbol', 'interval', 'quantity']):
                raise ValueError("Missing required strategy parameters (symbol, interval, quantity) in 'strategy' section.")

        # Basic validation for risk management parameters
        if 'risk_management' in config_data:
            if not all(k in config_data['risk_management'] for k in ['max_position_size', 'stop_loss_percent', 'take_profit_percent']):
                raise ValueError("Missing required risk management parameters (max_position_size, stop_loss_percent, take_profit_percent) in 'risk_management' section.")
            if not isinstance(config_data['risk_management']['max_position_size'], (int, float)) or config_data['risk_management']['max_position_size'] <= 0:
                raise ValueError("max_position_size must be a positive number.")
            if not isinstance(config_data['risk_management']['stop_loss_percent'], (int, float)) or not (0 < config_data['risk_management']['stop_loss_percent'] < 1):
                raise ValueError("stop_loss_percent must be a float between 0 and 1.")
            if not isinstance(config_data['risk_management']['take_profit_percent'], (int, float)) or not (0 < config_data['risk_management']['take_profit_percent'] < 1):
                raise ValueError("take_profit_percent must be a float between 0 and 1.")


    def get(self, key, default=None):
        keys = key.split('.')
        val = self.config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def __getitem__(self, key):
        return self.get(key)

    def __str__(self):
        return str(self.config)
