#!/usr/bin/env python3
"""
Enterprise AI Trading Bot - Configuration Loader

Loads and validates configuration from YAML files and environment variables.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """Load and validate configuration from YAML and environment variables"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config_enterprise.yaml"
        self.logger = logging.getLogger(__name__)
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file and environment variables"""
        config = {}
        
        # Load YAML config
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f)
                self.logger.info(f"Configuration loaded from {self.config_file}")
            except Exception as e:
                self.logger.error(f"Failed to load config file: {e}")
                config = {}
        else:
            self.logger.warning(f"Config file not found: {self.config_file}")
        
        # Override with environment variables
        config = self._apply_env_overrides(config)
        
        return config
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        env_mappings = {
            # Exchange
            'EXCHANGE': ['exchange', 'name'],
            'EXCHANGE_API_KEY': ['exchange', 'api_key'],
            'EXCHANGE_API_SECRET': ['exchange', 'api_secret'],
            'DRY_RUN': ['exchange', 'dry_run'],
            
            # Trading
            'SYMBOLS': ['trading', 'symbols'],
            'TIMEFRAME': ['trading', 'timeframe'],
            'INITIAL_CAPITAL': ['trading', 'initial_capital'],
            'MAX_POSITION_SIZE': ['trading', 'max_position_size'],
            'STOP_LOSS_PCT': ['trading', 'stop_loss_pct'],
            'TAKE_PROFIT_PCT': ['trading', 'take_profit_pct'],
            
            # AI/ML
            'USE_ENSEMBLE': ['ai_ml', 'use_ensemble'],
            'TRAINING_SYMBOLS_LIMIT': ['ai_ml', 'training_symbols_limit'],
            
            # Monitoring
            'LOG_LEVEL': ['system', 'log_level'],
            'ENABLE_DEBUG_LOGGING': ['system', 'enable_debug_logging'],
            'MAX_MEMORY_MB': ['system', 'max_memory_mb'],
            
            # InfluxDB
            'INFLUXDB_URL': ['monitoring', 'influxdb', 'url'],
            'INFLUXDB_TOKEN': ['monitoring', 'influxdb', 'token'],
            'INFLUXDB_ORG': ['monitoring', 'influxdb', 'org'],
            'INFLUXDB_BUCKET': ['monitoring', 'influxdb', 'bucket'],
            
            # Telegram
            'TELEGRAM_BOT_TOKEN': ['monitoring', 'telegram', 'bot_token'],
            'TELEGRAM_ADMIN_IDS': ['monitoring', 'telegram', 'admin_chat_ids'],
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                self._set_nested_value(config, config_path, self._convert_env_value(env_value))
                self.logger.debug(f"Environment override applied: {env_var} -> {'.'.join(config_path)}")
        
        return config
    
    def _set_nested_value(self, config: Dict[str, Any], path: list, value: Any):
        """Set nested dictionary value using path"""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type"""
        # Boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer
        try:
            if '.' not in value and not value.startswith('0') or value == '0':
                return int(value)
        except ValueError:
            pass
        
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        
        # List (comma-separated)
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        # String
        return value
    
    def save_template(self, output_file: str = "config_template.yaml"):
        """Save configuration template"""
        template = {
            'exchange': {
                'name': os.getenv('EXCHANGE', 'binance'),
                'sandbox': False,
                'dry_run': os.getenv('DRY_RUN', 'true').lower() == 'true',
                'api_key': '${EXCHANGE_API_KEY}',
                'api_secret': '${EXCHANGE_API_SECRET}'
            },
            'trading': {
                'symbols': os.getenv('SYMBOLS', 'BTC/USDT,ETH/USDT').split(','),
                'timeframe': os.getenv('TIMEFRAME', '1h'),
                'initial_capital': float(os.getenv('INITIAL_CAPITAL', '10000')),
                'max_position_size': float(os.getenv('MAX_POSITION_SIZE', '0.1')),
                'stop_loss_pct': float(os.getenv('STOP_LOSS_PCT', '0.02')),
                'take_profit_pct': float(os.getenv('TAKE_PROFIT_PCT', '0.04'))
            },
            'ai_ml': {
                'use_ensemble': os.getenv('USE_ENSEMBLE', 'true').lower() == 'true',
                'training_symbols_limit': int(os.getenv('TRAINING_SYMBOLS_LIMIT', '50'))
            },
            'monitoring': {
                'log_level': os.getenv('LOG_LEVEL', 'INFO'),
                'influxdb': {
                    'enabled': os.getenv('INFLUXDB_ENABLED', 'true').lower() == 'true',
                    'url': os.getenv('INFLUXDB_URL', 'http://localhost:8086'),
                    'token': '${INFLUXDB_TOKEN}',
                    'org': os.getenv('INFLUXDB_ORG', 'trading_bot'),
                    'bucket': os.getenv('INFLUXDB_BUCKET', 'trading_metrics')
                },
                'telegram': {
                    'enabled': os.getenv('TELEGRAM_ENABLED', 'true').lower() == 'true',
                    'bot_token': '${TELEGRAM_BOT_TOKEN}',
                    'admin_chat_ids': [int(x.strip()) for x in os.getenv('TELEGRAM_ADMIN_IDS', '').split(',') if x.strip()]
                }
            },
            'system': {
                'log_level': os.getenv('LOG_LEVEL', 'INFO'),
                'max_memory_mb': int(os.getenv('MAX_MEMORY_MB', '2000')),
                'enable_debug_logging': os.getenv('ENABLE_DEBUG_LOGGING', 'false').lower() == 'true'
            }
        }
        
        with open(output_file, 'w') as f:
            yaml.dump(template, f, default_flow_style=False, indent=2)
        
        print(f"Configuration template saved to {output_file}")


def main():
    """CLI interface for config loader"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enterprise Trading Bot Config Loader')
    parser.add_argument('--config', '-c', help='Config file path', default='config_enterprise.yaml')
    parser.add_argument('--template', '-t', help='Generate config template', action='store_true')
    parser.add_argument('--output', '-o', help='Template output file', default='config_template.yaml')
    parser.add_argument('--validate', '-v', help='Validate configuration', action='store_true')
    
    args = parser.parse_args()
    
    if args.template:
        ConfigLoader().save_template(args.output)
    elif args.validate:
        loader = ConfigLoader(args.config)
        config = loader.load_config()
        print("Configuration loaded successfully:")
        print(yaml.dump(config, default_flow_style=False, indent=2))
    else:
        loader = ConfigLoader(args.config)
        config = loader.load_config()
        print("Configuration summary:")
        print(f"  Exchange: {config.get('exchange', {}).get('name', 'Not configured')}")
        print(f"  Symbols: {config.get('trading', {}).get('symbols', [])}")
        print(f"  Initial Capital: ${config.get('trading', {}).get('initial_capital', 0):,.2f}")
        print(f"  AI/ML Enabled: {config.get('ai_ml', {}).get('use_ensemble', False)}")
        print(f"  Monitoring Enabled: {config.get('monitoring', {}).get('enable_monitoring', False)}")


if __name__ == "__main__":
    main()