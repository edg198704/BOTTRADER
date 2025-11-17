import asyncio
import logging
import signal
import sys
from typing import Dict, Any

import yaml
from pydantic import ValidationError

from bot_core.config import BotConfig
from bot_core.exchange_api import CCXTExchangeAPI, MockExchangeAPI, ExchangeAPI
from bot_core.position_manager import PositionManager
from bot_core.risk_manager import RiskManager
from bot_core.strategy import TradingStrategy, SimpleMACrossoverStrategy, AIEnsembleStrategy
from bot_core.order_manager import OrderManager
from bot_core.execution_handler import ExecutionHandler
from bot_core.bot import TradingBot
from bot_core.telegram_bot import TelegramBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config_enterprise.yaml") -> BotConfig:
    """Loads and validates the bot configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return BotConfig(**config_dict)
    except FileNotFoundError:
        logger.critical(f"Configuration file not found at {config_path}")
        sys.exit(1)
    except ValidationError as e:
        logger.critical(f"Configuration validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Error loading configuration: {e}")
        sys.exit(1)

def get_exchange_api(config: BotConfig) -> ExchangeAPI:
    """Factory function for creating an exchange API instance."""
    if config.exchange.name == "MockExchange":
        return MockExchangeAPI()
    return CCXTExchangeAPI(
        name=config.exchange.name,
        api_key=config.exchange.api_key,
        api_secret=config.exchange.api_secret,
        testnet=config.exchange.testnet
    )

def get_strategy(config: BotConfig) -> TradingStrategy:
    """Factory function for creating a trading strategy instance."""
    strategy_map = {
        "SimpleMACrossoverStrategy": SimpleMACrossoverStrategy,
        "AIEnsembleStrategy": AIEnsembleStrategy
    }
    strategy_class = strategy_map.get(config.strategy.name)
    if not strategy_class:
        raise ValueError(f"Unknown strategy: {config.strategy.name}")
    return strategy_class(config.strategy.dict())

async def main():
    """Main function to initialize and run the bot and its components."""
    config = load_config()
    
    # Initialize components
    exchange_api = get_exchange_api(config)
    position_manager = PositionManager(db_path=config.database.path)
    risk_manager = RiskManager(config.risk_management, position_manager, config.initial_capital)
    strategy = get_strategy(config)
    order_manager = OrderManager(exchange_api)
    execution_handler = ExecutionHandler(order_manager, risk_manager)

    # Initialize the main bot
    trading_bot = TradingBot(
        config=config,
        exchange_api=exchange_api,
        strategy=strategy,
        position_manager=position_manager,
        risk_manager=risk_manager,
        execution_handler=execution_handler,
        order_manager=order_manager
    )

    # Initialize the Telegram bot
    telegram_bot = TelegramBot(config.telegram, trading_bot)

    # Graceful shutdown handler
    shutdown_event = asyncio.Event()
    def _signal_handler(*_):
        logger.info("Shutdown signal received. Initiating graceful shutdown...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Start tasks
    bot_task = asyncio.create_task(trading_bot.run())
    telegram_task = asyncio.create_task(telegram_bot.start())

    logger.info("Bot and all components are running. Press Ctrl+C to stop.")

    # Wait for shutdown signal
    await shutdown_event.wait()

    # Perform cleanup
    logger.info("Stopping all services...")
    await trading_bot.stop()
    await telegram_bot.stop()
    position_manager.close()
    await exchange_api.close()

    bot_task.cancel()
    telegram_task.cancel()
    await asyncio.gather(bot_task, telegram_task, return_exceptions=True)

    logger.info("Shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application stopped by user.")
