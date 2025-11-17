import asyncio
import logging
import signal
from typing import Type

from bot_core.bot import TradingBot
from bot_core.config import BotConfig
from bot_core.exchange_api import ExchangeAPI, MockExchangeAPI, CCXTExchangeAPI
from bot_core.position_manager import PositionManager
from bot_core.risk_manager import RiskManager
from bot_core.strategy import TradingStrategy, SimpleMACrossoverStrategy, AIEnsembleStrategy
from config_loader import ConfigLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global bot instance to be managed by signal handlers
bot_instance: TradingBot = None

def get_exchange_api(config: BotConfig) -> ExchangeAPI:
    """Factory function to get an exchange API instance based on config."""
    exchange_config = config.exchange
    if exchange_config.name == "MockExchange":
        return MockExchangeAPI()
    else:
        return CCXTExchangeAPI(
            name=exchange_config.name,
            api_key=exchange_config.api_key,
            api_secret=exchange_config.api_secret,
            testnet=exchange_config.testnet
        )

def get_strategy(config: BotConfig) -> TradingStrategy:
    """Factory function to get a trading strategy instance based on config."""
    strategy_config = config.strategy
    strategy_map = {
        "SimpleMACrossoverStrategy": SimpleMACrossoverStrategy,
        "AIEnsembleStrategy": AIEnsembleStrategy
    }
    strategy_class = strategy_map.get(strategy_config.name)
    if not strategy_class:
        raise ValueError(f"Unknown strategy: {strategy_config.name}")
    return strategy_class(strategy_config.dict())

async def async_main():
    """Asynchronous main function to set up and run the bot."""
    global bot_instance
    try:
        # 1. Load Configuration
        config_loader = ConfigLoader(config_path='config_enterprise.yaml')
        config = config_loader.load_and_validate()

        # 2. Initialize Components
        exchange_api = get_exchange_api(config)
        strategy = get_strategy(config)
        position_manager = PositionManager(db_path=config.database.path)
        
        # RiskManager needs initial capital and access to the position manager
        initial_capital = 10000.0 # This should ideally come from the exchange or config
        risk_manager = RiskManager(
            config=config,
            position_manager=position_manager,
            initial_capital=initial_capital
        )

        # 3. Instantiate the Bot
        bot_instance = TradingBot(
            config=config,
            exchange_api=exchange_api,
            strategy=strategy,
            position_manager=position_manager,
            risk_manager=risk_manager
        )

        # 4. Run the Bot
        await bot_instance.run()

    except Exception as e:
        logger.critical(f"Failed to initialize or run the bot: {e}", exc_info=True)
    finally:
        if bot_instance and bot_instance.exchange_api:
            await bot_instance.exchange_api.close()
            logger.info("Exchange API connection closed.")

def shutdown_handler(signum, frame):
    logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
    if bot_instance:
        bot_instance.stop()

if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        asyncio.run(async_main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot shutdown sequence initiated.")
    
    logger.info("Bot has been shut down.")
