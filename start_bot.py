import asyncio
import logging
import signal
from typing import Dict, Any

from config_loader import ConfigLoader
from bot_core.config import BotConfig
from bot_core.exchange_api import ExchangeAPI, MockExchangeAPI, CCXTExchangeAPI
from bot_core.position_manager import PositionManager
from bot_core.risk_manager import RiskManager
from bot_core.strategy import TradingStrategy, SimpleMACrossoverStrategy, AIEnsembleStrategy
from bot_core.bot import TradingBot

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Factory Functions ---

def get_exchange_api(config: BotConfig) -> ExchangeAPI:
    """Factory function to create an exchange API instance."""
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
    """Factory function to create a trading strategy instance."""
    strategy_config = config.strategy
    strategy_map = {
        "SimpleMACrossoverStrategy": SimpleMACrossoverStrategy,
        "AIEnsembleStrategy": AIEnsembleStrategy
    }
    strategy_class = strategy_map.get(strategy_config.name)
    if not strategy_class:
        raise ValueError(f"Unknown strategy: {strategy_config.name}")
    return strategy_class(strategy_config.dict())

# --- Main Application ---

class BotRunner:
    def __init__(self, config: BotConfig):
        self.config = config
        self.bot: TradingBot = None
        self.tasks: list[asyncio.Task] = []

    async def start(self):
        """Initializes and starts the trading bot and its components."""
        try:
            # 1. Initialize Components
            exchange_api = get_exchange_api(self.config)
            strategy = get_strategy(self.config)
            position_manager = PositionManager(self.config.database.path)
            
            # Determine initial capital for RiskManager
            # In a real scenario, this would be fetched from the exchange or a starting config value.
            initial_capital = 10000.0 # Example starting capital
            risk_manager = RiskManager(self.config, position_manager, initial_capital)

            # 2. Create the Bot
            self.bot = TradingBot(
                config=self.config,
                exchange_api=exchange_api,
                strategy=strategy,
                position_manager=position_manager,
                risk_manager=risk_manager
            )

            # 3. Start the bot's main loop as a task
            self.tasks.append(asyncio.create_task(self.bot.run()))
            logger.info("Bot has started successfully.")
            await asyncio.gather(*self.tasks)

        except Exception as e:
            logger.critical(f"Failed to start bot runner: {e}", exc_info=True)
        finally:
            await self.shutdown()

    async def shutdown(self, signame: str = 'SHUTDOWN'):
        """Handles graceful shutdown of the bot and its components."""
        logger.info(f"Shutdown initiated by {signame}...")
        if self.bot:
            self.bot.stop()
        
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Allow tasks to finish cancellation
        await asyncio.gather(*self.tasks, return_exceptions=True)

        if self.bot and self.bot.exchange_api:
            await self.bot.exchange_api.close()
        
        logger.info("Shutdown complete.")

async def main():
    """Main entry point for the application."""
    try:
        config_loader = ConfigLoader('config_enterprise.yaml')
        config = config_loader.load_and_validate()
    except (FileNotFoundError, ValueError) as e:
        logger.critical(f"Configuration error: {e}")
        return

    runner = BotRunner(config)

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    for signame in {'SIGINT', 'SIGTERM'}:
        loop.add_signal_handler(
            getattr(signal, signame),
            lambda signame=signame: asyncio.create_task(runner.shutdown(signame))
        )

    await runner.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Application terminated.")
