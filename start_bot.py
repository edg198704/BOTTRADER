import asyncio
import logging
import signal
import sys

from config_loader import ConfigLoader
from bot_core.config import BotConfig, StrategyConfig
from bot_core.exchange_api import ExchangeAPI, MockExchangeAPI, CCXTExchangeAPI
from bot_core.position_manager import PositionManager
from bot_core.risk_manager import RiskManager
from bot_core.strategy import TradingStrategy, SimpleMACrossoverStrategy, AIEnsembleStrategy
from bot_core.bot import TradingBot
from bot_core.execution_handler import ExecutionHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Factory Functions ---

def get_exchange_api(config: BotConfig) -> ExchangeAPI:
    """Factory function to create an exchange API instance based on config."""
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

def get_strategy(config: StrategyConfig) -> TradingStrategy:
    """Factory function to create a trading strategy instance based on config."""
    strategy_map = {
        "SimpleMACrossoverStrategy": SimpleMACrossoverStrategy,
        "AIEnsembleStrategy": AIEnsembleStrategy
    }
    strategy_class = strategy_map.get(config.name)
    if not strategy_class:
        raise ValueError(f"Unknown strategy: {config.name}")
    return strategy_class(config.dict())

# --- Main Application Logic ---

class BotApplication:
    def __init__(self, config: BotConfig):
        self.config = config
        self.bot: Optional[TradingBot] = None
        self.tasks: List[asyncio.Task] = []

    async def start(self):
        """Initializes and starts the trading bot and its components."""
        try:
            # 1. Initialize Components
            exchange_api = get_exchange_api(self.config)
            strategy = get_strategy(self.config.strategy)
            position_manager = PositionManager(self.config.database.path)
            risk_manager = RiskManager(self.config.risk_management, position_manager, self.config.initial_capital)
            execution_handler = ExecutionHandler(exchange_api, position_manager, risk_manager)

            # 2. Initialize the main TradingBot
            self.bot = TradingBot(
                config=self.config,
                exchange_api=exchange_api,
                strategy=strategy,
                position_manager=position_manager,
                risk_manager=risk_manager,
                execution_handler=execution_handler
            )

            # 3. Start the bot's main loop as a task
            self.tasks.append(asyncio.create_task(self.bot.run()))
            logger.info("Bot application started. Press Ctrl+C to stop.")
            await asyncio.gather(*self.tasks)

        except Exception as e:
            logger.critical(f"Failed to start bot application: {e}", exc_info=True)
        finally:
            await self.shutdown()

    async def shutdown(self, signame: str = 'SIGTERM'):
        """Gracefully shuts down the application."""
        logger.info(f"Shutdown triggered by {signame}. Cleaning up...")
        if self.bot:
            self.bot.stop()
        
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Allow tasks to finish cancellation
        await asyncio.sleep(1)

        # Clean up resources
        if self.bot and self.bot.exchange_api:
            await self.bot.exchange_api.close()
        if self.bot and self.bot.position_manager:
            self.bot.position_manager.close()

        logger.info("Shutdown complete.")

async def main():
    try:
        config_loader = ConfigLoader('config_enterprise.yaml')
        config = config_loader.load_and_validate()
    except (FileNotFoundError, ValueError) as e:
        logger.critical(f"Configuration error: {e}")
        sys.exit(1)

    app = BotApplication(config)

    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    for signame in {'SIGINT', 'SIGTERM'}:
        loop.add_signal_handler(
            getattr(signal, signame),
            lambda signame=signame: asyncio.create_task(app.shutdown(signame))
        )

    await app.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application stopped by user.")
