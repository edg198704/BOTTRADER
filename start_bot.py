import asyncio
import signal
import logging
from typing import Dict, Any

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

# --- Component Factory Functions ---

def get_exchange_api(config: Dict[str, Any]) -> ExchangeAPI:
    """Factory function to create an exchange API instance."""
    name = config.get('name', 'MockExchange')
    if name == 'MockExchange':
        return MockExchangeAPI()
    else:
        return CCXTExchangeAPI(
            name=name,
            api_key=config.get('api_key'),
            api_secret=config.get('api_secret'),
            testnet=config.get('testnet', True)
        )

def get_strategy(config: Dict[str, Any]) -> TradingStrategy:
    """Factory function to create a trading strategy instance."""
    strategy_name = config.get('name')
    strategy_map = {
        'SimpleMACrossoverStrategy': SimpleMACrossoverStrategy,
        'AIEnsembleStrategy': AIEnsembleStrategy
    }
    strategy_class = strategy_map.get(strategy_name)
    if not strategy_class:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    return strategy_class(config)

# --- Main Application Logic ---

class BotRunner:
    def __init__(self, config: BotConfig):
        self.config = config
        self.bot: TradingBot = None
        self.main_task: asyncio.Task = None

    async def start(self):
        """Initializes and starts the trading bot."""
        try:
            # Dependency Injection
            exchange_api = get_exchange_api(self.config.exchange.dict())
            strategy = get_strategy(self.config.strategy.dict())
            position_manager = PositionManager(self.config.database.path)
            risk_manager = RiskManager(self.config, position_manager, self.config.strategy.initial_capital if hasattr(self.config.strategy, 'initial_capital') else 10000.0)

            self.bot = TradingBot(self.config, exchange_api, strategy, position_manager, risk_manager)
            
            self.main_task = asyncio.create_task(self.bot.run())
            await self.main_task

        except Exception as e:
            logger.critical(f"Failed to start bot: {e}", exc_info=True)
        finally:
            await self.shutdown()

    async def shutdown(self, signame: str = ''):
        """Performs a graceful shutdown of the bot and its components."""
        if signame:
            logger.info(f"Received signal {signame}. Starting graceful shutdown...")
        else:
            logger.info("Starting graceful shutdown...")

        if self.bot:
            if self.bot.running:
                self.bot.stop()
            
            if self.main_task and not self.main_task.done():
                self.main_task.cancel()
                try:
                    await self.main_task
                except asyncio.CancelledError:
                    pass # Expected

            if self.bot.exchange_api:
                await self.bot.exchange_api.close()
            
            if self.bot.position_manager:
                self.bot.position_manager.close()

        logger.info("Shutdown complete.")

async def main():
    """Main entry point for the application."""
    try:
        config_loader = ConfigLoader('config_enterprise.yaml')
        config = config_loader.load_and_validate()
    except Exception as e:
        logger.critical(f"Failed to load configuration: {e}")
        return

    runner = BotRunner(config)

    def signal_handler(signame):
        asyncio.create_task(runner.shutdown(signame))

    for signame in ('SIGINT', 'SIGTERM'):
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(getattr(signal, signame), lambda: signal_handler(signame))

    await runner.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user.")
