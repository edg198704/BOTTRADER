import asyncio
import logging
import signal
import sys

from config_loader import ConfigLoader
from bot_core.bot import TradingBot
from bot_core.exchange_api import MockExchangeAPI, CCXTExchangeAPI, ExchangeAPI
from bot_core.position_manager import PositionManager
from bot_core.risk_manager import RiskManager
from bot_core.strategy import SimpleMACrossoverStrategy, AIEnsembleStrategy, TradingStrategy
from bot_core.config import BotConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global bot instance to be accessible by signal handler
bot_instance: TradingBot = None

def get_exchange_api(config: BotConfig) -> ExchangeAPI:
    """Factory function to create an exchange API instance."""
    if config.exchange.name == "MockExchange":
        return MockExchangeAPI()
    else:
        return CCXTExchangeAPI(
            name=config.exchange.name,
            api_key=config.exchange.api_key,
            api_secret=config.exchange.api_secret,
            testnet=config.exchange.testnet
        )

def get_strategy(config: BotConfig) -> TradingStrategy:
    """Factory function to create a trading strategy instance."""
    strategy_map = {
        "SimpleMACrossoverStrategy": SimpleMACrossoverStrategy,
        "AIEnsembleStrategy": AIEnsembleStrategy,
    }
    strategy_class = strategy_map.get(config.strategy.name)
    if not strategy_class:
        raise ValueError(f"Unknown strategy: {config.strategy.name}")
    return strategy_class(config.strategy.dict())

async def main():
    """Main function to initialize and run the trading bot."""
    global bot_instance
    position_manager = None
    exchange_api = None

    try:
        # Load configuration
        config_loader = ConfigLoader('config_enterprise.yaml')
        config = config_loader.load_and_validate()

        # Initialize components
        position_manager = PositionManager(db_path=config.database.path)
        exchange_api = get_exchange_api(config)
        risk_manager = RiskManager(
            config=config.risk_management, 
            position_manager=position_manager, 
            initial_capital=config.initial_capital
        )
        strategy = get_strategy(config)

        # Create and run the bot
        bot_instance = TradingBot(config, exchange_api, strategy, position_manager, risk_manager)
        
        # Setup signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s)))

        await bot_instance.run()

    except Exception as e:
        logger.critical(f"Bot failed to start or crashed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if bot_instance and bot_instance.running:
            bot_instance.stop()
        if position_manager:
            position_manager.close()
        if exchange_api:
            await exchange_api.close()
        logger.info("Bot shutdown complete.")

async def shutdown(sig: signal.Signals):
    """Graceful shutdown handler."""
    logger.info(f"Received exit signal {sig.name}... initiating graceful shutdown.")
    if bot_instance:
        bot_instance.stop()
    
    # Allow time for tasks to clean up
    await asyncio.sleep(2)

    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if tasks:
        logger.info(f"Cancelling {len(tasks)} outstanding tasks")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.get_running_loop().stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.exceptions.CancelledError):
        logger.info("Event loop stopped.")
