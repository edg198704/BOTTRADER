import asyncio
import logging
import signal
import sys

from config_loader import ConfigLoader
from bot_core.config import BotConfig
from bot_core.exchange_api import MockExchangeAPI, CCXTExchangeAPI, ExchangeAPI
from bot_core.position_manager import PositionManager
from bot_core.risk_manager import RiskManager
from bot_core.strategy import TradingStrategy, AIEnsembleStrategy, SimpleMACrossoverStrategy
from bot_core.bot import TradingBot

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# --- Factory Functions ---

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
    strategy_name = config.strategy.name
    strategy_config = config.strategy.dict()

    if strategy_name == "AIEnsembleStrategy":
        return AIEnsembleStrategy(strategy_config)
    elif strategy_name == "SimpleMACrossoverStrategy":
        return SimpleMACrossoverStrategy(strategy_config)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

# --- Main Application ---

async def main():
    """Initializes and runs the trading bot."""
    logger.info("Starting the modular trading bot...")

    # 1. Load Configuration
    try:
        config_loader = ConfigLoader('config_enterprise.yaml')
        config = config_loader.load_and_validate()
    except Exception as e:
        logger.critical(f"Failed to load configuration: {e}")
        return

    # 2. Initialize Components
    try:
        exchange_api = get_exchange_api(config)
        position_manager = PositionManager(db_path=config.database.path)
        risk_manager = RiskManager(config, position_manager, initial_capital=10000) # Assuming initial capital from a portfolio manager in a real system
        strategy = get_strategy(config)
        
        bot = TradingBot(
            config=config,
            exchange_api=exchange_api,
            strategy=strategy,
            position_manager=position_manager,
            risk_manager=risk_manager
        )
    except Exception as e:
        logger.critical(f"Failed to initialize bot components: {e}")
        if 'exchange_api' in locals() and exchange_api:
            await exchange_api.close()
        return

    # 3. Setup Graceful Shutdown
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def shutdown_handler():
        logger.info("Shutdown signal received. Stopping bot...")
        bot.stop()
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_handler)

    # 4. Run the Bot
    try:
        bot_task = asyncio.create_task(bot.run())
        await stop_event.wait()
        await bot_task # Wait for the bot to finish its current loop and stop
    except Exception as e:
        logger.critical(f"Bot crashed with an unhandled exception: {e}", exc_info=True)
    finally:
        logger.info("Closing exchange connection...")
        await exchange_api.close()
        logger.info("Bot has been shut down.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user.")
