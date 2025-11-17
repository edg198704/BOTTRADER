# start_bot.py
import asyncio
import logging
import os
import sys
import signal

# Add the current directory to the Python path to allow importing bot_core
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config_loader import ConfigLoader
from bot_core.exchange_api import MockExchangeAPI, ExchangeAPI # Import base and mock
# from bot_core.exchange_api import BinanceExchangeAPI # Uncomment if using Binance
from bot_core.position_manager import PositionManager
from bot_core.strategy import SimpleMACrossoverStrategy, TradingStrategy # Import specific strategy and base
from bot_core.bot import TradingBot

# Configure logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

async def main():
    config_path = os.path.join(os.path.dirname(__file__), 'config_enterprise.yaml')
    try:
        config_loader = ConfigLoader(config_path)
        app_config = config_loader.config

        # Set global log level from config if available
        global_log_level = app_config.get("global", {}).get("log_level", LOG_LEVEL).upper()
        logging.getLogger().setLevel(global_log_level)
        logger.info(f"Starting bot with log level: {global_log_level}")

        # Initialize Exchange API
        exchange_config = config_loader.get_section("exchange")
        exchange_name = exchange_config.get("name", "MockExchange")
        exchange_api: ExchangeAPI

        if exchange_name == "MockExchange":
            exchange_api = MockExchangeAPI()
            logger.info("Using MockExchangeAPI for trading.")
        # elif exchange_name == "Binance":
        #     api_key = exchange_config.get("api_key")
        #     api_secret = exchange_config.get("api_secret")
        #     testnet = exchange_config.get("testnet", False)
        #     if not api_key or not api_secret:
        #         raise ValueError("Binance API key and secret must be provided in config.")
        #     exchange_api = BinanceExchangeAPI(api_key, api_secret, testnet)
        #     logger.info(f"Using BinanceExchangeAPI (Testnet: {testnet}) for trading.")
        else:
            raise ValueError(f"Unsupported exchange: {exchange_name}")

        # Initialize Position Manager
        db_path = config_loader.get("database.path", "position_ledger.db")
        position_manager = PositionManager(db_path)

        # Initialize Strategy
        strategy_name = config_loader.get("strategy.name", "SimpleMACrossoverStrategy")
        strategy_class: Type[TradingStrategy]
        if strategy_name == "SimpleMACrossoverStrategy":
            strategy_class = SimpleMACrossoverStrategy
        else:
            raise ValueError(f"Unsupported strategy: {strategy_name}")

        # Initialize and run the Trading Bot
        bot = TradingBot(app_config, exchange_api, strategy_class, position_manager)

        # Handle graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in ('SIGINT', 'SIGTERM'):
            loop.add_signal_handler(getattr(signal, sig), lambda: asyncio.create_task(shutdown(bot, loop)))

        await bot.run()

    except FileNotFoundError as e:
        logger.critical(f"Configuration error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.critical(f"Configuration or initialization error: {e}")
        sys.exit(1)
    except KeyError as e:
        logger.critical(f"Missing required configuration key: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

async def shutdown(bot: TradingBot, loop: asyncio.AbstractEventLoop):
    """Graceful shutdown handler."""
    logger.info("Shutdown signal received. Initiating graceful shutdown...")
    bot.stop()
    # Give some time for the bot loop to exit
    await asyncio.sleep(2)
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True) # Wait for tasks to cancel
    loop.stop()
    logger.info("Bot shutdown complete.")

if __name__ == "__main__":
    import signal
    asyncio.run(main())
