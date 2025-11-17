import asyncio
import logging
import signal
import sys
from typing import Dict, Any

from config_loader import ConfigLoader
from bot_core.bot import TradingBot
from bot_core.exchange_api import ExchangeAPI, MockExchangeAPI, CCXTExchangeAPI
from bot_core.position_manager import PositionManager
from bot_core.risk_manager import RiskManager
from bot_core.strategy import TradingStrategy, SimpleMACrossoverStrategy, AIEnsembleStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# --- Factory Functions for Dynamic Component Creation ---

def get_exchange_api(config: Dict[str, Any]) -> ExchangeAPI:
    exchange_name = config.get('name', 'MockExchange')
    if exchange_name == 'MockExchange':
        logger.info("Using MockExchangeAPI for testing.")
        return MockExchangeAPI()
    else:
        logger.info(f"Using CCXTExchangeAPI for {exchange_name}.")
        return CCXTExchangeAPI(
            name=exchange_name,
            api_key=config.get('api_key'),
            api_secret=config.get('api_secret'),
            testnet=config.get('testnet', True)
        )

def get_strategy(config: Dict[str, Any]) -> TradingStrategy:
    strategy_name = config.get('name', 'SimpleMACrossoverStrategy')
    if strategy_name == 'AIEnsembleStrategy':
        logger.info("Using AIEnsembleStrategy.")
        return AIEnsembleStrategy(config)
    elif strategy_name == 'SimpleMACrossoverStrategy':
        logger.info("Using SimpleMACrossoverStrategy.")
        return SimpleMACrossoverStrategy(config)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

async def main():
    """Main function to initialize and run the trading bot."""
    bot_instance = None
    try:
        # 1. Load Configuration
        config_loader = ConfigLoader('config_enterprise.yaml')
        config = config_loader.load_and_validate()

        # 2. Initialize Components (Dependency Injection)
        exchange_api = get_exchange_api(config.exchange.dict())
        position_manager = PositionManager(config.database.path)
        risk_manager = RiskManager(config.risk_management.dict(), position_manager, config.strategy.dict().get('initial_capital', 10000.0))
        strategy = get_strategy(config.strategy.dict())

        # 3. Create Bot Instance
        bot_instance = TradingBot(
            config=config,
            exchange_api=exchange_api,
            strategy=strategy,
            position_manager=position_manager,
            risk_manager=risk_manager
        )

        # 4. Setup Graceful Shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(shutdown(s, loop, bot_instance))
            )

        # 5. Run the Bot
        await bot_instance.run()

    except FileNotFoundError as e:
        logger.critical(f"Configuration file error: {e}")
    except ValueError as e:
        logger.critical(f"Configuration or initialization error: {e}")
    except Exception as e:
        logger.critical(f"An unexpected error occurred during startup: {e}", exc_info=True)
    finally:
        if bot_instance and bot_instance.running:
            bot_instance.stop()
        logger.info("Bot application has shut down.")

async def shutdown(sig: signal.Signals, loop: asyncio.AbstractEventLoop, bot: TradingBot):
    """Graceful shutdown handler."""
    logger.warning(f"Received exit signal {sig.name}...")
    
    if bot:
        bot.stop()

    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]

    logger.info(f"Cancelling {len(tasks)} outstanding tasks")
    await asyncio.gather(*tasks, return_exceptions=True)
    
    if bot and bot.exchange_api:
        await bot.exchange_api.close()

    loop.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot shutdown initiated by user.")
