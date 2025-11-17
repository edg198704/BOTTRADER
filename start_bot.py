import asyncio
import signal
from typing import Dict

from bot_core.logger import setup_logging, get_logger
from bot_core.config import BotConfig, StrategyConfig
from config_loader import load_config
from bot_core.exchange_api import ExchangeAPI, MockExchangeAPI, CCXTExchangeAPI
from bot_core.position_manager import PositionManager
from bot_core.risk_manager import RiskManager
from bot_core.strategy import TradingStrategy, AIEnsembleStrategy, SimpleMACrossoverStrategy
from bot_core.bot import TradingBot
from bot_core.telegram_bot import TelegramBot

logger = get_logger(__name__)

def get_exchange_api(config: BotConfig) -> ExchangeAPI:
    if config.exchange.name == "MockExchange":
        return MockExchangeAPI()
    else:
        return CCXTExchangeAPI(
            name=config.exchange.name,
            api_key=config.exchange.api_key,
            api_secret=config.exchange.api_secret,
            testnet=config.exchange.testnet
        )

def get_strategy(config: StrategyConfig) -> TradingStrategy:
    strategy_map: Dict[str, TradingStrategy] = {
        "AIEnsembleStrategy": AIEnsembleStrategy,
        "SimpleMACrossoverStrategy": SimpleMACrossoverStrategy
    }
    strategy_class = strategy_map.get(config.name)
    if not strategy_class:
        raise ValueError(f"Unknown strategy: {config.name}")
    return strategy_class(config)

async def main():
    bot = None
    telegram_bot = None
    tasks = []

    def shutdown_handler(signum, frame):
        logger.critical(f"Signal {signum} received, initiating graceful shutdown.")
        if bot:
            bot.running = False # Signal the main loop to stop

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        # 1. Load Configuration
        config = load_config()

        # 2. Setup Logging
        setup_logging(config.logging.level, config.logging.file_path, config.logging.use_json)

        # 3. Initialize Components (Dependency Injection)
        exchange_api = get_exchange_api(config)
        position_manager = PositionManager(config.database)
        risk_manager = RiskManager(config.risk_management)
        strategy = get_strategy(config.strategy)

        # 4. Initialize the Main Bot Orchestrator
        bot = TradingBot(config, exchange_api, strategy, position_manager, risk_manager)

        # 5. Initialize optional Telegram Bot
        telegram_bot = TelegramBot(config.telegram, bot)
        if telegram_bot.enabled:
            tasks.append(asyncio.create_task(telegram_bot.run()))

        # 6. Start the main bot task
        tasks.append(asyncio.create_task(bot.run()))
        
        # Wait for all tasks to complete (e.g., main loop to exit)
        await asyncio.gather(*tasks)

    except Exception as e:
        logger.critical("Bot failed to start or crashed unexpectedly.", error=str(e), exc_info=True)
    finally:
        logger.info("Performing final cleanup...")
        if bot:
            await bot.stop()
        if telegram_bot:
            await telegram_bot.stop()
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user.")
