import asyncio
import logging
import signal
from typing import Dict, Any

from config_loader import load_config
from bot_core.config import BotConfig, StrategyConfig
from bot_core.exchange_api import ExchangeAPI, MockExchangeAPI, CCXTExchangeAPI
from bot_core.position_manager import PositionManager
from bot_core.risk_manager import RiskManager
from bot_core.strategy import TradingStrategy, AIEnsembleStrategy, SimpleMACrossoverStrategy
from bot_core.order_manager import OrderManager
from bot_core.execution_handler import ExecutionHandler
from bot_core.bot import TradingBot
from bot_core.telegram_bot import TelegramBot

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global bot instance to be accessible by signal handler
bot_instance: TradingBot = None
telegram_bot_instance: TelegramBot = None

def get_exchange_api(config: BotConfig) -> ExchangeAPI:
    """Factory function to create an exchange API instance based on config."""
    exchange_config = config.exchange
    if exchange_config.name == "MockExchange":
        logger.info("Using MockExchangeAPI for simulation.")
        return MockExchangeAPI()
    else:
        logger.info(f"Using CCXTExchangeAPI for {exchange_config.name}.")
        return CCXTExchangeAPI(
            name=exchange_config.name,
            api_key=exchange_config.api_key,
            api_secret=exchange_config.api_secret,
            testnet=exchange_config.testnet
        )

def get_strategy(config: StrategyConfig) -> TradingStrategy:
    """Factory function to create a trading strategy instance based on config."""
    strategy_map = {
        "AIEnsembleStrategy": AIEnsembleStrategy,
        "SimpleMACrossoverStrategy": SimpleMACrossoverStrategy
    }
    strategy_class = strategy_map.get(config.name)
    if not strategy_class:
        raise ValueError(f"Unknown strategy '{config.name}' specified in configuration.")
    
    logger.info(f"Using strategy: {config.name}")
    return strategy_class(config.dict())

async def main():
    """Main function to initialize and run the trading bot."""
    global bot_instance, telegram_bot_instance
    try:
        # 1. Load Configuration
        config = load_config('config_enterprise.yaml')

        # 2. Initialize Components
        exchange_api = get_exchange_api(config)
        position_manager = PositionManager(config.database.path)
        risk_manager = RiskManager(config.risk_management, position_manager, config.initial_capital)
        strategy = get_strategy(config.strategy)
        order_manager = OrderManager(exchange_api)
        execution_handler = ExecutionHandler(order_manager, risk_manager)

        # 3. Initialize the Main Bot Orchestrator
        bot_instance = TradingBot(
            config=config,
            exchange_api=exchange_api,
            strategy=strategy,
            position_manager=position_manager,
            risk_manager=risk_manager,
            execution_handler=execution_handler,
            order_manager=order_manager
        )

        # 4. Initialize optional Telegram Bot
        try:
            telegram_bot_instance = TelegramBot(config.telegram, bot_instance)
            if telegram_bot_instance.enabled:
                await telegram_bot_instance.start()
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")

        # 5. Start the bot's main loop
        await bot_instance.run()

    except Exception as e:
        logger.critical(f"Bot failed to start: {e}", exc_info=True)
    finally:
        logger.info("Bot shutting down.")
        if bot_instance:
            await bot_instance.stop()
        if telegram_bot_instance:
            await telegram_bot_instance.stop()
        if 'position_manager' in locals() and position_manager:
            position_manager.close()
        if 'exchange_api' in locals() and exchange_api:
            await exchange_api.close()

async def shutdown(signal, loop):
    """Graceful shutdown handler."""
    logger.info(f"Received exit signal {signal.name}...")
    if bot_instance:
        await bot_instance.stop()
    
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]

    logger.info("Cancelling outstanding tasks")
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
    for s in signals:
        loop.add_signal_handler(
            s, lambda s=s: asyncio.create_task(shutdown(s, loop))
        )

    try:
        loop.run_until_complete(main())
    finally:
        logger.info("Shutdown complete.")
        loop.close()
