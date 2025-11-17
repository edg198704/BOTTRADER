import asyncio
import logging
import signal
from typing import Dict, Any

from config_loader import ConfigLoader
from bot_core.config import BotConfig, StrategyConfig
from bot_core.bot import TradingBot
from bot_core.exchange_api import ExchangeAPI, MockExchangeAPI, CCXTExchangeAPI
from bot_core.position_manager import PositionManager
from bot_core.risk_manager import RiskManager
from bot_core.strategy import TradingStrategy, SimpleMACrossoverStrategy, AIEnsembleStrategy
from bot_core.execution_handler import ExecutionHandler
from bot_core.order_manager import OrderManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

bots = []

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

def get_strategy(config: StrategyConfig) -> TradingStrategy:
    """Factory function to create a strategy instance."""
    strategy_map = {
        "SimpleMACrossoverStrategy": SimpleMACrossoverStrategy,
        "AIEnsembleStrategy": AIEnsembleStrategy
    }
    strategy_class = strategy_map.get(config.name)
    if not strategy_class:
        raise ValueError(f"Unknown strategy: {config.name}")
    return strategy_class(config.dict())

async def main():
    """Main function to initialize and run the trading bot."""
    config_loader = ConfigLoader('config_enterprise.yaml')
    try:
        config = config_loader.load_and_validate()
    except (FileNotFoundError, ValueError) as e:
        logger.critical(f"Failed to load configuration: {e}")
        return

    # Initialize shared components
    exchange_api = get_exchange_api(config)
    position_manager = PositionManager(config.database.path)
    risk_manager = RiskManager(config.risk_management, position_manager, config.initial_capital)
    order_manager = OrderManager(exchange_api)

    # Initialize strategy-specific components
    strategy = get_strategy(config.strategy)
    execution_handler = ExecutionHandler(order_manager, risk_manager)

    # Create and store the bot instance
    bot = TradingBot(
        config=config,
        exchange_api=exchange_api,
        strategy=strategy,
        position_manager=position_manager,
        risk_manager=risk_manager,
        execution_handler=execution_handler,
        order_manager=order_manager
    )
    bots.append(bot)

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s, loop)))

    try:
        await bot.run()
    finally:
        logger.info("Bot has finished its run.")
        position_manager.close()
        await exchange_api.close()

async def shutdown(sig, loop):
    logger.info(f"Received exit signal {sig.name}...")
    for bot in bots:
        await bot.stop()
    
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]

    logger.info(f"Cancelling {len(tasks)} outstanding tasks")
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Application shutdown gracefully.")
