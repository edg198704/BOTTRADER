import asyncio
import signal
import os
from typing import Dict, Any

from config_loader import load_config
from bot_core.logger import setup_logging, get_logger
from bot_core.config import BotConfig
from bot_core.exchange_api import ExchangeAPI, MockExchangeAPI, CCXTExchangeAPI
from bot_core.position_manager import PositionManager
from bot_core.risk_manager import RiskManager
from bot_core.strategy import TradingStrategy
from bot_core import strategy as strategy_module
from bot_core.bot import TradingBot
from bot_core.telegram_bot import TelegramBot
from bot_core.monitoring import HealthChecker, InfluxDBMetrics
from bot_core.data_handler import DataHandler
from bot_core.order_sizer import OrderSizer
from bot_core.position_monitor import PositionMonitor

# Shared state for communication between components (e.g., Telegram and Bot)
shared_bot_state: Dict[str, Any] = {}

async def main():
    config = load_config('config_enterprise.yaml')
    setup_logging(config.logging.level, config.logging.file_path, config.logging.use_json)
    logger = get_logger(__name__)

    # --- Shared State & Dependency Injection ---
    latest_prices: Dict[str, float] = {}

    exchange_api = get_exchange_api(config)
    position_manager = PositionManager(config.database, config.initial_capital)
    risk_manager = RiskManager(config.risk_management)
    strategy = get_strategy(config)
    order_sizer = OrderSizer()
    health_checker = HealthChecker()
    metrics_writer = InfluxDBMetrics(url=os.getenv('INFLUXDB_URL'), token=os.getenv('INFLUXDB_TOKEN'), org=os.getenv('INFLUXDB_ORG'), bucket=os.getenv('INFLUXDB_BUCKET'))

    position_monitor = PositionMonitor(
        config=config,
        position_manager=position_manager,
        shared_latest_prices=latest_prices
    )

    bot = TradingBot(
        config=config,
        exchange_api=exchange_api,
        data_handler=None, # Will be set after DataHandler is created
        strategy=strategy,
        position_manager=position_manager,
        risk_manager=risk_manager,
        order_sizer=order_sizer,
        health_checker=health_checker,
        position_monitor=position_monitor,
        shared_latest_prices=latest_prices,
        metrics_writer=metrics_writer,
        shared_bot_state=shared_bot_state
    )

    data_handler = DataHandler(
        exchange_api=exchange_api,
        config=config,
        shared_latest_prices=latest_prices
    )
    bot.data_handler = data_handler # Complete the dependency injection

    # Set the callback after bot is created to avoid circular dependency on init
    position_monitor.set_close_position_callback(bot._close_position)

    # --- Setup shared state and graceful shutdown ---
    shared_bot_state['stop_bot_callback'] = bot.stop
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(bot.stop()))

    # --- Initialize and run components ---
    await position_manager.initialize() # Load historical PnL before starting
    await data_handler.initialize_data() # Load historical data before starting
    telegram_bot = TelegramBot(config.telegram, shared_bot_state)
    
    try:
        bot_task = asyncio.create_task(bot.run())
        telegram_task = asyncio.create_task(telegram_bot.run())
        await asyncio.gather(bot_task, telegram_task)
    except asyncio.CancelledError:
        logger.info("Main task cancelled.")
    finally:
        logger.info("Shutting down all components.")
        await telegram_bot.stop()
        # The bot.stop() is already called by the signal handler

def get_exchange_api(config: BotConfig) -> ExchangeAPI:
    if config.exchange.name == "MockExchange":
        return MockExchangeAPI()
    return CCXTExchangeAPI(config.exchange)

def get_strategy(config: BotConfig) -> TradingStrategy:
    """Dynamically loads and instantiates a strategy class by name."""
    logger = get_logger(__name__)
    strategy_name = config.strategy.name
    try:
        strategy_class = getattr(strategy_module, strategy_name)
        if not issubclass(strategy_class, TradingStrategy):
            raise TypeError(f"Strategy '{strategy_name}' is not a valid subclass of TradingStrategy.")
        
        logger.info(f"Loading strategy: {strategy_name}")
        return strategy_class(config.strategy)
    except AttributeError:
        logger.critical(f"Strategy class '{strategy_name}' not found in bot_core/strategy.py.")
        raise ValueError(f"Unknown strategy: {strategy_name}")
    except TypeError as e:
        logger.critical(f"Strategy class '{strategy_name}' is not a valid strategy.", error=str(e))
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("\nBot shutdown requested. Exiting.")
