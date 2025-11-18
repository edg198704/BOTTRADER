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
from bot_core.strategy import TradingStrategy, SimpleMACrossoverStrategy, AIEnsembleStrategy
from bot_core.bot import TradingBot
from bot_core.telegram_bot import TelegramBot
from bot_core.monitoring import HealthChecker, InfluxDBMetrics

# Shared state for communication between components (e.g., Telegram and Bot)
shared_bot_state: Dict[str, Any] = {}

async def main():
    config = load_config('config_enterprise.yaml')
    setup_logging(config.logging.level, config.logging.file_path, config.logging.use_json)
    logger = get_logger(__name__)

    # --- Dependency Injection ---
    exchange_api = get_exchange_api(config)
    position_manager = PositionManager(config.database.path)
    risk_manager = RiskManager(config.risk_management)
    strategy = get_strategy(config)
    health_checker = HealthChecker()
    metrics_writer = InfluxDBMetrics(url=os.getenv('INFLUXDB_URL'), token=os.getenv('INFLUXDB_TOKEN'), org=os.getenv('INFLUXDB_ORG'), bucket=os.getenv('INFLUXDB_BUCKET'))

    bot = TradingBot(config, exchange_api, strategy, position_manager, risk_manager, health_checker, metrics_writer)

    # --- Setup shared state and graceful shutdown ---
    shared_bot_state['stop_bot_callback'] = bot.stop
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(bot.stop()))

    # --- Initialize and run components ---
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
    return CCXTExchangeAPI(
        name=config.exchange.name,
        api_key=config.exchange.api_key,
        api_secret=config.exchange.api_secret,
        testnet=config.exchange.testnet
    )

def get_strategy(config: BotConfig) -> TradingStrategy:
    strategy_map = {
        "SimpleMACrossoverStrategy": SimpleMACrossoverStrategy,
        "AIEnsembleStrategy": AIEnsembleStrategy
    }
    strategy_class = strategy_map.get(config.strategy.name)
    if not strategy_class:
        raise ValueError(f"Unknown strategy: {config.strategy.name}")
    return strategy_class(config.strategy)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("\nBot shutdown requested. Exiting.")
