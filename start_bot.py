import asyncio
import signal
import os
from typing import Optional

from bot_core.logger import setup_logging, get_logger
from bot_core.config import BotConfig
from bot_core.bot import TradingBot
from bot_core.exchange_api import ExchangeAPI, MockExchangeAPI, CCXTExchangeAPI
from bot_core.strategy import TradingStrategy, SimpleMACrossoverStrategy, AIEnsembleStrategy
from bot_core.position_manager import PositionManager
from bot_core.risk_manager import RiskManager
from bot_core.telegram_bot import TelegramBot
from bot_core.monitoring import HealthChecker, InfluxDBMetrics
from config_loader import load_config

logger: Optional[get_logger] = None

def get_exchange_api(config: BotConfig) -> ExchangeAPI:
    if config.exchange.name == "MockExchange":
        return MockExchangeAPI()
    return CCXTExchangeAPI(
        name=config.exchange.name,
        api_key=os.getenv('BOT_EXCHANGE_API_KEY', config.exchange.api_key),
        api_secret=os.getenv('BOT_EXCHANGE_API_SECRET', config.exchange.api_secret),
        testnet=config.exchange.testnet
    )

def get_strategy(config: BotConfig) -> TradingStrategy:
    strategy_map = {
        "AIEnsembleStrategy": AIEnsembleStrategy,
        "SimpleMACrossoverStrategy": SimpleMACrossoverStrategy
    }
    strategy_class = strategy_map.get(config.strategy.name)
    if not strategy_class:
        raise ValueError(f"Unknown strategy: {config.strategy.name}")
    return strategy_class(config.strategy)

async def main():
    global logger
    config = load_config('config_enterprise.yaml')
    setup_logging(config.logging.level, config.logging.file_path, config.logging.use_json)
    logger = get_logger(__name__)

    bot: Optional[TradingBot] = None
    telegram_bot: Optional[TelegramBot] = None

    async def shutdown(sig=None):
        if sig:
            logger.info(f"Received shutdown signal: {sig.name}")
        if bot and bot.running:
            await bot.stop()
        if telegram_bot:
            await telegram_bot.stop()
        # Allow time for cleanup tasks
        await asyncio.sleep(2)
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        [task.cancel() for task in tasks]
        await asyncio.gather(*tasks, return_exceptions=True)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s)))

    try:
        # Initialize components
        exchange_api = get_exchange_api(config)
        strategy = get_strategy(config)
        position_manager = PositionManager(config.database)
        risk_manager = RiskManager(config.risk_management)
        health_checker = HealthChecker()
        
        influx_url = os.getenv('INFLUXDB_URL')
        influx_token = os.getenv('INFLUXDB_TOKEN')
        influx_org = os.getenv('INFLUXDB_ORG')
        influx_bucket = os.getenv('INFLUXDB_BUCKET')
        metrics_writer = InfluxDBMetrics(influx_url, influx_token, influx_org, influx_bucket)

        bot = TradingBot(
            config=config,
            exchange_api=exchange_api,
            strategy=strategy,
            position_manager=position_manager,
            risk_manager=risk_manager,
            health_checker=health_checker,
            metrics_writer=metrics_writer
        )

        if config.telegram.bot_token and config.telegram.admin_chat_ids:
            telegram_bot = TelegramBot(config.telegram, bot)
            asyncio.create_task(telegram_bot.start())
        
        await bot.run()

    except Exception as e:
        logger.critical("Bot failed to start or crashed.", error=str(e), exc_info=True)
        await shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("\nBot shutdown gracefully.")
