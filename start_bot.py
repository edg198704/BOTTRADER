import asyncio
import signal
from typing import Dict, Any, Optional

from bot_core.config import BotConfig
from bot_core.logger import setup_logging, get_logger, set_correlation_id
from bot_core.exchange_api import MockExchangeAPI, CCXTExchangeAPI, ExchangeAPI
from bot_core.position_manager import PositionManager
from bot_core.risk_manager import RiskManager
from bot_core.strategy import TradingStrategy, SimpleMACrossoverStrategy, AIEnsembleStrategy
from bot_core.bot import TradingBot
from bot_core.telegram_bot import TelegramBot
from bot_core.data_handler import LiveAPIDataHandler
from bot_core.execution_handler import ExecutionHandler
from bot_core.order_manager import OrderManager
from bot_core.ai.ensemble_learner import EnsembleLearner
from bot_core.ai.regime_detector import MarketRegimeDetector
from config_loader import load_config

logger = get_logger(__name__)

# --- Factory Functions ---

def get_exchange_api(config: Dict[str, Any]) -> ExchangeAPI:
    """Factory function to create an exchange API instance."""
    exchange_config = config['exchange']
    if exchange_config['name'] == "MockExchange":
        return MockExchangeAPI()
    else:
        return CCXTExchangeAPI(
            name=exchange_config['name'],
            api_key=exchange_config.get('api_key'),
            api_secret=exchange_config.get('api_secret'),
            testnet=exchange_config.get('testnet', True)
        )

def get_strategy(event_queue: asyncio.Queue, config: Dict[str, Any]) -> TradingStrategy:
    """Factory function to create a trading strategy instance."""
    strategy_config = config['strategy']
    strategy_name = strategy_config['name']
    
    strategy_map = {
        "SimpleMACrossoverStrategy": SimpleMACrossoverStrategy,
        "AIEnsembleStrategy": AIEnsembleStrategy
    }

    if strategy_name not in strategy_map:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    if strategy_name == "AIEnsembleStrategy":
        # Create and inject AI components
        from bot_core.config import AIStrategyConfig
        ai_config = AIStrategyConfig(**strategy_config['ai_ensemble'])
        ensemble_learner = EnsembleLearner(ai_config)
        regime_detector = MarketRegimeDetector(ai_config)
        return AIEnsembleStrategy(event_queue, strategy_config, ensemble_learner, regime_detector)
    else:
        return strategy_map[strategy_name](event_queue, strategy_config)

# --- Main Application ---

async def main():
    """Main function to initialize and run the trading bot."""
    set_correlation_id()
    
    # Load and validate configuration
    try:
        raw_config = load_config('config_enterprise.yaml')
        config = BotConfig(**raw_config)
    except Exception as e:
        # Use basic logger since full one might not be configured
        import logging
        logging.basicConfig(level="ERROR")
        logging.error(f"Failed to load or validate configuration: {e}")
        return

    # Setup application-wide logging
    setup_logging(config.logging.level, config.logging.file_path, config.logging.use_json)
    logger.info("Configuration loaded and validated successfully.")

    bot = None
    telegram_bot = None
    try:
        # Initialize components with dependency injection
        event_queue = asyncio.Queue()
        exchange_api = get_exchange_api(config.dict())
        position_manager = PositionManager(db_path=config.database.path)
        risk_manager = RiskManager(config.risk_management, position_manager, config.initial_capital)
        strategy = get_strategy(event_queue, config.dict())
        
        order_manager = OrderManager(event_queue, exchange_api)
        execution_handler = ExecutionHandler(event_queue, order_manager, risk_manager)
        data_handler = LiveAPIDataHandler(event_queue, exchange_api, [config.strategy.symbol], config.strategy.interval_seconds)

        # Assemble the bot
        bot = TradingBot(
            config=config,
            data_handler=data_handler,
            exchange_api=exchange_api,
            strategy=strategy,
            position_manager=position_manager,
            risk_manager=risk_manager,
            execution_handler=execution_handler,
            order_manager=order_manager
        )

        # Initialize and start Telegram bot if configured
        if config.telegram.bot_token and config.telegram.admin_chat_ids:
            telegram_bot = TelegramBot(config.telegram, bot)
            asyncio.create_task(telegram_bot.start())
        else:
            logger.warning("Telegram bot not configured. Skipping.")

        # Handle graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s, bot, telegram_bot)))

        # Start the main bot loop
        await bot.run()

    except Exception as e:
        logger.critical("Fatal error during bot initialization or runtime", error=str(e), exc_info=True)
    finally:
        logger.info("Main function finished.")

async def shutdown(sig: signal.Signals, bot: TradingBot, telegram_bot: Optional[TelegramBot]):
    """Graceful shutdown handler."""
    logger.warning(f"Received shutdown signal: {sig.name}. Initiating graceful shutdown...")
    
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    
    if telegram_bot:
        await telegram_bot.stop()
    
    if bot:
        await bot.stop()

    for task in tasks:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("All tasks cancelled. Shutdown complete.")
    asyncio.get_running_loop().stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Application exiting.")
