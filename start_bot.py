import asyncio
import signal
import sys
from typing import Dict

from config_loader import load_config
from bot_core.logger import setup_logging, get_logger
from bot_core.config import BotConfig
from bot_core.exchange_api import ExchangeAPI, MockExchangeAPI, CCXTExchangeAPI
from bot_core.position_manager import PositionManager
from bot_core.risk_manager import RiskManager
from bot_core.strategy import TradingStrategy, SimpleMACrossoverStrategy, AIEnsembleStrategy
from bot_core.bot import TradingBot

# --- Factory Functions for Dependency Injection ---

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
    strategy_map: Dict[str, TradingStrategy] = {
        "SimpleMACrossoverStrategy": SimpleMACrossoverStrategy,
        "AIEnsembleStrategy": AIEnsembleStrategy,
    }
    strategy_class = strategy_map.get(config.strategy.name)
    if not strategy_class:
        raise ValueError(f"Unknown strategy: {config.strategy.name}")
    return strategy_class(config.strategy)

# --- Main Application --- 

async def main():
    """Main function to initialize and run the trading bot."""
    # 1. Load Configuration
    try:
        config = load_config()
    except Exception as e:
        print(f"FATAL: Could not load configuration. {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Setup Logging
    setup_logging(config.logging.level, config.logging.file_path, config.logging.use_json)
    logger = get_logger(__name__)

    # 3. Initialize Components
    try:
        exchange_api = get_exchange_api(config)
        position_manager = PositionManager(config.database)
        risk_manager = RiskManager(config.risk_management, position_manager, config.initial_capital)
        strategy = get_strategy(config)
        
        bot = TradingBot(
            config=config,
            exchange_api=exchange_api,
            strategy=strategy,
            position_manager=position_manager,
            risk_manager=risk_manager
        )
    except Exception as e:
        logger.critical("Failed to initialize bot components.", error=str(e), exc_info=True)
        sys.exit(1)

    # 4. Setup Graceful Shutdown
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def shutdown_handler():
        logger.warning("Shutdown signal received. Stopping bot...")
        if not stop_event.is_set():
            loop.call_soon_threadsafe(stop_event.set)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_handler)

    # 5. Run the Bot
    bot_task = asyncio.create_task(bot.run())
    
    await stop_event.wait()

    # 6. Clean up
    await bot.stop()
    bot_task.cancel()
    try:
        await bot_task
    except asyncio.CancelledError:
        pass

    logger.info("Bot has been shut down gracefully.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
