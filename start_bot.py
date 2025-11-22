import asyncio
import signal
import os
from typing import Dict, Any

from config_loader import load_config
from bot_core.logger import setup_logging, get_logger
from bot_core.config import BotConfig, AIEnsembleStrategyParams
from bot_core.exchange_api import ExchangeAPI, MockExchangeAPI, CCXTExchangeAPI
from bot_core.position_manager import PositionManager
from bot_core.risk_manager import RiskManager
from bot_core.strategy import TradingStrategy
from bot_core import strategy as strategy_module
from bot_core.bot import TradingBot
from bot_core.telegram_bot import TelegramBot
from bot_core.monitoring import HealthChecker, InfluxDBMetrics, AlertSystem
from bot_core.data_handler import DataHandler
from bot_core.order_sizer import OrderSizer
from bot_core.position_monitor import PositionMonitor
from bot_core.order_lifecycle_manager import OrderLifecycleService
from bot_core.trade_executor import TradeExecutor
from bot_core.utils import generate_indicator_rename_map
from bot_core.event_system import EventBus

shared_bot_state: Dict[str, Any] = {}

def validate_config(config: BotConfig):
    logger = get_logger(__name__)
    logger.info("Performing configuration validation...")
    if isinstance(config.strategy.params, AIEnsembleStrategyParams):
        base_columns = {'open', 'high', 'low', 'close', 'volume'}
        try:
            rename_map = generate_indicator_rename_map(config.strategy.indicators)
            generated_columns = set(rename_map.values())
            if config.strategy.secondary_timeframes:
                htf_cols = set()
                for tf in config.strategy.secondary_timeframes:
                    for col in generated_columns:
                        htf_cols.add(f"{tf}_{col}")
                generated_columns.update(htf_cols)
        except ValueError as e:
            logger.critical(f"Configuration error in indicators: {e}")
            raise e

        available_columns = base_columns.union(generated_columns)
        if config.strategy.params.features.use_time_features:
            available_columns.update(['time_hour_sin', 'time_hour_cos', 'time_dow_sin', 'time_dow_cos'])
        if config.strategy.params.features.use_price_action_features:
            available_columns.update(['pa_body_size', 'pa_upper_wick', 'pa_lower_wick'])
        if config.strategy.params.features.use_volatility_estimators:
            available_columns.update(['volatility_gk'])
        if config.strategy.params.features.use_microstructure_features:
            available_columns.update(['amihud_illiquidity', 'volatility_parkinson'])
        if config.strategy.params.features.use_frac_diff:
            available_columns.update(['close_frac'])

        required_columns = set(config.strategy.params.feature_columns)
        missing_columns = required_columns - available_columns
        if missing_columns:
            error_msg = f"Missing feature columns: {list(missing_columns)}"
            logger.critical(error_msg)
            raise ValueError(error_msg)
    logger.info("Configuration validation passed.")

async def main():
    config = load_config('config_enterprise.yaml')
    setup_logging(config.logging.level, config.logging.file_path, config.logging.use_json)
    logger = get_logger(__name__)

    try:
        validate_config(config)
    except ValueError as e:
        logger.critical("Configuration is invalid. Bot cannot start.", error=str(e))
        return

    latest_prices: Dict[str, float] = {}
    alert_system = AlertSystem()
    event_bus = EventBus()

    exchange_api = get_exchange_api(config)
    
    data_handler = DataHandler(
        exchange_api=exchange_api,
        config=config,
        shared_latest_prices=latest_prices,
        event_bus=event_bus
    )

    position_manager = PositionManager(
        config.database, 
        config.initial_capital,
        alert_system=alert_system
    )
    
    risk_manager = RiskManager(
        config.risk_management,
        position_manager=position_manager,
        data_handler=data_handler,
        alert_system=alert_system
    )
    
    # Initialize Metrics Writer (InfluxDB)
    metrics_writer = InfluxDBMetrics(
        url=os.getenv('INFLUXDB_URL'), 
        token=os.getenv('INFLUXDB_TOKEN'), 
        org=os.getenv('INFLUXDB_ORG'), 
        bucket=os.getenv('INFLUXDB_BUCKET')
    )

    strategy = get_strategy(config, metrics_writer)
    order_sizer = OrderSizer()
    health_checker = HealthChecker()

    position_monitor = PositionMonitor(
        config=config,
        position_manager=position_manager,
        risk_manager=risk_manager,
        data_handler=data_handler,
        shared_latest_prices=latest_prices
    )

    order_lifecycle_service = OrderLifecycleService(
        exchange_api=exchange_api,
        position_manager=position_manager,
        event_bus=event_bus,
        exec_config=config.execution,
        shared_latest_prices=latest_prices
    )

    trade_executor = TradeExecutor(
        config=config,
        exchange_api=exchange_api,
        position_manager=position_manager,
        risk_manager=risk_manager,
        order_sizer=order_sizer,
        order_lifecycle_service=order_lifecycle_service,
        alert_system=alert_system,
        shared_latest_prices=latest_prices,
        market_details={},
        data_handler=data_handler
    )

    bot = TradingBot(
        config=config,
        exchange_api=exchange_api,
        data_handler=data_handler,
        strategy=strategy,
        position_manager=position_manager,
        risk_manager=risk_manager,
        health_checker=health_checker,
        position_monitor=position_monitor,
        trade_executor=trade_executor,
        alert_system=alert_system,
        shared_latest_prices=latest_prices,
        event_bus=event_bus,
        order_lifecycle_service=order_lifecycle_service,
        metrics_writer=metrics_writer,
        shared_bot_state=shared_bot_state
    )

    position_monitor.set_close_position_callback(bot._close_position)
    shared_bot_state['stop_bot_callback'] = bot.stop
    
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Signal received, initiating shutdown...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    await position_manager.initialize()
    await risk_manager.initialize()
    await data_handler.initialize_data()
    
    telegram_bot = TelegramBot(config.telegram, shared_bot_state)
    if telegram_bot.application:
        alert_system.register_handler(telegram_bot.create_alert_handler())

    try:
        bot_task = asyncio.create_task(bot.run())
        telegram_task = asyncio.create_task(telegram_bot.run())
        await stop_event.wait()
        await bot.stop()
        await telegram_bot.stop()
        await asyncio.gather(bot_task, telegram_task, return_exceptions=True)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.critical("Unexpected error in main loop", error=str(e), exc_info=True)
    finally:
        if bot.shared_bot_state.get('status') != 'stopped':
            await bot.stop()
        await telegram_bot.stop()
        await metrics_writer.close()

def get_exchange_api(config: BotConfig) -> ExchangeAPI:
    if config.exchange.name == "MockExchange":
        return MockExchangeAPI()
    return CCXTExchangeAPI(config.exchange)

def get_strategy(config: BotConfig, metrics_writer: Optional[InfluxDBMetrics] = None) -> TradingStrategy:
    logger = get_logger(__name__)
    strategy_name = config.strategy.params.name
    strategy_params = config.strategy.params
    try:
        strategy_class = getattr(strategy_module, strategy_name)
        if not issubclass(strategy_class, TradingStrategy):
            raise TypeError(f"Strategy '{strategy_name}' is not a valid subclass of TradingStrategy.")
        return strategy_class(strategy_params, metrics_writer)
    except AttributeError:
        logger.critical(f"Strategy class '{strategy_name}' not found.")
        raise ValueError(f"Unknown strategy: {strategy_name}")
    except TypeError as e:
        logger.critical(f"Invalid strategy class.", error=str(e))
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
