import asyncio
import pandas as pd
import os
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any

from config_loader import load_config
from bot_core.logger import setup_logging, get_logger
from bot_core.exchange_api import BacktestExchangeAPI, CCXTExchangeAPI
from bot_core.data_handler import DataHandler, create_dataframe
from bot_core.position_manager import PositionManager
from bot_core.risk_manager import RiskManager
from bot_core.strategy import TradingStrategy
from bot_core import strategy as strategy_module
from bot_core.bot import TradingBot
from bot_core.monitoring import HealthChecker, AlertSystem
from bot_core.order_sizer import OrderSizer
from bot_core.position_monitor import PositionMonitor
from bot_core.order_lifecycle_manager import OrderLifecycleManager
from bot_core.trade_executor import TradeExecutor
from bot_core.utils import Clock
from bot_core.reporting import PerformanceAnalyzer
from bot_core.config import AIEnsembleStrategyParams
from bot_core.event_system import EventBus

logger = get_logger(__name__)

async def load_historical_data(config, exchange_api) -> Dict[str, pd.DataFrame]:
    """Loads historical data for backtesting. Tries cache first, then downloads."""
    data = {}
    # Use a temporary real exchange API to download data if needed
    real_exchange = CCXTExchangeAPI(config.exchange)
    
    for symbol in config.strategy.symbols:
        logger.info(f"Loading data for {symbol}...")
        # Try to load from a local CSV first
        safe_symbol = symbol.replace('/', '_')
        path = f"backtest_data/{safe_symbol}.csv"
        os.makedirs("backtest_data", exist_ok=True)
        
        if os.path.exists(path):
            df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
            logger.info(f"Loaded {len(df)} rows from {path}")
        else:
            logger.info(f"Downloading history for {symbol}...")
            # Download 10000 candles (approx 35 days of 5m data)
            ohlcv = await real_exchange.get_market_data(symbol, config.strategy.timeframe, 10000)
            df = create_dataframe(ohlcv)
            if df is not None:
                df.to_csv(path)
                logger.info(f"Saved {len(df)} rows to {path}")
            else:
                logger.error(f"Failed to download data for {symbol}")
                continue
        
        data[symbol] = df
    
    await real_exchange.close()
    return data

async def run_backtest():
    setup_logging("INFO", "logs/backtest.log", False)
    config = load_config('config_enterprise.yaml')
    
    # --- Backtest Configuration Overrides ---
    # 1. Database: Use a separate DB for backtesting
    config.database.path = "backtest_ledger.db"
    if os.path.exists(config.database.path):
        os.remove(config.database.path)

    # 2. AI Models: Use a separate directory for backtest models to avoid overwriting prod
    if isinstance(config.strategy.params, AIEnsembleStrategyParams):
        config.strategy.params.model_path = config.backtest.model_path
        # Clean up previous backtest models
        if os.path.exists(config.backtest.model_path):
            shutil.rmtree(config.backtest.model_path)
        os.makedirs(config.backtest.model_path, exist_ok=True)
        logger.info(f"Using backtest model path: {config.backtest.model_path}")

    # 3. Optimizer: Use a separate state file for backtesting
    config.optimizer.state_file_path = "backtest_optimizer_state.json"
    if os.path.exists(config.optimizer.state_file_path):
        os.remove(config.optimizer.state_file_path)

    # 1. Load Data
    historical_data = await load_historical_data(config, None)
    if not historical_data:
        logger.error("No historical data found. Exiting.")
        return

    # 2. Initialize Backtest Exchange
    initial_balances = {"USDT": config.backtest.initial_balance}
    exchange_api = BacktestExchangeAPI(historical_data, initial_balances, config.backtest)

    # 3. Initialize Components
    shared_latest_prices = {}
    alert_system = AlertSystem()
    event_bus = EventBus()
    
    data_handler = DataHandler(exchange_api, config, shared_latest_prices)
    
    position_manager = PositionManager(config.database, config.backtest.initial_balance, alert_system)
    await position_manager.initialize()

    risk_manager = RiskManager(config.risk_management, position_manager, data_handler, alert_system)
    await risk_manager.initialize()

    # Load Strategy (with the modified config)
    strategy_class = getattr(strategy_module, config.strategy.params.name)
    strategy = strategy_class(config.strategy.params)

    order_sizer = OrderSizer()
    health_checker = HealthChecker()
    
    position_monitor = PositionMonitor(config, position_manager, risk_manager, data_handler, shared_latest_prices)
    
    order_lifecycle_manager = OrderLifecycleManager(exchange_api, config.execution, shared_latest_prices)
    
    trade_executor = TradeExecutor(
        config, exchange_api, position_manager, risk_manager, order_sizer, 
        order_lifecycle_manager, alert_system, shared_latest_prices, 
        market_details={}, # Will be loaded in setup
        data_handler=data_handler, # Injected for ATR calculation
        event_bus=event_bus
    )

    bot = TradingBot(
        config, exchange_api, data_handler, strategy, position_manager, risk_manager,
        health_checker, position_monitor, trade_executor, alert_system, shared_latest_prices, event_bus
    )
    
    position_monitor.set_close_position_callback(bot._close_position)

    # 4. Run Setup
    await bot.setup()

    # 5. Main Backtest Loop
    start_times = [df.index[0] for df in historical_data.values()]
    end_times = [df.index[-1] for df in historical_data.values()]
    
    # Warmup period: Allow enough data for indicators and initial training
    warmup_candles = max(config.data_handler.history_limit, 500)
    # Estimate warmup time based on timeframe (approximate)
    # We'll just start 24h in, assuming that's enough for 5m candles (288 candles)
    start_time = max(start_times) + timedelta(hours=24) 
    end_time = min(end_times)
    
    current_time = start_time
    interval = timedelta(seconds=config.strategy.interval_seconds)
    
    logger.info(f"Starting backtest from {start_time} to {end_time}")
    
    # --- Initial Training / Warmup ---
    # Set the clock to start time so data handler fetches data up to this point
    Clock.set_time(current_time)
    
    # Pre-load data into DataHandler buffers for the warmup period
    for symbol in config.strategy.symbols:
        await data_handler.update_symbol_data(symbol)

    # If it's an AI strategy, perform initial training now
    if isinstance(config.strategy.params, AIEnsembleStrategyParams):
        logger.info("Performing initial AI model training (Walk-Forward Warmup)...")
        for symbol in config.strategy.symbols:
            training_limit = strategy.get_training_data_limit()
            # Fetch full history available up to current_time
            training_df = await data_handler.fetch_full_history_for_symbol(symbol, training_limit)
            if training_df is not None and not training_df.empty:
                # Use the bot's executor to train
                await strategy.retrain(symbol, training_df, bot.process_executor)
            else:
                logger.warning(f"Insufficient data for initial training of {symbol}")

    equity_curve: List[Dict[str, Any]] = []
    last_optimization_time = current_time

    try:
        while current_time <= end_time:
            Clock.set_time(current_time)
            
            # 1. Update Data
            for symbol in config.strategy.symbols:
                await data_handler.update_symbol_data(symbol)
            
            # 2. Check for Retraining (Walk-Forward Analysis)
            # We do this BEFORE processing ticks to ensure the model is up-to-date
            for symbol in config.strategy.symbols:
                if strategy.needs_retraining(symbol):
                    logger.info(f"Retraining triggered for {symbol} at {current_time}")
                    training_limit = strategy.get_training_data_limit()
                    training_df = await data_handler.fetch_full_history_for_symbol(symbol, training_limit)
                    if training_df is not None and not training_df.empty:
                        await strategy.retrain(symbol, training_df, bot.process_executor)

            # 3. Run Strategy Optimizer (Self-Optimization Simulation)
            if config.optimizer.enabled:
                time_since_opt = (current_time - last_optimization_time).total_seconds()
                if time_since_opt >= (config.optimizer.interval_hours * 3600):
                    logger.info(f"Running Strategy Optimizer at {current_time}...")
                    await bot.optimizer.optimize()
                    last_optimization_time = current_time

            # 4. Run Bot Logic
            for symbol in config.strategy.symbols:
                await bot.process_symbol_tick(symbol)
                
            # 5. Run Position Monitor
            open_positions = await position_manager.get_all_open_positions()
            for pos in open_positions:
                await position_monitor._check_position(pos)
                
            # 6. Record Equity
            equity = position_manager.get_portfolio_value(shared_latest_prices, open_positions)
            equity_curve.append({'timestamp': current_time, 'equity': equity})

            # 7. Advance Time
            current_time += interval
            
            if current_time.hour == 0 and current_time.minute == 0 and current_time.second == 0:
                 logger.info(f"Progress: {current_time} | Equity: ${equity:.2f}")

    finally:
        # Ensure we shut down the bot's executor to prevent hanging
        await bot.stop()

    # 8. Final Report
    logger.info("Backtest Complete. Generating Report...")
    
    closed_positions = await position_manager.get_all_closed_positions()
    
    metrics = PerformanceAnalyzer.generate_report(
        closed_positions,
        equity_curve,
        config.backtest.initial_balance
    )
    
    PerformanceAnalyzer.print_report(metrics)

if __name__ == "__main__":
    asyncio.run(run_backtest())
