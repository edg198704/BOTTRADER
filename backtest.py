import asyncio
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Dict

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
    
    # Override DB path for backtest to avoid messing up live DB
    config.database.path = "backtest_ledger.db"
    if os.path.exists(config.database.path):
        os.remove(config.database.path)

    # 1. Load Data
    historical_data = await load_historical_data(config, None)
    if not historical_data:
        logger.error("No historical data found. Exiting.")
        return

    # 2. Initialize Backtest Exchange
    initial_balances = {"USDT": config.initial_capital}
    exchange_api = BacktestExchangeAPI(historical_data, initial_balances)

    # 3. Initialize Components
    shared_latest_prices = {}
    alert_system = AlertSystem()
    
    data_handler = DataHandler(exchange_api, config, shared_latest_prices)
    # Pre-load data handler buffers manually since we won't run its loop
    # We don't need to do anything yet, update_symbol_data will handle it

    position_manager = PositionManager(config.database, config.initial_capital, alert_system)
    await position_manager.initialize()

    risk_manager = RiskManager(config.risk_management, position_manager, data_handler, alert_system)
    await risk_manager.initialize()

    # Load Strategy
    strategy_class = getattr(strategy_module, config.strategy.params.name)
    strategy = strategy_class(config.strategy.params)

    order_sizer = OrderSizer()
    health_checker = HealthChecker()
    
    position_monitor = PositionMonitor(config, position_manager, risk_manager, shared_latest_prices)
    
    order_lifecycle_manager = OrderLifecycleManager(exchange_api, config.execution, shared_latest_prices)
    
    trade_executor = TradeExecutor(
        config, exchange_api, position_manager, risk_manager, order_sizer, 
        order_lifecycle_manager, alert_system, shared_latest_prices, 
        market_details={} # Will be loaded in setup
    )

    bot = TradingBot(
        config, exchange_api, data_handler, strategy, position_manager, risk_manager,
        health_checker, position_monitor, trade_executor, alert_system, shared_latest_prices
    )
    
    # Set callback for position monitor
    position_monitor.set_close_position_callback(bot._close_position)

    # 4. Run Setup (Loads market details, etc.)
    await bot.setup()

    # 5. Main Backtest Loop
    # Determine start and end times based on data intersection
    start_times = [df.index[0] for df in historical_data.values()]
    end_times = [df.index[-1] for df in historical_data.values()]
    start_time = max(start_times) + timedelta(hours=24) # Warmup period
    end_time = min(end_times)
    
    current_time = start_time
    interval = timedelta(seconds=config.strategy.interval_seconds)
    
    logger.info(f"Starting backtest from {start_time} to {end_time}")
    
    while current_time <= end_time:
        Clock.set_time(current_time)
        
        # Update DataHandler for all symbols at this timestamp
        for symbol in config.strategy.symbols:
            await data_handler.update_symbol_data(symbol)
        
        # Run Bot Logic
        for symbol in config.strategy.symbols:
            await bot.process_symbol_tick(symbol)
            
        # Run Position Monitor (Check SL/TP)
        # We manually trigger the check logic
        open_positions = await position_manager.get_all_open_positions()
        for pos in open_positions:
            await position_monitor._check_position(pos)
            
        # Advance Time
        current_time += interval
        
        # Optional: Print progress every day
        if current_time.hour == 0 and current_time.minute == 0 and current_time.second == 0:
             equity = position_manager.get_portfolio_value(shared_latest_prices, open_positions)
             logger.info(f"Progress: {current_time} | Equity: ${equity:.2f}")

    # 6. Final Report
    final_equity = position_manager.get_portfolio_value(shared_latest_prices, await position_manager.get_all_open_positions())
    logger.info("Backtest Complete.")
    logger.info(f"Initial Capital: ${config.initial_capital}")
    logger.info(f"Final Equity: ${final_equity:.2f}")
    logger.info(f"Return: {((final_equity - config.initial_capital) / config.initial_capital) * 100:.2f}%")

if __name__ == "__main__":
    asyncio.run(run_backtest())
