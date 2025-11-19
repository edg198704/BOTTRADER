import asyncio
import pandas as pd
import os
import shutil
from datetime import datetime, timedelta
from typing import Dict
from concurrent.futures import ProcessPoolExecutor

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
from bot_core.utils import Clock, PerformanceMetrics

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
    config.database.path = "backtest_ledger.db"
    if os.path.exists(config.database.path):
        os.remove(config.database.path)

    # Use a temporary directory for backtest models to avoid overwriting production models
    backtest_model_path = "backtest_models"
    if os.path.exists(backtest_model_path):
        shutil.rmtree(backtest_model_path)
    os.makedirs(backtest_model_path, exist_ok=True)
    
    # Inject the temp path into the strategy params
    if hasattr(config.strategy.params, 'model_path'):
        config.strategy.params.model_path = backtest_model_path

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
    
    data_handler = DataHandler(exchange_api, config, shared_latest_prices)
    
    position_manager = PositionManager(config.database, config.backtest.initial_balance, alert_system)
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
    
    position_monitor.set_close_position_callback(bot._close_position)

    # 4. Run Setup
    await bot.setup()

    # 5. Main Backtest Loop
    start_times = [df.index[0] for df in historical_data.values()]
    end_times = [df.index[-1] for df in historical_data.values()]
    start_time = max(start_times) + timedelta(hours=24) # Warmup period
    end_time = min(end_times)
    
    current_time = start_time
    interval = timedelta(seconds=config.strategy.interval_seconds)
    
    logger.info(f"Starting backtest from {start_time} to {end_time}")
    
    # Executor for AI training simulation
    process_executor = ProcessPoolExecutor(max_workers=2)

    try:
        while current_time <= end_time:
            Clock.set_time(current_time)
            
            # Update DataHandler (Simulates receiving new candles)
            for symbol in config.strategy.symbols:
                await data_handler.update_symbol_data(symbol)
            
            # --- AI Retraining Simulation ---
            # Check if strategy needs retraining based on the simulated time
            for symbol in config.strategy.symbols:
                if strategy.needs_retraining(symbol):
                    logger.info(f"[Backtest] Retraining model for {symbol} at {current_time}")
                    limit = strategy.get_training_data_limit()
                    if limit > 0:
                        # Fetch data available UP TO current_time
                        training_df = await data_handler.fetch_full_history_for_symbol(symbol, limit)
                        if training_df is not None and not training_df.empty:
                            await strategy.retrain(symbol, training_df, process_executor)

            # Run Bot Logic
            for symbol in config.strategy.symbols:
                await bot.process_symbol_tick(symbol)
                
            # Run Position Monitor (Check SL/TP)
            open_positions = await position_manager.get_all_open_positions()
            for pos in open_positions:
                await position_monitor._check_position(pos)
                
            # Advance Time
            current_time += interval
            
            # Progress Log
            if current_time.minute == 0 and current_time.second == 0:
                 equity = position_manager.get_portfolio_value(shared_latest_prices, open_positions)
                 print(f"\rProgress: {current_time} | Equity: ${equity:.2f}", end="")

    finally:
        process_executor.shutdown()
        print("\n")

    # 6. Final Report
    logger.info("Backtest Complete. Generating Report...")
    
    closed_positions = await position_manager.get_all_closed_positions()
    trades_data = []
    for pos in closed_positions:
        trades_data.append({
            'symbol': pos.symbol,
            'side': pos.side,
            'entry_price': pos.entry_price,
            'close_price': pos.close_price,
            'quantity': pos.quantity,
            'pnl': pos.pnl,
            'open_timestamp': pos.open_timestamp,
            'close_timestamp': pos.close_timestamp
        })

    metrics = PerformanceMetrics.calculate(trades_data, config.backtest.initial_balance)
    
    print("="*40)
    print("       BACKTEST PERFORMANCE REPORT       ")
    print("="*40)
    print(f"Initial Capital:   ${config.backtest.initial_balance:,.2f}")
    print(f"Final Equity:      ${metrics['final_equity']:,.2f}")
    print(f"Total Return:      {metrics['total_return_pct']}%")
    print(f"Total Trades:      {metrics['total_trades']}")
    print(f"Win Rate:          {metrics['win_rate']}%")
    print(f"Profit Factor:     {metrics['profit_factor']}")
    print(f"Sharpe Ratio:      {metrics['sharpe_ratio']}")
    print(f"Sortino Ratio:     {metrics['sortino_ratio']}")
    print(f"Max Drawdown:      {metrics['max_drawdown']}%")
    print("="*40)

    # Cleanup
    if os.path.exists(backtest_model_path):
        shutil.rmtree(backtest_model_path)
        logger.info("Cleaned up backtest models.")

if __name__ == "__main__":
    asyncio.run(run_backtest())
