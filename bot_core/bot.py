import asyncio
import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import pandas as pd

from bot_core.logger import get_logger, set_correlation_id
from bot_core.exchange_api import ExchangeAPI
from bot_core.position_manager import PositionManager, Position
from bot_core.risk_manager import RiskManager
from bot_core.strategy import TradingStrategy
from bot_core.config import BotConfig
from bot_core.data_handler import DataHandler
from bot_core.monitoring import HealthChecker, InfluxDBMetrics, AlertSystem
from bot_core.position_monitor import PositionMonitor
from bot_core.trade_executor import TradeExecutor

logger = get_logger(__name__)

class TradingBot:
    def __init__(self, config: BotConfig, exchange_api: ExchangeAPI, data_handler: DataHandler, 
                 strategy: TradingStrategy, position_manager: PositionManager, risk_manager: RiskManager,
                 health_checker: HealthChecker, position_monitor: PositionMonitor,
                 trade_executor: TradeExecutor, alert_system: AlertSystem,
                 shared_latest_prices: Dict[str, float],
                 metrics_writer: Optional[InfluxDBMetrics] = None,
                 shared_bot_state: Optional[Dict[str, Any]] = None):
        self.config = config
        self.exchange_api = exchange_api
        self.data_handler = data_handler
        self.strategy = strategy
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.health_checker = health_checker
        self.position_monitor = position_monitor
        self.trade_executor = trade_executor
        self.alert_system = alert_system
        self.metrics_writer = metrics_writer
        self.shared_bot_state = shared_bot_state if shared_bot_state is not None else {}
        
        self.running = False
        self.start_time = datetime.now(timezone.utc)
        self.latest_prices = shared_latest_prices
        self.market_details: Dict[str, Dict[str, Any]] = {}
        self.tasks: list[asyncio.Task] = []
        
        self._initialize_shared_state()
        logger.info("TradingBot orchestrator initialized.")

    def _initialize_shared_state(self):
        """Initializes the shared state dictionary for external components like Telegram."""
        self.shared_bot_state['status'] = 'initializing'
        self.shared_bot_state['start_time'] = self.start_time
        self.shared_bot_state['position_manager'] = self.position_manager
        self.shared_bot_state['risk_manager'] = self.risk_manager
        self.shared_bot_state['latest_prices'] = self.latest_prices
        self.shared_bot_state['config'] = self.config

    async def _load_market_details(self):
        """Fetches and caches exchange trading rules for all configured symbols."""
        logger.info("Loading market details for all symbols...")
        for symbol in self.config.strategy.symbols:
            try:
                details = await self.exchange_api.fetch_market_details(symbol)
                if details:
                    self.market_details[symbol] = details
                    logger.info("Successfully loaded market details", symbol=symbol)
                else:
                    logger.error("Failed to load market details for symbol, trading may fail.", symbol=symbol)
            except Exception as e:
                logger.critical("Could not load market details for symbol due to an exception.", symbol=symbol, error=str(e))
        # Pass the loaded market details to the trade executor
        self.trade_executor.market_details = self.market_details
        logger.info("Market details loading complete and passed to TradeExecutor.")

    async def run(self):
        """Main entry point to start all bot activities."""
        self.running = True
        self.shared_bot_state['status'] = 'running'
        logger.info("Starting TradingBot...", symbols=self.config.strategy.symbols)
        
        await self._load_market_details()

        # Start shared loops
        self.tasks.append(asyncio.create_task(self.data_handler.run()))
        self.tasks.append(asyncio.create_task(self._monitoring_loop()))
        self.tasks.append(asyncio.create_task(self.position_monitor.run()))
        self.tasks.append(asyncio.create_task(self._retraining_loop()))

        # Start a trading cycle for each symbol
        for symbol in self.config.strategy.symbols:
            self.tasks.append(asyncio.create_task(self._trading_cycle_for_symbol(symbol)))
        
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def _trading_cycle_for_symbol(self, symbol: str):
        """Runs the trading logic loop for a single symbol to find entry/exit signals."""
        logger.info("Starting trading cycle for symbol", symbol=symbol)
        while self.running:
            set_correlation_id()
            try:
                await self._run_single_trade_check(symbol)
            except asyncio.CancelledError:
                logger.info("Trading loop cancelled for symbol", symbol=symbol)
                break
            except Exception as e:
                logger.critical("Unhandled exception in trading cycle", symbol=symbol, error=str(e), exc_info=True)
            
            await asyncio.sleep(self.config.strategy.interval_seconds)

    async def _monitoring_loop(self):
        """Periodically runs health checks and portfolio monitoring."""
        while self.running:
            try:
                health_status = self.health_checker.get_health_status()
                logger.info("Health Check", **health_status)
                if self.metrics_writer and self.metrics_writer.enabled:
                    await self.metrics_writer.write_metric('health', fields=health_status)
                
                if not self.latest_prices:
                    await asyncio.sleep(5)
                    continue

                open_positions = self.position_manager.get_all_open_positions()
                portfolio_value = self.position_manager.get_portfolio_value(self.latest_prices, open_positions)
                
                daily_pnl = await self.position_manager.get_daily_realized_pnl()
                await self.risk_manager.update_portfolio_risk(portfolio_value, daily_pnl)

                # Update shared state for Telegram bot
                self.shared_bot_state['portfolio_equity'] = portfolio_value
                self.shared_bot_state['open_positions_count'] = len(open_positions)
                self.shared_bot_state['daily_pnl'] = daily_pnl

                if self.metrics_writer and self.metrics_writer.enabled:
                    await self.metrics_writer.write_metric('portfolio', fields={'equity': portfolio_value, 'daily_pnl': daily_pnl})

            except asyncio.CancelledError:
                logger.info("Monitoring loop cancelled.")
                break
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
            await asyncio.sleep(60) # Monitoring interval

    async def _retraining_loop(self):
        """Periodically checks if the strategy models need retraining and triggers it."""
        logger.info("Starting model retraining loop.")
        # Wait a bit on startup to let data load
        await asyncio.sleep(60)

        while self.running:
            try:
                for symbol in self.config.strategy.symbols:
                    if self.strategy.needs_retraining(symbol):
                        logger.info("Retraining needed for symbol, initiating process.", symbol=symbol)
                        
                        training_data_limit = self.strategy.get_training_data_limit()
                        if training_data_limit <= 0:
                            logger.info("Strategy does not require training data, skipping.", symbol=symbol, strategy=self.strategy.__class__.__name__)
                            continue

                        training_df = await self.data_handler.fetch_full_history_for_symbol(
                            symbol, training_data_limit
                        )

                        if training_df is not None and not training_df.empty:
                            # Run the potentially CPU-intensive training in a separate thread
                            # to avoid blocking the main async event loop.
                            loop = asyncio.get_running_loop()
                            await loop.run_in_executor(
                                None, 
                                self.strategy.retrain,
                                symbol,
                                training_df
                            )
                        else:
                            logger.error("Could not fetch training data, skipping retraining for now.", symbol=symbol)
                
                # Check again in an hour
                await asyncio.sleep(3600)

            except asyncio.CancelledError:
                logger.info("Retraining loop cancelled.")
                break
            except Exception as e:
                logger.error("Error in retraining loop", error=str(e), exc_info=True)
                await asyncio.sleep(3600) # Wait before retrying on error

    async def _run_single_trade_check(self, symbol: str):
        logger.debug("Starting new trade check", symbol=symbol)

        if self.risk_manager.is_halted:
            logger.warning("Trading is halted by RiskManager.")
            return

        df_with_indicators = self.data_handler.get_market_data(symbol)
        if df_with_indicators is None:
            logger.warning("Could not get market data from handler.", symbol=symbol)
            return
        
        if symbol not in self.latest_prices:
            logger.warning("Latest price for symbol not available yet.", symbol=symbol)
            return

        position = self.position_manager.get_open_position(symbol)
        signal = await self.strategy.analyze_market(symbol, df_with_indicators, position)

        if signal: await self._handle_signal(signal, df_with_indicators, position)

    async def _handle_signal(self, signal: Dict, df_with_indicators: pd.DataFrame, position: Optional[Position]):
        """Delegates signal handling to the TradeExecutor."""
        await self.trade_executor.execute_trade_signal(signal, df_with_indicators, position)

    async def _close_position(self, position: Position, reason: str):
        """Delegates position closing to the TradeExecutor."""
        await self.trade_executor.close_position(position, reason)

    async def stop(self):
        logger.info("Stopping TradingBot...")
        self.running = False
        self.shared_bot_state['status'] = 'stopping'
        
        await self.data_handler.stop()
        await self.position_monitor.stop()

        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        await asyncio.gather(*[task for task in self.tasks if not task.done()], return_exceptions=True)

        if self.exchange_api: await self.exchange_api.close()
        if self.position_manager: self.position_manager.close()
        if self.metrics_writer: await self.metrics_writer.close()
        logger.info("TradingBot stopped gracefully.")
        self.shared_bot_state['status'] = 'stopped'
