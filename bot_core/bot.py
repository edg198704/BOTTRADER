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
from bot_core.data_handler import create_dataframe, calculate_technical_indicators
from bot_core.monitoring import HealthChecker, InfluxDBMetrics

logger = get_logger(__name__)

class TradingBot:
    def __init__(self, config: BotConfig, exchange_api: ExchangeAPI, strategy: TradingStrategy, 
                 position_manager: PositionManager, risk_manager: RiskManager,
                 health_checker: HealthChecker, metrics_writer: Optional[InfluxDBMetrics] = None):
        self.config = config
        self.exchange_api = exchange_api
        self.strategy = strategy
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.health_checker = health_checker
        self.metrics_writer = metrics_writer
        
        self.running = False
        self.start_time = datetime.now(timezone.utc)
        self.latest_prices: Dict[str, float] = {}
        self.tasks: list[asyncio.Task] = []
        logger.info("TradingBot orchestrator initialized.")

    async def run(self):
        """Main entry point to start all bot activities."""
        self.running = True
        logger.info("Starting TradingBot...", symbols=self.config.strategy.symbols)
        
        # Start shared loops
        self.tasks.append(asyncio.create_task(self._monitoring_loop()))
        self.tasks.append(asyncio.create_task(self._ticker_update_loop()))
        self.tasks.append(asyncio.create_task(self._position_management_loop()))

        # Start a trading cycle for each symbol
        for symbol in self.config.strategy.symbols:
            self.tasks.append(asyncio.create_task(self._trading_cycle_for_symbol(symbol)))
        
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def _ticker_update_loop(self):
        """Periodically updates the latest prices for all configured symbols."""
        while self.running:
            try:
                for symbol in self.config.strategy.symbols:
                    ticker_data = await self.exchange_api.get_ticker_data(symbol)
                    if ticker_data and ticker_data.get('lastPrice'):
                        self.latest_prices[symbol] = float(ticker_data['lastPrice'])
                await asyncio.sleep(5) # Update prices more frequently for SL/TP checks
            except asyncio.CancelledError:
                logger.info("Ticker update loop cancelled.")
                break
            except Exception as e:
                logger.error("Error in ticker update loop", error=str(e))
                await asyncio.sleep(30) # Wait longer on error

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

    async def _position_management_loop(self):
        """Monitors open positions for stop-loss and take-profit triggers."""
        logger.info("Starting position management loop.")
        while self.running:
            try:
                open_positions = self.position_manager.get_all_open_positions()
                for pos in open_positions:
                    current_price = self.latest_prices.get(pos.symbol)
                    if not current_price:
                        continue

                    # Check for stop-loss
                    if pos.side == 'BUY' and pos.stop_loss_price and current_price <= pos.stop_loss_price:
                        logger.info("Stop-loss triggered for position", symbol=pos.symbol, price=current_price, sl=pos.stop_loss_price)
                        await self._close_position(pos, current_price, "Stop-Loss")
                    
                    # Check for take-profit
                    if pos.side == 'BUY' and pos.take_profit_price and current_price >= pos.take_profit_price:
                        logger.info("Take-profit triggered for position", symbol=pos.symbol, price=current_price, tp=pos.take_profit_price)
                        await self._close_position(pos, current_price, "Take-Profit")

            except asyncio.CancelledError:
                logger.info("Position management loop cancelled.")
                break
            except Exception as e:
                logger.error("Error in position management loop", error=str(e))
            await asyncio.sleep(5) # Check positions frequently

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
                portfolio_value = self.position_manager.get_portfolio_value(self.latest_prices, self.config.initial_capital, open_positions)
                self.risk_manager.update_portfolio_risk(portfolio_value)

                if self.metrics_writer and self.metrics_writer.enabled:
                    await self.metrics_writer.write_metric('portfolio', fields={'equity': portfolio_value})

            except asyncio.CancelledError:
                logger.info("Monitoring loop cancelled.")
                break
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
            await asyncio.sleep(60) # Monitoring interval

    async def _run_single_trade_check(self, symbol: str):
        logger.debug("Starting new trade check", symbol=symbol)

        if self.risk_manager.is_halted:
            logger.warning("Trading is halted by RiskManager circuit breaker.")
            return

        try:
            ohlcv_data = await self.exchange_api.get_market_data(symbol, '1h', 200)
            if not ohlcv_data:
                logger.warning("Could not fetch OHLCV data.", symbol=symbol)
                return
            if symbol not in self.latest_prices:
                logger.warning("Latest price for symbol not available yet.", symbol=symbol)
                return
        except Exception as e:
            logger.error("Failed to fetch market data", symbol=symbol, error=str(e))
            return

        df = create_dataframe(ohlcv_data)
        if df is None: return
        df_with_indicators = calculate_technical_indicators(df)

        position = self.position_manager.get_open_position(symbol)
        signal = await self.strategy.analyze_market(df_with_indicators, symbol, position)

        if signal: await self._handle_signal(signal, df_with_indicators)

    async def _handle_signal(self, signal: Dict, df_with_indicators: pd.DataFrame):
        action = signal.get('action')
        symbol = signal.get('symbol')
        current_price = self.latest_prices.get(symbol)

        if not all([action, symbol, current_price]):
            logger.warning("Received invalid signal", signal=signal)
            return

        position = self.position_manager.get_open_position(symbol)

        if action == 'BUY' and not position:
            open_positions = self.position_manager.get_all_open_positions()
            if not self.risk_manager.check_trade_allowed(symbol, open_positions):
                return

            portfolio_equity = self.position_manager.get_portfolio_value(self.latest_prices, self.config.initial_capital, open_positions)
            position_size_usd = self.risk_manager.calculate_position_size(portfolio_equity)
            quantity = position_size_usd / current_price

            stop_loss = self.risk_manager.calculate_stop_loss('BUY', current_price, df_with_indicators)
            take_profit = self.risk_manager.calculate_take_profit('BUY', current_price, stop_loss)

            order_result = await self.exchange_api.place_order(symbol, 'BUY', 'MARKET', quantity)
            if order_result and order_result.get('orderId'):
                self.position_manager.open_position(symbol, 'BUY', quantity, current_price, stop_loss, take_profit)
        
        elif action == 'SELL' and position:
            await self._close_position(position, current_price, "Strategy Signal")

    async def _close_position(self, position: Position, close_price: float, reason: str):
        order_result = await self.exchange_api.place_order(position.symbol, 'SELL', 'MARKET', position.quantity)
        if order_result and order_result.get('orderId'):
            self.position_manager.close_position(position.symbol, close_price, reason)

    async def stop(self):
        logger.info("Stopping TradingBot...")
        self.running = False
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        await asyncio.gather(*[task for task in self.tasks if not task.done()], return_exceptions=True)

        if self.exchange_api: await self.exchange_api.close()
        if self.position_manager: self.position_manager.close()
        if self.metrics_writer: await self.metrics_writer.close()
        logger.info("TradingBot stopped gracefully.")
