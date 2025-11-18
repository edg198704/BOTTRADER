import asyncio
import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import pandas as pd

from bot_core.logger import get_logger, set_correlation_id
from bot_core.exchange_api import ExchangeAPI
from bot_core.position_manager import PositionManager
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
        """Main event loop of the trading bot."""
        self.running = True
        logger.info("Starting TradingBot main loop.")
        self.tasks.append(asyncio.create_task(self._trading_loop()))
        self.tasks.append(asyncio.create_task(self._position_monitoring_loop()))
        self.tasks.append(asyncio.create_task(self._health_monitoring_loop()))
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def _trading_loop(self):
        while self.running:
            set_correlation_id() # New ID for each trading cycle
            try:
                await self._run_trading_cycle()
            except asyncio.CancelledError:
                logger.info("Trading loop cancelled.")
                break
            except Exception as e:
                logger.critical("Unhandled exception in trading loop", error=str(e), exc_info=True)
            
            await asyncio.sleep(self.config.strategy.interval_seconds)

    async def _position_monitoring_loop(self):
        while self.running:
            try:
                open_positions = self.position_manager.get_all_open_positions()
                if not open_positions:
                    await asyncio.sleep(10) # Sleep longer if no positions to check
                    continue

                symbols_to_check = {p.symbol for p in open_positions}
                for symbol in symbols_to_check:
                    ticker_data = await self.exchange_api.get_ticker_data(symbol)
                    if ticker_data and ticker_data.get('lastPrice'):
                        self.latest_prices[symbol] = float(ticker_data['lastPrice'])
                
                for pos in open_positions:
                    current_price = self.latest_prices.get(pos.symbol)
                    if not current_price: continue

                    # Check stop-loss
                    if pos.stop_loss_price and pos.side == 'BUY' and current_price <= pos.stop_loss_price:
                        logger.info("Stop-loss triggered for position", symbol=pos.symbol, trigger_price=pos.stop_loss_price)
                        await self._close_position(pos.symbol, current_price, reason="Stop-Loss")

            except asyncio.CancelledError:
                logger.info("Position monitoring loop cancelled.")
                break
            except Exception as e:
                logger.error("Error in position monitoring loop", error=str(e))
            await asyncio.sleep(15) # Check positions every 15 seconds

    async def _health_monitoring_loop(self):
        while self.running:
            try:
                health_status = self.health_checker.get_health_status()
                if self.metrics_writer and self.metrics_writer.enabled:
                    await self.metrics_writer.write_metric('health', fields=health_status)
                
                portfolio_value = self.position_manager.get_portfolio_value(self.latest_prices, self.config.initial_capital)
                open_positions = self.position_manager.get_all_open_positions()
                self.risk_manager.update_portfolio_risk(portfolio_value, open_positions)

                if self.metrics_writer and self.metrics_writer.enabled:
                    await self.metrics_writer.write_metric('portfolio', fields={'equity': portfolio_value, 'open_positions': len(open_positions)})

            except asyncio.CancelledError:
                logger.info("Health monitoring loop cancelled.")
                break
            except Exception as e:
                logger.error("Error in health monitoring loop", error=str(e))
            await asyncio.sleep(60) # Monitoring interval

    async def _run_trading_cycle(self):
        symbol = self.config.strategy.symbol
        logger.debug("Starting new trading cycle", symbol=symbol)

        try:
            ohlcv_data = await self.exchange_api.get_market_data(symbol, '1h', 200)
            if not ohlcv_data:
                logger.warning("Could not fetch OHLCV data.", symbol=symbol)
                return
        except Exception as e:
            logger.error("Failed to fetch market data", symbol=symbol, error=str(e))
            return

        df = create_dataframe(ohlcv_data)
        if df is None: return
        df_with_indicators = calculate_technical_indicators(df)

        if self.risk_manager.is_halted:
            logger.warning("Trading is halted by RiskManager.")
            return

        open_positions = self.position_manager.get_all_open_positions()
        signal = await self.strategy.analyze_market(df_with_indicators, open_positions)

        if signal: await self._handle_signal(signal, df_with_indicators)

    async def _handle_signal(self, signal: Dict, df_with_indicators: pd.DataFrame):
        action = signal.get('action')
        symbol = signal.get('symbol')
        position = self.position_manager.get_open_position(symbol)

        if action == 'BUY' and not position:
            await self._open_position(symbol, 'BUY', df_with_indicators)
        
        elif action == 'SELL' and position:
            current_price = self.latest_prices.get(symbol)
            if current_price:
                await self._close_position(symbol, current_price, reason="Strategy Signal")

    async def _open_position(self, symbol: str, side: str, df: pd.DataFrame):
        open_positions = self.position_manager.get_all_open_positions()
        if not self.risk_manager.check_trade_allowed(symbol, open_positions):
            return

        ticker_data = await self.exchange_api.get_ticker_data(symbol)
        if not ticker_data or not ticker_data.get('lastPrice'):
            logger.error("Could not get current price to open position", symbol=symbol)
            return
        current_price = float(ticker_data['lastPrice'])
        self.latest_prices[symbol] = current_price

        portfolio_equity = self.position_manager.get_portfolio_value(self.latest_prices, self.config.initial_capital)
        position_size_usd = self.risk_manager.calculate_position_size(portfolio_equity)
        quantity = position_size_usd / current_price

        stop_loss_price = self.risk_manager.calculate_stop_loss(side, current_price, df)
        # Take profit can be added here similarly
        take_profit_price = current_price * 1.1 # Placeholder TP

        order_result = await self.exchange_api.place_order(symbol, side, 'MARKET', quantity)
        if order_result and order_result.get('orderId'):
            logger.info("Successfully placed market order", order=order_result)
            self.position_manager.open_position(symbol, side, quantity, current_price, stop_loss_price, take_profit_price)
        else:
            logger.error("Failed to place order", symbol=symbol, side=side)

    async def _close_position(self, symbol: str, price: float, reason: str):
        position = self.position_manager.get_open_position(symbol)
        if not position:
            logger.warning("Attempted to close a position that does not exist.", symbol=symbol)
            return

        order_result = await self.exchange_api.place_order(symbol, 'SELL', 'MARKET', position.quantity)
        if order_result and order_result.get('orderId'):
            closed_pos = self.position_manager.close_position(symbol, price)
            if closed_pos and self.metrics_writer and self.metrics_writer.enabled:
                await self.metrics_writer.write_metric(
                    'trade', 
                    fields={'pnl': closed_pos.pnl, 'close_price': closed_pos.close_price}, 
                    tags={'symbol': symbol, 'reason': reason}
                )
        else:
            logger.error("Failed to place closing order", symbol=symbol)

    async def stop(self):
        logger.info("Stopping TradingBot...")
        self.running = False
        for task in self.tasks:
            task.cancel()
        await asyncio.sleep(1) # Allow tasks to process cancellation
        if self.exchange_api: await self.exchange_api.close()
        if self.position_manager: self.position_manager.close()
        if self.metrics_writer: await self.metrics_writer.close()
