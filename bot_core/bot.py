import asyncio
import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from bot_core.logger import get_logger, set_correlation_id
from bot_core.exchange_api import ExchangeAPI
from bot_core.position_manager import PositionManager
from bot_core.risk_manager import RiskManager
from bot_core.strategy import TradingStrategy
from bot_core.config import BotConfig
from bot_core.data_handler import create_dataframe, calculate_technical_indicators

logger = get_logger(__name__)

class TradingBot:
    def __init__(self, config: BotConfig, exchange_api: ExchangeAPI, strategy: TradingStrategy, 
                 position_manager: PositionManager, risk_manager: RiskManager):
        self.config = config
        self.exchange_api = exchange_api
        self.strategy = strategy
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        
        self.running = False
        self.start_time = datetime.now(timezone.utc)
        self.latest_prices: Dict[str, float] = {}
        logger.info("TradingBot orchestrator initialized.")

    async def run(self):
        """Main event loop of the trading bot."""
        self.running = True
        logger.info("Starting TradingBot main loop.")

        while self.running:
            set_correlation_id() # New ID for each trading cycle
            try:
                await self._run_trading_cycle()
            except asyncio.CancelledError:
                logger.info("Trading loop cancelled.")
                break
            except Exception as e:
                logger.critical("Unhandled exception in main trading loop", error=str(e), exc_info=True)
            
            await asyncio.sleep(self.config.strategy.interval_seconds)

    async def _run_trading_cycle(self):
        """Executes a single cycle of the trading logic for the configured symbol."""
        symbol = self.config.strategy.symbol
        logger.debug("Starting new trading cycle", symbol=symbol)

        # 1. Fetch market data
        try:
            ohlcv_data = await self.exchange_api.get_market_data(symbol, '1h', 200)
            ticker_data = await self.exchange_api.get_ticker_data(symbol)
            if not ohlcv_data or not ticker_data or not ticker_data.get('lastPrice'):
                logger.warning("Could not fetch complete market data.", symbol=symbol)
                return
            self.latest_prices[symbol] = float(ticker_data['lastPrice'])
        except Exception as e:
            logger.error("Failed to fetch market data", symbol=symbol, error=str(e))
            return

        # 2. Process data
        df = create_dataframe(ohlcv_data)
        if df is None:
            return
        df_with_indicators = calculate_technical_indicators(df)

        # 3. Update portfolio and risk
        portfolio_value = self.position_manager.get_portfolio_value(self.latest_prices, self.config.initial_capital)
        self.risk_manager.update_portfolio_risk(portfolio_value)

        if self.risk_manager.is_halted:
            logger.warning("Trading is halted by RiskManager.")
            return

        # 4. Get signal from strategy
        open_positions = self.position_manager.get_all_open_positions()
        signal = await self.strategy.analyze_market(df_with_indicators, open_positions)

        if not signal:
            return

        # 5. Execute trade based on signal
        await self._handle_signal(signal)

    async def _handle_signal(self, signal: Dict):
        action = signal.get('action')
        symbol = signal.get('symbol')
        current_price = self.latest_prices.get(symbol)

        if not all([action, symbol, current_price]):
            logger.warning("Received invalid signal", signal=signal)
            return

        position = self.position_manager.get_open_position(symbol)

        if action == 'BUY' and not position:
            # Calculate size and check risk
            portfolio_equity = self.position_manager.get_portfolio_value(self.latest_prices, self.config.initial_capital)
            position_size_usd = self.risk_manager.calculate_position_size(portfolio_equity)
            quantity = position_size_usd / current_price

            if self.risk_manager.check_trade_allowed(symbol, quantity, current_price):
                # Place order
                order_result = await self.exchange_api.place_order(symbol, 'BUY', 'MARKET', quantity)
                if order_result and order_result.get('orderId'):
                    self.position_manager.open_position(symbol, 'BUY', quantity, current_price)
        
        elif action == 'SELL' and position:
            # Place order to close position
            order_result = await self.exchange_api.place_order(symbol, 'SELL', 'MARKET', position.quantity)
            if order_result and order_result.get('orderId'):
                self.position_manager.close_position(symbol, current_price)

    async def stop(self):
        """Stops the trading bot gracefully."""
        logger.info("Stopping TradingBot...")
        self.running = False
        if self.exchange_api:
            await self.exchange_api.close()
        if self.position_manager:
            self.position_manager.close()
