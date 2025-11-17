import asyncio
import logging
import time
from typing import Dict, Any, Type, Optional, List

from bot_core.exchange_api import ExchangeAPI
from bot_core.position_manager import PositionManager
from bot_core.risk_manager import RiskManager
from bot_core.strategy import TradingStrategy
from bot_core.config import BotConfig

logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, config: BotConfig, exchange_api: ExchangeAPI,
                 strategy: TradingStrategy, position_manager: PositionManager, risk_manager: RiskManager):
        self.config = config
        self.exchange_api = exchange_api
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.strategy = strategy
        self.symbol = self.strategy.symbol
        self.trade_interval_seconds = self.strategy.interval_seconds
        self.running = False
        logger.info(f"TradingBot initialized for symbol: {self.symbol}, interval: {self.trade_interval_seconds}s")

    async def _fetch_and_process_market_data(self) -> Optional[List[List[float]]]:
        """Fetches OHLCV data for the strategy."""
        try:
            # Strategies might need historical data, so fetch a reasonable number of candles
            market_data = await self.exchange_api.get_market_data(self.symbol)
            if not market_data or not market_data.get('ohlcv'):
                logger.warning(f"Received invalid or empty market data for {self.symbol}")
                return None
            logger.debug(f"Fetched {len(market_data['ohlcv'])} candles for {self.symbol}")
            return market_data['ohlcv']
        except Exception as e:
            logger.error(f"Error fetching market data for {self.symbol}: {e}", exc_info=True)
            return None

    async def _execute_trade_action(self, action: Dict[str, Any]):
        """Executes a trade action (BUY/SELL/CLOSE)."""
        action_type = action.get('action')
        symbol = self.symbol
        quantity = action.get('quantity')
        price = action.get('price') # For limit orders
        order_type = action.get('order_type', 'MARKET')

        if action_type in ['BUY', 'SELL']:
            current_price = (await self.exchange_api.get_market_data(symbol)).get('lastPrice', 0)
            if not self.risk_manager.check_trade_allowed(symbol, action_type, quantity, float(current_price)):
                logger.warning(f"Trade {action_type} {quantity} {symbol} denied by risk manager.")
                return

            try:
                order_response = await self.exchange_api.place_order(symbol, action_type, order_type, quantity, price)
                logger.info(f"Order placed: {order_response}")
                if order_response.get('status') == 'FILLED':
                    entry_price = float(order_response['cummulativeQuoteQty']) / float(order_response['executedQty'])
                    self.position_manager.add_position(
                        symbol=symbol,
                        side=action_type,
                        quantity=float(order_response['executedQty']),
                        entry_price=entry_price
                    )
            except Exception as e:
                logger.error(f"Error placing {action_type} order for {symbol}: {e}", exc_info=True)

        elif action_type == 'CLOSE':
            position_id = action.get('position_id')
            position = next((p for p in self.position_manager.get_open_positions() if p.id == position_id), None)
            if not position:
                logger.warning(f"Position {position_id} not found or already closed.")
                return

            close_side = 'SELL' if position.side == 'BUY' else 'BUY'
            try:
                order_response = await self.exchange_api.place_order(symbol, close_side, order_type, position.quantity, price)
                logger.info(f"Closing order placed for position {position_id}: {order_response}")
                if order_response.get('status') == 'FILLED':
                    close_price = float(order_response['cummulativeQuoteQty']) / float(order_response['executedQty'])
                    self.position_manager.close_position(position_id, close_price)
            except Exception as e:
                logger.error(f"Error closing position {position_id} for {symbol}: {e}", exc_info=True)

    async def run(self):
        """Main execution loop of the trading bot."""
        self.running = True
        logger.info(f"Starting TradingBot for {self.symbol}...")

        while self.running:
            start_time = time.monotonic()
            try:
                # 1. Fetch Market Data (OHLCV)
                ohlcv_data = await self._fetch_and_process_market_data()
                if ohlcv_data is None:
                    await asyncio.sleep(self.trade_interval_seconds)
                    continue
                
                current_price = ohlcv_data[-1][4] # Get latest close price

                # 2. Update Open Positions PnL
                open_positions = self.position_manager.get_open_positions(self.symbol)
                for pos in open_positions:
                    self.position_manager.update_position_pnl(pos.id, current_price)

                # 3. Update Risk Metrics
                self.risk_manager.update_risk_metrics()
                if self.risk_manager.is_trading_halted:
                    logger.warning("Trading halted by risk manager. Monitoring only.")
                    await asyncio.sleep(self.trade_interval_seconds)
                    continue

                # 4. Strategy Analysis for New Trades
                trade_signal = await self.strategy.analyze_market(ohlcv_data, open_positions)
                if trade_signal:
                    logger.info(f"Strategy generated new trade signal: {trade_signal}")
                    await self._execute_trade_action(trade_signal)

                # 5. Position Management
                position_management_actions = await self.strategy.manage_positions(ohlcv_data, open_positions)
                for action in position_management_actions:
                    logger.info(f"Strategy generated position management action: {action}")
                    await self._execute_trade_action(action)

            except Exception as e:
                logger.critical(f"Unhandled exception in main bot loop: {e}", exc_info=True)

            finally:
                elapsed_time = time.monotonic() - start_time
                sleep_duration = max(0, self.trade_interval_seconds - elapsed_time)
                await asyncio.sleep(sleep_duration)

    def stop(self):
        """Stops the trading bot gracefully."""
        logger.info("Stopping TradingBot...")
        self.running = False
