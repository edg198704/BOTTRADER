import asyncio
import logging
import time
import json
from typing import Dict, Any, Optional, List

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

    async def _fetch_and_process_market_data(self) -> Optional[Dict[str, Any]]:
        """Fetches and returns a dictionary with OHLCV and latest price."""
        try:
            market_data = await self.exchange_api.get_market_data(self.symbol)
            if not market_data or not market_data.get('ohlcv') or not market_data.get('lastPrice'):
                logger.warning(f"Received invalid or empty market data for {self.symbol}")
                return None
            logger.debug(f"Fetched {len(market_data['ohlcv'])} candles for {self.symbol}")
            return market_data
        except Exception as e:
            logger.error(f"Error fetching market data for {self.symbol}: {e}", exc_info=True)
            return None

    async def _execute_trade_action(self, action: Dict[str, Any], ohlcv_df):
        """Executes a trade action (BUY/SELL/CLOSE)."""
        action_type = action.get('action')
        symbol = self.symbol

        if action_type in ['BUY', 'SELL']:
            current_price_data = await self.exchange_api.get_market_data(symbol)
            current_price = float(current_price_data.get('lastPrice', 0))
            if current_price == 0:
                logger.error(f"Could not get current price for {symbol} to execute trade.")
                return

            # 1. Calculate position size from risk manager
            confidence = action.get('confidence', 0.5) # Strategy should provide confidence
            portfolio_value = self.risk_manager.initial_capital + self.position_manager.get_daily_realized_pnl() + self.position_manager.get_total_unrealized_pnl()
            quantity = self.risk_manager.calculate_position_size(symbol, current_price, confidence, portfolio_value)

            # 2. Pre-trade risk check
            if not self.risk_manager.check_trade_allowed(symbol, action_type, quantity, current_price):
                logger.warning(f"Trade {action_type} {quantity} {symbol} denied by risk manager.")
                return

            try:
                # 3. Place order
                order_response = await self.exchange_api.place_order(symbol, action_type, 'MARKET', quantity)
                logger.info(f"Order placed: {order_response}")
                if order_response and order_response.get('status') == 'FILLED':
                    entry_price = float(order_response['cummulativeQuoteQty']) / float(order_response['executedQty'])
                    
                    # 4. Calculate SL/TP
                    stop_loss = self.risk_manager.calculate_stop_loss(symbol, entry_price, action_type, ohlcv_df)
                    take_profit_levels = self.risk_manager.calculate_take_profit_levels(entry_price, action_type, confidence)

                    # 5. Persist position
                    self.position_manager.add_position(
                        symbol=symbol,
                        side=action_type,
                        quantity=float(order_response['executedQty']),
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit_levels=take_profit_levels
                    )
            except Exception as e:
                logger.error(f"Error placing {action_type} order for {symbol}: {e}", exc_info=True)

        elif action_type == 'CLOSE':
            position_id = action.get('position_id')
            await self._close_position_by_id(position_id)

    async def _close_position_by_id(self, position_id: int, reason: str = "strategy signal"):
        position = next((p for p in self.position_manager.get_open_positions() if p.id == position_id), None)
        if not position:
            logger.warning(f"Position {position_id} not found or already closed.")
            return

        close_side = 'SELL' if position.side == 'BUY' else 'BUY'
        try:
            order_response = await self.exchange_api.place_order(position.symbol, close_side, 'MARKET', position.quantity)
            logger.info(f"Closing order for position {position_id} ({reason}): {order_response}")
            if order_response and order_response.get('status') == 'FILLED':
                close_price = float(order_response['cummulativeQuoteQty']) / float(order_response['executedQty'])
                self.position_manager.close_position(position_id, close_price)
        except Exception as e:
            logger.error(f"Error closing position {position_id} for {position.symbol}: {e}", exc_info=True)

    async def run(self):
        """Main execution loop of the trading bot."""
        self.running = True
        logger.info(f"Starting TradingBot for {self.symbol}...")

        while self.running:
            start_time = time.monotonic()
            try:
                # 1. Fetch Market Data
                market_data = await self._fetch_and_process_market_data()
                if market_data is None:
                    await asyncio.sleep(self.trade_interval_seconds)
                    continue
                
                ohlcv_data = market_data['ohlcv']
                current_price = float(market_data['lastPrice'])
                ohlcv_df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

                # 2. Update Open Positions PnL
                open_positions = self.position_manager.get_open_positions(self.symbol)
                for pos in open_positions:
                    self.position_manager.update_position_pnl(pos.id, current_price)

                # 3. Update Risk Metrics & Trailing Stops
                self.risk_manager.update_risk_metrics()
                self.risk_manager.update_trailing_stops()
                if self.risk_manager.is_trading_halted:
                    logger.warning("Trading halted by risk manager. Monitoring only.")
                    await asyncio.sleep(self.trade_interval_seconds)
                    continue

                # 4. Check for SL/TP hits
                for pos in open_positions:
                    if (pos.side == 'BUY' and current_price <= pos.stop_loss) or (pos.side == 'SELL' and current_price >= pos.stop_loss):
                        logger.info(f"Stop loss hit for position {pos.id} at price {current_price}")
                        await self._close_position_by_id(pos.id, reason="stop loss")
                        continue # Re-fetch open positions after closing one
                    # (TP logic would be more complex with multiple levels, simplified here)

                # Re-fetch open positions after any SL/TP closures
                open_positions = self.position_manager.get_open_positions(self.symbol)

                # 5. Strategy Analysis for New Trades
                trade_signal = await self.strategy.analyze_market(ohlcv_data, open_positions)
                if trade_signal:
                    logger.info(f"Strategy generated new trade signal: {trade_signal}")
                    await self._execute_trade_action(trade_signal, ohlcv_df)

                # 6. Position Management from Strategy
                position_management_actions = await self.strategy.manage_positions(ohlcv_data, open_positions)
                for action in position_management_actions:
                    logger.info(f"Strategy generated position management action: {action}")
                    await self._execute_trade_action(action, ohlcv_df)

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
