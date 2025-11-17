import asyncio
import logging
import time
import pandas as pd
from typing import Dict, Any, Optional
import ccxt

from bot_core.exchange_api import ExchangeAPI
from bot_core.position_manager import PositionManager
from bot_core.risk_manager import RiskManager
from bot_core.strategy import TradingStrategy
from bot_core.config import BotConfig
from bot_core.execution_handler import ExecutionHandler
from bot_core.data_handler import create_dataframe, calculate_technical_indicators

logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, config: BotConfig, exchange_api: ExchangeAPI,
                 strategy: TradingStrategy, position_manager: PositionManager, 
                 risk_manager: RiskManager, execution_handler: ExecutionHandler):
        self.config = config
        self.exchange_api = exchange_api
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.strategy = strategy
        self.execution_handler = execution_handler
        self.symbol = self.strategy.symbol
        self.trade_interval_seconds = self.strategy.interval_seconds
        self.running = False
        logger.info(f"TradingBot initialized for symbol: {self.symbol}, interval: {self.trade_interval_seconds}s")

    async def _fetch_and_process_market_data(self) -> Optional[Dict[str, Any]]:
        """Fetches, processes, and enriches market data with technical indicators."""
        try:
            # Fetch raw data
            ohlcv_data_list = await self.exchange_api.get_market_data(self.symbol, '1h', 200)
            ticker_data = await self.exchange_api.get_market_data(self.symbol)
            
            if not isinstance(ohlcv_data_list, list) or not ticker_data or not ticker_data.get('lastPrice'):
                logger.warning(f"Received invalid or empty market data for {self.symbol}")
                return None
            
            # Create DataFrame
            df = create_dataframe(ohlcv_data_list)
            if df is None:
                return None

            # Calculate indicators
            df_with_indicators = calculate_technical_indicators(df)

            logger.debug(f"Fetched and processed {len(df_with_indicators)} candles for {self.symbol}")
            return {"ohlcv_df": df_with_indicators, "last_price": float(ticker_data['lastPrice'])}
        except Exception as e:
            logger.error(f"Error fetching or processing market data for {self.symbol}: {e}", exc_info=True)
            return None

    async def _execute_trade_action(self, action: Dict[str, Any], ohlcv_df: pd.DataFrame):
        """Creates a trade proposal and delegates to the ExecutionHandler."""
        trade_proposal = {
            'symbol': self.symbol,
            'action': action.get('action'),
            'confidence': action.get('confidence')
        }
        await self.execution_handler.execute_trade_proposal(trade_proposal, ohlcv_df)

    async def _close_position_by_id(self, position_id: int, reason: str = "strategy signal"):
        position = next((p for p in self.position_manager.get_open_positions() if p.id == position_id), None)
        if not position:
            logger.warning(f"Attempted to close position {position_id}, but it was not found.")
            return

        close_side = 'SELL' if position.side == 'BUY' else 'BUY'
        try:
            order_response = await self.exchange_api.place_order(position.symbol, close_side, 'MARKET', position.quantity)
            logger.info(f"Closing order for position {position_id} ({reason}): {order_response}")
            if order_response and order_response.get('status') == 'FILLED':
                close_price = float(order_response['cummulativeQuoteQty']) / float(order_response['executedQty'])
                self.position_manager.close_position(position_id, close_price)
            else:
                logger.error(f"Failed to close position {position_id}. Order status: {order_response.get('status')}")
        except Exception as e:
            logger.error(f"Error closing position {position_id} for {position.symbol}: {e}", exc_info=True)

    async def run(self):
        """Main execution loop of the trading bot."""
        self.running = True
        logger.info(f"Starting TradingBot for {self.symbol}...")

        while self.running:
            start_time = time.monotonic()
            try:
                market_data = await self._fetch_and_process_market_data()
                if not market_data:
                    await asyncio.sleep(self.trade_interval_seconds)
                    continue
                
                ohlcv_df = market_data['ohlcv_df']
                current_price = market_data['last_price']

                open_positions = self.position_manager.get_open_positions(self.symbol)
                for pos in open_positions:
                    self.position_manager.update_position_pnl(pos.id, current_price)
                
                self.risk_manager.update_risk_metrics()
                self.risk_manager.update_trailing_stops()

                if self.risk_manager.is_trading_halted:
                    logger.warning("Trading halted by risk manager. Monitoring only.")
                    await asyncio.sleep(self.trade_interval_seconds)
                    continue

                positions_to_check = self.position_manager.get_open_positions(self.symbol)
                for pos in positions_to_check:
                    if (pos.side == 'BUY' and current_price <= pos.stop_loss) or (pos.side == 'SELL' and current_price >= pos.stop_loss):
                        logger.info(f"Stop loss hit for position {pos.id} at price {current_price}")
                        await self._close_position_by_id(pos.id, reason="stop loss")
                        continue

                open_positions = self.position_manager.get_open_positions(self.symbol)
                
                trade_signal = await self.strategy.analyze_market(ohlcv_df, open_positions)
                if trade_signal:
                    logger.info(f"Strategy generated new trade signal: {trade_signal}")
                    await self._execute_trade_action(trade_signal, ohlcv_df)

                position_management_actions = await self.strategy.manage_positions(ohlcv_df, open_positions)
                for action in position_management_actions:
                    if action.get('action') == 'CLOSE':
                        logger.info(f"Strategy generated position management action: {action}")
                        await self._close_position_by_id(action.get('position_id'))

            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, asyncio.TimeoutError) as e:
                logger.warning(f"Network/Exchange issue in main loop: {e}. Retrying in 30s...")
                await asyncio.sleep(30)
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error in main loop: {e}. Retrying in 60s...")
                await asyncio.sleep(60)
            except Exception as e:
                logger.critical(f"Unhandled exception in main bot loop: {e}", exc_info=True)
                await asyncio.sleep(self.trade_interval_seconds)

            finally:
                elapsed_time = time.monotonic() - start_time
                sleep_duration = max(0, self.trade_interval_seconds - elapsed_time)
                await asyncio.sleep(sleep_duration)

    def stop(self):
        """Stops the trading bot gracefully."""
        logger.info("Stopping TradingBot...")
        self.running = False
