import asyncio
import logging
import time
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import ccxt

from bot_core.exchange_api import ExchangeAPI
from bot_core.position_manager import PositionManager
from bot_core.risk_manager import RiskManager
from bot_core.strategy import TradingStrategy
from bot_core.config import BotConfig
from bot_core.execution_handler import ExecutionHandler
from bot_core.order_manager import OrderManager, FillEvent
from bot_core.data_handler import create_dataframe, calculate_technical_indicators

logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, config: BotConfig, exchange_api: ExchangeAPI,
                 strategy: TradingStrategy, position_manager: PositionManager, 
                 risk_manager: RiskManager, execution_handler: ExecutionHandler,
                 order_manager: OrderManager):
        self.config = config
        self.exchange_api = exchange_api
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.strategy = strategy
        self.execution_handler = execution_handler
        self.order_manager = order_manager
        self.symbol = self.strategy.symbol
        self.trade_interval_seconds = self.strategy.interval_seconds
        self.running = False
        self.is_halted = False
        self.start_time = datetime.now(timezone.utc)
        self.tasks = []
        logger.info(f"TradingBot initialized for symbol: {self.symbol}, interval: {self.trade_interval_seconds}s")

    def halt_trading(self):
        """Halts the bot from opening new positions."""
        logger.warning("Trading HALTED by external command.")
        self.is_halted = True

    def resume_trading(self):
        """Resumes normal trading operations."""
        logger.info("Trading RESUMED by external command.")
        self.is_halted = False

    def get_status(self) -> Dict[str, Any]:
        """Returns a dictionary with the current status of the bot."""
        uptime_delta = datetime.now(timezone.utc) - self.start_time
        return {
            "is_running": self.running,
            "is_halted": self.is_halted,
            "uptime": str(uptime_delta).split('.')[0],
            "equity": self.risk_manager.get_portfolio_value(),
            "unrealized_pnl": self.position_manager.get_total_unrealized_pnl(),
            "open_positions_count": len(self.position_manager.get_open_positions(self.symbol)),
        }

    def get_open_positions_summary(self) -> List[Dict[str, Any]]:
        """Returns a summary of all open positions."""
        positions = self.position_manager.get_open_positions()
        summary = []
        for pos in positions:
            pnl_pct = (pos.unrealized_pnl / (pos.entry_price * pos.quantity)) * 100 if pos.entry_price > 0 else 0
            summary.append({
                "symbol": pos.symbol,
                "side": pos.side,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "pnl": pos.unrealized_pnl,
                "pnl_pct": pnl_pct
            })
        return summary

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Returns key performance and risk metrics."""
        return {
            "daily_realized_pnl": self.position_manager.get_daily_realized_pnl(),
            "total_unrealized_pnl": self.position_manager.get_total_unrealized_pnl(),
            "portfolio_value": self.risk_manager.get_portfolio_value(),
            "portfolio_drawdown": (self.risk_manager.get_portfolio_value() - self.config.initial_capital) / self.config.initial_capital
        }

    async def _fetch_and_process_market_data(self) -> Optional[Dict[str, Any]]:
        """Fetches, processes, and enriches market data with technical indicators."""
        try:
            ohlcv_data_list = await self.exchange_api.get_market_data(self.symbol, '1h', 200)
            ticker_data = await self.exchange_api.get_ticker_data(self.symbol)
            
            if not isinstance(ohlcv_data_list, list) or not ticker_data or not ticker_data.get('lastPrice'):
                logger.warning(f"Received invalid or empty market data for {self.symbol}")
                return None
            
            df = create_dataframe(ohlcv_data_list)
            if df is None: return None

            df_with_indicators = calculate_technical_indicators(df)
            logger.debug(f"Fetched and processed {len(df_with_indicators)} candles for {self.symbol}")
            return {"ohlcv_df": df_with_indicators, "last_price": float(ticker_data['lastPrice'])}
        except Exception as e:
            logger.error(f"Error fetching or processing market data for {self.symbol}: {e}", exc_info=True)
            return None

    async def _process_fill_events(self):
        """Processes fill events from the OrderManager to update positions."""
        fills = await self.order_manager.get_fill_events()
        for fill in fills:
            logger.info(f"Processing fill event: {fill}")
            try:
                self.position_manager.update_from_fill(fill)
            except Exception as e:
                logger.error(f"Error processing fill event in PositionManager: {e}", exc_info=True)

    async def _close_position_by_id(self, position_id: int, reason: str = "strategy signal"):
        position = self.position_manager.get_position_by_id(position_id)
        if not position:
            logger.warning(f"Attempted to close position {position_id}, but it was not found.")
            return

        close_side = 'SELL' if position.side == 'BUY' else 'BUY'
        try:
            metadata = {"intent": "CLOSE", "position_id_to_close": position.id}
            await self.order_manager.submit_order(
                symbol=position.symbol, 
                side=close_side, 
                order_type='MARKET', 
                quantity=position.quantity,
                metadata=metadata
            )
            logger.info(f"Submitted closing order for position {position_id} ({reason})")
        except Exception as e:
            logger.error(f"Error submitting closing order for position {position_id}: {e}", exc_info=True)

    async def run(self):
        """Main execution loop of the trading bot."""
        self.running = True
        await self.order_manager.start()
        logger.info(f"Starting TradingBot for {self.symbol}...")

        while self.running:
            start_time = time.monotonic()
            try:
                await self._process_fill_events()

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

                if self.risk_manager.is_halted:
                    logger.warning("Trading halted by risk manager. Monitoring only.")
                
                positions_to_check = self.position_manager.get_open_positions(self.symbol)
                for pos in positions_to_check:
                    if pos.stop_loss and ((pos.side == 'BUY' and current_price <= pos.stop_loss) or \
                       (pos.side == 'SELL' and current_price >= pos.stop_loss)):
                        logger.info(f"Stop loss hit for position {pos.id} at price {current_price}")
                        await self._close_position_by_id(pos.id, reason="stop loss")
                        continue

                if not self.risk_manager.is_halted and not self.is_halted:
                    current_open_positions = self.position_manager.get_open_positions(self.symbol)
                    
                    trade_signal = await self.strategy.analyze_market(ohlcv_df, current_open_positions)
                    if trade_signal:
                        logger.info(f"Strategy generated new trade signal: {trade_signal}")
                        await self.execution_handler.execute_trade_proposal(trade_signal, ohlcv_df)

                    position_management_actions = await self.strategy.manage_positions(ohlcv_df, current_open_positions)
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

    async def stop(self):
        """Stops the trading bot gracefully."""
        logger.info("Stopping TradingBot...")
        self.running = False
        await self.order_manager.stop()
