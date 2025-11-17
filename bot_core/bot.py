import asyncio
import time
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from bot_core.logger import get_logger
from bot_core.exchange_api import ExchangeAPI
from bot_core.position_manager import PositionManager
from bot_core.risk_manager import RiskManager
from bot_core.strategy import TradingStrategy
from bot_core.config import BotConfig
from bot_core.execution_handler import ExecutionHandler
from bot_core.order_manager import OrderManager
from bot_core.data_handler import DataHandler, MarketEvent, SignalEvent, OrderEvent, FillEvent

logger = get_logger(__name__)

class TradingBot:
    def __init__(self, config: BotConfig, data_handler: DataHandler, exchange_api: ExchangeAPI,
                 strategy: TradingStrategy, position_manager: PositionManager, 
                 risk_manager: RiskManager, execution_handler: ExecutionHandler,
                 order_manager: OrderManager):
        self.config = config
        self.event_queue = asyncio.Queue()
        self.data_handler = data_handler
        self.exchange_api = exchange_api
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.strategy = strategy
        self.execution_handler = execution_handler
        self.order_manager = order_manager
        
        self.running = False
        self.is_halted = False
        self.start_time = datetime.now(timezone.utc)
        self.latest_market_data: Dict[str, Dict[str, Any]] = {}
        logger.info("TradingBot initialized for event-driven architecture.")

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
            "is_halted": self.is_halted or self.risk_manager.is_halted,
            "uptime": str(uptime_delta).split('.')[0],
            "equity": self.risk_manager.get_portfolio_value(),
            "unrealized_pnl": self.position_manager.get_total_unrealized_pnl(),
            "open_positions_count": len(self.position_manager.get_open_positions()),
        }

    async def run(self):
        """Main event loop of the trading bot."""
        self.running = True
        await self.data_handler.start_streaming()
        await self.order_manager.start()
        logger.info("Starting TradingBot event loop.")

        while self.running:
            try:
                event = await self.event_queue.get()
            except asyncio.CancelledError:
                break

            try:
                if isinstance(event, MarketEvent):
                    logger.debug("Processing MarketEvent", symbol=event.symbol)
                    self.latest_market_data[event.symbol] = {
                        "ohlcv_df": event.ohlcv_df,
                        "last_price": event.last_price
                    }
                    open_positions = self.position_manager.get_open_positions()
                    for pos in open_positions:
                        if pos.symbol == event.symbol:
                            self.position_manager.update_position_pnl(pos.id, event.last_price)
                    
                    self.risk_manager.update_risk_metrics()
                    self.risk_manager.update_trailing_stops()

                    await self.strategy.on_market_event(event, open_positions)

                elif isinstance(event, SignalEvent):
                    logger.debug("Processing SignalEvent", symbol=event.symbol, action=event.action)
                    if self.is_halted or self.risk_manager.is_halted:
                        logger.warning("Signal ignored: Trading is halted.", signal=event)
                        continue
                    market_data = self.latest_market_data.get(event.symbol)
                    if market_data:
                        await self.execution_handler.on_signal_event(event, market_data['ohlcv_df'])
                    else:
                        logger.warning("No market data available to process signal", symbol=event.symbol)

                elif isinstance(event, OrderEvent):
                    logger.debug("Processing OrderEvent", symbol=event.symbol, side=event.side)
                    await self.order_manager.on_order_event(event)

                elif isinstance(event, FillEvent):
                    logger.debug("Processing FillEvent", symbol=event.symbol, price=event.price)
                    self.position_manager.update_from_fill(event)
                    await self.strategy.on_fill_event(event)

            except Exception as e:
                logger.critical("Unhandled exception in event loop", event_type=type(event).__name__, error=str(e), exc_info=True)

    async def stop(self):
        """Stops the trading bot gracefully."""
        logger.info("Stopping TradingBot...")
        self.running = False
        await self.data_handler.stop_streaming()
        await self.order_manager.stop()
        self.position_manager.close()
