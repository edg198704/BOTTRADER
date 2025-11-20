import asyncio
from typing import Dict, Callable, Awaitable

from bot_core.logger import get_logger
from bot_core.config import BotConfig
from bot_core.position_manager import PositionManager, Position
from bot_core.risk_manager import RiskManager
from bot_core.data_handler import DataHandler
from bot_core.strategy import TradingStrategy

logger = get_logger(__name__)

class PositionMonitor:
    """
    Monitors all open positions for stop-loss, take-profit, and trailing stop updates.
    This component is responsible for triggering the closure of positions based on risk parameters.
    """
    def __init__(self, config: BotConfig, position_manager: PositionManager, risk_manager: RiskManager, data_handler: DataHandler, shared_latest_prices: Dict[str, float]):
        self.config = config
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.data_handler = data_handler
        self.latest_prices = shared_latest_prices
        self.running = False
        self._task: asyncio.Task = None
        self._close_position_callback: Callable[[Position, str], Awaitable[None]] = None
        self.strategy: TradingStrategy = None # Injected via setter or bot init
        logger.info("PositionMonitor initialized.")

    def set_strategy(self, strategy: TradingStrategy):
        self.strategy = strategy

    def set_close_position_callback(self, callback: Callable[[Position, str], Awaitable[None]]):
        """Sets the callback function to be executed when a position needs to be closed."""
        self._close_position_callback = callback

    async def run(self):
        """Main loop to monitor open positions."""
        if not self._close_position_callback:
            logger.critical("PositionMonitor cannot run without a close_position_callback set.")
            return

        self.running = True
        self._task = asyncio.current_task()
        logger.info("Starting position monitoring loop.")
        while self.running:
            try:
                open_positions = await self.position_manager.get_all_open_positions()
                check_tasks = [self._check_position(pos) for pos in open_positions]
                await asyncio.gather(*check_tasks)

            except asyncio.CancelledError:
                logger.info("Position monitoring loop cancelled.")
                break
            except Exception as e:
                logger.error("Error in position monitoring loop", error=str(e), exc_info=True)
            
            await asyncio.sleep(self.config.risk_management.monitor_interval_seconds)

    async def _check_position(self, pos: Position):
        """Checks a single position for TSL, SL, TP, and Stagnation triggers."""
        current_price = self.latest_prices.get(pos.symbol)
        if not current_price:
            return

        if self.config.risk_management.breakeven.enabled:
            pos = await self.position_manager.manage_breakeven_stop(
                pos, current_price, self.config.risk_management.breakeven
            )

        if self.config.risk_management.use_trailing_stop:
            atr_value = 0.0
            if self.config.risk_management.use_atr_for_trailing:
                df = self.data_handler.get_market_data(pos.symbol)
                if df is not None and not df.empty:
                    atr_col = self.config.risk_management.atr_column_name
                    if atr_col in df.columns:
                        atr_value = df[atr_col].iloc[-1]
            
            market_regime = None
            if self.strategy:
                market_regime = self.strategy.get_latest_regime(pos.symbol)

            new_stop, new_ref, activated = self.risk_manager.calculate_dynamic_trailing_stop(
                pos, current_price, atr_value, market_regime
            )
            
            if new_stop is not None or new_ref is not None or activated != pos.trailing_stop_active:
                pos = await self.position_manager.update_position_stop_loss(
                    pos, new_stop, new_ref, activated
                )

        if self.config.risk_management.time_based_exit.enabled:
             should_exit = self.risk_manager.check_time_based_exit(pos, current_price)
             if should_exit:
                 await self._close_position_callback(pos, "Time-Based Stagnation")
                 return

        if pos.side == 'BUY':
            if current_price <= pos.stop_loss_price:
                logger.info("Stop-loss triggered for LONG position", symbol=pos.symbol, price=current_price, sl=pos.stop_loss_price)
                await self._close_position_callback(pos, "Stop-Loss")
                return
            if pos.take_profit_price and current_price >= pos.take_profit_price:
                logger.info("Take-profit triggered for LONG position", symbol=pos.symbol, price=current_price, tp=pos.take_profit_price)
                await self._close_position_callback(pos, "Take-Profit")
                return
        elif pos.side == 'SELL':
            if current_price >= pos.stop_loss_price:
                logger.info("Stop-loss triggered for SHORT position", symbol=pos.symbol, price=current_price, sl=pos.stop_loss_price)
                await self._close_position_callback(pos, "Stop-Loss")
                return
            if pos.take_profit_price and current_price <= pos.take_profit_price:
                logger.info("Take-profit triggered for SHORT position", symbol=pos.symbol, price=current_price, tp=pos.take_profit_price)
                await self._close_position_callback(pos, "Take-Profit")
                return

    async def stop(self):
        """Stops the monitoring loop."""
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("PositionMonitor stopped.")
