# bot_core/risk_manager.py
import logging
from typing import Dict, Any, List
from bot_core.position_manager import PositionManager, Position

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, config: Dict[str, Any], position_manager: PositionManager):
        self.max_position_size_usd = config.get("max_position_size_usd", 1000)
        self.max_daily_loss_usd = config.get("max_daily_loss_usd", 500)
        self.max_open_positions = config.get("max_open_positions", 5)
        self.position_manager = position_manager
        self.is_trading_halted = False
        logger.info(f"RiskManager initialized with config: Max Pos Size=${self.max_position_size_usd}, "
                    f"Max Daily Loss=${self.max_daily_loss_usd}, Max Open Positions={self.max_open_positions}")

    def check_trade_allowed(self, symbol: str, side: str, quantity: float, price: float) -> bool:
        """
        Performs pre-trade risk checks.
        Returns True if trade is allowed, False otherwise.
        """
        if self.is_trading_halted:
            logger.warning("Trading is halted by risk manager.")
            return False

        # Check max open positions
        open_positions = self.position_manager.get_open_positions()
        if len(open_positions) >= self.max_open_positions:
            logger.warning(f"Trade denied: Max open positions ({self.max_open_positions}) reached.")
            return False

        # Check position size
        trade_value_usd = quantity * price
        if trade_value_usd > self.max_position_size_usd:
            logger.warning(f"Trade denied for {symbol}: Proposed trade value ${trade_value_usd:.2f} exceeds "
                           f"max position size ${self.max_position_size_usd:.2f}.")
            return False

        # Check daily loss limit
        daily_pnl = self.position_manager.get_daily_realized_pnl()
        if daily_pnl < -self.max_daily_loss_usd:
            logger.critical(f"Daily loss limit of ${self.max_daily_loss_usd:.2f} exceeded. "
                            f"Current daily PnL: ${daily_pnl:.2f}. Halting trading.")
            self.is_trading_halted = True
            return False

        logger.info(f"Trade for {symbol} {side} {quantity}@{price} passed initial risk checks.")
        return True

    def update_risk_metrics(self):
        """
        Updates internal risk metrics and checks for conditions that might halt trading.
        This should be called periodically.
        """
        daily_pnl = self.position_manager.get_daily_realized_pnl()
        if daily_pnl < -self.max_daily_loss_usd and not self.is_trading_halted:
            logger.critical(f"Daily loss limit of ${self.max_daily_loss_usd:.2f} exceeded. "
                            f"Current daily PnL: ${daily_pnl:.2f}. Halting trading.")
            self.is_trading_halted = True
        elif daily_pnl >= -self.max_daily_loss_usd and self.is_trading_halted:
            # Optionally, re-enable trading if conditions improve, or require manual reset
            # For safety, we'll keep it halted once triggered.
            pass
        logger.debug(f"Risk metrics updated. Daily PnL: ${daily_pnl:.2f}. Trading Halted: {self.is_trading_halted}")

    def get_current_risk_status(self) -> Dict[str, Any]:
        """Returns a dictionary of current risk status."""
        daily_pnl = self.position_manager.get_daily_realized_pnl()
        open_positions = self.position_manager.get_open_positions()
        return {
            "is_trading_halted": self.is_trading_halted,
            "current_daily_pnl_usd": daily_pnl,
            "max_daily_loss_usd": self.max_daily_loss_usd,
            "open_positions_count": len(open_positions),
            "max_open_positions": self.max_open_positions,
            "max_position_size_usd": self.max_position_size_usd
        }
