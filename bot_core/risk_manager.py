from datetime import datetime, timezone, date
from typing import Dict

from bot_core.logger import get_logger
from bot_core.config import RiskManagementConfig

logger = get_logger(__name__)

class RiskManager:
    """Implements pre-trade and portfolio-level risk controls."""
    def __init__(self, config: RiskManagementConfig):
        self.config = config
        self.daily_loss_usd = 0.0
        self.last_trade_date = date.today()
        self.is_halted = False
        logger.info("RiskManager initialized", config=config.dict())

    def _check_daily_reset(self):
        """Resets daily loss counter if a new day has started."""
        today = date.today()
        if today != self.last_trade_date:
            logger.info("New trading day. Resetting daily loss counter.", last_day_loss=self.daily_loss_usd)
            self.daily_loss_usd = 0.0
            self.last_trade_date = today
            if self.is_halted:
                logger.info("Re-enabling trading after daily reset.")
                self.is_halted = False

    def update_portfolio_risk(self, portfolio_value: float):
        """Updates portfolio-level risk, like the main circuit breaker."""
        # This is a placeholder for more complex portfolio risk logic
        # For now, the main halt is based on daily loss.
        pass

    def calculate_position_size(self, portfolio_equity: float) -> float:
        """Calculates position size in USD based on risk parameters."""
        # Risk 1% of total equity per trade
        size_from_risk = portfolio_equity * self.config.risk_per_trade_pct
        # Cap by max position size
        return min(size_from_risk, self.config.max_position_size_usd)

    def check_trade_allowed(self, symbol: str, quantity: float, price: float) -> bool:
        """Performs pre-trade risk checks."""
        self._check_daily_reset()

        if self.is_halted:
            logger.warning("Trade rejected: RiskManager is halted.", symbol=symbol)
            return False

        if self.daily_loss_usd <= -self.config.max_daily_loss_usd:
            logger.critical("DAILY LOSS LIMIT HIT. Halting all trading for the day.", daily_loss=self.daily_loss_usd)
            self.is_halted = True
            return False
        
        # Additional checks (e.g., max exposure per asset) could be added here

        return True

    def record_pnl(self, pnl: float):
        """Records the PnL of a closed trade to update risk metrics."""
        self._check_daily_reset()
        self.daily_loss_usd += pnl
        logger.info("PnL recorded for risk management.", pnl=pnl, new_daily_pnl=self.daily_loss_usd)
