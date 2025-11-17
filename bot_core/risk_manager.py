from typing import List, Dict
from datetime import datetime, timezone, timedelta

from bot_core.logger import get_logger
from bot_core.config import RiskManagementConfig
from bot_core.position_manager import PositionManager, Position

logger = get_logger(__name__)

class RiskManager:
    def __init__(self, config: RiskManagementConfig, position_manager: PositionManager, initial_capital: float):
        self.config = config
        self.position_manager = position_manager
        self.initial_capital = initial_capital
        self.portfolio_peak_value = initial_capital
        self.daily_pnl = 0.0
        self.last_pnl_reset_time = datetime.now(timezone.utc)
        self.is_halted = False
        logger.info("RiskManager initialized.")

    def _update_daily_pnl(self):
        now = datetime.now(timezone.utc)
        if now.date() > self.last_pnl_reset_time.date():
            logger.info("Resetting daily PnL for new day.", previous_pnl=self.daily_pnl)
            self.daily_pnl = 0.0
            self.last_pnl_reset_time = now

    def check_trade_allowed(self, symbol: str, quantity: float, price: float) -> bool:
        """Performs all pre-trade risk checks."""
        if self.is_halted:
            logger.warning("Trade denied: RiskManager is halted.", symbol=symbol)
            return False

        self._update_daily_pnl()

        open_positions = self.position_manager.get_all_open_positions()

        # Check max open positions
        if len(open_positions) >= self.config.max_open_positions:
            logger.warning("Trade denied: Max open positions reached.", symbol=symbol, limit=self.config.max_open_positions)
            return False

        # Check max position size
        position_value = quantity * price
        if position_value > self.config.max_position_size_usd:
            logger.warning("Trade denied: Position size exceeds limit.", symbol=symbol, value=position_value, limit=self.config.max_position_size_usd)
            return False

        # Check daily loss limit
        if self.daily_pnl < -abs(self.config.max_daily_loss_usd):
            logger.critical("Trade denied: Daily loss limit exceeded. Halting trading.", daily_pnl=self.daily_pnl)
            self.is_halted = True
            return False

        return True

    def update_portfolio_risk(self, portfolio_value: float):
        """Updates portfolio-level risk metrics like drawdown and circuit breaker."""
        # Update peak value for drawdown calculation
        self.portfolio_peak_value = max(self.portfolio_peak_value, portfolio_value)

        # Check circuit breaker
        drawdown = (portfolio_value - self.portfolio_peak_value) / self.portfolio_peak_value
        if drawdown < self.config.circuit_breaker_threshold:
            if not self.is_halted:
                logger.critical("CIRCUIT BREAKER TRIPPED: Portfolio drawdown exceeded threshold.", 
                              drawdown=f"{drawdown:.2%}", threshold=f"{self.config.circuit_breaker_threshold:.2%}")
                self.is_halted = True
        
        # Update daily PnL based on closed trades (simplified)
        # A more robust implementation would get this from PositionManager

    def calculate_position_size(self, portfolio_equity: float) -> float:
        """Calculates position size based on risk per trade percentage."""
        # Simple fixed fractional sizing
        risk_amount = portfolio_equity * self.config.risk_per_trade_pct
        # This is a simplified version. A full implementation would use stop loss distance.
        # For now, we assume it's a fraction of the max size.
        return self.config.max_position_size_usd * 0.5 # Return 50% of max size as a default

    def calculate_stop_loss(self, entry_price: float, side: str, atr: Optional[float] = None) -> float:
        """Calculates stop loss using ATR if available, otherwise fallback percentage."""
        if atr and atr > 0:
            if side.upper() == 'BUY':
                return entry_price - (atr * self.config.atr_stop_multiplier)
            else:
                return entry_price + (atr * self.config.atr_stop_multiplier)
        else:
            # Fallback to percentage
            if side.upper() == 'BUY':
                return entry_price * (1 - self.config.stop_loss_fallback_pct)
            else:
                return entry_price * (1 + self.config.stop_loss_fallback_pct)
