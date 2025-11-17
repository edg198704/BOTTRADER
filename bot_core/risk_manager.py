from typing import Dict, Any, List
import pandas as pd

from bot_core.logger import get_logger
from bot_core.config import RiskManagementConfig

logger = get_logger(__name__)

class RiskManager:
    def __init__(self, config: RiskManagementConfig):
        self.config = config
        self.daily_loss = 0.0
        self.is_halted = False
        logger.info("RiskManager initialized.")

    def check_trade_allowed(self, symbol: str, quantity: float, price: float) -> bool:
        """Performs pre-trade risk checks."""
        if self.is_halted:
            logger.warning("Trade denied: RiskManager is halted.", symbol=symbol)
            return False

        position_value = quantity * price
        if position_value > self.config.max_position_size_usd:
            logger.warning("Trade denied: Exceeds max position size.", symbol=symbol, value=position_value)
            return False

        if self.daily_loss <= -self.config.max_daily_loss_usd:
            logger.critical("Trade denied: Daily loss limit reached.", daily_loss=self.daily_loss)
            self.is_halted = True
            return False

        return True

    def calculate_position_size(self, portfolio_equity: float) -> float:
        """Calculates position size in USD based on portfolio equity and risk per trade."""
        size_usd = portfolio_equity * self.config.risk_per_trade_pct
        return min(size_usd, self.config.max_position_size_usd)

    def calculate_stop_loss(self, symbol: str, entry_price: float, side: str, df: pd.DataFrame) -> float:
        """Calculates stop loss using ATR or a fallback percentage."""
        if 'atr' in df.columns and not df['atr'].empty:
            atr = df['atr'].iloc[-1]
            if side.upper() == 'BUY':
                stop_loss = entry_price - (atr * self.config.atr_stop_multiplier)
            else:
                stop_loss = entry_price + (atr * self.config.atr_stop_multiplier)
            logger.debug("Calculated ATR-based stop loss", symbol=symbol, stop_loss=stop_loss)
        else:
            if side.upper() == 'BUY':
                stop_loss = entry_price * (1 - self.config.stop_loss_fallback_pct)
            else:
                stop_loss = entry_price * (1 + self.config.stop_loss_fallback_pct)
            logger.warning("Using fallback percentage for stop loss", symbol=symbol, stop_loss=stop_loss)
        return stop_loss

    def calculate_take_profit_levels(self, entry_price: float, side: str, confidence: float) -> List[Dict[str, float]]:
        """Calculates multiple take-profit levels based on confidence."""
        # Example: More aggressive targets for higher confidence
        base_tp_pct = 0.02 * (1 + confidence) # e.g., 2% to 4%
        levels = [
            {'price': entry_price * (1 + base_tp_pct) if side.upper() == 'BUY' else entry_price * (1 - base_tp_pct), 'quantity_pct': 0.5},
            {'price': entry_price * (1 + base_tp_pct * 2) if side.upper() == 'BUY' else entry_price * (1 - base_tp_pct * 2), 'quantity_pct': 0.5}
        ]
        return levels

    def update_portfolio_risk(self, portfolio_value: float):
        """Updates portfolio-level risk metrics, like the circuit breaker."""
        # This is a simplified implementation. A real one would track initial capital.
        # For now, we assume it's managed by the PositionManager's PnL tracking.
        pass

    def reset_daily_limits(self):
        """Resets daily loss counters and un-halts the system if needed."""
        self.daily_loss = 0.0
        self.is_halted = False
        logger.info("Daily risk limits have been reset.")
