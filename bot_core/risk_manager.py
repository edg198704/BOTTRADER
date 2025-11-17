import logging
from typing import Dict, Any
from bot_core.position_manager import PositionManager

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, config: Dict[str, Any], position_manager: PositionManager, initial_capital: float):
        self.max_position_size_usd = config.get("max_position_size_usd", 1000.0)
        self.max_daily_loss_usd = config.get("max_daily_loss_usd", 500.0)
        self.max_open_positions = config.get("max_open_positions", 5)
        self.circuit_breaker_threshold = config.get("circuit_breaker_threshold", -0.10) # -10%
        self.position_manager = position_manager
        self.initial_capital = initial_capital
        self.is_trading_halted = False
        logger.info(f"RiskManager initialized with config: {config}")

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
            logger.warning(f"Trade denied for {symbol}: Proposed trade value ${trade_value_usd:.2f} exceeds max position size ${self.max_position_size_usd:.2f}.")
            return False

        # Check daily loss limit
        daily_pnl = self.position_manager.get_daily_realized_pnl()
        if daily_pnl < -abs(self.max_daily_loss_usd):
            logger.critical(f"Daily loss limit of ${self.max_daily_loss_usd:.2f} exceeded. Halting trading.")
            self.is_trading_halted = True
            return False

        logger.debug(f"Trade for {symbol} {side} {quantity}@{price} passed initial risk checks.")
        return True

    def update_risk_metrics(self):
        """
        Updates portfolio-level risk metrics and checks for circuit breaker conditions.
        """
        realized_pnl = self.position_manager.get_daily_realized_pnl()
        unrealized_pnl = self.position_manager.get_total_unrealized_pnl()
        
        portfolio_value = self.initial_capital + realized_pnl + unrealized_pnl
        daily_drawdown = (portfolio_value - self.initial_capital) / self.initial_capital if self.initial_capital > 0 else 0

        # Check daily loss limit
        if realized_pnl < -abs(self.max_daily_loss_usd) and not self.is_trading_halted:
            logger.critical(f"Daily loss limit of ${self.max_daily_loss_usd:.2f} exceeded. Halting trading.")
            self.is_trading_halted = True

        # Check circuit breaker for total portfolio drawdown
        if daily_drawdown < self.circuit_breaker_threshold and not self.is_trading_halted:
            logger.critical(f"Portfolio drawdown {daily_drawdown:.2%} exceeded circuit breaker threshold of {self.circuit_breaker_threshold:.2%}. Halting trading.")
            self.is_trading_halted = True
        
        logger.debug(f"Risk metrics updated. Daily PnL: ${realized_pnl:.2f}, Portfolio Value: ${portfolio_value:.2f}, Drawdown: {daily_drawdown:.2%}, Trading Halted: {self.is_trading_halted}")

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
