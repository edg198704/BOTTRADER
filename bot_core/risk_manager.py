import logging
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime, time
from enum import Enum, auto

from bot_core.position_manager import PositionManager
from bot_core.config import RiskManagementConfig

logger = logging.getLogger(__name__)

class HaltReason(Enum):
    NONE = auto()
    DAILY_LOSS_LIMIT = auto()
    CIRCUIT_BREAKER = auto()

class RiskManager:
    def __init__(self, config: RiskManagementConfig, position_manager: PositionManager, initial_capital: float):
        self.config = config
        self.position_manager = position_manager
        self.initial_capital = initial_capital
        self._halt_reason: HaltReason = HaltReason.NONE
        self._current_trading_day: datetime.date = datetime.utcnow().date()
        logger.info(f"RiskManager initialized with config: {self.config.dict()}")

    @property
    def is_halted(self) -> bool:
        """Public property to check if trading is currently halted."""
        return self._halt_reason != HaltReason.NONE

    def get_portfolio_value(self) -> float:
        """Calculates the current total value of the portfolio."""
        realized_pnl = self.position_manager.get_daily_realized_pnl(datetime.combine(self._current_trading_day, time.min))
        unrealized_pnl = self.position_manager.get_total_unrealized_pnl()
        return self.initial_capital + realized_pnl + unrealized_pnl

    def calculate_position_size(self, current_price: float, stop_loss_price: float) -> float:
        """Calculates position size based on a fixed percentage of portfolio equity to risk."""
        try:
            if current_price <= 0 or stop_loss_price <= 0:
                logger.warning("Invalid prices for position size calculation.")
                return 0.0

            risk_per_unit = abs(current_price - stop_loss_price)
            if risk_per_unit == 0:
                logger.warning("Risk per unit is zero, cannot calculate position size.")
                return 0.0

            portfolio_value = self.get_portfolio_value()
            risk_amount_usd = portfolio_value * self.config.risk_per_trade_pct
            size = risk_amount_usd / risk_per_unit

            max_size_from_cap = self.config.max_position_size_usd / current_price
            final_size = min(size, max_size_from_cap)

            logger.info(f"Calculated position size: {final_size:.8f} (Portfolio Value: ${portfolio_value:.2f}, Risk Amount: ${risk_amount_usd:.2f})")
            return float(final_size)

        except Exception as e:
            logger.error(f"Error calculating position size: {e}", exc_info=True)
            return 0.0

    def calculate_stop_loss(self, symbol: str, entry_price: float, side: str, df: pd.DataFrame) -> float:
        """Calculate stop loss using ATR for volatility, with a fallback to percentage."""
        try:
            if 'atr' in df.columns and pd.notna(df['atr'].iloc[-1]) and df['atr'].iloc[-1] > 0:
                atr = df['atr'].iloc[-1]
                if side.upper() == 'BUY':
                    stop_loss = entry_price - (atr * self.config.atr_stop_multiplier)
                else:
                    stop_loss = entry_price + (atr * self.config.atr_stop_multiplier)
                logger.info(f"ATR-based stop loss for {symbol}: {stop_loss:.4f} (ATR: {atr:.4f})")
                return float(stop_loss)

            logger.warning(f"Could not calculate ATR for {symbol}. Falling back to percentage-based stop loss.")
            if side.upper() == 'BUY':
                return entry_price * (1 - self.config.stop_loss_fallback_pct)
            else:
                return entry_price * (1 + self.config.stop_loss_fallback_pct)

        except Exception as e:
            logger.error(f"Error calculating stop loss for {symbol}: {e}", exc_info=True)
            if side.upper() == 'BUY':
                return entry_price * (1 - self.config.stop_loss_fallback_pct)
            else:
                return entry_price * (1 + self.config.stop_loss_fallback_pct)

    def calculate_take_profit_levels(self, entry_price: float, side: str, confidence: float) -> List[Dict[str, Any]]:
        """Calculate multiple take profit levels based on confidence and risk/reward."""
        if confidence > 0.8:
            tp_multipliers = [1.5, 3.0, 5.0]  # Risk:Reward
            size_distribution = [0.5, 0.3, 0.2] # 50%, 30%, 20%
        elif confidence > 0.6:
            tp_multipliers = [1.5, 3.0]
            size_distribution = [0.6, 0.4]
        else:
            tp_multipliers = [1.5]
            size_distribution = [1.0]

        levels = []
        risk_pct = self.config.stop_loss_fallback_pct # Use fallback as a baseline risk measure
        for mult, frac in zip(tp_multipliers, size_distribution):
            if side.upper() == 'BUY':
                tp_price = entry_price * (1 + risk_pct * mult)
            else:
                tp_price = entry_price * (1 - risk_pct * mult)
            levels.append({"price": tp_price, "fraction": frac})
        return levels

    def check_trade_allowed(self, symbol: str, side: str, quantity: float, price: float) -> bool:
        """Performs pre-trade risk checks."""
        if self.is_halted:
            logger.warning(f"Trading is halted due to {self._halt_reason.name}. Trade denied.")
            return False

        open_positions = self.position_manager.get_open_positions()
        if len(open_positions) >= self.config.max_open_positions:
            logger.warning(f"Trade denied: Max open positions ({self.config.max_open_positions}) reached.")
            return False

        trade_value_usd = quantity * price
        if trade_value_usd > self.config.max_position_size_usd:
            logger.warning(f"Trade denied for {symbol}: Proposed value ${trade_value_usd:.2f} exceeds max size ${self.config.max_position_size_usd:.2f}.")
            return False

        # Re-check metrics just before trade to ensure no limits were just breached
        self.update_risk_metrics()
        if self.is_halted:
            logger.warning(f"Trade denied: Risk limit breached just before execution ({self._halt_reason.name}).")
            return False

        logger.debug(f"Trade for {symbol} {side} {quantity}@{price} passed initial risk checks.")
        return True

    def update_risk_metrics(self):
        """Updates portfolio-level risk metrics and manages trading halt state machine."""
        today = datetime.utcnow().date()
        if today > self._current_trading_day:
            logger.info(f"New trading day. Resetting daily limits from {self._current_trading_day} to {today}.")
            self._current_trading_day = today
            if self._halt_reason == HaltReason.DAILY_LOSS_LIMIT:
                logger.info("Resetting trading halt due to new day.")
                self._halt_reason = HaltReason.NONE

        # If trading is halted by circuit breaker, no further checks are needed.
        if self._halt_reason == HaltReason.CIRCUIT_BREAKER:
            return

        portfolio_value = self.get_portfolio_value()
        realized_pnl = self.position_manager.get_daily_realized_pnl(datetime.combine(self._current_trading_day, time.min))
        portfolio_drawdown = (portfolio_value - self.initial_capital) / self.initial_capital if self.initial_capital > 0 else 0

        if realized_pnl < -abs(self.config.max_daily_loss_usd):
            if self._halt_reason != HaltReason.DAILY_LOSS_LIMIT:
                logger.critical(f"Daily loss limit of ${self.config.max_daily_loss_usd:.2f} exceeded. Halting trading.")
                self._halt_reason = HaltReason.DAILY_LOSS_LIMIT

        if portfolio_drawdown < self.config.circuit_breaker_threshold:
            if self._halt_reason != HaltReason.CIRCUIT_BREAKER:
                logger.critical(f"Portfolio drawdown {portfolio_drawdown:.2%} exceeded circuit breaker threshold of {self.config.circuit_breaker_threshold:.2%}. Halting trading permanently.")
                self._halt_reason = HaltReason.CIRCUIT_BREAKER
        
        logger.debug(f"Risk metrics updated. Daily PnL: ${realized_pnl:.2f}, Portfolio Value: ${portfolio_value:.2f}, Drawdown: {portfolio_drawdown:.2%}, Halt Status: {self._halt_reason.name}")

    def update_trailing_stops(self):
        """Iterates through open positions and updates their trailing stops."""
        if not self.config.use_trailing_stop:
            return

        open_positions = self.position_manager.get_open_positions()
        for pos in open_positions:
            if pos.stop_loss is None:
                continue
            
            trailing_distance = pos.current_price * 0.015 
            new_stop = None
            if pos.side == 'BUY' and pos.current_price > pos.entry_price:
                potential_stop = pos.current_price - trailing_distance
                if potential_stop > pos.stop_loss:
                    new_stop = potential_stop
            elif pos.side == 'SELL' and pos.current_price < pos.entry_price:
                potential_stop = pos.current_price + trailing_distance
                if potential_stop < pos.stop_loss:
                    new_stop = potential_stop
            
            if new_stop:
                self.position_manager.update_position_risk(pos.id, new_stop)
