from typing import Dict, Any, List
import pandas as pd
from datetime import datetime, time
from enum import Enum, auto

from bot_core.logger import get_logger
from bot_core.position_manager import PositionManager
from bot_core.config import RiskManagementConfig

logger = get_logger(__name__)

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
        logger.info("RiskManager initialized", config=self.config.dict())

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
                logger.warning("Invalid prices for position size calculation.", current_price=current_price, stop_loss_price=stop_loss_price)
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

            logger.info("Calculated position size", final_size=final_size, portfolio_value=portfolio_value, risk_amount_usd=risk_amount_usd)
            return float(final_size)

        except Exception as e:
            logger.error("Error calculating position size", error=str(e), exc_info=True)
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
                logger.info("ATR-based stop loss calculated", symbol=symbol, stop_loss=stop_loss, atr=atr)
                return float(stop_loss)

            logger.warning("Could not calculate ATR, falling back to percentage-based stop loss.", symbol=symbol)
            if side.upper() == 'BUY':
                return entry_price * (1 - self.config.stop_loss_fallback_pct)
            else:
                return entry_price * (1 + self.config.stop_loss_fallback_pct)

        except Exception as e:
            logger.error("Error calculating stop loss", symbol=symbol, error=str(e), exc_info=True)
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
        """Performs pre-trade risk checks, accounting for existing positions."""
        if self.is_halted:
            logger.warning("Trade denied: Trading is halted.", reason=self._halt_reason.name)
            return False

        self.update_risk_metrics()
        if self.is_halted:
            logger.warning("Trade denied: Risk limit breached just before execution.", reason=self._halt_reason.name)
            return False

        open_positions = self.position_manager.get_open_positions()
        existing_position = next((p for p in open_positions if p.symbol == symbol), None)
        
        trade_value_usd = quantity * price

        if existing_position:
            if existing_position.side == side.upper():
                new_total_quantity = existing_position.quantity + quantity
                new_total_value = new_total_quantity * price
                if new_total_value > self.config.max_position_size_usd:
                    logger.warning("Trade denied: Increasing position would exceed max size.", symbol=symbol, new_value=new_total_value, max_value=self.config.max_position_size_usd)
                    return False
        else:
            if len(open_positions) >= self.config.max_open_positions:
                logger.warning("Trade denied: Max open positions reached.", max_positions=self.config.max_open_positions)
                return False
            
            if trade_value_usd > self.config.max_position_size_usd:
                logger.warning("Trade denied: Proposed value exceeds max size.", symbol=symbol, trade_value=trade_value_usd, max_value=self.config.max_position_size_usd)
                return False

        logger.debug("Trade passed risk checks.", symbol=symbol, side=side, quantity=quantity, price=price)
        return True

    def update_risk_metrics(self):
        """Updates portfolio-level risk metrics and manages trading halt state machine."""
        today = datetime.utcnow().date()
        if today > self._current_trading_day:
            logger.info("New trading day. Resetting daily limits.", old_day=self._current_trading_day, new_day=today)
            self._current_trading_day = today
            if self._halt_reason == HaltReason.DAILY_LOSS_LIMIT:
                logger.info("Resetting trading halt due to new day.")
                self._halt_reason = HaltReason.NONE

        if self._halt_reason == HaltReason.CIRCUIT_BREAKER:
            return

        portfolio_value = self.get_portfolio_value()
        realized_pnl = self.position_manager.get_daily_realized_pnl(datetime.combine(self._current_trading_day, time.min))
        portfolio_drawdown = (portfolio_value - self.initial_capital) / self.initial_capital if self.initial_capital > 0 else 0

        if realized_pnl < -abs(self.config.max_daily_loss_usd):
            if self._halt_reason != HaltReason.DAILY_LOSS_LIMIT:
                logger.critical("Daily loss limit exceeded. Halting trading.", daily_pnl=realized_pnl, limit=self.config.max_daily_loss_usd)
                self._halt_reason = HaltReason.DAILY_LOSS_LIMIT

        if portfolio_drawdown < self.config.circuit_breaker_threshold:
            if self._halt_reason != HaltReason.CIRCUIT_BREAKER:
                logger.critical("Portfolio drawdown exceeded circuit breaker. Halting trading permanently.", drawdown=portfolio_drawdown, threshold=self.config.circuit_breaker_threshold)
                self._halt_reason = HaltReason.CIRCUIT_BREAKER
        
        logger.debug("Risk metrics updated", daily_pnl=realized_pnl, portfolio_value=portfolio_value, drawdown=portfolio_drawdown, halt_status=self._halt_reason.name)

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
