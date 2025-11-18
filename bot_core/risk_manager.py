from typing import List
import pandas as pd

from bot_core.config import RiskManagementConfig
from bot_core.logger import get_logger
from bot_core.position_manager import Position

logger = get_logger(__name__)

class RiskManager:
    def __init__(self, config: RiskManagementConfig):
        self.config = config
        self.is_halted = False
        self.initial_capital = None # Will be set on first update
        self.peak_portfolio_value = None
        logger.info("RiskManager initialized.")

    def update_portfolio_risk(self, portfolio_value: float):
        if self.initial_capital is None:
            self.initial_capital = portfolio_value
            self.peak_portfolio_value = portfolio_value

        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value

        drawdown = (portfolio_value - self.peak_portfolio_value) / self.peak_portfolio_value if self.peak_portfolio_value > 0 else 0

        if drawdown < self.config.circuit_breaker_threshold:
            if not self.is_halted:
                self.is_halted = True
                logger.critical("CIRCUIT BREAKER TRIPPED! Trading halted due to excessive drawdown.", 
                              drawdown=f"{drawdown:.2%}", threshold=f"{self.config.circuit_breaker_threshold:.2%}")
        else:
            if self.is_halted:
                # Note: A manual resume process is safer. This is a simple auto-resume for now.
                self.is_halted = False
                logger.info("Trading resumed as portfolio recovered from drawdown.")

    def calculate_position_size(self, portfolio_equity: float, entry_price: float, stop_loss_price: float) -> float:
        """Calculates position size in asset quantity based on risk."""
        if entry_price <= 0 or stop_loss_price <= 0:
            logger.warning("Cannot calculate position size with zero or negative prices.", entry=entry_price, sl=stop_loss_price)
            return 0.0

        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit == 0:
            logger.warning("Risk per unit is zero, cannot calculate position size. Check stop-loss logic.")
            return 0.0

        risk_amount_usd = portfolio_equity * self.config.risk_per_trade_pct
        quantity = risk_amount_usd / risk_per_unit
        
        # Cap the position size based on the max USD value allowed
        max_quantity_by_usd_cap = self.config.max_position_size_usd / entry_price
        
        final_quantity = min(quantity, max_quantity_by_usd_cap)

        if final_quantity < quantity:
            logger.info("Position size capped by max_position_size_usd.",
                        risk_based_qty=quantity,
                        capped_qty=final_quantity,
                        max_usd=self.config.max_position_size_usd)

        return final_quantity

    def check_trade_allowed(self, symbol: str, open_positions: List[Position]) -> bool:
        if self.is_halted:
            logger.warning("Trade rejected: Trading is halted by circuit breaker.", symbol=symbol)
            return False
        
        if len(open_positions) >= self.config.max_open_positions:
            logger.warning("Trade rejected: Max open positions reached.", symbol=symbol, limit=self.config.max_open_positions)
            return False
        
        return True

    def calculate_stop_loss(self, side: str, entry_price: float, df_with_indicators: pd.DataFrame) -> float:
        atr = df_with_indicators['atr'].iloc[-1] if 'atr' in df_with_indicators.columns and not df_with_indicators['atr'].empty else 0
        
        if atr > 0:
            stop_loss_offset = atr * self.config.atr_stop_multiplier
        else:
            stop_loss_offset = entry_price * self.config.stop_loss_fallback_pct

        if side == 'BUY':
            return entry_price - stop_loss_offset
        else: # SELL
            return entry_price + stop_loss_offset

    def calculate_take_profit(self, side: str, entry_price: float, stop_loss_price: float) -> float:
        risk_per_unit = abs(entry_price - stop_loss_price)
        profit_target = risk_per_unit * self.config.reward_to_risk_ratio

        if side == 'BUY':
            return entry_price + profit_target
        else: # SELL
            return entry_price - profit_target
