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

    def update_portfolio_risk(self, portfolio_value: float, open_positions: List[Position]):
        if self.initial_capital is None:
            self.initial_capital = portfolio_value
            self.peak_portfolio_value = portfolio_value

        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value

        drawdown = (portfolio_value - self.peak_portfolio_value) / self.peak_portfolio_value

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

    def calculate_position_size(self, portfolio_equity: float) -> float:
        size_usd = portfolio_equity * self.config.risk_per_trade_pct
        return min(size_usd, self.config.max_position_size_usd)

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
