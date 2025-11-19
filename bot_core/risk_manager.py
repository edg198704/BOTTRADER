from typing import List, Optional, Any, TYPE_CHECKING
import pandas as pd
from datetime import datetime

from bot_core.config import RiskManagementConfig
from bot_core.logger import get_logger
from bot_core.position_manager import Position

if TYPE_CHECKING:
    from bot_core.monitoring import AlertSystem
    from bot_core.position_manager import PositionManager
    from bot_core.data_handler import DataHandler

logger = get_logger(__name__)

class RiskManager:
    def __init__(self, config: RiskManagementConfig, position_manager: 'PositionManager', data_handler: 'DataHandler', alert_system: Optional['AlertSystem'] = None):
        self.config = config
        self.position_manager = position_manager
        self.data_handler = data_handler
        self.alert_system = alert_system
        
        self.circuit_breaker_halted = False
        self.daily_loss_halted = False
        self.initial_capital = None 
        self.peak_portfolio_value = None
        self.current_drawdown = 0.0 # Track current drawdown state
        
        # Flags for emergency liquidation
        self.liquidation_needed = False
        self.liquidation_triggered = False
        
        logger.info("RiskManager initialized.")

    async def initialize(self):
        """Loads the initial capital and peak portfolio value from the persistent store."""
        state = await self.position_manager.get_portfolio_state()
        if state:
            self.initial_capital = state['initial_capital']
            self.peak_portfolio_value = state['peak_equity']
            logger.info("RiskManager loaded persistent state.", 
                        initial_capital=self.initial_capital, 
                        peak_equity=self.peak_portfolio_value)
        else:
            logger.warning("RiskManager could not load persistent state. Will initialize on first update.")

    @property
    def is_halted(self) -> bool:
        """Returns True if any risk halt condition is met."""
        return self.circuit_breaker_halted or self.daily_loss_halted

    async def update_portfolio_risk(self, portfolio_value: float, daily_realized_pnl: float):
        # Initialize on first run if DB was empty or initialize wasn't called
        if self.peak_portfolio_value is None:
            self.peak_portfolio_value = portfolio_value
        if self.initial_capital is None:
            self.initial_capital = portfolio_value

        # 1. Check Circuit Breaker based on drawdown from High Water Mark
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
            # Persist the new high water mark
            await self.position_manager.update_portfolio_high_water_mark(portfolio_value)

        # Calculate Drawdown (Negative Float, e.g., -0.05)
        self.current_drawdown = (portfolio_value - self.peak_portfolio_value) / self.peak_portfolio_value if self.peak_portfolio_value > 0 else 0.0

        if self.current_drawdown < self.config.circuit_breaker_threshold:
            if not self.circuit_breaker_halted:
                self.circuit_breaker_halted = True
                message = f"CIRCUIT BREAKER TRIPPED! Trading halted due to excessive drawdown."
                details = {
                    'drawdown': f"{self.current_drawdown:.2%}", 
                    'threshold': f"{self.config.circuit_breaker_threshold:.2%}",
                    'portfolio_value': portfolio_value,
                    'peak_portfolio_value': self.peak_portfolio_value
                }
                logger.critical(message, **details)
                if self.alert_system:
                    await self.alert_system.send_alert(level='critical', message=message, details=details)
                
                # Trigger Liquidation if configured
                if self.config.close_positions_on_halt and not self.liquidation_triggered:
                    self.liquidation_needed = True
                    self.liquidation_triggered = True
                    logger.warning("Emergency liquidation triggered by Circuit Breaker.")

        else:
            if self.circuit_breaker_halted:
                # Note: A manual resume process is safer. This is a simple auto-resume for now.
                self.circuit_breaker_halted = False
                self.liquidation_triggered = False # Reset trigger so it can fire again if needed
                logger.info("Trading resumed as portfolio recovered from drawdown.")

        # 2. Check Max Daily Loss
        # Note: daily_realized_pnl will be negative for a loss.
        if self.config.max_daily_loss_usd > 0 and daily_realized_pnl < -self.config.max_daily_loss_usd:
            if not self.daily_loss_halted:
                self.daily_loss_halted = True
                message = f"MAX DAILY LOSS REACHED! Trading halted for the day."
                details = {
                    'daily_pnl': f"${daily_realized_pnl:.2f}",
                    'limit': f"-${self.config.max_daily_loss_usd:.2f}"
                }
                logger.critical(message, **details)
                if self.alert_system:
                    await self.alert_system.send_alert(level='critical', message=message, details=details)
                
                # Trigger Liquidation if configured
                if self.config.close_positions_on_halt and not self.liquidation_triggered:
                    self.liquidation_needed = True
                    self.liquidation_triggered = True
                    logger.warning("Emergency liquidation triggered by Max Daily Loss.")

    def _get_regime_param(self, param_name: str, regime: Optional[str]) -> Any:
        """Gets a risk parameter, using a regime-specific value if available, otherwise the default."""
        default_value = getattr(self.config, param_name)
        if regime and hasattr(self.config.regime_based_risk, regime):
            regime_config = getattr(self.config.regime_based_risk, regime)
            if hasattr(regime_config, param_name):
                override_value = getattr(regime_config, param_name)
                if override_value is not None:
                    logger.debug(f"Using regime-based risk parameter.", parameter=param_name, regime=regime, value=override_value)
                    return override_value
        return default_value

    def calculate_position_size(self, symbol: str, portfolio_equity: float, entry_price: float, stop_loss_price: float, open_positions: List[Position], market_regime: Optional[str] = None) -> float:
        """Calculates position size in asset quantity based on risk and correlation."""
        if entry_price <= 0 or stop_loss_price <= 0:
            logger.warning("Cannot calculate position size with zero or negative prices.", entry=entry_price, sl=stop_loss_price)
            return 0.0

        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit == 0:
            logger.warning("Risk per unit is zero, cannot calculate position size. Check stop-loss logic.")
            return 0.0

        risk_pct = self._get_regime_param('risk_per_trade_pct', market_regime)
        
        # --- Drawdown Scaling ---
        # Reduce risk as drawdown increases to preserve capital.
        drawdown_scaling = max(0.2, 1.0 + self.current_drawdown)
        adjusted_risk_pct = risk_pct * drawdown_scaling
        
        if drawdown_scaling < 1.0:
            logger.info("Risk scaled down due to drawdown", 
                        drawdown=f"{self.current_drawdown:.2%}", 
                        original_risk=risk_pct, 
                        adjusted_risk=adjusted_risk_pct)

        # --- Correlation Scaling ---
        correlation_scaling = 1.0
        if self.config.correlation.enabled:
            max_corr = 0.0
            correlated_symbol = None
            for pos in open_positions:
                if pos.symbol == symbol:
                    continue # Should not happen for new trades, but safe check
                
                corr = self.data_handler.get_correlation(symbol, pos.symbol, self.config.correlation.lookback_periods)
                if corr > max_corr:
                    max_corr = corr
                    correlated_symbol = pos.symbol
            
            if max_corr > self.config.correlation.max_correlation:
                correlation_scaling = self.config.correlation.penalty_factor
                logger.info("High correlation detected. Applying penalty to position size.", 
                            symbol=symbol, 
                            correlated_with=correlated_symbol, 
                            correlation=f"{max_corr:.2f}", 
                            penalty=correlation_scaling)

        # Apply all scalings
        final_risk_pct = adjusted_risk_pct * correlation_scaling

        risk_amount_usd = portfolio_equity * final_risk_pct
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
            reason = "Circuit Breaker" if self.circuit_breaker_halted else "Max Daily Loss"
            logger.warning(f"Trade rejected: Trading is halted by {reason}.", symbol=symbol)
            return False
        
        if len(open_positions) >= self.config.max_open_positions:
            logger.warning("Trade rejected: Max open positions reached.", symbol=symbol, limit=self.config.max_open_positions)
            return False
        
        return True

    def calculate_stop_loss(self, side: str, entry_price: float, df_with_indicators: pd.DataFrame, market_regime: Optional[str] = None) -> float:
        atr_col = 'ATRr_14' # Default ATR column name from pandas-ta
        atr = df_with_indicators[atr_col].iloc[-1] if atr_col in df_with_indicators.columns and not df_with_indicators[atr_col].empty else 0
        
        atr_multiplier = self._get_regime_param('atr_stop_multiplier', market_regime)

        if atr > 0:
            stop_loss_offset = atr * atr_multiplier
        else:
            stop_loss_offset = entry_price * self.config.stop_loss_fallback_pct

        if side == 'BUY':
            return entry_price - stop_loss_offset
        else: # SELL
            return entry_price + stop_loss_offset

    def calculate_take_profit(self, side: str, entry_price: float, stop_loss_price: float, market_regime: Optional[str] = None) -> float:
        risk_per_unit = abs(entry_price - stop_loss_price)
        rr_ratio = self._get_regime_param('reward_to_risk_ratio', market_regime)
        profit_target = risk_per_unit * rr_ratio

        if side == 'BUY':
            return entry_price + profit_target
        else: # SELL
            return entry_price - profit_target

    def check_time_based_exit(self, position: Position, current_price: float) -> bool:
        """Checks if a position has been open too long without sufficient profit."""
        cfg = self.config.time_based_exit
        if not cfg.enabled: return False
        
        now = datetime.utcnow()
        duration = now - position.open_timestamp
        if duration.total_seconds() > (cfg.max_hold_time_hours * 3600):
            # Calculate PnL %
            pnl_pct = (current_price - position.entry_price) / position.entry_price
            if position.side == 'SELL':
                pnl_pct = -pnl_pct
            
            if pnl_pct < cfg.threshold_pct:
                logger.info("Time-based exit triggered", 
                            symbol=position.symbol, 
                            duration_hours=duration.total_seconds()/3600, 
                            pnl_pct=pnl_pct)
                return True
        return False
