import logging
from typing import Dict, Any, List
import pandas as pd
from bot_core.position_manager import PositionManager
from bot_core.config import RiskManagementConfig, BotConfig

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, config: BotConfig, position_manager: PositionManager, initial_capital: float):
        self.config = config.risk_management
        self.position_manager = position_manager
        self.initial_capital = initial_capital
        self.is_trading_halted = False
        logger.info(f"RiskManager initialized with config: {self.config.dict()}")

    def calculate_position_size(self, symbol: str, current_price: float, confidence: float, available_equity: float) -> float:
        """Calculate position size based on risk management rules."""
        try:
            if current_price <= 0:
                return 0.0
            # Base position size based on max value in USD, scaled by confidence
            # Confidence acts as a conviction score, from 0.5 to 1.0, scaling the position size.
            confidence_scaler = (confidence - 0.5) * 2 if confidence > 0.5 else 0.1
            position_value_usd = self.config.max_position_size_usd * confidence_scaler
            
            # Ensure position value does not exceed a fraction of total equity
            max_value_from_equity = available_equity * 0.1 # Max 10% of equity per trade
            final_position_value = min(position_value_usd, max_value_from_equity)

            size = final_position_value / current_price
            logger.info(f"Calculated position size for {symbol}: {size:.8f} based on confidence {confidence:.2f} and equity {available_equity:.2f}")
            return float(size)

        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}", exc_info=True)
            return 0.0

    def calculate_stop_loss(self, symbol: str, entry_price: float, side: str, df: pd.DataFrame) -> float:
        """Calculate stop loss using ATR for volatility, with a fallback to percentage."""
        try:
            # ATR-based stop loss
            if df is not None and not df.empty and all(c in df.columns for c in ['high', 'low', 'close']) and len(df) >= 14:
                high = df['high']
                low = df['low']
                close = df['close']
                tr1 = high - low
                tr2 = abs(high - close.shift())
                tr3 = abs(low - close.shift())
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1]

                if pd.notna(atr) and atr > 0:
                    if side.upper() == 'BUY':
                        stop_loss = entry_price - (atr * self.config.atr_stop_multiplier)
                    else:
                        stop_loss = entry_price + (atr * self.config.atr_stop_multiplier)
                    logger.info(f"ATR-based stop loss for {symbol}: {stop_loss:.4f} (ATR: {atr:.4f})")
                    return float(stop_loss)

            # Fallback to percentage-based stop loss
            logger.warning(f"Could not calculate ATR for {symbol}. Falling back to percentage-based stop loss.")
            if side.upper() == 'BUY':
                return entry_price * (1 - self.config.stop_loss_fallback_pct)
            else:
                return entry_price * (1 + self.config.stop_loss_fallback_pct)

        except Exception as e:
            logger.error(f"Error calculating stop loss for {symbol}: {e}", exc_info=True)
            # Safe fallback
            if side.upper() == 'BUY':
                return entry_price * (1 - self.config.stop_loss_fallback_pct)
            else:
                return entry_price * (1 + self.config.stop_loss_fallback_pct)

    def calculate_take_profit_levels(self, entry_price: float, side: str, confidence: float) -> List[Dict[str, Any]]:
        """Calculate multiple take profit levels based on confidence and risk/reward."""
        levels = []
        stop_pct = self.config.stop_loss_fallback_pct
        base_rr = 2.0 # Base risk-reward ratio

        # Define TP levels based on confidence
        if confidence > 0.8:
            tp_ratios = [base_rr * 1.0, base_rr * 2.0]
            size_fractions = [0.6, 0.4]
        elif confidence > 0.6:
            tp_ratios = [base_rr * 1.25]
            size_fractions = [1.0]
        else:
            tp_ratios = [base_rr * 1.0]
            size_fractions = [1.0]

        for ratio, frac in zip(tp_ratios, size_fractions):
            if side.upper() == 'BUY':
                tp_price = entry_price * (1 + stop_pct * ratio)
            else:
                tp_price = entry_price * (1 - stop_pct * ratio)
            levels.append({"price": tp_price, "fraction": frac})
        return levels

    def check_trade_allowed(self, symbol: str, side: str, quantity: float, price: float) -> bool:
        """Performs pre-trade risk checks."""
        if self.is_trading_halted:
            logger.warning("Trading is halted by risk manager.")
            return False

        open_positions = self.position_manager.get_open_positions()
        if len(open_positions) >= self.config.max_open_positions:
            logger.warning(f"Trade denied: Max open positions ({self.config.max_open_positions}) reached.")
            return False

        trade_value_usd = quantity * price
        if trade_value_usd > self.config.max_position_size_usd:
            logger.warning(f"Trade denied for {symbol}: Proposed value ${trade_value_usd:.2f} exceeds max size ${self.config.max_position_size_usd:.2f}.")
            return False

        # Check daily loss limit
        daily_pnl = self.position_manager.get_daily_realized_pnl()
        if daily_pnl < -abs(self.config.max_daily_loss_usd):
            logger.critical(f"Daily loss limit of ${self.config.max_daily_loss_usd:.2f} exceeded. Halting trading.")
            self.is_trading_halted = True
            return False

        logger.debug(f"Trade for {symbol} {side} {quantity}@{price} passed initial risk checks.")
        return True

    def update_risk_metrics(self):
        """Updates portfolio-level risk metrics and checks for circuit breaker conditions."""
        realized_pnl = self.position_manager.get_daily_realized_pnl()
        unrealized_pnl = self.position_manager.get_total_unrealized_pnl()
        
        portfolio_value = self.initial_capital + realized_pnl + unrealized_pnl
        daily_drawdown = (portfolio_value - self.initial_capital) / self.initial_capital if self.initial_capital > 0 else 0

        if daily_pnl < -abs(self.config.max_daily_loss_usd) and not self.is_trading_halted:
            logger.critical(f"Daily loss limit of ${self.config.max_daily_loss_usd:.2f} exceeded. Halting trading.")
            self.is_trading_halted = True

        if daily_drawdown < self.config.circuit_breaker_threshold and not self.is_trading_halted:
            logger.critical(f"Portfolio drawdown {daily_drawdown:.2%} exceeded circuit breaker threshold of {self.config.circuit_breaker_threshold:.2%}. Halting trading.")
            self.is_trading_halted = True
        
        logger.debug(f"Risk metrics updated. Daily PnL: ${realized_pnl:.2f}, Portfolio Value: ${portfolio_value:.2f}, Drawdown: {daily_drawdown:.2%}")

    def update_trailing_stops(self):
        """Iterates through open positions and updates their trailing stops."""
        if not self.config.use_trailing_stop:
            return

        open_positions = self.position_manager.get_open_positions()
        for pos in open_positions:
            if pos.stop_loss is None:
                continue
            
            # Example: 1.5% trailing distance from current price, only if in profit
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
