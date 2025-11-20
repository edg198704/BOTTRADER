from typing import List, Optional, Any, Dict, TYPE_CHECKING, Tuple
import pandas as pd
import json
import os
from datetime import datetime, timedelta

from bot_core.config import RiskManagementConfig
from bot_core.logger import get_logger
from bot_core.position_manager import Position, PositionStatus
from bot_core.utils import Clock

if TYPE_CHECKING:
    from bot_core.monitoring import AlertSystem
    from bot_core.position_manager import PositionManager
    from bot_core.data_handler import DataHandler

logger = get_logger(__name__)

class RiskManager:
    """
    The authoritative Gatekeeper for all trading decisions.
    Enforces portfolio-wide constraints, circuit breakers, and position sizing.
    Persists critical risk state (consecutive losses, cooldowns) to disk.
    """
    def __init__(self, config: RiskManagementConfig, position_manager: 'PositionManager', data_handler: 'DataHandler', alert_system: Optional['AlertSystem'] = None):
        self.config = config
        self.position_manager = position_manager
        self.data_handler = data_handler
        self.alert_system = alert_system
        
        # Circuit Breaker State
        self.circuit_breaker_halted = False
        self.daily_loss_halted = False
        self.initial_capital = None 
        self.peak_portfolio_value = None
        self.current_drawdown = 0.0
        
        # Liquidation State
        self.liquidation_needed = False
        self.liquidation_triggered = False
        
        # Symbol-Specific State (Persisted)
        self.symbol_consecutive_losses: Dict[str, int] = {}
        self.symbol_cooldowns: Dict[str, datetime] = {}
        
        self.state_file = "risk_state.json"
        
        logger.info("RiskManager initialized.")

    async def initialize(self):
        """Loads persistent risk state from the database and local file."""
        # Load Portfolio State from DB
        state = await self.position_manager.get_portfolio_state()
        if state:
            self.initial_capital = state['initial_capital']
            self.peak_portfolio_value = state['peak_equity']
            logger.info("RiskManager loaded portfolio state.", 
                        initial_capital=self.initial_capital, 
                        peak_equity=self.peak_portfolio_value)
        else:
            logger.warning("RiskManager could not load portfolio state. Will initialize on first update.")
            
        # Load Risk Counters from File
        self._load_state()

    def _load_state(self):
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
                self.symbol_consecutive_losses = data.get('consecutive_losses', {})
                cooldowns = data.get('cooldowns', {})
                self.symbol_cooldowns = {k: datetime.fromisoformat(v) for k, v in cooldowns.items()}
            logger.info("RiskManager loaded persistent risk counters.")
        except Exception as e:
            logger.error("Failed to load risk state", error=str(e))

    def _save_state(self):
        try:
            data = {
                'consecutive_losses': self.symbol_consecutive_losses,
                'cooldowns': {k: v.isoformat() for k, v in self.symbol_cooldowns.items()}
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed to save risk state", error=str(e))

    @property
    def is_halted(self) -> bool:
        return self.circuit_breaker_halted or self.daily_loss_halted

    def validate_entry(self, symbol: str, open_positions: List[Position]) -> Tuple[bool, str]:
        """
        The Master Gatekeeper for new entries.
        Checks ALL constraints: System Halt, Cooldowns, Position Limits.
        Returns (Allowed: bool, Reason: str)
        """
        # 1. Global System Halts
        if self.circuit_breaker_halted:
            return False, "Trading halted: Circuit Breaker Active"
        if self.daily_loss_halted:
            return False, "Trading halted: Max Daily Loss Reached"

        # 2. Symbol Cooldowns
        if symbol in self.symbol_cooldowns:
            if Clock.now() < self.symbol_cooldowns[symbol]:
                return False, f"Symbol in cooldown until {self.symbol_cooldowns[symbol]}"
            else:
                # Cooldown expired, cleanup
                del self.symbol_cooldowns[symbol]
                self.symbol_consecutive_losses[symbol] = 0
                self._save_state()
                logger.info(f"Cooldown expired for {symbol}. Resuming trading.")

        # 3. Position Limits
        if len(open_positions) >= self.config.max_open_positions:
            return False, f"Max open positions reached ({self.config.max_open_positions})"

        # 4. Duplicate Position Check
        for pos in open_positions:
            if pos.symbol == symbol:
                return False, f"Position already open for {symbol}"

        return True, "OK"

    def update_trade_outcome(self, symbol: str, pnl: float):
        """Updates internal state based on closed trade PnL."""
        if pnl < 0:
            self.symbol_consecutive_losses[symbol] = self.symbol_consecutive_losses.get(symbol, 0) + 1
            logger.info(f"Recorded loss for {symbol}. Consecutive: {self.symbol_consecutive_losses[symbol]}")
            
            if self.symbol_consecutive_losses[symbol] >= self.config.max_consecutive_losses:
                cooldown_duration = timedelta(minutes=self.config.consecutive_loss_cooldown_minutes)
                self.symbol_cooldowns[symbol] = Clock.now() + cooldown_duration
                logger.warning(f"Symbol {symbol} halted due to consecutive losses.", 
                               count=self.symbol_consecutive_losses[symbol], 
                               until=self.symbol_cooldowns[symbol])
                
                if self.alert_system:
                    import asyncio
                    asyncio.create_task(self.alert_system.send_alert(
                        level='warning',
                        message=f"⚠️ Symbol {symbol} halted for {self.config.consecutive_loss_cooldown_minutes}m due to {self.symbol_consecutive_losses[symbol]} consecutive losses.",
                        details={'symbol': symbol, 'losses': self.symbol_consecutive_losses[symbol]}
                    ))
            self._save_state()
            
        elif pnl > 0:
            if symbol in self.symbol_consecutive_losses and self.symbol_consecutive_losses[symbol] > 0:
                self.symbol_consecutive_losses[symbol] = 0
                self._save_state()
                logger.info(f"Recorded win for {symbol}. Consecutive losses reset.")

    async def update_portfolio_risk(self, portfolio_value: float, daily_realized_pnl: float):
        """Updates portfolio-level risk metrics and triggers circuit breakers."""
        if self.peak_portfolio_value is None:
            self.peak_portfolio_value = portfolio_value
        if self.initial_capital is None:
            self.initial_capital = portfolio_value

        # Update High Water Mark
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
            await self.position_manager.update_portfolio_high_water_mark(portfolio_value)

        # Calculate Drawdown
        self.current_drawdown = (portfolio_value - self.peak_portfolio_value) / self.peak_portfolio_value if self.peak_portfolio_value > 0 else 0.0

        # Check Circuit Breaker
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
                
                if self.config.close_positions_on_halt and not self.liquidation_triggered:
                    self.liquidation_needed = True
                    self.liquidation_triggered = True
                    logger.warning("Emergency liquidation triggered by Circuit Breaker.")
        else:
            if self.circuit_breaker_halted:
                self.circuit_breaker_halted = False
                self.liquidation_triggered = False
                logger.info("Trading resumed as portfolio recovered from drawdown.")

        # Check Daily Loss Limit
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
                
                if self.config.close_positions_on_halt and not self.liquidation_triggered:
                    self.liquidation_needed = True
                    self.liquidation_triggered = True
                    logger.warning("Emergency liquidation triggered by Max Daily Loss.")

    def _get_regime_param(self, param_name: str, regime: Optional[str]) -> Any:
        """Helper to get risk parameter with regime override."""
        default_value = getattr(self.config, param_name)
        if regime and hasattr(self.config.regime_based_risk, regime):
            regime_config = getattr(self.config.regime_based_risk, regime)
            if hasattr(regime_config, param_name):
                override_value = getattr(regime_config, param_name)
                if override_value is not None:
                    return override_value
        return default_value

    def calculate_dynamic_trailing_stop(self, pos: Position, current_price: float, atr: Optional[float], market_regime: Optional[str]) -> Tuple[Optional[float], Optional[float], bool]:
        """
        Calculates the new stop loss and reference price based on the current market regime.
        Returns (new_stop_price, new_ref_price, activated_flag) or (None, None, False) if no update needed.
        """
        # 1. Determine Activation
        activated = pos.trailing_stop_active
        if not activated:
            activation_pct = self._get_regime_param('trailing_stop_activation_pct', market_regime)
            if pos.side == 'BUY':
                activation_price = pos.entry_price * (1 + activation_pct)
                if current_price >= activation_price:
                    activated = True
            elif pos.side == 'SELL':
                activation_price = pos.entry_price * (1 - activation_pct)
                if current_price <= activation_price:
                    activated = True
        
        # Update Reference Price (High Water Mark)
        new_ref_price = pos.trailing_ref_price
        if pos.side == 'BUY':
            new_ref_price = max(pos.trailing_ref_price, current_price)
        else:
            new_ref_price = min(pos.trailing_ref_price, current_price)

        if not activated:
            # If not active yet, we just update the ref price if it improved
            if new_ref_price != pos.trailing_ref_price:
                return None, new_ref_price, False
            return None, None, False

        # 2. Calculate Trailing Distance (Regime Aware)
        use_atr = self._get_regime_param('use_atr_for_trailing', market_regime)
        
        trailing_dist = 0.0
        if use_atr and atr is not None and atr > 0:
            multiplier = self._get_regime_param('atr_trailing_multiplier', market_regime)
            trailing_dist = atr * multiplier
        else:
            pct = self._get_regime_param('trailing_stop_pct', market_regime)
            trailing_dist = new_ref_price * pct

        # 3. Calculate New Stop
        new_stop_price = pos.stop_loss_price
        if pos.side == 'BUY':
            potential_stop = new_ref_price - trailing_dist
            # Only move UP
            if potential_stop > pos.stop_loss_price:
                new_stop_price = potential_stop
        else:
            potential_stop = new_ref_price + trailing_dist
            # Only move DOWN
            if potential_stop < pos.stop_loss_price:
                new_stop_price = potential_stop
        
        # Return update if changed
        if new_stop_price != pos.stop_loss_price or new_ref_price != pos.trailing_ref_price or activated != pos.trailing_stop_active:
            return new_stop_price, new_ref_price, activated
        
        return None, None, False

    def calculate_position_size(self, 
                                symbol: str, 
                                portfolio_equity: float, 
                                entry_price: float, 
                                stop_loss_price: float, 
                                open_positions: List[Position], 
                                market_regime: Optional[str] = None,
                                confidence: Optional[float] = None,
                                confidence_threshold: Optional[float] = None,
                                model_metrics: Optional[Dict[str, Any]] = None) -> float:
        """Calculates the optimal position size based on risk parameters, Kelly Criterion, and AI confidence."""
        if entry_price <= 0 or stop_loss_price <= 0:
            logger.warning("Cannot calculate position size with zero or negative prices.", entry=entry_price, sl=stop_loss_price)
            return 0.0

        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit == 0:
            logger.warning("Risk per unit is zero, cannot calculate position size. Check stop-loss logic.")
            return 0.0

        base_risk_pct = self._get_regime_param('risk_per_trade_pct', market_regime)
        base_risk_pct = max(0.001, base_risk_pct)
        
        # --- Kelly Criterion ---
        kelly_risk_pct = None
        if self.config.use_kelly_criterion and model_metrics:
            ensemble_metrics = model_metrics.get('ensemble', model_metrics)
            win_rate = ensemble_metrics.get('win_rate')
            profit_factor = ensemble_metrics.get('profit_factor')
            
            if win_rate is not None and profit_factor is not None and win_rate > 0:
                if win_rate < 1.0:
                    r_ratio = profit_factor * (1.0 - win_rate) / win_rate
                else:
                    r_ratio = 999.0
                
                if r_ratio > 0:
                    raw_kelly = win_rate - (1.0 - win_rate) / r_ratio
                    kelly_risk_pct = raw_kelly * self.config.kelly_fraction
                    kelly_risk_pct = max(0.0, kelly_risk_pct)
        
        if kelly_risk_pct is not None:
            safety_cap = max(0.05, base_risk_pct * 5.0)
            risk_pct = min(kelly_risk_pct, safety_cap)
        else:
            risk_pct = base_risk_pct

        # --- Drawdown Scaling ---
        drawdown_scaling = max(0.2, 1.0 + self.current_drawdown)
        adjusted_risk_pct = risk_pct * drawdown_scaling

        # --- Correlation Penalty ---
        correlation_scaling = 1.0
        if self.config.correlation.enabled:
            max_corr = 0.0
            for pos in open_positions:
                if pos.symbol == symbol:
                    continue
                corr = self.data_handler.get_correlation(symbol, pos.symbol, self.config.correlation.lookback_periods)
                if corr > max_corr:
                    max_corr = corr
            
            if max_corr > self.config.correlation.max_correlation:
                correlation_scaling = self.config.correlation.penalty_factor

        # --- Confidence Scaling ---
        confidence_scaling = 1.0
        if kelly_risk_pct is None and (self.config.confidence_scaling_factor > 0 and 
            confidence is not None and 
            confidence_threshold is not None and 
            confidence > confidence_threshold):
            
            surplus = confidence - confidence_threshold
            raw_scaling = 1.0 + (surplus * self.config.confidence_scaling_factor)
            confidence_scaling = min(raw_scaling, self.config.max_confidence_risk_multiplier)

        final_risk_pct = adjusted_risk_pct * correlation_scaling * confidence_scaling
        risk_amount_usd = portfolio_equity * final_risk_pct
        
        # --- Portfolio Risk Cap ---
        if self.config.max_portfolio_risk_pct > 0:
            current_portfolio_risk = 0.0
            for pos in open_positions:
                if pos.status == PositionStatus.OPEN and pos.stop_loss_price is not None and pos.quantity > 0:
                    p_risk = abs(pos.entry_price - pos.stop_loss_price) * pos.quantity
                    current_portfolio_risk += p_risk
            
            max_allowed_portfolio_risk = portfolio_equity * self.config.max_portfolio_risk_pct
            remaining_risk_budget = max(0.0, max_allowed_portfolio_risk - current_portfolio_risk)
            
            if risk_amount_usd > remaining_risk_budget:
                risk_amount_usd = remaining_risk_budget

        quantity = risk_amount_usd / risk_per_unit
        
        # --- Hard Caps ---
        max_quantity_by_pct = (portfolio_equity * self.config.max_position_size_portfolio_pct) / entry_price
        max_quantity_by_usd_cap = self.config.max_position_size_usd / entry_price

        liquidity_cap_quantity = float('inf')
        if self.config.max_volume_participation_pct > 0:
            df = self.data_handler.get_market_data(symbol)
            if df is not None and not df.empty and 'volume' in df.columns:
                lookback = self.config.volume_lookback_periods
                recent_vol = df['volume'].iloc[-lookback:]
                avg_vol = recent_vol.mean()
                if avg_vol > 0:
                    liquidity_cap_quantity = avg_vol * self.config.max_volume_participation_pct
        
        final_quantity = min(quantity, max_quantity_by_pct, max_quantity_by_usd_cap, liquidity_cap_quantity)
        return final_quantity

    def calculate_stop_loss(self, side: str, entry_price: float, df_with_indicators: Optional[pd.DataFrame], market_regime: Optional[str] = None) -> float:
        """Calculates initial stop loss based on ATR or Swing points."""
        atr_col = self.config.atr_column_name
        atr = 0.0
        if df_with_indicators is not None and not df_with_indicators.empty:
            if atr_col in df_with_indicators.columns:
                atr = df_with_indicators[atr_col].iloc[-1]
        
        if self.config.stop_loss_type == 'SWING' and df_with_indicators is not None and not df_with_indicators.empty:
            lookback = self.config.swing_lookback
            if len(df_with_indicators) >= lookback:
                window = df_with_indicators.iloc[-lookback:]
                buffer = atr * self.config.swing_buffer_atr_multiplier
                
                if side == 'BUY':
                    swing_low = window['low'].min()
                    sl_price = swing_low - buffer
                    if sl_price < entry_price: return sl_price
                else:
                    swing_high = window['high'].max()
                    sl_price = swing_high + buffer
                    if sl_price > entry_price: return sl_price

        atr_multiplier = self._get_regime_param('atr_stop_multiplier', market_regime)

        if atr > 0:
            stop_loss_offset = atr * atr_multiplier
        else:
            stop_loss_offset = entry_price * self.config.stop_loss_fallback_pct

        if side == 'BUY':
            return entry_price - stop_loss_offset
        else:
            return entry_price + stop_loss_offset

    def calculate_take_profit(self, 
                              side: str, 
                              entry_price: float, 
                              stop_loss_price: float, 
                              market_regime: Optional[str] = None,
                              confidence: Optional[float] = None,
                              confidence_threshold: Optional[float] = None) -> float:
        """Calculates take profit target based on Risk:Reward ratio."""
        risk_per_unit = abs(entry_price - stop_loss_price)
        base_rr_ratio = self._get_regime_param('reward_to_risk_ratio', market_regime)
        
        rr_multiplier = 1.0
        if (self.config.confidence_rr_scaling_factor > 0 and 
            confidence is not None and 
            confidence_threshold is not None and 
            confidence > confidence_threshold):
            
            surplus = confidence - confidence_threshold
            raw_scaling = 1.0 + (surplus * self.config.confidence_rr_scaling_factor)
            rr_multiplier = min(raw_scaling, self.config.max_confidence_rr_multiplier)

        final_rr_ratio = base_rr_ratio * rr_multiplier
        profit_target = risk_per_unit * final_rr_ratio

        if side == 'BUY':
            return entry_price + profit_target
        else:
            return entry_price - profit_target

    def check_time_based_exit(self, position: Position, current_price: float) -> bool:
        """Checks if a position has stagnated for too long."""
        cfg = self.config.time_based_exit
        if not cfg.enabled: return False
        
        now = Clock.now()
        duration = now - position.open_timestamp
        if duration.total_seconds() > (cfg.max_hold_time_hours * 3600):
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
