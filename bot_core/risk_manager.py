from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict, TYPE_CHECKING, Tuple
from datetime import datetime, timedelta
import numpy as np

from bot_core.config import RiskManagementConfig
from bot_core.logger import get_logger
from bot_core.position_manager import Position
from bot_core.utils import Clock

if TYPE_CHECKING:
    from bot_core.monitoring import AlertSystem
    from bot_core.position_manager import PositionManager
    from bot_core.data_handler import DataHandler

logger = get_logger(__name__)

# --- Risk Rules (Chain of Responsibility) ---

class RiskRule(ABC):
    @abstractmethod
    async def check(self, symbol: str, open_positions: List[Position], context: 'RiskManager') -> Tuple[bool, str]:
        pass

class CircuitBreakerRule(RiskRule):
    async def check(self, symbol: str, open_positions: List[Position], context: 'RiskManager') -> Tuple[bool, str]:
        if context.circuit_breaker_halted:
            return False, "Circuit Breaker Active"
        if context.daily_loss_halted:
            return False, "Max Daily Loss Reached"
        return True, "OK"

class CooldownRule(RiskRule):
    async def check(self, symbol: str, open_positions: List[Position], context: 'RiskManager') -> Tuple[bool, str]:
        if symbol in context.symbol_cooldowns:
            if Clock.now() < context.symbol_cooldowns[symbol]:
                return False, f"Cooldown active until {context.symbol_cooldowns[symbol]}"
            else:
                del context.symbol_cooldowns[symbol]
                context.symbol_consecutive_losses[symbol] = 0
                await context._save_symbol_state(symbol)
        return True, "OK"

class MaxPositionsRule(RiskRule):
    async def check(self, symbol: str, open_positions: List[Position], context: 'RiskManager') -> Tuple[bool, str]:
        if len(open_positions) >= context.config.max_open_positions:
            return False, "Max open positions reached"
        return True, "OK"

class DuplicatePositionRule(RiskRule):
    async def check(self, symbol: str, open_positions: List[Position], context: 'RiskManager') -> Tuple[bool, str]:
        for pos in open_positions:
            if pos.symbol == symbol:
                return False, "Position already open"
        return True, "OK"

class CorrelationRule(RiskRule):
    async def check(self, symbol: str, open_positions: List[Position], context: 'RiskManager') -> Tuple[bool, str]:
        if not context.config.correlation.enabled:
            return True, "OK"
        
        if not open_positions:
            return True, "OK"

        max_corr = 0.0
        conflicting_symbol = ""
        
        for pos in open_positions:
            if pos.symbol == symbol: 
                continue
            
            c = context.data_handler.get_correlation(
                symbol, 
                pos.symbol, 
                context.config.correlation.lookback_periods
            )
            
            if c > max_corr:
                max_corr = c
                conflicting_symbol = pos.symbol

        if max_corr > context.config.correlation.max_correlation:
            return False, f"High Correlation ({max_corr:.2f}) with {conflicting_symbol}"
            
        return True, "OK"

class VaRCheckRule(RiskRule):
    """Predictive check: Will adding this position breach the Portfolio VaR limit?"""
    async def check(self, symbol: str, open_positions: List[Position], context: 'RiskManager') -> Tuple[bool, str]:
        if not context.config.var.enabled:
            return True, "OK"
        
        # We can't fully check the *future* VaR here without knowing the size,
        # but we can check if the *current* VaR is already breached.
        current_var = context.calculate_portfolio_var(open_positions)
        portfolio_value = context.position_manager.get_portfolio_value({}, open_positions) # Approx
        
        if portfolio_value > 0:
            var_pct = current_var / portfolio_value
            if var_pct > context.config.var.max_portfolio_var_pct:
                return False, f"Portfolio VaR ({var_pct:.2%}) exceeds limit ({context.config.var.max_portfolio_var_pct:.2%})"
        
        return True, "OK"

# --- Position Sizer Component ---

class PositionSizer:
    def __init__(self, config: RiskManagementConfig, data_handler: 'DataHandler', risk_manager: 'RiskManager'):
        self.config = config
        self.data_handler = data_handler
        self.risk_manager = risk_manager

    def calculate(self, 
                  symbol: str, 
                  equity: float, 
                  entry_price: float, 
                  stop_loss: float, 
                  open_positions: List[Position], 
                  regime: Optional[str],
                  confidence: Optional[float],
                  confidence_threshold: Optional[float],
                  metrics: Optional[Dict]) -> float:
        
        if entry_price <= 0 or stop_loss <= 0: return 0.0
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit == 0: return 0.0

        # 1. Base Risk
        base_risk_pct = self._get_regime_param('risk_per_trade_pct', regime)
        
        # 2. Kelly Criterion
        risk_pct = self._apply_kelly(base_risk_pct, metrics)
        
        # 3. Confidence Scaling
        risk_pct = self._apply_confidence(risk_pct, confidence, confidence_threshold)

        # 4. Correlation Penalty
        risk_pct = self._apply_correlation_penalty(symbol, risk_pct, open_positions)

        # 5. Calculate USD Risk
        risk_usd = equity * risk_pct

        # 6. Portfolio Risk Cap (Hard Dollar Stop)
        risk_usd = self._apply_portfolio_cap(risk_usd, equity, open_positions, entry_price, stop_loss)

        # 7. VaR-Based Sizing Cap (Statistical Risk)
        risk_usd = self._apply_var_cap(symbol, risk_usd, equity, open_positions)

        # 8. Convert to Quantity
        quantity = risk_usd / risk_per_unit

        # 9. Hard Caps (Liquidity, Max Size)
        quantity = self._apply_hard_caps(symbol, quantity, equity, entry_price)

        return quantity

    def _get_regime_param(self, param: str, regime: Optional[str]) -> Any:
        default = getattr(self.config, param)
        if regime and hasattr(self.config.regime_based_risk, regime):
            override = getattr(getattr(self.config.regime_based_risk, regime), param)
            if override is not None: return override
        return default

    def _apply_kelly(self, base_risk: float, metrics: Optional[Dict]) -> float:
        if not self.config.use_kelly_criterion or not metrics:
            return base_risk
        ensemble = metrics.get('ensemble', metrics)
        win_rate = ensemble.get('win_rate', 0.0)
        profit_factor = ensemble.get('profit_factor', 0.0)
        if win_rate <= 0 or profit_factor <= 0: return base_risk
        if win_rate >= 1.0: return base_risk * 2.0
        r_ratio = profit_factor * (1.0 - win_rate) / win_rate
        if r_ratio <= 0: return base_risk
        kelly = win_rate - (1.0 - win_rate) / r_ratio
        return max(0.0, min(kelly * self.config.kelly_fraction, base_risk * 5.0, 0.05))

    def _apply_confidence(self, risk_pct: float, confidence: Optional[float], threshold: Optional[float]) -> float:
        if self.config.confidence_scaling_factor <= 0 or not confidence or not threshold:
            return risk_pct
        if confidence <= threshold: return risk_pct
        surplus = confidence - threshold
        scaler = 1.0 + (surplus * self.config.confidence_scaling_factor)
        return risk_pct * min(scaler, self.config.max_confidence_risk_multiplier)

    def _apply_correlation_penalty(self, symbol: str, risk_pct: float, open_positions: List[Position]) -> float:
        if not self.config.correlation.enabled: return risk_pct
        max_corr = 0.0
        for pos in open_positions:
            if pos.symbol == symbol: continue
            c = self.data_handler.get_correlation(symbol, pos.symbol, self.config.correlation.lookback_periods)
            if c > max_corr: max_corr = c
        
        if max_corr > (self.config.correlation.max_correlation * 0.8):
            return risk_pct * self.config.correlation.penalty_factor
        return risk_pct

    def _apply_portfolio_cap(self, risk_usd: float, equity: float, open_positions: List[Position], entry: float, sl: float) -> float:
        if self.config.max_portfolio_risk_pct <= 0: return risk_usd
        current_risk = sum(abs(p.entry_price - (p.stop_loss_price or p.entry_price)) * p.quantity for p in open_positions)
        max_risk = equity * self.config.max_portfolio_risk_pct
        return min(risk_usd, max(0.0, max_risk - current_risk))

    def _apply_var_cap(self, symbol: str, risk_usd: float, equity: float, open_positions: List[Position]) -> float:
        if not self.config.var.enabled: return risk_usd
        
        # Calculate current Portfolio VaR
        current_var = self.risk_manager.calculate_portfolio_var(open_positions)
        max_var = equity * self.config.var.max_portfolio_var_pct
        remaining_var_budget = max(0.0, max_var - current_var)
        
        # Estimate Marginal VaR of the new trade
        # Simplified: VaR_trade = Position_Value * Volatility * Z_score
        # We use the asset's standalone VaR as a conservative estimate for Marginal VaR
        # (assuming correlation=1 in worst case, or we could use actual correlation)
        
        df = self.data_handler.get_market_data(symbol)
        if df is None or df.empty:
            return risk_usd # Cannot calculate, default to base risk
            
        returns = df['close'].pct_change().dropna()
        if len(returns) < 20: return risk_usd
        
        # Historical VaR of the asset
        asset_var_pct = np.percentile(returns, (1.0 - self.config.var.confidence_level) * 100)
        asset_var_pct = abs(asset_var_pct)
        
        if asset_var_pct == 0: return risk_usd
        
        # Max position value allowed by VaR budget
        # VaR_trade = Position_Value * asset_var_pct
        # Position_Value = VaR_trade / asset_var_pct
        max_pos_value_var = remaining_var_budget / asset_var_pct
        
        # Convert Position Value to Risk USD (approximate via Stop Loss distance)
        # This is tricky because Risk USD != Position Value.
        # We return the Risk USD that corresponds to this Position Value cap.
        # Since we don't have the exact SL distance here easily without passing it around,
        # we simply cap the risk_usd if it implies a position size larger than max_pos_value_var.
        
        # Actually, we can just return risk_usd, but we need to ensure the resulting quantity
        # doesn't create a position value > max_pos_value_var.
        # We'll handle this by returning a capped risk_usd, assuming risk_usd is proportional to size.
        
        return risk_usd # The hard cap is better applied at quantity level, but we do what we can here.
        # Better approach: We calculate the max quantity allowed by VaR in _apply_hard_caps logic or similar.
        # For now, let's just use the remaining budget to scale down if needed.
        
        return min(risk_usd, remaining_var_budget * 5.0) # Heuristic: Risk is usually ~1-5% of Pos Value. 

    def _apply_hard_caps(self, symbol: str, qty: float, equity: float, price: float) -> float:
        max_by_pct = (equity * self.config.max_position_size_portfolio_pct) / price
        max_by_usd = self.config.max_position_size_usd / price
        liquidity_cap = float('inf')
        if self.config.max_volume_participation_pct > 0:
            df = self.data_handler.get_market_data(symbol)
            if df is not None and 'volume' in df.columns:
                avg_vol = df['volume'].iloc[-self.config.volume_lookback_periods:].mean()
                liquidity_cap = avg_vol * self.config.max_volume_participation_pct
        
        # VaR Cap (Quantity based)
        var_cap = float('inf')
        if self.config.var.enabled:
             current_var = self.risk_manager.calculate_portfolio_var(self.risk_manager.position_manager._position_cache.values())
             max_var = equity * self.config.var.max_portfolio_var_pct
             remaining = max(0.0, max_var - current_var)
             
             df = self.data_handler.get_market_data(symbol)
             if df is not None:
                 returns = df['close'].pct_change().dropna()
                 if not returns.empty:
                     asset_var_pct = abs(np.percentile(returns, (1.0 - self.config.var.confidence_level) * 100))
                     if asset_var_pct > 0:
                         var_cap = remaining / (asset_var_pct * price)

        return min(qty, max_by_pct, max_by_usd, liquidity_cap, var_cap)

# --- Main Risk Manager ---

class RiskManager:
    def __init__(self, config: RiskManagementConfig, position_manager: 'PositionManager', data_handler: 'DataHandler', alert_system: Optional['AlertSystem'] = None):
        self.config = config
        self.position_manager = position_manager
        self.data_handler = data_handler
        self.alert_system = alert_system
        
        self.circuit_breaker_halted = False
        self.daily_loss_halted = False
        self.peak_portfolio_value = None
        self.current_drawdown = 0.0
        
        self.symbol_consecutive_losses: Dict[str, int] = {}
        self.symbol_cooldowns: Dict[str, datetime] = {}
        
        self.liquidation_needed = False
        
        self.sizer = PositionSizer(config, data_handler, self)
        self.rules: List[RiskRule] = [
            CircuitBreakerRule(),
            CooldownRule(),
            MaxPositionsRule(),
            DuplicatePositionRule(),
            CorrelationRule(),
            VaRCheckRule()
        ]
        
        logger.info("RiskManager initialized with modular Risk Engine.")

    async def initialize(self):
        state = await self.position_manager.get_portfolio_state()
        if state:
            self.peak_portfolio_value = state['peak_equity']
        await self._load_state()

    async def _load_state(self):
        try:
            risk_states = await self.position_manager.get_all_risk_states()
            for symbol, state in risk_states.items():
                self.symbol_consecutive_losses[symbol] = state['consecutive_losses']
                if state['cooldown_until']:
                    self.symbol_cooldowns[symbol] = state['cooldown_until']
            logger.info("Risk state loaded from database.")
        except Exception as e:
            logger.error("Failed to load risk state", error=str(e))

    async def _save_symbol_state(self, symbol: str):
        try:
            losses = self.symbol_consecutive_losses.get(symbol, 0)
            cooldown = self.symbol_cooldowns.get(symbol)
            await self.position_manager.update_risk_state(symbol, losses, cooldown)
        except Exception as e:
            logger.error("Failed to save risk state", symbol=symbol, error=str(e))

    @property
    def is_halted(self) -> bool:
        return self.circuit_breaker_halted or self.daily_loss_halted

    async def validate_entry(self, symbol: str, open_positions: List[Position]) -> Tuple[bool, str]:
        for rule in self.rules:
            passed, reason = await rule.check(symbol, open_positions, self)
            if not passed:
                return False, reason
        return True, "OK"

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
        
        qty = self.sizer.calculate(
            symbol, portfolio_equity, entry_price, stop_loss_price, 
            open_positions, market_regime, confidence, confidence_threshold, model_metrics
        )
        
        if self.current_drawdown < -0.05:
            qty *= 0.75
            
        return qty

    def calculate_portfolio_var(self, open_positions: List[Position]) -> float:
        """
        Calculates the Historical Value at Risk (VaR) for the current portfolio.
        Uses historical simulation of the current portfolio weights applied to past returns.
        """
        if not open_positions:
            return 0.0
            
        # 1. Collect historical returns for all active assets
        returns_map = {}
        min_len = float('inf')
        
        for pos in open_positions:
            df = self.data_handler.get_market_data(pos.symbol)
            if df is not None and not df.empty:
                # Get recent returns
                rets = df['close'].pct_change().tail(self.config.var.lookback_periods).dropna()
                if not rets.empty:
                    returns_map[pos.symbol] = rets
                    min_len = min(min_len, len(rets))
        
        if not returns_map:
            return 0.0
            
        # 2. Align returns (truncate to shortest length)
        aligned_returns = {}
        for sym, rets in returns_map.items():
            aligned_returns[sym] = rets.iloc[-min_len:].values
            
        # 3. Simulate Portfolio PnL history
        # PnL_t = Sum(Position_Value_i * Return_i_t)
        portfolio_pnl_history = np.zeros(min_len)
        
        for pos in open_positions:
            if pos.symbol in aligned_returns:
                pos_value = pos.quantity * pos.entry_price # Approx current value base
                asset_returns = aligned_returns[pos.symbol]
                portfolio_pnl_history += (pos_value * asset_returns)
        
        # 4. Calculate VaR (Percentile of PnL history)
        # 95% VaR is the 5th percentile of the PnL distribution
        var_threshold = (1.0 - self.config.var.confidence_level) * 100
        portfolio_var = np.percentile(portfolio_pnl_history, var_threshold)
        
        # VaR is typically expressed as a positive loss number
        return abs(min(0.0, portfolio_var))

    def calculate_stop_loss(self, side: str, entry_price: float, df: Any, market_regime: Optional[str] = None) -> float:
        atr = 0.0
        if df is not None and self.config.atr_column_name in df.columns:
            atr = df[self.config.atr_column_name].iloc[-1]
            
        if self.config.stop_loss_type == 'SWING' and df is not None:
            lookback = self.config.swing_lookback
            if len(df) >= lookback:
                window = df.iloc[-lookback:]
                buffer = atr * self.config.swing_buffer_atr_multiplier
                if side == 'BUY':
                    return min(entry_price * 0.99, window['low'].min() - buffer)
                else:
                    return max(entry_price * 1.01, window['high'].max() + buffer)

        mult = self.sizer._get_regime_param('atr_stop_multiplier', market_regime)
        dist = (atr * mult) if atr > 0 else (entry_price * self.config.stop_loss_fallback_pct)
        return entry_price - dist if side == 'BUY' else entry_price + dist

    def calculate_take_profit(self, side: str, entry: float, sl: float, market_regime: Optional[str], confidence: Optional[float], confidence_threshold: Optional[float]) -> float:
        risk = abs(entry - sl)
        rr = self.sizer._get_regime_param('reward_to_risk_ratio', market_regime)
        
        if self.config.confidence_rr_scaling_factor > 0 and confidence and confidence_threshold and confidence > confidence_threshold:
            surplus = confidence - confidence_threshold
            scaler = 1.0 + (surplus * self.config.confidence_rr_scaling_factor)
            rr *= min(scaler, self.config.max_confidence_rr_multiplier)
            
        dist = risk * rr
        return entry + dist if side == 'BUY' else entry - dist

    async def update_trade_outcome(self, symbol: str, pnl: float):
        if pnl < 0:
            self.symbol_consecutive_losses[symbol] = self.symbol_consecutive_losses.get(symbol, 0) + 1
            if self.symbol_consecutive_losses[symbol] >= self.config.max_consecutive_losses:
                self.symbol_cooldowns[symbol] = Clock.now() + timedelta(minutes=self.config.consecutive_loss_cooldown_minutes)
                logger.warning(f"Symbol {symbol} halted due to consecutive losses.")
        else:
            self.symbol_consecutive_losses[symbol] = 0
        await self._save_symbol_state(symbol)

    async def update_portfolio_risk(self, portfolio_value: float, daily_pnl: float):
        if self.peak_portfolio_value is None or portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
            await self.position_manager.update_portfolio_high_water_mark(portfolio_value)
            
        self.current_drawdown = (portfolio_value - self.peak_portfolio_value) / self.peak_portfolio_value
        
        if self.current_drawdown < self.config.circuit_breaker_threshold:
            if not self.circuit_breaker_halted:
                self.circuit_breaker_halted = True
                self.liquidation_needed = self.config.close_positions_on_halt
                logger.critical("CIRCUIT BREAKER TRIPPED.")
        elif self.circuit_breaker_halted:
            self.circuit_breaker_halted = False
            logger.info("Circuit breaker reset.")
            
        if self.config.max_daily_loss_usd > 0 and daily_pnl < -self.config.max_daily_loss_usd:
            if not self.daily_loss_halted:
                self.daily_loss_halted = True
                self.liquidation_needed = self.config.close_positions_on_halt
                logger.critical("MAX DAILY LOSS REACHED.")

    def calculate_dynamic_trailing_stop(self, pos: Position, current_price: float, atr: float, regime: str) -> Tuple[Optional[float], Optional[float], bool]:
        if not self.config.use_trailing_stop:
            return None, None, False

        base_multiplier = self.config.atr_trailing_multiplier
        if regime == 'volatile': base_multiplier *= 1.5
        elif regime == 'bull' and pos.side == 'BUY': base_multiplier *= 0.8
        elif regime == 'bear' and pos.side == 'SELL': base_multiplier *= 0.8

        roi_pct = 0.0
        if pos.entry_price > 0:
            roi_pct = (current_price - pos.entry_price) / pos.entry_price if pos.side == 'BUY' else (pos.entry_price - current_price) / pos.entry_price
        
        acceleration = max(0.0, (roi_pct * 100) * 0.1)
        final_multiplier = max(1.0, base_multiplier - acceleration)
        distance = (atr * final_multiplier) if atr > 0 else (current_price * self.config.trailing_stop_pct)

        new_stop, new_ref = None, None
        current_ref = pos.trailing_ref_price or pos.entry_price

        if pos.side == 'BUY':
            new_ref = max(current_ref, current_price)
            potential_stop = new_ref - distance
            if potential_stop > (pos.stop_loss_price or 0.0):
                new_stop = potential_stop
        else:
            new_ref = min(current_ref, current_price)
            potential_stop = new_ref + distance
            if potential_stop < (pos.stop_loss_price or float('inf')):
                new_stop = potential_stop

        activated = pos.trailing_stop_active or (roi_pct >= self.config.trailing_stop_activation_pct)
        if not activated:
            return None, new_ref, False

        return new_stop, new_ref, True

    def check_time_based_exit(self, pos: Position, price: float) -> bool:
        if not self.config.time_based_exit.enabled: return False
        duration = Clock.now() - pos.open_timestamp
        if duration.total_seconds() > (self.config.time_based_exit.max_hold_time_hours * 3600):
            pnl_pct = (price - pos.entry_price) / pos.entry_price
            if pos.side == 'SELL': pnl_pct = -pnl_pct
            return pnl_pct < self.config.time_based_exit.threshold_pct
        return False
