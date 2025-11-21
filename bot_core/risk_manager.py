from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict, TYPE_CHECKING, Tuple
from datetime import datetime, timedelta

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
                # Cooldown expired, clear it
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
    """
    Rejects new positions if the asset is highly correlated with existing positions.
    Acts as a Fail-Fast mechanism before sizing.
    """
    async def check(self, symbol: str, open_positions: List[Position], context: 'RiskManager') -> Tuple[bool, str]:
        if not context.config.correlation.enabled:
            return True, "OK"
        
        # Skip if no open positions
        if not open_positions:
            return True, "OK"

        max_corr = 0.0
        conflicting_symbol = ""
        
        for pos in open_positions:
            if pos.symbol == symbol: 
                continue
            
            # Calculate correlation via DataHandler
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

# --- Position Sizer Component ---

class PositionSizer:
    def __init__(self, config: RiskManagementConfig, data_handler: 'DataHandler'):
        self.config = config
        self.data_handler = data_handler

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

        # 4. Correlation Penalty (Sizing Reduction)
        # Even if CorrelationRule passes, we might still want to reduce size if correlation is moderate
        risk_pct = self._apply_correlation_penalty(symbol, risk_pct, open_positions)

        # 5. Calculate USD Risk
        risk_usd = equity * risk_pct

        # 6. Portfolio Risk Cap
        risk_usd = self._apply_portfolio_cap(risk_usd, equity, open_positions, entry_price, stop_loss)

        # 7. Convert to Quantity
        quantity = risk_usd / risk_per_unit

        # 8. Hard Caps
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
        # Note: High correlations are rejected by CorrelationRule. 
        # This applies a penalty for moderate correlations if configured.
        max_corr = 0.0
        for pos in open_positions:
            if pos.symbol == symbol: continue
            c = self.data_handler.get_correlation(symbol, pos.symbol, self.config.correlation.lookback_periods)
            if c > max_corr: max_corr = c
        
        # If correlation is high but passed the rule (e.g. rule threshold is 0.9, this is 0.8)
        # we apply a penalty factor.
        if max_corr > (self.config.correlation.max_correlation * 0.8):
            return risk_pct * self.config.correlation.penalty_factor
        return risk_pct

    def _apply_portfolio_cap(self, risk_usd: float, equity: float, open_positions: List[Position], entry: float, sl: float) -> float:
        if self.config.max_portfolio_risk_pct <= 0: return risk_usd
        current_risk = sum(abs(p.entry_price - (p.stop_loss_price or p.entry_price)) * p.quantity for p in open_positions)
        max_risk = equity * self.config.max_portfolio_risk_pct
        return min(risk_usd, max(0.0, max_risk - current_risk))

    def _apply_hard_caps(self, symbol: str, qty: float, equity: float, price: float) -> float:
        max_by_pct = (equity * self.config.max_position_size_portfolio_pct) / price
        max_by_usd = self.config.max_position_size_usd / price
        liquidity_cap = float('inf')
        if self.config.max_volume_participation_pct > 0:
            df = self.data_handler.get_market_data(symbol)
            if df is not None and 'volume' in df.columns:
                avg_vol = df['volume'].iloc[-self.config.volume_lookback_periods:].mean()
                liquidity_cap = avg_vol * self.config.max_volume_participation_pct
        return min(qty, max_by_pct, max_by_usd, liquidity_cap)

# --- Main Risk Manager ---

class RiskManager:
    """
    Authoritative Gatekeeper for trading decisions.
    Uses a modular 'Risk Engine' architecture.
    """
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
        
        # Initialize Components
        self.sizer = PositionSizer(config, data_handler)
        self.rules: List[RiskRule] = [
            CircuitBreakerRule(),
            CooldownRule(),
            MaxPositionsRule(),
            DuplicatePositionRule(),
            CorrelationRule() # Added Correlation Rule
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
        """Executes the chain of risk rules."""
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
        
        # Drawdown Scaling (Global override)
        if self.current_drawdown < -0.05:
            qty *= 0.75
            
        return qty

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
