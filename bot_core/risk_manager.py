from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict, TYPE_CHECKING, Tuple
from datetime import datetime, timedelta
import asyncio
import numpy as np
from decimal import Decimal

from bot_core.config import RiskManagementConfig
from bot_core.logger import get_logger
from bot_core.position_manager import Position
from bot_core.utils import Clock
from bot_core.common import to_decimal, ZERO, ONE

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
        
        cached_corrs = context.cached_correlations.get(symbol, {})
        for pos_symbol, corr_value in cached_corrs.items():
            if corr_value > context.config.correlation.max_correlation:
                return False, f"High Correlation ({corr_value:.2f}) with {pos_symbol}"
            
        return True, "OK"

class VaRCheckRule(RiskRule):
    async def check(self, symbol: str, open_positions: List[Position], context: 'RiskManager') -> Tuple[bool, str]:
        if not context.config.var.enabled:
            return True, "OK"
        
        current_var_pct = context.cached_portfolio_var_pct
        limit_pct = context.config.var.max_portfolio_var_pct
        
        if current_var_pct > limit_pct:
            return False, f"Portfolio VaR ({current_var_pct:.2%}) exceeds limit ({limit_pct:.2%})"
        
        return True, "OK"

# --- Position Sizer Component ---

class PositionSizer:
    def __init__(self, config: RiskManagementConfig, data_handler: 'DataHandler', risk_manager: 'RiskManager'):
        self.config = config
        self.data_handler = data_handler
        self.risk_manager = risk_manager

    def calculate(self, 
                  symbol: str, 
                  equity: Decimal, 
                  entry_price: Decimal, 
                  stop_loss: Decimal, 
                  open_positions: List[Position], 
                  regime: Optional[str],
                  confidence: Optional[float],
                  confidence_threshold: Optional[float],
                  metrics: Optional[Dict]) -> Decimal:
        
        if entry_price <= ZERO or stop_loss <= ZERO: return ZERO
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit == ZERO: return ZERO

        # 1. Base Risk
        base_risk_pct = to_decimal(self._get_regime_param('risk_per_trade_pct', regime))
        
        # 2. Kelly Criterion
        risk_pct = self._apply_kelly(base_risk_pct, metrics)
        
        # 3. Confidence Scaling
        risk_pct = self._apply_confidence(risk_pct, confidence, confidence_threshold)

        # 4. Correlation Penalty
        risk_pct = self._apply_correlation_penalty(symbol, risk_pct)

        # 5. Calculate USD Risk
        risk_usd = equity * risk_pct

        # 6. Portfolio Risk Cap (Hard Dollar Stop)
        risk_usd = self._apply_portfolio_cap(risk_usd, equity, open_positions)

        # 7. VaR-Based Sizing Cap (Statistical Risk)
        risk_usd = self._apply_var_cap(symbol, risk_usd, equity)

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

    def _apply_kelly(self, base_risk: Decimal, metrics: Optional[Dict]) -> Decimal:
        if not self.config.use_kelly_criterion or not metrics:
            return base_risk
        ensemble = metrics.get('ensemble', metrics)
        win_rate = to_decimal(ensemble.get('win_rate', 0.0))
        profit_factor = to_decimal(ensemble.get('profit_factor', 0.0))
        
        if win_rate <= ZERO or profit_factor <= ZERO: return base_risk
        if win_rate >= ONE: return base_risk * Decimal("2.0")
        
        r_ratio = profit_factor * (ONE - win_rate) / win_rate
        if r_ratio <= ZERO: return base_risk
        
        kelly = win_rate - (ONE - win_rate) / r_ratio
        kelly_fraction = to_decimal(self.config.kelly_fraction)
        
        # Cap Kelly at 5x base risk or 5% total equity for safety
        max_risk = min(base_risk * Decimal("5.0"), Decimal("0.05"))
        return max(ZERO, min(kelly * kelly_fraction, max_risk))

    def _apply_confidence(self, risk_pct: Decimal, confidence: Optional[float], threshold: Optional[float]) -> Decimal:
        if self.config.confidence_scaling_factor <= 0 or not confidence or not threshold:
            return risk_pct
        
        conf_dec = to_decimal(confidence)
        thresh_dec = to_decimal(threshold)
        
        if conf_dec <= thresh_dec: return risk_pct
        
        surplus = conf_dec - thresh_dec
        scaler = ONE + (surplus * to_decimal(self.config.confidence_scaling_factor))
        max_mult = to_decimal(self.config.max_confidence_risk_multiplier)
        return risk_pct * min(scaler, max_mult)

    def _apply_correlation_penalty(self, symbol: str, risk_pct: Decimal) -> Decimal:
        if not self.config.correlation.enabled: return risk_pct
        
        cached_corrs = self.risk_manager.cached_correlations.get(symbol, {})
        max_corr = 0.0
        if cached_corrs:
            max_corr = max(cached_corrs.values())
        
        if max_corr > (self.config.correlation.max_correlation * 0.8):
            return risk_pct * to_decimal(self.config.correlation.penalty_factor)
        return risk_pct

    def _apply_portfolio_cap(self, risk_usd: Decimal, equity: Decimal, open_positions: List[Position]) -> Decimal:
        if self.config.max_portfolio_risk_pct <= 0: return risk_usd
        
        current_risk = ZERO
        for p in open_positions:
            sl = p.stop_loss_price or p.entry_price
            current_risk += abs(p.entry_price - sl) * p.quantity
            
        max_risk = equity * to_decimal(self.config.max_portfolio_risk_pct)
        return min(risk_usd, max(ZERO, max_risk - current_risk))

    def _apply_var_cap(self, symbol: str, risk_usd: Decimal, equity: Decimal) -> Decimal:
        if not self.config.var.enabled: return risk_usd
        
        current_var_pct = self.risk_manager.cached_portfolio_var_pct
        max_var_pct = self.config.var.max_portfolio_var_pct
        
        if current_var_pct >= max_var_pct:
            return ZERO
        return risk_usd

    def _apply_hard_caps(self, symbol: str, qty: Decimal, equity: Decimal, price: Decimal) -> Decimal:
        max_by_pct = (equity * to_decimal(self.config.max_position_size_portfolio_pct)) / price
        max_by_usd = to_decimal(self.config.max_position_size_usd) / price
        liquidity_cap = Decimal('Infinity')
        
        if self.config.max_volume_participation_pct > 0:
            df = self.data_handler.get_market_data(symbol)
            if df is not None and 'volume' in df.columns:
                avg_vol = df['volume'].iloc[-self.config.volume_lookback_periods:].mean()
                liquidity_cap = to_decimal(avg_vol) * to_decimal(self.config.max_volume_participation_pct)
        
        return min(qty, max_by_pct, max_by_usd, liquidity_cap)

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
        
        self.cached_portfolio_var_pct: float = 0.0
        self.cached_correlations: Dict[str, Dict[str, float]] = {}
        self._metrics_lock = asyncio.Lock()
        self.running = False
        
        self.sizer = PositionSizer(config, data_handler, self)
        self.rules: List[RiskRule] = [
            CircuitBreakerRule(),
            CooldownRule(),
            MaxPositionsRule(),
            DuplicatePositionRule(),
            CorrelationRule(),
            VaRCheckRule()
        ]
        
        logger.info("RiskManager initialized with Asynchronous Risk Engine.")

    async def initialize(self):
        state = await self.position_manager.get_portfolio_state()
        if state:
            self.peak_portfolio_value = float(state['peak_equity'])
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

    async def run_metrics_loop(self):
        self.running = True
        logger.info("Starting Risk Metrics background loop.")
        while self.running:
            try:
                await self._recalculate_portfolio_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in Risk Metrics loop", error=str(e))
            await asyncio.sleep(5)

    async def stop(self):
        self.running = False
        logger.info("RiskManager stopped.")

    async def _recalculate_portfolio_metrics(self):
        open_positions = await self.position_manager.get_all_open_positions()
        
        # VaR Calculation (using floats for statistical estimation)
        var_val = self._calculate_portfolio_var_internal(open_positions)
        portfolio_val = float(self.position_manager.get_portfolio_value({}, open_positions))
        var_pct = var_val / portfolio_val if portfolio_val > 0 else 0.0
        
        corrs = {}
        if self.config.correlation.enabled and len(open_positions) > 0:
            symbols = list(set([p.symbol for p in open_positions]))
            for i, sym_a in enumerate(symbols):
                corrs[sym_a] = {}
                for sym_b in symbols:
                    if sym_a == sym_b: continue
                    c = self.data_handler.get_correlation(sym_a, sym_b, self.config.correlation.lookback_periods)
                    corrs[sym_a][sym_b] = c

        async with self._metrics_lock:
            self.cached_portfolio_var_pct = var_pct
            self.cached_correlations = corrs
            
        if var_pct > self.config.var.max_portfolio_var_pct * 0.8:
            logger.warning("Portfolio VaR is high", var_pct=f"{var_pct:.2%}")

    def _calculate_portfolio_var_internal(self, open_positions: List[Position]) -> float:
        if not open_positions:
            return 0.0
            
        returns_map = {}
        min_len = float('inf')
        
        for pos in open_positions:
            df = self.data_handler.get_market_data(pos.symbol)
            if df is not None and not df.empty:
                rets = df['close'].pct_change().tail(self.config.var.lookback_periods).dropna()
                if not rets.empty:
                    returns_map[pos.symbol] = rets
                    min_len = min(min_len, len(rets))
        
        if not returns_map:
            return 0.0
            
        aligned_returns = {sym: rets.iloc[-min_len:].values for sym, rets in returns_map.items()}
        portfolio_pnl_history = np.zeros(min_len)
        
        for pos in open_positions:
            if pos.symbol in aligned_returns:
                pos_value = float(pos.quantity * pos.entry_price)
                portfolio_pnl_history += (pos_value * aligned_returns[pos.symbol])
        
        var_threshold = (1.0 - self.config.var.confidence_level) * 100
        portfolio_var = np.percentile(portfolio_pnl_history, var_threshold)
        return abs(min(0.0, portfolio_var))

    async def validate_entry(self, symbol: str, open_positions: List[Position]) -> Tuple[bool, str]:
        for rule in self.rules:
            passed, reason = await rule.check(symbol, open_positions, self)
            if not passed:
                return False, reason
        return True, "OK"

    def calculate_position_size(self, 
                                symbol: str, 
                                portfolio_equity: Decimal, 
                                entry_price: Decimal, 
                                stop_loss_price: Decimal, 
                                open_positions: List[Position], 
                                market_regime: Optional[str] = None,
                                confidence: Optional[float] = None,
                                confidence_threshold: Optional[float] = None,
                                model_metrics: Optional[Dict[str, Any]] = None) -> Decimal:
        
        qty = self.sizer.calculate(
            symbol, portfolio_equity, entry_price, stop_loss_price, 
            open_positions, market_regime, confidence, confidence_threshold, model_metrics
        )
        
        if self.current_drawdown < -0.05:
            qty *= Decimal("0.75")
            
        return qty

    def calculate_stop_loss(self, side: str, entry_price: Decimal, df: Any, market_regime: Optional[str] = None) -> Decimal:
        atr = ZERO
        if df is not None and self.config.atr_column_name in df.columns:
            atr = to_decimal(df[self.config.atr_column_name].iloc[-1])
            
        if self.config.stop_loss_type == 'SWING' and df is not None:
            lookback = self.config.swing_lookback
            if len(df) >= lookback:
                window = df.iloc[-lookback:]
                buffer = atr * to_decimal(self.config.swing_buffer_atr_multiplier)
                if side == 'BUY':
                    low_min = to_decimal(window['low'].min())
                    return min(entry_price * Decimal("0.99"), low_min - buffer)
                else:
                    high_max = to_decimal(window['high'].max())
                    return max(entry_price * Decimal("1.01"), high_max + buffer)

        mult = to_decimal(self.sizer._get_regime_param('atr_stop_multiplier', market_regime))
        dist = (atr * mult) if atr > ZERO else (entry_price * to_decimal(self.config.stop_loss_fallback_pct))
        return entry_price - dist if side == 'BUY' else entry_price + dist

    def calculate_take_profit(self, side: str, entry: Decimal, sl: Decimal, market_regime: Optional[str], confidence: Optional[float], confidence_threshold: Optional[float]) -> Decimal:
        risk = abs(entry - sl)
        rr = to_decimal(self.sizer._get_regime_param('reward_to_risk_ratio', market_regime))
        
        if self.config.confidence_rr_scaling_factor > 0 and confidence and confidence_threshold and confidence > confidence_threshold:
            surplus = to_decimal(confidence - confidence_threshold)
            scaler = ONE + (surplus * to_decimal(self.config.confidence_rr_scaling_factor))
            rr *= min(scaler, to_decimal(self.config.max_confidence_rr_multiplier))
            
        dist = risk * rr
        return entry + dist if side == 'BUY' else entry - dist

    async def update_trade_outcome(self, symbol: str, pnl: Decimal):
        if pnl < ZERO:
            self.symbol_consecutive_losses[symbol] = self.symbol_consecutive_losses.get(symbol, 0) + 1
            if self.symbol_consecutive_losses[symbol] >= self.config.max_consecutive_losses:
                self.symbol_cooldowns[symbol] = Clock.now() + timedelta(minutes=self.config.consecutive_loss_cooldown_minutes)
                logger.warning(f"Symbol {symbol} halted due to consecutive losses.")
        else:
            self.symbol_consecutive_losses[symbol] = 0
        await self._save_symbol_state(symbol)

    async def update_portfolio_risk(self, portfolio_value: Decimal, daily_pnl: Decimal):
        pv_float = float(portfolio_value)
        if self.peak_portfolio_value is None or pv_float > self.peak_portfolio_value:
            self.peak_portfolio_value = pv_float
            await self.position_manager.update_portfolio_high_water_mark(portfolio_value)
            
        self.current_drawdown = (pv_float - self.peak_portfolio_value) / self.peak_portfolio_value
        
        if self.current_drawdown < self.config.circuit_breaker_threshold:
            if not self.circuit_breaker_halted:
                self.circuit_breaker_halted = True
                self.liquidation_needed = self.config.close_positions_on_halt
                logger.critical("CIRCUIT BREAKER TRIPPED.")
        elif self.circuit_breaker_halted:
            self.circuit_breaker_halted = False
            logger.info("Circuit breaker reset.")
            
        if self.config.max_daily_loss_usd > 0 and daily_pnl < -to_decimal(self.config.max_daily_loss_usd):
            if not self.daily_loss_halted:
                self.daily_loss_halted = True
                self.liquidation_needed = self.config.close_positions_on_halt
                logger.critical("MAX DAILY LOSS REACHED.")

    def calculate_dynamic_trailing_stop(self, pos: Position, current_price: Decimal, atr: Decimal, regime: str) -> Tuple[Optional[Decimal], Optional[Decimal], bool]:
        if not self.config.use_trailing_stop:
            return None, None, False

        base_multiplier = to_decimal(self.config.atr_trailing_multiplier)
        if regime == 'volatile': base_multiplier *= Decimal("1.5")
        elif regime == 'bull' and pos.side == 'BUY': base_multiplier *= Decimal("0.8")
        elif regime == 'bear' and pos.side == 'SELL': base_multiplier *= Decimal("0.8")

        roi_pct = ZERO
        if pos.entry_price > ZERO:
            roi_pct = (current_price - pos.entry_price) / pos.entry_price if pos.side == 'BUY' else (pos.entry_price - current_price) / pos.entry_price
        
        acceleration = max(ZERO, (roi_pct * 100) * Decimal("0.1"))
        final_multiplier = max(ONE, base_multiplier - acceleration)
        distance = (atr * final_multiplier) if atr > ZERO else (current_price * to_decimal(self.config.trailing_stop_pct))

        new_stop, new_ref = None, None
        current_ref = pos.trailing_ref_price or pos.entry_price

        if pos.side == 'BUY':
            new_ref = max(current_ref, current_price)
            potential_stop = new_ref - distance
            if potential_stop > (pos.stop_loss_price or ZERO):
                new_stop = potential_stop
        else:
            new_ref = min(current_ref, current_price)
            potential_stop = new_ref + distance
            if potential_stop < (pos.stop_loss_price or Decimal('Infinity')):
                new_stop = potential_stop

        activated = pos.trailing_stop_active or (roi_pct >= to_decimal(self.config.trailing_stop_activation_pct))
        if not activated:
            return None, new_ref, False

        return new_stop, new_ref, True

    def check_time_based_exit(self, pos: Position, price: Decimal) -> bool:
        if not self.config.time_based_exit.enabled: return False
        duration = Clock.now() - pos.open_timestamp
        if duration.total_seconds() > (self.config.time_based_exit.max_hold_time_hours * 3600):
            pnl_pct = (price - pos.entry_price) / pos.entry_price
            if pos.side == 'SELL': pnl_pct = -pnl_pct
            return pnl_pct < to_decimal(self.config.time_based_exit.threshold_pct)
        return False
