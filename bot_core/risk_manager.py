from typing import List, Optional, Any, Dict, TYPE_CHECKING, Tuple
import json
import os
from datetime import datetime, timedelta

from bot_core.config import RiskManagementConfig
from bot_core.logger import get_logger
from bot_core.position_manager import Position
from bot_core.utils import Clock, AsyncAtomicJsonStore

if TYPE_CHECKING:
    from bot_core.monitoring import AlertSystem
    from bot_core.position_manager import PositionManager
    from bot_core.data_handler import DataHandler

logger = get_logger(__name__)

class RiskManager:
    """
    Authoritative Gatekeeper for trading decisions.
    Enforces portfolio constraints, circuit breakers, and dynamic position sizing.
    Uses async I/O for state persistence to avoid blocking the event loop.
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
        self.state_store = AsyncAtomicJsonStore("risk_state.json")
        
        self.liquidation_needed = False
        self.liquidation_triggered = False
        
        logger.info("RiskManager initialized.")

    async def initialize(self):
        state = await self.position_manager.get_portfolio_state()
        if state:
            self.peak_portfolio_value = state['peak_equity']
        await self._load_state()

    async def _load_state(self):
        try:
            data = await self.state_store.load()
            if data:
                self.symbol_consecutive_losses = data.get('consecutive_losses', {})
                self.symbol_cooldowns = {k: datetime.fromisoformat(v) for k, v in data.get('cooldowns', {}).items()}
        except Exception as e:
            logger.error("Failed to load risk state", error=str(e))

    async def _save_state(self):
        try:
            data = {
                'consecutive_losses': self.symbol_consecutive_losses,
                'cooldowns': {k: v.isoformat() for k, v in self.symbol_cooldowns.items()}
            }
            await self.state_store.save(data)
        except Exception as e:
            logger.error("Failed to save risk state", error=str(e))

    @property
    def is_halted(self) -> bool:
        return self.circuit_breaker_halted or self.daily_loss_halted

    async def validate_entry(self, symbol: str, open_positions: List[Position]) -> Tuple[bool, str]:
        """
        Checks if a new trade entry is permitted.
        Async because it may need to save state (clearing cooldowns).
        """
        if self.circuit_breaker_halted: return False, "Circuit Breaker Active"
        if self.daily_loss_halted: return False, "Max Daily Loss Reached"

        if symbol in self.symbol_cooldowns:
            if Clock.now() < self.symbol_cooldowns[symbol]:
                return False, f"Cooldown active until {self.symbol_cooldowns[symbol]}"
            else:
                del self.symbol_cooldowns[symbol]
                self.symbol_consecutive_losses[symbol] = 0
                await self._save_state()

        if len(open_positions) >= self.config.max_open_positions:
            return False, "Max open positions reached"

        for pos in open_positions:
            if pos.symbol == symbol:
                return False, "Position already open"

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
        
        if entry_price <= 0 or stop_loss_price <= 0: return 0.0
        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit == 0: return 0.0

        # 1. Base Risk
        base_risk_pct = self._get_regime_param('risk_per_trade_pct', market_regime)
        
        # 2. Kelly Criterion Adjustment
        risk_pct = self._apply_kelly_criterion(base_risk_pct, model_metrics)
        
        # 3. Drawdown Scaling (Reduce risk in drawdown)
        if self.current_drawdown < -0.05:
            risk_pct *= 0.75

        # 4. Confidence Scaling
        risk_pct = self._apply_confidence_scaling(risk_pct, confidence, confidence_threshold)

        # 5. Correlation Penalty
        risk_pct = self._apply_correlation_penalty(symbol, risk_pct, open_positions)

        # 6. Calculate USD Risk
        risk_amount_usd = portfolio_equity * risk_pct

        # 7. Portfolio Risk Cap
        risk_amount_usd = self._apply_portfolio_risk_cap(risk_amount_usd, portfolio_equity, open_positions)

        # 8. Convert to Quantity
        quantity = risk_amount_usd / risk_per_unit

        # 9. Hard Caps (Max Size, Liquidity)
        quantity = self._apply_hard_caps(symbol, quantity, portfolio_equity, entry_price)

        return quantity

    def _apply_kelly_criterion(self, base_risk: float, metrics: Optional[Dict]) -> float:
        if not self.config.use_kelly_criterion or not metrics:
            return base_risk
        
        ensemble = metrics.get('ensemble', metrics)
        win_rate = ensemble.get('win_rate', 0.0)
        profit_factor = ensemble.get('profit_factor', 0.0)
        
        if win_rate <= 0 or profit_factor <= 0: return base_risk
        
        # Kelly = W - (1-W)/R
        # R approx Profit Factor * (1-W)/W ? No, R is AvgWin/AvgLoss.
        # We approximate R from PF and WR: PF = (W*AvgWin) / ((1-W)*AvgLoss) => R = PF * (1-W)/W
        if win_rate >= 1.0: return base_risk * 2.0
        
        r_ratio = profit_factor * (1.0 - win_rate) / win_rate
        if r_ratio <= 0: return base_risk
        
        kelly = win_rate - (1.0 - win_rate) / r_ratio
        kelly_fraction = kelly * self.config.kelly_fraction
        
        # Safety clamp: Never exceed 5x base risk or 5% total equity
        return max(0.0, min(kelly_fraction, base_risk * 5.0, 0.05))

    def _apply_confidence_scaling(self, risk_pct: float, confidence: Optional[float], threshold: Optional[float]) -> float:
        if self.config.confidence_scaling_factor <= 0 or not confidence or not threshold:
            return risk_pct
        if confidence <= threshold: return risk_pct
        
        surplus = confidence - threshold
        scaler = 1.0 + (surplus * self.config.confidence_scaling_factor)
        scaler = min(scaler, self.config.max_confidence_risk_multiplier)
        return risk_pct * scaler

    def _apply_correlation_penalty(self, symbol: str, risk_pct: float, open_positions: List[Position]) -> float:
        if not self.config.correlation.enabled: return risk_pct
        
        max_corr = 0.0
        for pos in open_positions:
            if pos.symbol == symbol: continue
            c = self.data_handler.get_correlation(symbol, pos.symbol, self.config.correlation.lookback_periods)
            if c > max_corr: max_corr = c
            
        if max_corr > self.config.correlation.max_correlation:
            return risk_pct * self.config.correlation.penalty_factor
        return risk_pct

    def _apply_portfolio_risk_cap(self, risk_usd: float, equity: float, open_positions: List[Position]) -> float:
        if self.config.max_portfolio_risk_pct <= 0: return risk_usd
        
        current_risk = 0.0
        for pos in open_positions:
            if pos.stop_loss_price:
                current_risk += abs(pos.entry_price - pos.stop_loss_price) * pos.quantity
        
        max_risk = equity * self.config.max_portfolio_risk_pct
        remaining = max(0.0, max_risk - current_risk)
        return min(risk_usd, remaining)

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

    def calculate_stop_loss(self, side: str, entry_price: float, df: Any, market_regime: Optional[str] = None) -> float:
        atr = 0.0
        if df is not None and self.config.atr_column_name in df.columns:
            atr = df[self.config.atr_column_name].iloc[-1]
            
        # Swing Low/High Logic
        if self.config.stop_loss_type == 'SWING' and df is not None:
            lookback = self.config.swing_lookback
            if len(df) >= lookback:
                window = df.iloc[-lookback:]
                buffer = atr * self.config.swing_buffer_atr_multiplier
                if side == 'BUY':
                    return min(entry_price * 0.99, window['low'].min() - buffer)
                else:
                    return max(entry_price * 1.01, window['high'].max() + buffer)

        # ATR Logic
        mult = self._get_regime_param('atr_stop_multiplier', market_regime)
        dist = (atr * mult) if atr > 0 else (entry_price * self.config.stop_loss_fallback_pct)
        
        return entry_price - dist if side == 'BUY' else entry_price + dist

    def calculate_take_profit(self, side: str, entry: float, sl: float, market_regime: Optional[str], confidence: Optional[float], confidence_threshold: Optional[float]) -> float:
        risk = abs(entry - sl)
        rr = self._get_regime_param('reward_to_risk_ratio', market_regime)
        
        # Dynamic RR based on confidence
        if self.config.confidence_rr_scaling_factor > 0 and confidence and confidence_threshold and confidence > confidence_threshold:
            surplus = confidence - confidence_threshold
            scaler = 1.0 + (surplus * self.config.confidence_rr_scaling_factor)
            rr *= min(scaler, self.config.max_confidence_rr_multiplier)
            
        dist = risk * rr
        return entry + dist if side == 'BUY' else entry - dist

    async def update_trade_outcome(self, symbol: str, pnl: float):
        """
        Updates risk state based on trade outcome.
        Async because it saves state to disk.
        """
        if pnl < 0:
            self.symbol_consecutive_losses[symbol] = self.symbol_consecutive_losses.get(symbol, 0) + 1
            if self.symbol_consecutive_losses[symbol] >= self.config.max_consecutive_losses:
                self.symbol_cooldowns[symbol] = Clock.now() + timedelta(minutes=self.config.consecutive_loss_cooldown_minutes)
                logger.warning(f"Symbol {symbol} halted due to consecutive losses.")
        else:
            self.symbol_consecutive_losses[symbol] = 0
        await self._save_state()

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

    def _get_regime_param(self, param: str, regime: Optional[str]) -> Any:
        default = getattr(self.config, param)
        if regime and hasattr(self.config.regime_based_risk, regime):
            override = getattr(getattr(self.config.regime_based_risk, regime), param)
            if override is not None: return override
        return default

    def calculate_dynamic_trailing_stop(self, pos: Position, current_price: float, atr: float, regime: str) -> Tuple[Optional[float], Optional[float], bool]:
        # Placeholder for full implementation
        return None, None, False

    def check_time_based_exit(self, pos: Position, price: float) -> bool:
        if not self.config.time_based_exit.enabled: return False
        duration = Clock.now() - pos.open_timestamp
        if duration.total_seconds() > (self.config.time_based_exit.max_hold_time_hours * 3600):
            pnl_pct = (price - pos.entry_price) / pos.entry_price
            if pos.side == 'SELL': pnl_pct = -pnl_pct
            return pnl_pct < self.config.time_based_exit.threshold_pct
        return False
