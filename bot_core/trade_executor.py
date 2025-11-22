import uuid
import asyncio
import os
from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel

from bot_core.logger import get_logger
from bot_core.config import BotConfig
from bot_core.exchange_api import ExchangeAPI, BotInsufficientFundsError, BotInvalidOrderError
from bot_core.position_manager import PositionManager, Position
from bot_core.risk_manager import RiskManager
from bot_core.order_sizer import OrderSizer
from bot_core.order_lifecycle_manager import OrderLifecycleService, ActiveOrderContext
from bot_core.monitoring import AlertSystem
from bot_core.data_handler import DataHandler
from bot_core.common import TradeSignal

logger = get_logger(__name__)

class TradeExecutionResult(BaseModel):
    symbol: str
    action: str
    quantity: float
    price: float
    order_id: str
    status: str
    metadata: Dict[str, Any] = {}

class TradeExecutor:
    """
    Institutional-grade execution engine.
    Enforces strict pre-trade checks, atomic execution locking, and delegates lifecycle management.
    """
    def __init__(self,
                 config: BotConfig,
                 exchange_api: ExchangeAPI,
                 position_manager: PositionManager,
                 risk_manager: RiskManager,
                 order_sizer: OrderSizer,
                 order_lifecycle_service: OrderLifecycleService,
                 alert_system: AlertSystem,
                 shared_latest_prices: Dict[str, float],
                 market_details: Dict[str, Dict[str, Any]],
                 data_handler: DataHandler):
        self.config = config
        self.exchange_api = exchange_api
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.order_sizer = order_sizer
        self.order_lifecycle_service = order_lifecycle_service
        self.alert_system = alert_system
        self.latest_prices = shared_latest_prices
        self.market_details = market_details
        self.data_handler = data_handler
        
        self._symbol_locks: Dict[str, asyncio.Lock] = {}
        logger.info("TradeExecutor initialized with atomic execution engine.")

    def _get_lock(self, symbol: str) -> asyncio.Lock:
        if symbol not in self._symbol_locks:
            self._symbol_locks[symbol] = asyncio.Lock()
        return self._symbol_locks[symbol]

    async def execute_trade_signal(self, signal: TradeSignal, df_with_indicators: Any, position_snapshot: Optional[Position]) -> Optional[TradeExecutionResult]:
        """
        Router for trade signals. Ensures atomic execution per symbol using Double-Check Locking.
        """
        # 0. Signal TTL Check
        signal_age = datetime.now(timezone.utc) - signal.generated_at
        if signal_age > timedelta(seconds=60):
            logger.warning("Signal rejected: TTL expired.", symbol=signal.symbol, age_seconds=signal_age.total_seconds())
            return None

        symbol = signal.symbol
        
        # 1. Data Freshness Check (Critical for HFT/Algorithmic trading)
        latency = self.data_handler.get_latency(symbol)
        max_latency = 5.0 # Max allowable latency in seconds
        if latency > max_latency:
            logger.warning("Execution skipped: Data stale.", symbol=symbol, latency=latency, threshold=max_latency)
            return None

        async with self._get_lock(symbol):
            # 2. Re-fetch State (Double-Check Locking)
            # The snapshot passed from the pipeline might be stale by the time we acquire the lock.
            # We must query the authoritative source (PositionManager) inside the critical section.
            current_position = await self.position_manager.get_open_position(symbol)
            
            # 3. Re-fetch Price
            current_price = self.latest_prices.get(symbol)
            if not current_price:
                logger.warning("Execution skipped: No price data available.", symbol=symbol)
                return None

            # --- Entry Logic ---
            if not current_position:
                if signal.action in ['BUY', 'SELL']:
                    return await self._handle_entry(signal, current_price, df_with_indicators)
            
            # --- Exit Logic ---
            elif current_position:
                # Check if signal contradicts current position
                is_close_long = (signal.action == 'SELL') and current_position.side == 'BUY'
                is_close_short = (signal.action == 'BUY') and current_position.side == 'SELL'
                
                if is_close_long or is_close_short:
                    await self.close_position(current_position, "Strategy Signal")
                    return None
        return None

    async def _handle_entry(self, signal: TradeSignal, current_price: float, df: Any) -> Optional[TradeExecutionResult]:
        symbol = signal.symbol
        side = signal.action

        # 1. Market Pre-Flight Check (Spread & Depth)
        if not await self._check_liquidity(symbol, current_price):
            return None

        # 2. Risk Gatekeeper
        # Re-fetch active positions inside lock to ensure global limits (e.g. Max Open Positions) are respected atomically
        active_positions = await self.position_manager.get_all_active_positions()
        allowed, reason = await self.risk_manager.validate_entry(symbol, active_positions)
        if not allowed:
            logger.info("Entry rejected by Risk Manager.", symbol=symbol, reason=reason)
            return None

        # 3. Sizing Calculation
        portfolio_equity = self.position_manager.get_portfolio_value(self.latest_prices, active_positions)
        stop_loss = self.risk_manager.calculate_stop_loss(side, current_price, df, market_regime=signal.regime)
        
        confidence_threshold = signal.metadata.get('effective_threshold')
        ideal_quantity = self.risk_manager.calculate_position_size(
            symbol=symbol,
            portfolio_equity=portfolio_equity, 
            entry_price=current_price, 
            stop_loss_price=stop_loss, 
            open_positions=active_positions,
            market_regime=signal.regime,
            confidence=signal.confidence,
            confidence_threshold=confidence_threshold,
            model_metrics=signal.metadata.get('metrics')
        )

        # 4. Wallet & Exchange Constraints
        market_details = self.market_details.get(symbol)
        if not market_details:
            logger.error("Market details missing, aborting entry.", symbol=symbol)
            return None

        purchasing_power_qty = await self._get_purchasing_power_qty(symbol, side, current_price)
        capped_quantity = min(ideal_quantity, purchasing_power_qty)
        final_quantity = self.order_sizer.adjust_order_quantity(symbol, capped_quantity, current_price, market_details)
        
        if final_quantity <= 0:
            logger.info("Quantity adjusted to zero (below limits or insufficient funds).", symbol=symbol)
            return None

        # 5. Impact Cost Check (Depth)
        # Perform this check with the FINAL calculated quantity
        if not await self._check_impact_cost(symbol, side, final_quantity, current_price):
            logger.warning("Entry rejected: High estimated impact cost.", symbol=symbol, qty=final_quantity)
            return None

        # 6. Order Construction
        order_type = self.config.execution.default_order_type
        limit_price = None
        if order_type == 'LIMIT':
            limit_price = await self._calculate_limit_price(symbol, side, current_price, df)

        # 7. Execution (Initial Placement)
        trade_id = str(uuid.uuid4())
        # Create pending position record BEFORE placing order to handle partial fills/crashes
        await self.position_manager.create_pending_position(
            symbol, side, trade_id, trade_id, 
            decision_price=current_price, 
            strategy_metadata=signal.metadata
        )

        try:
            extra_params = {'clientOrderId': trade_id}
            # Aggressive signals might skip Post-Only to ensure fill
            is_aggressive = signal.confidence >= 0.8
            if self.config.execution.post_only and order_type == 'LIMIT' and not is_aggressive:
                extra_params['postOnly'] = True

            order_result = await self.exchange_api.place_order(
                symbol, side, order_type, final_quantity, price=limit_price, extra_params=extra_params
            )
        except Exception as e:
            logger.error("Order placement failed.", error=str(e))
            await self.position_manager.mark_position_failed(symbol, trade_id, str(e))
            return None

        exchange_order_id = order_result.get('orderId', trade_id)
        if exchange_order_id != trade_id:
            await self.position_manager.update_pending_order_id(symbol, trade_id, exchange_order_id)

        # 8. Handoff to Lifecycle Service
        ctx = ActiveOrderContext(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            total_quantity=final_quantity,
            initial_price=limit_price or current_price,
            strategy_metadata=signal.metadata,
            current_order_id=exchange_order_id,
            current_price=limit_price or current_price,
            market_details=market_details,
            intent='OPEN'
        )
        
        await self.order_lifecycle_service.register_order(ctx)

        return TradeExecutionResult(
            symbol=symbol, action=side, quantity=final_quantity, price=limit_price or current_price,
            order_id=exchange_order_id, status='PENDING', metadata=signal.metadata
        )

    async def close_position(self, position: Position, reason: str):
        async with self._get_lock(position.symbol):
            # Re-fetch to ensure we don't double close
            fresh_pos = await self.position_manager.get_open_position(position.symbol)
            if not fresh_pos: return

            close_side = 'SELL' if position.side == 'BUY' else 'BUY'
            current_price = self.latest_prices.get(position.symbol, position.entry_price)
            market_details = self.market_details.get(position.symbol)
            if not market_details: return

            close_qty = self.order_sizer.adjust_order_quantity(position.symbol, position.quantity, current_price, market_details)
            if close_qty <= 0:
                await self.position_manager.close_position(position.symbol, current_price, f"{reason} (Dust)")
                return

            order_type = self.config.execution.default_order_type
            limit_price = None
            if order_type == 'LIMIT':
                limit_price = await self._calculate_limit_price(position.symbol, close_side, current_price, None)
            else:
                order_type = 'MARKET'

            try:
                extra_params = {'reduceOnly': True} if order_type == 'LIMIT' else {}
                order_result = await self.exchange_api.place_order(
                    position.symbol, close_side, order_type, close_qty, price=limit_price, extra_params=extra_params
                )
                
                close_trade_id = f"close_{position.trade_id}_{int(datetime.now().timestamp())}"
                exchange_order_id = order_result.get('orderId')
                
                if exchange_order_id:
                    await self.position_manager.register_closing_order(position.symbol, exchange_order_id)
                
                ctx = ActiveOrderContext(
                    trade_id=close_trade_id,
                    symbol=position.symbol,
                    side=close_side,
                    total_quantity=close_qty,
                    initial_price=limit_price or current_price,
                    strategy_metadata={},
                    current_order_id=exchange_order_id,
                    current_price=limit_price or current_price,
                    market_details=market_details,
                    intent='CLOSE'
                )
                
                await self.order_lifecycle_service.register_order(ctx)
                logger.info("Closing order placed and registered", symbol=position.symbol, order_id=exchange_order_id)

            except Exception as e:
                logger.error("Close order placement failed.", error=str(e))

    async def _check_liquidity(self, symbol: str, current_price: float) -> bool:
        book = self.data_handler.get_latest_order_book(symbol)
        if not book or not book.get('bids') or not book.get('asks'): 
            logger.warning("Liquidity check failed: Empty order book.", symbol=symbol)
            return False
        
        spread = (book['asks'][0][0] - book['bids'][0][0]) / current_price
        if spread > self.config.execution.max_entry_spread_pct:
            logger.warning("Spread too high.", symbol=symbol, spread=f"{spread:.4%}", limit=f"{self.config.execution.max_entry_spread_pct:.4%}")
            return False
        return True

    async def _check_impact_cost(self, symbol: str, side: str, quantity: float, current_price: float) -> bool:
        """Calculates estimated slippage based on order book depth."""
        book = self.data_handler.get_latest_order_book(symbol)
        if not book: 
            # Fail open if no book to avoid stalling, but log warning. 
            # In strict HFT, this should fail closed.
            logger.warning("Impact check skipped: No order book.", symbol=symbol)
            return True 
        
        levels = book['asks'] if side == 'BUY' else book['bids']
        if not levels: return False
        
        remaining_qty = quantity
        weighted_price_sum = 0.0
        
        for price, vol in levels:
            take = min(remaining_qty, vol)
            weighted_price_sum += take * price
            remaining_qty -= take
            if remaining_qty <= 0: break
            
        if remaining_qty > 0:
            logger.warning("Insufficient depth for order size.", symbol=symbol, qty=quantity, remaining=remaining_qty)
            return False
            
        avg_fill_price = weighted_price_sum / quantity
        impact_pct = abs(avg_fill_price - current_price) / current_price
        
        if impact_pct > self.config.execution.max_impact_cost_pct:
            logger.warning("High impact cost detected.", symbol=symbol, impact=f"{impact_pct:.4%}", limit=f"{self.config.execution.max_impact_cost_pct:.4%}")
            return False
            
        return True

    async def _get_purchasing_power_qty(self, symbol: str, side: str, price: float) -> float:
        try:
            base, quote = symbol.split('/')
            balances = await self.exchange_api.get_balance()
            if side == 'BUY':
                free_quote = balances.get(quote, {}).get('free', 0.0)
                # Reserve fees
                return (free_quote / (1 + self.config.exchange.taker_fee_pct)) / price if free_quote > 0 else 0.0
            else:
                return balances.get(base, {}).get('free', 0.0)
        except Exception:
            return 0.0

    async def _calculate_limit_price(self, symbol: str, side: str, current_price: float, df: Any) -> float:
        ticker = await self.exchange_api.get_ticker_data(symbol)
        ref_price = ticker.get('bid' if side == 'BUY' else 'ask') or current_price
        offset = ref_price * self.config.execution.limit_price_offset_pct
        # For BUY: Bid - Offset (Passive) or Bid + Offset (Aggressive - if offset is negative)
        # Config usually defines positive offset as 'better' price (more passive)
        return ref_price - offset if side == 'BUY' else ref_price + offset
