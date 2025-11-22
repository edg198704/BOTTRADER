import uuid
import asyncio
import os
from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel
from decimal import Decimal

from bot_core.logger import get_logger
from bot_core.config import BotConfig, ExecutionProfile
from bot_core.exchange_api import ExchangeAPI, BotInsufficientFundsError, BotInvalidOrderError, OrderStateUnknownError
from bot_core.position_manager import PositionManager, Position
from bot_core.risk_manager import RiskManager
from bot_core.order_sizer import OrderSizer
from bot_core.order_lifecycle_manager import OrderLifecycleService, ActiveOrderContext
from bot_core.monitoring import AlertSystem
from bot_core.data_handler import DataHandler
from bot_core.common import TradeSignal, TradeExecutionResult, to_decimal, ZERO

logger = get_logger(__name__)

class PreTradeValidator:
    """Encapsulates all pre-trade validation logic."""
    def __init__(self, config: BotConfig, data_handler: DataHandler, risk_manager: RiskManager):
        self.config = config
        self.data_handler = data_handler
        self.risk_manager = risk_manager

    def check_signal_ttl(self, signal: TradeSignal) -> bool:
        signal_age = datetime.now(timezone.utc) - signal.generated_at
        if signal_age > timedelta(seconds=60):
            logger.warning("Signal rejected: TTL expired.", symbol=signal.symbol, age_seconds=signal_age.total_seconds())
            return False
        return True

    def check_data_latency(self, symbol: str) -> bool:
        latency = self.data_handler.get_latency(symbol)
        max_latency = 5.0
        if latency > max_latency:
            logger.warning("Execution skipped: Data stale.", symbol=symbol, latency=latency, threshold=max_latency)
            return False
        return True

    async def check_liquidity(self, symbol: str, current_price: Decimal) -> bool:
        book = self.data_handler.get_latest_order_book(symbol)
        if not book or not book.get('bids') or not book.get('asks'): 
            logger.warning("Liquidity check failed: Empty order book.", symbol=symbol)
            return False
        
        ask0 = to_decimal(book['asks'][0][0])
        bid0 = to_decimal(book['bids'][0][0])
        
        spread = (ask0 - bid0) / current_price
        if spread > to_decimal(self.config.execution.max_entry_spread_pct):
            logger.warning("Spread too high.", symbol=symbol, spread=f"{spread:.4%}", limit=f"{self.config.execution.max_entry_spread_pct:.4%}")
            return False
        return True

    async def check_impact_cost(self, symbol: str, side: str, quantity: Decimal, current_price: Decimal) -> bool:
        book = self.data_handler.get_latest_order_book(symbol)
        if not book: 
            logger.warning("Impact check skipped: No order book.", symbol=symbol)
            return True 
        
        levels = book['asks'] if side == 'BUY' else book['bids']
        if not levels: return False
        
        remaining_qty = quantity
        weighted_price_sum = ZERO
        
        for price_raw, vol_raw in levels:
            price = to_decimal(price_raw)
            vol = to_decimal(vol_raw)
            
            take = min(remaining_qty, vol)
            weighted_price_sum += take * price
            remaining_qty -= take
            if remaining_qty <= ZERO: break
            
        if remaining_qty > ZERO:
            logger.warning("Insufficient depth for order size.", symbol=symbol, qty=str(quantity), remaining=str(remaining_qty))
            return False
            
        avg_fill_price = weighted_price_sum / quantity
        impact_pct = abs(avg_fill_price - current_price) / current_price
        
        if impact_pct > to_decimal(self.config.execution.max_impact_cost_pct):
            logger.warning("High impact cost detected.", symbol=symbol, impact=f"{impact_pct:.4%}", limit=f"{self.config.execution.max_impact_cost_pct:.4%}")
            return False
            
        return True

    async def check_risk_limits(self, symbol: str, position_manager: PositionManager) -> bool:
        active_positions = await position_manager.get_all_active_positions()
        allowed, reason = await self.risk_manager.validate_entry(symbol, active_positions)
        if not allowed:
            logger.info("Entry rejected by Risk Manager.", symbol=symbol, reason=reason)
            return False
        return True

class TradeExecutor:
    """
    Institutional-grade execution engine.
    Enforces strict pre-trade checks, atomic execution locking, and delegates lifecycle management.
    Uses a Fire-and-Forget model for order placement to avoid blocking the tick pipeline.
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
        
        self.validator = PreTradeValidator(config, data_handler, risk_manager)
        self._symbol_locks: Dict[str, asyncio.Lock] = {}
        logger.info("TradeExecutor initialized with atomic execution engine.")

    def _get_lock(self, symbol: str) -> asyncio.Lock:
        if symbol not in self._symbol_locks:
            self._symbol_locks[symbol] = asyncio.Lock()
        return self._symbol_locks[symbol]

    def _get_execution_profile(self, urgency: str) -> ExecutionProfile:
        return self.config.execution.profiles.get(urgency, self.config.execution.profiles[self.config.execution.default_profile])

    async def execute_trade_signal(self, signal: TradeSignal, df_with_indicators: Any, position_snapshot: Optional[Position]) -> Optional[TradeExecutionResult]:
        """
        Router for trade signals. Ensures atomic execution per symbol using Double-Check Locking.
        Initiates the trade asynchronously to prevent blocking the data pipeline.
        """
        # 1. Basic Validation
        if not self.validator.check_signal_ttl(signal):
            return None
        if not self.validator.check_data_latency(signal.symbol):
            return None

        symbol = signal.symbol
        async with self._get_lock(symbol):
            # 2. Re-fetch State (Double-Check Locking)
            current_position = await self.position_manager.get_open_position(symbol)
            
            price_float = self.latest_prices.get(symbol)
            if not price_float:
                logger.warning("Execution skipped: No price data available.", symbol=symbol)
                return None
            current_price = to_decimal(price_float)

            # --- Entry Logic ---
            if not current_position:
                if signal.action in ['BUY', 'SELL']:
                    return await self._initiate_entry(signal, current_price, df_with_indicators)
            
            # --- Exit Logic ---
            elif current_position:
                is_close_long = (signal.action == 'SELL') and current_position.side == 'BUY'
                is_close_short = (signal.action == 'BUY') and current_position.side == 'SELL'
                
                if is_close_long or is_close_short:
                    asyncio.create_task(self.close_position(current_position, "Strategy Signal"))
                    return None
        return None

    async def _initiate_entry(self, signal: TradeSignal, current_price: Decimal, df: Any) -> Optional[TradeExecutionResult]:
        symbol = signal.symbol
        side = signal.action
        profile = self._get_execution_profile(signal.urgency)

        # 1. Advanced Validation
        if not await self.validator.check_liquidity(symbol, current_price):
            return None
        if not await self.validator.check_risk_limits(symbol, self.position_manager):
            return None

        # 2. Sizing Calculation
        active_positions = await self.position_manager.get_all_active_positions()
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

        # 3. Wallet & Exchange Constraints
        market_details = self.market_details.get(symbol)
        if not market_details:
            logger.error("Market details missing, aborting entry.", symbol=symbol)
            return None

        purchasing_power_qty = await self._get_purchasing_power_qty(symbol, side, current_price)
        capped_quantity = min(ideal_quantity, purchasing_power_qty)
        final_quantity = self.order_sizer.adjust_order_quantity(symbol, capped_quantity, current_price, market_details)
        
        if final_quantity <= ZERO:
            logger.info("Quantity adjusted to zero (below limits or insufficient funds).", symbol=symbol)
            return None

        # 4. Impact Cost Check (Final Qty)
        if not await self.validator.check_impact_cost(symbol, side, final_quantity, current_price):
            return None

        # 5. Order Construction
        order_type = profile.order_type
        limit_price = None
        if order_type == 'LIMIT':
            limit_price = await self._calculate_limit_price(symbol, side, current_price, profile.limit_offset_pct)

        # 6. State Reservation (Synchronous/Fast)
        trade_id = str(uuid.uuid4()) # Unique ID for the entire trade lifecycle
        await self.position_manager.create_pending_position(
            symbol, side, trade_id, trade_id, 
            decision_price=current_price, 
            strategy_metadata=signal.metadata
        )

        # 7. Background Execution (Fire-and-Forget)
        asyncio.create_task(self._submit_order_task(
            trade_id, symbol, side, order_type, final_quantity, limit_price, profile, market_details, signal.metadata
        ))

        return TradeExecutionResult(
            symbol=symbol, action=side, quantity=final_quantity, price=limit_price or current_price,
            trade_id=trade_id, status='INITIATED', metadata=signal.metadata
        )

    async def _submit_order_task(self, 
                                 trade_id: str, 
                                 symbol: str, 
                                 side: str, 
                                 order_type: str, 
                                 quantity: Decimal, 
                                 price: Optional[Decimal], 
                                 profile: ExecutionProfile, 
                                 market_details: Dict[str, Any],
                                 metadata: Dict[str, Any]):
        """Background task to handle the network I/O for order placement."""
        try:
            # Use the new Idempotent Gateway
            order_result = await self.exchange_api.place_order_idempotent(
                symbol, side, order_type, quantity, price, 
                client_order_id=trade_id,
                safety_config=self.config.execution.safety
            )
            
            exchange_order_id = order_result.get('orderId', trade_id)
            if exchange_order_id != trade_id:
                await self.position_manager.update_pending_order_id(symbol, trade_id, exchange_order_id)

            # Handoff to Lifecycle Service
            ctx = ActiveOrderContext(
                trade_id=trade_id,
                symbol=symbol,
                side=side,
                total_quantity=quantity,
                initial_price=price or ZERO,
                strategy_metadata=metadata,
                current_order_id=exchange_order_id,
                current_price=price or ZERO,
                market_details=market_details,
                intent='OPEN',
                profile=profile
            )
            
            await self.order_lifecycle_service.register_order(ctx)
            logger.info("Order submitted successfully.", trade_id=trade_id, exchange_id=exchange_order_id)

        except OrderStateUnknownError as e:
            logger.critical("Order state unknown after max verification attempts!", trade_id=trade_id, error=str(e))
            # Do NOT mark failed. Leave as PENDING. 
            # The OrderLifecycleManager's reconciliation logic will eventually pick it up or expire it.
            # We alert the human operator.
            if self.alert_system:
                asyncio.create_task(self.alert_system.send_alert(
                    "CRITICAL", 
                    f"Order State Unknown for {symbol}", 
                    details={'trade_id': trade_id, 'error': str(e)}
                ))

        except Exception as e:
            logger.error("Order placement failed in background task.", error=str(e), trade_id=trade_id)
            await self.position_manager.mark_position_failed(symbol, trade_id, str(e))

    async def close_position(self, position: Position, reason: str):
        async with self._get_lock(position.symbol):
            fresh_pos = await self.position_manager.get_open_position(position.symbol)
            if not fresh_pos: return

            close_side = 'SELL' if position.side == 'BUY' else 'BUY'
            current_price = to_decimal(self.latest_prices.get(position.symbol, float(position.entry_price)))
            market_details = self.market_details.get(position.symbol)
            if not market_details: return

            close_qty = self.order_sizer.adjust_order_quantity(position.symbol, position.quantity, current_price, market_details)
            if close_qty <= ZERO:
                await self.position_manager.close_position(position.symbol, current_price, f"{reason} (Dust)")
                return

            profile = self._get_execution_profile('neutral')
            
            order_type = profile.order_type
            limit_price = None
            if order_type == 'LIMIT':
                limit_price = await self._calculate_limit_price(position.symbol, close_side, current_price, profile.limit_offset_pct)
            else:
                order_type = 'MARKET'

            try:
                extra_params = {'reduceOnly': True} if order_type == 'LIMIT' else {}
                # Closing orders are less critical for idempotency (reduceOnly protects us), 
                # but we still use standard placement.
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
                    intent='CLOSE',
                    profile=profile
                )
                
                await self.order_lifecycle_service.register_order(ctx)
                logger.info("Closing order placed and registered", symbol=position.symbol, order_id=exchange_order_id)

            except Exception as e:
                logger.error("Close order placement failed.", error=str(e))

    async def _get_purchasing_power_qty(self, symbol: str, side: str, price: Decimal) -> Decimal:
        try:
            base, quote = symbol.split('/')
            balances = await self.exchange_api.get_balance()
            if side == 'BUY':
                free_quote = balances.get(quote, {}).get('free', ZERO)
                fee_mult = ONE + to_decimal(self.config.exchange.taker_fee_pct)
                return (free_quote / fee_mult) / price if free_quote > ZERO else ZERO
            else:
                return balances.get(base, {}).get('free', ZERO)
        except Exception:
            return ZERO

    async def _calculate_limit_price(self, symbol: str, side: str, current_price: Decimal, offset_pct: float) -> Decimal:
        ticker = await self.exchange_api.get_ticker_data(symbol)
        bid = ticker.get('bid')
        ask = ticker.get('ask')
        
        ref_price = bid if side == 'BUY' and bid else (ask if side == 'SELL' and ask else current_price)
        offset = ref_price * to_decimal(offset_pct)
        return ref_price - offset if side == 'BUY' else ref_price + offset
