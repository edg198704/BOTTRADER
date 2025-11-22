import uuid
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pydantic import BaseModel

from bot_core.logger import get_logger
from bot_core.config import BotConfig, ExecutionProfile
from bot_core.exchange_api import ExchangeAPI
from bot_core.position_manager import PositionManager, Position
from bot_core.risk_manager import RiskManager
from bot_core.order_sizer import OrderSizer
from bot_core.order_lifecycle_manager import OrderLifecycleService, ActiveOrderContext
from bot_core.monitoring import AlertSystem
from bot_core.data_handler import DataHandler
from bot_core.common import TradeSignal, Arith

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

    def _get_execution_profile(self, urgency: str) -> ExecutionProfile:
        return self.config.execution.profiles.get(urgency, self.config.execution.profiles[self.config.execution.default_profile])

    async def execute_trade_signal(self, signal: TradeSignal, df_with_indicators: Any, position_snapshot: Optional[Position]) -> Optional[TradeExecutionResult]:
        """
        Router for trade signals. Ensures atomic execution per symbol using Double-Check Locking.
        """
        symbol = signal.symbol
        async with self._get_lock(symbol):
            # 1. Re-fetch State (Double-Check Locking)
            current_position = await self.position_manager.get_open_position(symbol)
            current_price_float = self.latest_prices.get(symbol)
            
            if not current_price_float:
                logger.warning("Execution skipped: No price data available.", symbol=symbol)
                return None
            
            current_price = Arith.decimal(current_price_float)

            # --- Entry Logic ---
            if not current_position:
                if signal.action in ['BUY', 'SELL']:
                    return await self._handle_entry(signal, current_price, df_with_indicators)
            
            # --- Exit Logic ---
            elif current_position:
                is_close_long = (signal.action == 'SELL') and current_position.side == 'BUY'
                is_close_short = (signal.action == 'BUY') and current_position.side == 'SELL'
                
                if is_close_long or is_close_short:
                    await self.close_position(current_position, "Strategy Signal")
                    return None
        return None

    async def _handle_entry(self, signal: TradeSignal, current_price: Decimal, df: Any) -> Optional[TradeExecutionResult]:
        symbol = signal.symbol
        side = signal.action
        profile = self._get_execution_profile(signal.urgency)

        # 1. Risk Sizing (Legacy RiskManager expects floats, we convert back and forth)
        # In a full refactor, RiskManager would be Decimal-native.
        active_positions = await self.position_manager.get_all_active_positions()
        portfolio_equity = self.position_manager.get_portfolio_value(self.latest_prices, active_positions)
        
        # Calculate Stop Loss (Float)
        stop_loss_float = self.risk_manager.calculate_stop_loss(side, float(current_price), df, market_regime=signal.regime)
        
        # Calculate Quantity (Float)
        ideal_qty_float = self.risk_manager.calculate_position_size(
            symbol=symbol,
            portfolio_equity=portfolio_equity, 
            entry_price=float(current_price), 
            stop_loss_price=stop_loss_float, 
            open_positions=active_positions,
            market_regime=signal.regime,
            confidence=signal.confidence,
            confidence_threshold=signal.metadata.get('effective_threshold'),
            model_metrics=signal.metadata.get('metrics')
        )
        
        ideal_quantity = Arith.decimal(ideal_qty_float)

        # 2. Wallet & Exchange Constraints
        market_details = self.market_details.get(symbol)
        if not market_details:
            logger.error("Market details missing, aborting entry.", symbol=symbol)
            return None

        purchasing_power_qty = await self._get_purchasing_power_qty(symbol, side, current_price)
        capped_quantity = min(ideal_quantity, purchasing_power_qty)
        
        # Adjust for precision (OrderSizer needs update to Decimal, assuming it handles floats for now, we cast)
        # Ideally OrderSizer is also refactored. Here we do a safe cast.
        final_quantity_float = self.order_sizer.adjust_order_quantity(symbol, float(capped_quantity), float(current_price), market_details)
        final_quantity = Arith.decimal(final_quantity_float)
        
        if final_quantity <= 0:
            return None

        # 3. Order Construction
        order_type = profile.order_type
        limit_price = None
        if order_type == 'LIMIT':
            # Calculate limit price
            offset = current_price * Arith.decimal(profile.limit_offset_pct)
            limit_price = (current_price - offset) if side == 'BUY' else (current_price + offset)

        # 4. Execution (Initial Placement)
        trade_id = str(uuid.uuid4())
        await self.position_manager.create_pending_position(
            symbol, side, trade_id, trade_id, 
            decision_price=float(current_price), 
            strategy_metadata=signal.metadata
        )

        try:
            extra_params = {'clientOrderId': trade_id}
            if profile.post_only and order_type == 'LIMIT':
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

        # 5. Handoff to Lifecycle Service
        ctx = ActiveOrderContext(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            total_quantity=float(final_quantity),
            initial_price=float(limit_price or current_price),
            strategy_metadata=signal.metadata,
            current_order_id=exchange_order_id,
            current_price=float(limit_price or current_price),
            market_details=market_details,
            intent='OPEN',
            profile=profile
        )
        
        await self.order_lifecycle_service.register_order(ctx)

        return TradeExecutionResult(
            symbol=symbol, action=side, quantity=float(final_quantity), price=float(limit_price or current_price),
            order_id=exchange_order_id, status='PENDING', metadata=signal.metadata
        )

    async def close_position(self, position: Position, reason: str):
        async with self._get_lock(position.symbol):
            fresh_pos = await self.position_manager.get_open_position(position.symbol)
            if not fresh_pos: return

            close_side = 'SELL' if position.side == 'BUY' else 'BUY'
            current_price = Arith.decimal(self.latest_prices.get(position.symbol, float(position.entry_price_dec)))
            market_details = self.market_details.get(position.symbol)
            if not market_details: return

            close_qty_float = self.order_sizer.adjust_order_quantity(position.symbol, float(position.quantity_dec), float(current_price), market_details)
            close_qty = Arith.decimal(close_qty_float)
            
            if close_qty <= 0:
                await self.position_manager.close_position(position.symbol, float(current_price), f"{reason} (Dust)")
                return

            profile = self._get_execution_profile('neutral')
            order_type = profile.order_type
            limit_price = None
            if order_type == 'LIMIT':
                offset = current_price * Arith.decimal(profile.limit_offset_pct)
                limit_price = (current_price - offset) if close_side == 'BUY' else (current_price + offset)
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
                    total_quantity=float(close_qty),
                    initial_price=float(limit_price or current_price),
                    strategy_metadata={},
                    current_order_id=exchange_order_id,
                    current_price=float(limit_price or current_price),
                    market_details=market_details,
                    intent='CLOSE',
                    profile=profile
                )
                
                await self.order_lifecycle_service.register_order(ctx)

            except Exception as e:
                logger.error("Close order placement failed.", error=str(e))

    async def _get_purchasing_power_qty(self, symbol: str, side: str, price: Decimal) -> Decimal:
        try:
            base, quote = symbol.split('/')
            balances = await self.exchange_api.get_balance()
            if side == 'BUY':
                free_quote = balances.get(quote, {}).get('free', Decimal(0))
                fee_buffer = Decimal(1) + Arith.decimal(self.config.exchange.taker_fee_pct)
                return (free_quote / fee_buffer) / price if free_quote > 0 else Decimal(0)
            else:
                return balances.get(base, {}).get('free', Decimal(0))
        except Exception:
            return Decimal(0)
