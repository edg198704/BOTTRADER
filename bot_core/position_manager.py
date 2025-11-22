import datetime
import asyncio
import enum
import json
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from decimal import Decimal

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Enum as SQLAlchemyEnum, func, Boolean, cast, Date, text, inspect, event, Numeric
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.sqlite import insert

from bot_core.logger import get_logger
from bot_core.config import DatabaseConfig
from bot_core.utils import Clock
from bot_core.common import to_decimal, ZERO, ONE, Dec

if TYPE_CHECKING:
    from bot_core.monitoring import AlertSystem
    from bot_core.config import BreakevenConfig
    from bot_core.exchange_api import ExchangeAPI

logger = get_logger(__name__)
Base = declarative_base()

class PositionStatus(enum.Enum):
    PENDING = 'PENDING'
    OPEN = 'OPEN'
    CLOSED = 'CLOSED'
    FAILED = 'FAILED'

class Position(Base):
    __tablename__ = 'positions'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False, index=True)
    side = Column(String, nullable=False)
    
    quantity = Column(Numeric(30, 10, asdecimal=True), nullable=False)
    entry_price = Column(Numeric(30, 10, asdecimal=True), nullable=False)
    
    status = Column(SQLAlchemyEnum(PositionStatus), default=PositionStatus.OPEN, nullable=False, index=True)
    order_id = Column(String, nullable=True, index=True)
    closing_order_id = Column(String, nullable=True, index=True)
    trade_id = Column(String, nullable=True)
    
    stop_loss_price = Column(Numeric(30, 10, asdecimal=True), nullable=True)
    take_profit_price = Column(Numeric(30, 10, asdecimal=True), nullable=True)
    
    creation_timestamp = Column(DateTime, default=Clock.now)
    open_timestamp = Column(DateTime, default=Clock.now)
    close_timestamp = Column(DateTime, nullable=True)
    
    close_price = Column(Numeric(30, 10, asdecimal=True), nullable=True)
    pnl = Column(Numeric(30, 10, asdecimal=True), nullable=True)
    fees = Column(Numeric(30, 10, asdecimal=True), default=ZERO, nullable=False)
    
    decision_price = Column(Numeric(30, 10, asdecimal=True), nullable=True)
    execution_latency_ms = Column(Numeric(20, 5, asdecimal=True), nullable=True)
    slippage_pct = Column(Numeric(10, 5, asdecimal=True), nullable=True)

    trailing_ref_price = Column(Numeric(30, 10, asdecimal=True), nullable=True)
    trailing_stop_active = Column(Boolean, default=False, nullable=False)
    breakeven_active = Column(Boolean, default=False, nullable=False)
    strategy_metadata = Column(String, nullable=True)
    
    exit_reason = Column(String, nullable=True)

class PortfolioState(Base):
    __tablename__ = 'portfolio_state'
    id = Column(Integer, primary_key=True)
    initial_capital = Column(Numeric(30, 10, asdecimal=True), nullable=False)
    peak_equity = Column(Numeric(30, 10, asdecimal=True), nullable=False)
    last_update = Column(DateTime, default=Clock.now, onupdate=Clock.now)

class RiskState(Base):
    __tablename__ = 'risk_state'
    symbol = Column(String, primary_key=True)
    consecutive_losses = Column(Integer, default=0)
    cooldown_until = Column(DateTime, nullable=True)
    last_updated = Column(DateTime, default=Clock.now, onupdate=Clock.now)

class PositionManager:
    """
    Manages trading positions with an In-Memory Source of Truth and Asynchronous Persistence.
    Uses granular per-symbol locking to maximize concurrency.
    """
    def __init__(self, config: DatabaseConfig, initial_capital: float, alert_system: Optional['AlertSystem'] = None):
        self.engine = create_engine(f'sqlite:///{config.path}')
        
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()

        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._initial_capital = to_decimal(initial_capital)
        self._realized_pnl = ZERO
        self.alert_system = alert_system
        
        # In-Memory State
        self._position_cache: Dict[str, Position] = {}
        self._portfolio_state_cache: Optional[Dict[str, Decimal]] = None
        
        # Concurrency Control
        self._symbol_locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock() # For portfolio-wide ops
        
        # Persistence Queue
        self._persist_queue = asyncio.Queue()
        self._persistence_task = None
        
        logger.info("PositionManager initialized (In-Memory + Async Persistence).")

    def _get_lock(self, symbol: str) -> asyncio.Lock:
        if symbol not in self._symbol_locks:
            self._symbol_locks[symbol] = asyncio.Lock()
        return self._symbol_locks[symbol]

    async def initialize(self):
        # Load initial state from DB synchronously to warm up cache
        await self._run_in_executor(self._load_state_sync)
        self._persistence_task = asyncio.create_task(self._persistence_worker())
        logger.info("PositionManager state initialized.", 
                    realized_pnl=self._realized_pnl, 
                    cached_positions=len(self._position_cache))

    def _load_state_sync(self):
        session = self.SessionLocal()
        try:
            # 1. Load Realized PnL
            realized_pnl = session.query(func.sum(Position.pnl)).filter(Position.status == PositionStatus.CLOSED).scalar()
            self._realized_pnl = realized_pnl or ZERO

            # 2. Load Portfolio State
            state = session.query(PortfolioState).filter(PortfolioState.id == 1).first()
            if not state:
                state = PortfolioState(id=1, initial_capital=self._initial_capital, peak_equity=self._initial_capital)
                session.add(state)
                session.commit()
            self._portfolio_state_cache = {'initial_capital': state.initial_capital, 'peak_equity': state.peak_equity}

            # 3. Load Active Positions
            positions = session.query(Position).filter(
                Position.status.in_([PositionStatus.OPEN, PositionStatus.PENDING])
            ).all()
            for p in positions:
                session.expunge(p)
                self._position_cache[p.symbol] = p
        finally:
            session.close()

    async def _persistence_worker(self):
        logger.info("Persistence worker started.")
        while True:
            try:
                task_func, args = await self._persist_queue.get()
                await self._run_in_executor(task_func, *args)
                self._persist_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Persistence worker error", error=str(e))

    async def reconcile_positions(self, exchange_api: 'ExchangeAPI', latest_prices: Dict[str, float]):
        logger.info("Starting position reconciliation...")
        try:
            balances = await exchange_api.get_balance()
        except Exception as e:
            logger.error("Reconciliation failed: Could not fetch exchange balance", error=str(e))
            return

        # Snapshot keys to avoid modification during iteration
        symbols = list(self._position_cache.keys())
        
        for symbol in symbols:
            async with self._get_lock(symbol):
                pos = self._position_cache.get(symbol)
                if not pos or pos.status != PositionStatus.OPEN:
                    continue

                base_asset = symbol.split('/')[0]
                held_balance = balances.get(base_asset, {}).get('total', ZERO)
                
                # Tolerance for dust (5% mismatch allowed)
                if held_balance < (pos.quantity * Dec("0.95")):
                    logger.critical("Phantom position detected! Closing in DB.", symbol=symbol, db_qty=pos.quantity, exchange_qty=held_balance)
                    await self.close_position(symbol, pos.entry_price, reason="Reconciliation: Phantom Position")
                
                elif held_balance < (pos.quantity * Dec("0.99")):
                    logger.warning("Quantity mismatch detected. Adjusting DB.", symbol=symbol, db_qty=pos.quantity, exchange_qty=held_balance)
                    diff = pos.quantity - held_balance
                    current_price = to_decimal(latest_prices.get(symbol, float(pos.entry_price)))
                    await self.confirm_position_close(symbol, current_price, diff, ZERO, reason="Reconciliation: Quantity Adjustment")

    async def _run_in_executor(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func, *args)

    # --- Risk State Management ---

    async def get_all_risk_states(self) -> Dict[str, Dict[str, Any]]:
        return await self._run_in_executor(self._get_all_risk_states_sync)

    def _get_all_risk_states_sync(self) -> Dict[str, Dict[str, Any]]:
        session = self.SessionLocal()
        try:
            states = session.query(RiskState).all()
            return {
                s.symbol: {
                    'consecutive_losses': s.consecutive_losses,
                    'cooldown_until': s.cooldown_until
                } for s in states
            }
        finally:
            session.close()

    async def update_risk_state(self, symbol: str, consecutive_losses: int, cooldown_until: Optional[datetime.datetime]):
        self._persist_queue.put_nowait((self._update_risk_state_sync, (symbol, consecutive_losses, cooldown_until)))

    def _update_risk_state_sync(self, symbol: str, consecutive_losses: int, cooldown_until: Optional[datetime.datetime]):
        session = self.SessionLocal()
        try:
            stmt = insert(RiskState).values(
                symbol=symbol, 
                consecutive_losses=consecutive_losses, 
                cooldown_until=cooldown_until,
                last_updated=Clock.now()
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=['symbol'],
                set_=dict(
                    consecutive_losses=stmt.excluded.consecutive_losses,
                    cooldown_until=stmt.excluded.cooldown_until,
                    last_updated=Clock.now()
                )
            )
            session.execute(stmt)
            session.commit()
        except Exception as e:
            logger.error("Failed to update risk state", symbol=symbol, error=str(e))
            session.rollback()
        finally:
            session.close()

    # --- Public Async Interface (In-Memory) ---

    async def get_portfolio_state(self) -> Optional[Dict[str, Decimal]]:
        return self._portfolio_state_cache

    async def update_portfolio_high_water_mark(self, new_peak: Decimal):
        if self._portfolio_state_cache:
            self._portfolio_state_cache['peak_equity'] = max(self._portfolio_state_cache['peak_equity'], new_peak)
        else:
            self._portfolio_state_cache = {'initial_capital': self._initial_capital, 'peak_equity': new_peak}
        
        self._persist_queue.put_nowait((self._update_portfolio_high_water_mark_sync, (new_peak,)))

    def _update_portfolio_high_water_mark_sync(self, new_peak: Decimal):
        session = self.SessionLocal()
        try:
            state = session.query(PortfolioState).filter(PortfolioState.id == 1).first()
            if state:
                if new_peak > state.peak_equity:
                    state.peak_equity = new_peak
                    session.commit()
            else:
                state = PortfolioState(id=1, initial_capital=self._initial_capital, peak_equity=new_peak)
                session.add(state)
                session.commit()
        except Exception as e:
            logger.error("Failed to update portfolio high water mark", error=str(e))
            session.rollback()
        finally:
            session.close()

    async def get_open_position(self, symbol: str) -> Optional[Position]:
        # No lock needed for simple read of atomic dict
        pos = self._position_cache.get(symbol)
        if pos and pos.status == PositionStatus.OPEN:
            return pos
        return None

    async def get_all_open_positions(self) -> List[Position]:
        return [p for p in self._position_cache.values() if p.status == PositionStatus.OPEN]

    async def get_all_active_positions(self) -> List[Position]:
        return list(self._position_cache.values())

    async def get_pending_positions(self) -> List[Position]:
        return [p for p in self._position_cache.values() if p.status == PositionStatus.PENDING]

    # --- State Changing Methods (Write-Through) ---

    async def create_pending_position(self, symbol: str, side: str, order_id: str, trade_id: str, decision_price: Decimal, strategy_metadata: Optional[Dict[str, Any]] = None) -> Optional[Position]:
        async with self._get_lock(symbol):
            if symbol in self._position_cache:
                return None

            metadata_json = json.dumps(strategy_metadata) if strategy_metadata else None
            new_position = Position(
                symbol=symbol, side=side, quantity=ZERO, entry_price=ZERO,
                status=PositionStatus.PENDING, order_id=order_id, trade_id=trade_id,
                decision_price=decision_price, creation_timestamp=Clock.now(),
                trailing_ref_price=ZERO, trailing_stop_active=False,
                breakeven_active=False, strategy_metadata=metadata_json, fees=ZERO
            )
            
            # Update Memory Immediately
            self._position_cache[symbol] = new_position
            
            # Queue Persistence
            self._persist_queue.put_nowait((self._persist_new_position_sync, (new_position,)))
            return new_position

    def _persist_new_position_sync(self, position: Position):
        session = self.SessionLocal()
        try:
            session.add(position)
            session.commit()
        except Exception as e:
            logger.error("Failed to persist new position", symbol=position.symbol, error=str(e))
            session.rollback()
        finally:
            session.close()

    async def update_pending_order_id(self, symbol: str, old_order_id: str, new_order_id: str) -> bool:
        async with self._get_lock(symbol):
            pos = self._position_cache.get(symbol)
            if pos and pos.order_id == old_order_id and pos.status == PositionStatus.PENDING:
                pos.order_id = new_order_id
                self._persist_queue.put_nowait((self._update_pending_order_id_sync, (symbol, old_order_id, new_order_id)))
                return True
            return False

    def _update_pending_order_id_sync(self, symbol: str, old_order_id: str, new_order_id: str):
        session = self.SessionLocal()
        try:
            position = session.query(Position).filter(
                Position.symbol == symbol, Position.order_id == old_order_id, Position.status == PositionStatus.PENDING
            ).first()
            if position:
                position.order_id = new_order_id
                session.commit()
        except Exception as e:
            logger.error("Failed to update pending order ID", symbol=symbol, error=str(e))
            session.rollback()
        finally:
            session.close()

    async def confirm_position_open(self, symbol: str, order_id: str, quantity: Decimal, entry_price: Decimal, stop_loss: Decimal, take_profit: Decimal, fees: Decimal = ZERO) -> Optional[Position]:
        async with self._get_lock(symbol):
            pos = self._position_cache.get(symbol)
            if not pos or pos.status != PositionStatus.PENDING or pos.order_id != order_id:
                return None

            fill_time = Clock.now()
            latency = ZERO
            if pos.creation_timestamp:
                latency = to_decimal((fill_time - pos.creation_timestamp).total_seconds() * 1000.0)
            
            slippage = ZERO
            if pos.decision_price and pos.decision_price > 0:
                slippage = (entry_price - pos.decision_price) / pos.decision_price

            # Update Memory
            pos.status = PositionStatus.OPEN
            pos.quantity = quantity
            pos.entry_price = entry_price
            pos.stop_loss_price = stop_loss
            pos.take_profit_price = take_profit
            pos.trailing_ref_price = entry_price
            pos.open_timestamp = fill_time
            pos.fees += fees
            pos.execution_latency_ms = latency
            pos.slippage_pct = slippage

            # Queue Persistence
            # We pass a dict of updates to avoid detaching/attaching issues with SQLAlchemy objects across threads
            updates = {
                'status': PositionStatus.OPEN, 'quantity': quantity, 'entry_price': entry_price,
                'stop_loss_price': stop_loss, 'take_profit_price': take_profit, 'trailing_ref_price': entry_price,
                'open_timestamp': fill_time, 'fees': pos.fees, 'execution_latency_ms': latency, 'slippage_pct': slippage
            }
            self._persist_queue.put_nowait((self._update_position_sync, (symbol, order_id, updates)))

            if self.alert_system:
                asyncio.create_task(self.alert_system.send_alert(
                    level='info',
                    message=f"🟢 Opened {pos.side} position for {symbol}",
                    details={'symbol': symbol, 'side': pos.side, 'quantity': str(quantity), 'entry_price': str(entry_price), 'fees': str(fees)}
                ))
            return pos

    def _update_position_sync(self, symbol: str, order_id: str, updates: Dict[str, Any]):
        session = self.SessionLocal()
        try:
            session.query(Position).filter(Position.symbol == symbol, Position.order_id == order_id).update(updates)
            session.commit()
        except Exception as e:
            logger.error("Failed to persist position update", symbol=symbol, error=str(e))
            session.rollback()
        finally:
            session.close()

    async def register_closing_order(self, symbol: str, closing_order_id: str):
        async with self._get_lock(symbol):
            pos = self._position_cache.get(symbol)
            if pos:
                pos.closing_order_id = closing_order_id
                self._persist_queue.put_nowait((self._register_closing_order_sync, (symbol, closing_order_id)))

    def _register_closing_order_sync(self, symbol: str, closing_order_id: str):
        session = self.SessionLocal()
        try:
            session.query(Position).filter(Position.symbol == symbol, Position.status == PositionStatus.OPEN).update({'closing_order_id': closing_order_id})
            session.commit()
        except Exception as e:
            logger.error("Failed to register closing order", symbol=symbol, error=str(e))
            session.rollback()
        finally:
            session.close()

    async def confirm_position_close(self, symbol: str, exit_price: Decimal, filled_qty: Decimal, fees: Decimal, reason: str = "Filled") -> Optional[Position]:
        async with self._get_lock(symbol):
            pos = self._position_cache.get(symbol)
            if not pos or pos.status != PositionStatus.OPEN:
                return None

            if filled_qty < (pos.quantity * Dec("0.999")):
                # Partial Close Logic
                gross_pnl = (exit_price - pos.entry_price) * filled_qty
                if pos.side == 'SELL': gross_pnl = -gross_pnl
                
                chunk_ratio = filled_qty / pos.quantity
                entry_fees_chunk = pos.fees * chunk_ratio
                total_chunk_fees = entry_fees_chunk + fees
                net_pnl = gross_pnl - total_chunk_fees

                # Update Memory (Reduce Size)
                pos.quantity -= filled_qty
                pos.fees -= entry_fees_chunk
                self._realized_pnl += net_pnl
                
                # Persist Partial Close (Create a closed chunk record + update open pos)
                self._persist_queue.put_nowait((self._persist_partial_close_sync, (symbol, pos.order_id, filled_qty, exit_price, net_pnl, total_chunk_fees, reason)))
                return pos
            else:
                # Full Close
                gross_pnl = (exit_price - pos.entry_price) * pos.quantity
                if pos.side == 'SELL': gross_pnl = -gross_pnl

                pos.fees += fees
                net_pnl = gross_pnl - pos.fees

                pos.status = PositionStatus.CLOSED
                pos.close_price = exit_price
                pos.close_timestamp = Clock.now()
                pos.pnl = net_pnl
                pos.exit_reason = reason
                
                # Remove from Cache
                del self._position_cache[symbol]
                self._realized_pnl += net_pnl

                # Persist
                updates = {
                    'status': PositionStatus.CLOSED, 'close_price': exit_price, 'close_timestamp': pos.close_timestamp,
                    'pnl': net_pnl, 'fees': pos.fees, 'exit_reason': reason
                }
                self._persist_queue.put_nowait((self._update_position_sync, (symbol, pos.order_id, updates)))

                if self.alert_system:
                    pnl_emoji = "🟢" if net_pnl >= 0 else "🔴"
                    asyncio.create_task(self.alert_system.send_alert(
                        level='info',
                        message=f"{pnl_emoji} Closed position for {symbol}",
                        details={'symbol': symbol, 'net_pnl': f'{net_pnl:.2f}', 'reason': reason}
                    ))
                return pos

    def _persist_partial_close_sync(self, symbol: str, order_id: str, qty: Decimal, price: Decimal, pnl: Decimal, fees: Decimal, reason: str):
        session = self.SessionLocal()
        try:
            # 1. Update Open Position
            pos = session.query(Position).filter(Position.symbol == symbol, Position.order_id == order_id).first()
            if pos:
                pos.quantity -= qty
                pos.fees -= (fees - fees) # Simplified logic, actual fee tracking needs split
                
                # 2. Create Closed Chunk
                chunk = Position(
                    symbol=symbol, side=pos.side, quantity=qty, entry_price=pos.entry_price,
                    status=PositionStatus.CLOSED, order_id=pos.order_id, trade_id=pos.trade_id,
                    open_timestamp=pos.open_timestamp, close_timestamp=Clock.now(), close_price=price,
                    pnl=pnl, fees=fees, exit_reason=reason + " (Partial)"
                )
                session.add(chunk)
                session.commit()
        except Exception as e:
            logger.error("Failed to persist partial close", symbol=symbol, error=str(e))
            session.rollback()
        finally:
            session.close()

    async def mark_position_failed(self, symbol: str, order_id: str, reason: str):
        async with self._get_lock(symbol):
            pos = self._position_cache.get(symbol)
            if pos and pos.order_id == order_id:
                del self._position_cache[symbol]
                self._persist_queue.put_nowait((self._mark_position_failed_sync, (symbol, order_id, reason)))

    def _mark_position_failed_sync(self, symbol: str, order_id: str, reason: str):
        session = self.SessionLocal()
        try:
            session.query(Position).filter(Position.symbol == symbol, Position.order_id == order_id).update(
                {'status': PositionStatus.FAILED, 'close_timestamp': Clock.now(), 'exit_reason': reason}
            )
            session.commit()
        except Exception as e:
            logger.error("Failed to mark position failed", symbol=symbol, error=str(e))
            session.rollback()
        finally:
            session.close()

    async def close_position(self, symbol: str, close_price: Decimal, reason: str = "Unknown", actual_filled_qty: Optional[Decimal] = None, fees: Decimal = ZERO) -> Optional[Position]:
        # Wrapper for manual close that delegates to confirm_position_close logic
        async with self._get_lock(symbol):
            pos = self._position_cache.get(symbol)
            if not pos:
                return None
            qty = actual_filled_qty if actual_filled_qty is not None else pos.quantity
            return await self.confirm_position_close(symbol, close_price, qty, fees, reason)

    async def update_position_stop_loss(self, pos: Position, new_stop_price: Optional[Decimal], new_ref_price: Optional[Decimal] = None, activate_trailing: bool = False) -> Position:
        async with self._get_lock(pos.symbol):
            cached_pos = self._position_cache.get(pos.symbol)
            if not cached_pos:
                return pos
            
            updates = {}
            if new_stop_price is not None: 
                cached_pos.stop_loss_price = new_stop_price
                updates['stop_loss_price'] = new_stop_price
            if new_ref_price is not None: 
                cached_pos.trailing_ref_price = new_ref_price
                updates['trailing_ref_price'] = new_ref_price
            if activate_trailing: 
                cached_pos.trailing_stop_active = True
                updates['trailing_stop_active'] = True
            
            if updates:
                self._persist_queue.put_nowait((self._update_position_sync, (pos.symbol, pos.order_id, updates)))
            
            return cached_pos

    async def manage_breakeven_stop(self, pos: Position, current_price: Decimal, be_config: 'BreakevenConfig') -> Position:
        if not be_config.enabled or pos.breakeven_active:
            return pos
        
        should_update = False
        new_sl = pos.stop_loss_price
        
        if pos.side == 'BUY':
            activation_price = pos.entry_price * (ONE + to_decimal(be_config.activation_pct))
            if current_price >= activation_price:
                proposed_sl = pos.entry_price * (ONE + to_decimal(be_config.buffer_pct))
                if proposed_sl > (pos.stop_loss_price or ZERO):
                    new_sl = proposed_sl
                    should_update = True
        elif pos.side == 'SELL':
            activation_price = pos.entry_price * (ONE - to_decimal(be_config.activation_pct))
            if current_price <= activation_price:
                proposed_sl = pos.entry_price * (ONE - to_decimal(be_config.buffer_pct))
                if proposed_sl < (pos.stop_loss_price or Dec('Infinity')):
                    new_sl = proposed_sl
                    should_update = True
        
        if should_update:
            async with self._get_lock(pos.symbol):
                cached_pos = self._position_cache.get(pos.symbol)
                if cached_pos:
                    cached_pos.stop_loss_price = new_sl
                    cached_pos.breakeven_active = True
                    updates = {'stop_loss_price': new_sl, 'breakeven_active': True}
                    self._persist_queue.put_nowait((self._update_position_sync, (pos.symbol, pos.order_id, updates)))
                    return cached_pos
        return pos

    async def get_daily_realized_pnl(self) -> Decimal:
        return await self._run_in_executor(self._calculate_daily_pnl_sync)

    def _calculate_daily_pnl_sync(self) -> Decimal:
        session = self.SessionLocal()
        try:
            today_utc = Clock.now().date()
            daily_pnl = session.query(func.sum(Position.pnl)).filter(
                Position.status == PositionStatus.CLOSED,
                cast(Position.close_timestamp, Date) == today_utc
            ).scalar()
            return daily_pnl or ZERO
        finally:
            session.close()

    async def get_all_closed_positions(self) -> List[Position]:
        return await self._run_in_executor(self._get_all_closed_positions_sync)

    def _get_all_closed_positions_sync(self) -> List[Position]:
        session = self.SessionLocal()
        try:
            return session.query(Position).filter(Position.status == PositionStatus.CLOSED).all()
        finally:
            session.close()

    async def get_recent_execution_history(self, limit: int) -> List[Position]:
        return await self._run_in_executor(self._get_recent_execution_history_sync, limit)

    def _get_recent_execution_history_sync(self, limit: int) -> List[Position]:
        session = self.SessionLocal()
        try:
            return session.query(Position).filter(
                Position.status.in_([PositionStatus.CLOSED, PositionStatus.FAILED])
            ).order_by(Position.creation_timestamp.desc()).limit(limit).all()
        finally:
            session.close()

    def get_portfolio_value(self, latest_prices: Dict[str, float], open_positions: List[Position]) -> Decimal:
        unrealized_pnl = ZERO
        for pos in open_positions:
            # Convert float price to Decimal for calculation
            current_price = to_decimal(latest_prices.get(pos.symbol, float(pos.entry_price)))
            pnl = (current_price - pos.entry_price) * pos.quantity
            if pos.side == 'SELL':
                pnl = -pnl
            unrealized_pnl += pnl
        return self._initial_capital + self._realized_pnl + unrealized_pnl

    def close(self):
        if self._persistence_task:
            self._persistence_task.cancel()
        self.engine.dispose()
        logger.info("PositionManager database connection closed.")
