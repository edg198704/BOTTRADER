import datetime
import asyncio
import enum
import json
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Enum as SQLAlchemyEnum, func, Boolean, cast, Date, text, inspect, event
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.sqlite import insert

from bot_core.logger import get_logger
from bot_core.config import DatabaseConfig
from bot_core.utils import Clock

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
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    status = Column(SQLAlchemyEnum(PositionStatus), default=PositionStatus.OPEN, nullable=False, index=True)
    order_id = Column(String, nullable=True, index=True)
    closing_order_id = Column(String, nullable=True, index=True)
    trade_id = Column(String, nullable=True)
    stop_loss_price = Column(Float, nullable=True)
    take_profit_price = Column(Float, nullable=True)
    
    creation_timestamp = Column(DateTime, default=Clock.now)
    open_timestamp = Column(DateTime, default=Clock.now)
    close_timestamp = Column(DateTime, nullable=True)
    
    close_price = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    fees = Column(Float, default=0.0, nullable=False)
    
    decision_price = Column(Float, nullable=True)
    execution_latency_ms = Column(Float, nullable=True)
    slippage_pct = Column(Float, nullable=True)

    trailing_ref_price = Column(Float, nullable=True)
    trailing_stop_active = Column(Boolean, default=False, nullable=False)
    breakeven_active = Column(Boolean, default=False, nullable=False)
    strategy_metadata = Column(String, nullable=True)
    
    exit_reason = Column(String, nullable=True)

class PortfolioState(Base):
    __tablename__ = 'portfolio_state'
    id = Column(Integer, primary_key=True)
    initial_capital = Column(Float, nullable=False)
    peak_equity = Column(Float, nullable=False)
    last_update = Column(DateTime, default=Clock.now, onupdate=Clock.now)

class RiskState(Base):
    __tablename__ = 'risk_state'
    symbol = Column(String, primary_key=True)
    consecutive_losses = Column(Integer, default=0)
    cooldown_until = Column(DateTime, nullable=True)
    last_updated = Column(DateTime, default=Clock.now, onupdate=Clock.now)

class PositionManager:
    """
    Manages trading positions and risk state with a high-performance Write-Through Cache.
    Reads are served from memory (O(1)). Writes are persisted to DB and updated in cache.
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
        self._ensure_schema_updates()
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._initial_capital = initial_capital
        self._realized_pnl = 0.0
        self.alert_system = alert_system
        self._db_lock = asyncio.Lock()
        
        self._position_cache: Dict[str, Position] = {}
        self._portfolio_state_cache: Optional[Dict[str, float]] = None
        
        logger.info("PositionManager initialized (WAL mode, Write-Through Cache enabled).")

    def _ensure_schema_updates(self):
        try:
            insp = inspect(self.engine)
            with self.engine.connect() as conn:
                if insp.has_table('positions'):
                    columns = [c['name'] for c in insp.get_columns('positions')]
                    if 'strategy_metadata' not in columns:
                        conn.execute(text("ALTER TABLE positions ADD COLUMN strategy_metadata TEXT"))
                    if 'trade_id' not in columns:
                        conn.execute(text("ALTER TABLE positions ADD COLUMN trade_id TEXT"))
                    if 'fees' not in columns:
                        conn.execute(text("ALTER TABLE positions ADD COLUMN fees FLOAT DEFAULT 0.0"))
                    if 'breakeven_active' not in columns:
                        conn.execute(text("ALTER TABLE positions ADD COLUMN breakeven_active BOOLEAN DEFAULT 0"))
                    if 'exit_reason' not in columns:
                        conn.execute(text("ALTER TABLE positions ADD COLUMN exit_reason TEXT"))
                    if 'decision_price' not in columns:
                        conn.execute(text("ALTER TABLE positions ADD COLUMN decision_price FLOAT"))
                    if 'creation_timestamp' not in columns:
                        conn.execute(text("ALTER TABLE positions ADD COLUMN creation_timestamp DATETIME"))
                    if 'execution_latency_ms' not in columns:
                        conn.execute(text("ALTER TABLE positions ADD COLUMN execution_latency_ms FLOAT"))
                    if 'slippage_pct' not in columns:
                        conn.execute(text("ALTER TABLE positions ADD COLUMN slippage_pct FLOAT"))
                    if 'closing_order_id' not in columns:
                        conn.execute(text("ALTER TABLE positions ADD COLUMN closing_order_id TEXT"))
                
                if not insp.has_table('risk_state'):
                    RiskState.__table__.create(self.engine)

        except Exception as e:
            logger.warning(f"Schema update check failed: {e}")

    async def initialize(self):
        async with self._db_lock:
            self._realized_pnl = await self._run_in_executor(self._calculate_initial_realized_pnl_sync)
            await self._run_in_executor(self._initialize_portfolio_state_sync)
            self._portfolio_state_cache = await self._run_in_executor(self._get_portfolio_state_sync)
            active_positions = await self._run_in_executor(self._get_all_active_positions_sync)
            self._position_cache = {p.symbol: p for p in active_positions}
            
        logger.info("PositionManager state initialized.", 
                    realized_pnl=self._realized_pnl, 
                    cached_positions=len(self._position_cache))

    async def reconcile_positions(self, exchange_api: 'ExchangeAPI', latest_prices: Dict[str, float]):
        logger.info("Starting position reconciliation...")
        
        # 1. Get Exchange State
        try:
            # For Spot, we use balances. For Futures, we would use fetch_positions.
            # Assuming Spot/Margin mix based on config structure.
            balances = await exchange_api.get_balance()
        except Exception as e:
            logger.error("Reconciliation failed: Could not fetch exchange balance", error=str(e))
            return

        # 2. Get DB State
        open_positions = await self.get_all_open_positions()
        db_map = {p.symbol: p for p in open_positions}
        
        # 3. Check for Phantoms (In DB, not in Exchange)
        for symbol, pos in db_map.items():
            base_asset = symbol.split('/')[0]
            held_balance = balances.get(base_asset, {}).get('total', 0.0)
            
            # Tolerance for dust (5% mismatch allowed)
            if held_balance < (pos.quantity * 0.95):
                logger.critical("Phantom position detected! Closing in DB.", symbol=symbol, db_qty=pos.quantity, exchange_qty=held_balance)
                # We close it in DB with a special reason, effectively "writing off" the position
                await self.close_position(symbol, pos.entry_price, reason="Reconciliation: Phantom Position")
                if self.alert_system:
                    await self.alert_system.send_alert("CRITICAL", f"Phantom position detected and removed for {symbol}")
            
            # Check for Quantity Mismatch (Exchange has LESS than DB, but not zero)
            elif held_balance < (pos.quantity * 0.99):
                logger.warning("Quantity mismatch detected. Adjusting DB.", symbol=symbol, db_qty=pos.quantity, exchange_qty=held_balance)
                # Adjust quantity in DB (Partial Close logic without PnL realization for the missing part? 
                # Or just update? Updating is safer to reflect reality.)
                # For simplicity in this architecture, we treat it as a partial close at current price.
                diff = pos.quantity - held_balance
                current_price = latest_prices.get(symbol, pos.entry_price)
                await self.confirm_position_close(symbol, current_price, diff, 0.0, reason="Reconciliation: Quantity Adjustment")

        # 4. Check for Shadows (In Exchange, not in DB)
        # Note: On Spot, holding an asset doesn't always mean a "Position" (could be HODL).
        # We log significant balances that are not tracked.
        for asset, balance_data in balances.items():
            total = balance_data.get('total', 0.0)
            if total > 0:
                # Construct potential symbol (e.g., BTC -> BTC/USDT)
                # This is heuristic and depends on the quote currency used in config.
                # Assuming USDT for now.
                potential_symbol = f"{asset}/USDT"
                if potential_symbol not in db_map and asset != 'USDT':
                    # Check if value is significant (> $10)
                    price = latest_prices.get(potential_symbol, 0.0)
                    if price * total > 10.0:
                        logger.warning("Shadow position detected (Untracked Balance).", asset=asset, amount=total, value_usd=price*total)
                        # Optional: Auto-import logic could go here if configured.

    async def _run_in_executor(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func, *args)

    # --- Synchronous DB Helpers ---

    def _calculate_initial_realized_pnl_sync(self) -> float:
        session = self.SessionLocal()
        try:
            realized_pnl = session.query(func.sum(Position.pnl)).filter(Position.status == PositionStatus.CLOSED).scalar()
            return realized_pnl or 0.0
        finally:
            session.close()

    def _initialize_portfolio_state_sync(self):
        session = self.SessionLocal()
        try:
            state = session.query(PortfolioState).filter(PortfolioState.id == 1).first()
            if not state:
                state = PortfolioState(id=1, initial_capital=self._initial_capital, peak_equity=self._initial_capital)
                session.add(state)
                session.commit()
        except Exception as e:
            logger.error("Failed to initialize portfolio state", error=str(e))
            session.rollback()
        finally:
            session.close()

    def _get_portfolio_state_sync(self) -> Optional[Dict[str, float]]:
        session = self.SessionLocal()
        try:
            state = session.query(PortfolioState).filter(PortfolioState.id == 1).first()
            if state:
                return {'initial_capital': state.initial_capital, 'peak_equity': state.peak_equity}
            return None
        finally:
            session.close()

    def _update_portfolio_high_water_mark_sync(self, new_peak: float):
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

    def _get_all_active_positions_sync(self) -> List[Position]:
        session = self.SessionLocal()
        try:
            positions = session.query(Position).filter(
                Position.status.in_([PositionStatus.OPEN, PositionStatus.PENDING])
            ).all()
            for p in positions:
                session.expunge(p)
            return positions
        finally:
            session.close()

    # --- Risk State Management ---

    async def get_all_risk_states(self) -> Dict[str, Dict[str, Any]]:
        async with self._db_lock:
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
        async with self._db_lock:
            await self._run_in_executor(self._update_risk_state_sync, symbol, consecutive_losses, cooldown_until)

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

    # --- Public Async Interface (Cached Reads) ---

    async def get_portfolio_state(self) -> Optional[Dict[str, float]]:
        return self._portfolio_state_cache

    async def update_portfolio_high_water_mark(self, new_peak: float):
        if self._portfolio_state_cache:
            self._portfolio_state_cache['peak_equity'] = max(self._portfolio_state_cache['peak_equity'], new_peak)
        else:
            self._portfolio_state_cache = {'initial_capital': self._initial_capital, 'peak_equity': new_peak}
        async with self._db_lock:
            await self._run_in_executor(self._update_portfolio_high_water_mark_sync, new_peak)

    async def get_open_position(self, symbol: str) -> Optional[Position]:
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

    async def get_aggregated_open_positions(self) -> Dict[str, float]:
        agg = {}
        for p in self._position_cache.values():
            if p.status == PositionStatus.OPEN:
                agg[p.symbol] = agg.get(p.symbol, 0.0) + p.quantity
        return agg

    # --- State Changing Methods (Write-Through) ---

    async def create_pending_position(self, symbol: str, side: str, order_id: str, trade_id: str, decision_price: float, strategy_metadata: Optional[Dict[str, Any]] = None) -> Optional[Position]:
        async with self._db_lock:
            new_pos = await self._run_in_executor(self._create_pending_position_sync, symbol, side, order_id, trade_id, decision_price, strategy_metadata)
            if new_pos:
                self._position_cache[symbol] = new_pos
            return new_pos

    def _create_pending_position_sync(self, symbol: str, side: str, order_id: str, trade_id: str, decision_price: float, strategy_metadata: Optional[Dict[str, Any]]) -> Optional[Position]:
        session = self.SessionLocal()
        try:
            existing = session.query(Position).filter(
                Position.symbol == symbol, 
                Position.status.in_([PositionStatus.OPEN, PositionStatus.PENDING])
            ).first()
            
            if existing:
                return None

            metadata_json = json.dumps(strategy_metadata) if strategy_metadata else None
            new_position = Position(
                symbol=symbol, side=side, quantity=0.0, entry_price=0.0,
                status=PositionStatus.PENDING, order_id=order_id, trade_id=trade_id,
                decision_price=decision_price, creation_timestamp=Clock.now(),
                trailing_ref_price=0.0, trailing_stop_active=False,
                breakeven_active=False, strategy_metadata=metadata_json, fees=0.0
            )
            session.add(new_position)
            session.commit()
            session.refresh(new_position)
            session.expunge(new_position)
            return new_position
        except Exception as e:
            logger.error("Failed to create pending position", symbol=symbol, error=str(e))
            session.rollback()
            return None
        finally:
            session.close()

    async def update_pending_order_id(self, symbol: str, old_order_id: str, new_order_id: str) -> bool:
        async with self._db_lock:
            success = await self._run_in_executor(self._update_pending_order_id_sync, symbol, old_order_id, new_order_id)
            if success and symbol in self._position_cache:
                self._position_cache[symbol].order_id = new_order_id
            return success

    def _update_pending_order_id_sync(self, symbol: str, old_order_id: str, new_order_id: str) -> bool:
        session = self.SessionLocal()
        try:
            position = session.query(Position).filter(
                Position.symbol == symbol, Position.order_id == old_order_id, Position.status == PositionStatus.PENDING
            ).first()
            if position:
                position.order_id = new_order_id
                session.commit()
                return True
            return False
        except Exception as e:
            logger.error("Failed to update pending order ID", symbol=symbol, error=str(e))
            session.rollback()
            return False
        finally:
            session.close()

    async def confirm_position_open(self, symbol: str, order_id: str, quantity: float, entry_price: float, stop_loss: float, take_profit: float, fees: float = 0.0) -> Optional[Position]:
        async with self._db_lock:
            updated_pos = await self._run_in_executor(self._confirm_position_open_sync, symbol, order_id, quantity, entry_price, stop_loss, take_profit, fees)
            if updated_pos:
                self._position_cache[symbol] = updated_pos
                if self.alert_system:
                    asyncio.run_coroutine_threadsafe(self.alert_system.send_alert(
                        level='info',
                        message=f"🟢 Opened {updated_pos.side} position for {symbol}",
                        details={'symbol': symbol, 'side': updated_pos.side, 'quantity': quantity, 'entry_price': entry_price, 'fees': fees}
                    ), asyncio.get_running_loop())
            return updated_pos

    def _confirm_position_open_sync(self, symbol: str, order_id: str, quantity: float, entry_price: float, stop_loss: float, take_profit: float, fees: float) -> Optional[Position]:
        session = self.SessionLocal()
        try:
            position = session.query(Position).filter(
                Position.symbol == symbol, Position.order_id == order_id, Position.status == PositionStatus.PENDING
            ).first()

            if not position:
                return None

            fill_time = Clock.now()
            latency = 0.0
            if position.creation_timestamp:
                latency = (fill_time - position.creation_timestamp).total_seconds() * 1000.0
            
            slippage = 0.0
            if position.decision_price and position.decision_price > 0:
                slippage = (entry_price - position.decision_price) / position.decision_price

            position.status = PositionStatus.OPEN
            position.quantity = quantity
            position.entry_price = entry_price
            position.stop_loss_price = stop_loss
            position.take_profit_price = take_profit
            position.trailing_ref_price = entry_price
            position.open_timestamp = fill_time
            position.fees += fees
            position.execution_latency_ms = latency
            position.slippage_pct = slippage
            
            session.commit()
            session.refresh(position)
            session.expunge(position)
            return position
        except Exception as e:
            logger.error("Failed to confirm position open", symbol=symbol, error=str(e))
            session.rollback()
            return None
        finally:
            session.close()

    async def register_closing_order(self, symbol: str, closing_order_id: str):
        """Registers the order ID that is intended to close the position."""
        async with self._db_lock:
            await self._run_in_executor(self._register_closing_order_sync, symbol, closing_order_id)
            if symbol in self._position_cache:
                self._position_cache[symbol].closing_order_id = closing_order_id

    def _register_closing_order_sync(self, symbol: str, closing_order_id: str):
        session = self.SessionLocal()
        try:
            position = session.query(Position).filter(Position.symbol == symbol, Position.status == PositionStatus.OPEN).first()
            if position:
                position.closing_order_id = closing_order_id
                session.commit()
        except Exception as e:
            logger.error("Failed to register closing order", symbol=symbol, error=str(e))
            session.rollback()
        finally:
            session.close()

    async def confirm_position_close(self, symbol: str, exit_price: float, filled_qty: float, fees: float, reason: str = "Filled") -> Optional[Position]:
        """Confirms a position closure based on a filled closing order."""
        async with self._db_lock:
            closed_pos = await self._run_in_executor(self._confirm_position_close_sync, symbol, exit_price, filled_qty, fees, reason)
            if closed_pos:
                if closed_pos.status == PositionStatus.CLOSED:
                    self._realized_pnl += closed_pos.pnl
                    if symbol in self._position_cache:
                        del self._position_cache[symbol]
                    if self.alert_system:
                        pnl_emoji = "🟢" if closed_pos.pnl >= 0 else "🔴"
                        asyncio.run_coroutine_threadsafe(self.alert_system.send_alert(
                            level='info',
                            message=f"{pnl_emoji} Closed position for {symbol}",
                            details={'symbol': symbol, 'net_pnl': f'{closed_pos.pnl:.2f}', 'reason': reason}
                        ), asyncio.get_running_loop())
                else:
                    # Partial close, update cache
                    self._position_cache[symbol] = closed_pos
            return closed_pos

    def _confirm_position_close_sync(self, symbol: str, exit_price: float, filled_qty: float, fees: float, reason: str) -> Optional[Position]:
        session = self.SessionLocal()
        try:
            position = session.query(Position).filter(Position.symbol == symbol, Position.status == PositionStatus.OPEN).first()
            if not position:
                return None

            # Handle Partial Fills
            if filled_qty < (position.quantity * 0.999):
                # Reduce Position
                gross_pnl = (exit_price - position.entry_price) * filled_qty
                if position.side == 'SELL':
                    gross_pnl = -gross_pnl
                
                chunk_ratio = filled_qty / position.quantity
                entry_fees_chunk = position.fees * chunk_ratio
                total_chunk_fees = entry_fees_chunk + fees
                net_pnl = gross_pnl - total_chunk_fees

                # Create a closed record for the chunk
                closed_chunk = Position(
                    symbol=symbol, side=position.side, quantity=filled_qty, entry_price=position.entry_price,
                    status=PositionStatus.CLOSED, order_id=position.order_id, trade_id=position.trade_id,
                    open_timestamp=position.open_timestamp, close_timestamp=Clock.now(), close_price=exit_price,
                    pnl=net_pnl, fees=total_chunk_fees, exit_reason=reason + " (Partial)"
                )
                session.add(closed_chunk)

                # Update remaining position
                position.quantity -= filled_qty
                position.fees -= entry_fees_chunk
                self._realized_pnl += net_pnl
                
                session.commit()
                session.refresh(position)
                session.expunge(position)
                return position
            else:
                # Full Close
                gross_pnl = (exit_price - position.entry_price) * position.quantity
                if position.side == 'SELL':
                    gross_pnl = -gross_pnl

                position.fees += fees
                net_pnl = gross_pnl - position.fees

                position.status = PositionStatus.CLOSED
                position.close_price = exit_price
                position.close_timestamp = Clock.now()
                position.pnl = net_pnl
                position.exit_reason = reason
                
                session.commit()
                session.refresh(position)
                session.expunge(position)
                return position
        except Exception as e:
            logger.error("Failed to confirm position close", symbol=symbol, error=str(e))
            session.rollback()
            return None
        finally:
            session.close()

    async def mark_position_failed(self, symbol: str, order_id: str, reason: str):
        async with self._db_lock:
            await self._run_in_executor(self._mark_position_failed_sync, symbol, order_id, reason)
            if symbol in self._position_cache and self._position_cache[symbol].order_id == order_id:
                del self._position_cache[symbol]

    def _mark_position_failed_sync(self, symbol: str, order_id: str, reason: str):
        session = self.SessionLocal()
        try:
            position = session.query(Position).filter(
                Position.symbol == symbol, Position.order_id == order_id, Position.status == PositionStatus.PENDING
            ).first()
            if position:
                position.status = PositionStatus.FAILED
                position.close_timestamp = Clock.now()
                position.exit_reason = reason
                session.commit()
        except Exception as e:
            logger.error("Failed to mark position as failed", symbol=symbol, error=str(e))
            session.rollback()
        finally:
            session.close()

    async def close_position(self, symbol: str, close_price: float, reason: str = "Unknown", actual_filled_qty: Optional[float] = None, fees: float = 0.0) -> Optional[Position]:
        """Immediate close (e.g. for dust or reconciliation)."""
        async with self._db_lock:
            closed_pos = await self._run_in_executor(self._close_position_sync, symbol, close_price, reason, actual_filled_qty, fees)
            if closed_pos:
                self._realized_pnl += closed_pos.pnl
                if symbol in self._position_cache:
                    del self._position_cache[symbol]
                if self.alert_system:
                    pnl_emoji = "🟢" if closed_pos.pnl >= 0 else "🔴"
                    asyncio.run_coroutine_threadsafe(self.alert_system.send_alert(
                        level='info',
                        message=f"{pnl_emoji} Closed position for {symbol} due to {reason}",
                        details={'symbol': symbol, 'net_pnl': f'{closed_pos.pnl:.2f}', 'reason': reason}
                    ), asyncio.get_running_loop())
            return closed_pos

    def _close_position_sync(self, symbol: str, close_price: float, reason: str, actual_filled_qty: Optional[float], fees: float) -> Optional[Position]:
        session = self.SessionLocal()
        try:
            position = session.query(Position).filter(Position.symbol == symbol, Position.status == PositionStatus.OPEN).first()
            if not position:
                return None

            calc_qty = actual_filled_qty if actual_filled_qty is not None else position.quantity
            gross_pnl = (close_price - position.entry_price) * calc_qty
            if position.side == 'SELL':
                gross_pnl = -gross_pnl

            position.fees += fees
            net_pnl = gross_pnl - position.fees

            position.status = PositionStatus.CLOSED
            position.close_price = close_price
            position.close_timestamp = Clock.now()
            position.pnl = net_pnl
            position.exit_reason = reason
            
            session.commit()
            session.refresh(position)
            session.expunge(position)
            return position
        except Exception as e:
            logger.error("Failed to close position", symbol=symbol, error=str(e))
            session.rollback()
            return None
        finally:
            session.close()

    async def update_position_stop_loss(self, pos: Position, new_stop_price: Optional[float], new_ref_price: Optional[float] = None, activate_trailing: bool = False) -> Position:
        async with self._db_lock:
            updated_pos = await self._run_in_executor(self._update_position_stop_loss_sync, pos.symbol, new_stop_price, new_ref_price, activate_trailing)
            if updated_pos:
                self._position_cache[pos.symbol] = updated_pos
            return updated_pos

    def _update_position_stop_loss_sync(self, symbol: str, new_stop_price: Optional[float], new_ref_price: Optional[float], activate_trailing: bool) -> Optional[Position]:
        session = self.SessionLocal()
        try:
            pos = session.query(Position).filter(Position.symbol == symbol, Position.status == PositionStatus.OPEN).first()
            if not pos: return None
            
            if new_stop_price is not None: pos.stop_loss_price = new_stop_price
            if new_ref_price is not None: pos.trailing_ref_price = new_ref_price
            if activate_trailing: pos.trailing_stop_active = True

            session.commit()
            session.refresh(pos)
            session.expunge(pos)
            return pos
        except Exception as e:
            logger.error("Failed to update stop loss", symbol=symbol, error=str(e))
            session.rollback()
            return None
        finally:
            session.close()

    async def manage_breakeven_stop(self, pos: Position, current_price: float, be_config: 'BreakevenConfig') -> Position:
        if not be_config.enabled or pos.breakeven_active:
            return pos
        
        should_update = False
        new_sl = pos.stop_loss_price
        
        if pos.side == 'BUY':
            activation_price = pos.entry_price * (1 + be_config.activation_pct)
            if current_price >= activation_price:
                proposed_sl = pos.entry_price * (1 + be_config.buffer_pct)
                if proposed_sl > pos.stop_loss_price:
                    new_sl = proposed_sl
                    should_update = True
        elif pos.side == 'SELL':
            activation_price = pos.entry_price * (1 - be_config.activation_pct)
            if current_price <= activation_price:
                proposed_sl = pos.entry_price * (1 - be_config.buffer_pct)
                if proposed_sl < pos.stop_loss_price:
                    new_sl = proposed_sl
                    should_update = True
        
        if should_update:
            async with self._db_lock:
                updated_pos = await self._run_in_executor(self._activate_breakeven_sync, pos.symbol, new_sl)
                if updated_pos:
                    self._position_cache[pos.symbol] = updated_pos
                    return updated_pos
        return pos

    def _activate_breakeven_sync(self, symbol: str, new_sl: float) -> Optional[Position]:
        session = self.SessionLocal()
        try:
            pos = session.query(Position).filter(Position.symbol == symbol, Position.status == PositionStatus.OPEN).first()
            if pos:
                pos.stop_loss_price = new_sl
                pos.breakeven_active = True
                session.commit()
                session.refresh(pos)
                session.expunge(pos)
                return pos
            return None
        except Exception:
            session.rollback()
            return None
        finally:
            session.close()

    async def get_daily_realized_pnl(self) -> float:
        async with self._db_lock:
            return await self._run_in_executor(self._calculate_daily_pnl_sync)

    def _calculate_daily_pnl_sync(self) -> float:
        session = self.SessionLocal()
        try:
            today_utc = Clock.now().date()
            daily_pnl = session.query(func.sum(Position.pnl)).filter(
                Position.status == PositionStatus.CLOSED,
                cast(Position.close_timestamp, Date) == today_utc
            ).scalar()
            return daily_pnl or 0.0
        finally:
            session.close()

    async def get_all_closed_positions(self) -> List[Position]:
        async with self._db_lock:
            return await self._run_in_executor(self._get_all_closed_positions_sync)

    def _get_all_closed_positions_sync(self) -> List[Position]:
        session = self.SessionLocal()
        try:
            return session.query(Position).filter(Position.status == PositionStatus.CLOSED).all()
        finally:
            session.close()

    async def get_recent_execution_history(self, limit: int) -> List[Position]:
        async with self._db_lock:
            return await self._run_in_executor(self._get_recent_execution_history_sync, limit)

    def _get_recent_execution_history_sync(self, limit: int) -> List[Position]:
        session = self.SessionLocal()
        try:
            return session.query(Position).filter(
                Position.status.in_([PositionStatus.CLOSED, PositionStatus.FAILED])
            ).order_by(Position.creation_timestamp.desc()).limit(limit).all()
        finally:
            session.close()

    def get_portfolio_value(self, latest_prices: Dict[str, float], open_positions: List[Position]) -> float:
        unrealized_pnl = 0.0
        for pos in open_positions:
            current_price = latest_prices.get(pos.symbol, pos.entry_price)
            pnl = (current_price - pos.entry_price) * pos.quantity
            if pos.side == 'SELL':
                pnl = -pnl
            unrealized_pnl += pnl
        return self._initial_capital + self._realized_pnl + unrealized_pnl

    def close(self):
        self.engine.dispose()
        logger.info("PositionManager database connection closed.")
