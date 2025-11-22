import asyncio
import json
import enum
from typing import List, Optional, Dict, Any
from decimal import Decimal
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Enum as SQLAlchemyEnum, func, Boolean, cast, Date, text, inspect, event
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.sqlite import insert

from bot_core.logger import get_logger
from bot_core.config import DatabaseConfig
from bot_core.utils import Clock
from bot_core.common import Arith, OrderSide

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
    # Store Decimals as Strings in SQLite to preserve precision
    quantity = Column(String, nullable=False)
    entry_price = Column(String, nullable=False)
    status = Column(SQLAlchemyEnum(PositionStatus), default=PositionStatus.OPEN, nullable=False, index=True)
    order_id = Column(String, nullable=True, index=True)
    closing_order_id = Column(String, nullable=True, index=True)
    trade_id = Column(String, nullable=True)
    stop_loss_price = Column(String, nullable=True)
    take_profit_price = Column(String, nullable=True)
    
    creation_timestamp = Column(DateTime, default=Clock.now)
    open_timestamp = Column(DateTime, default=Clock.now)
    close_timestamp = Column(DateTime, nullable=True)
    
    close_price = Column(String, nullable=True)
    pnl = Column(String, nullable=True)
    fees = Column(String, default="0.0", nullable=False)
    
    decision_price = Column(String, nullable=True)
    execution_latency_ms = Column(Float, nullable=True)
    slippage_pct = Column(Float, nullable=True)

    trailing_ref_price = Column(String, nullable=True)
    trailing_stop_active = Column(Boolean, default=False, nullable=False)
    breakeven_active = Column(Boolean, default=False, nullable=False)
    strategy_metadata = Column(String, nullable=True)
    exit_reason = Column(String, nullable=True)

    # --- In-Memory Helpers ---
    @property
    def quantity_dec(self) -> Decimal:
        return Arith.decimal(self.quantity)
    
    @property
    def entry_price_dec(self) -> Decimal:
        return Arith.decimal(self.entry_price)

    @property
    def pnl_dec(self) -> Decimal:
        return Arith.decimal(self.pnl) if self.pnl else Decimal(0)

class PositionManager:
    """
    Manages trading positions with a Write-Behind Cache pattern.
    In-memory state is authoritative and uses Decimal.
    Persistence happens asynchronously.
    """
    def __init__(self, config: DatabaseConfig, initial_capital: float, alert_system: Any = None):
        self.engine = create_engine(f'sqlite:///{config.path}')
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        self._initial_capital = Arith.decimal(initial_capital)
        self._realized_pnl = Decimal(0)
        self.alert_system = alert_system
        
        # In-Memory State (The Truth)
        self._position_cache: Dict[str, Position] = {}
        self._persistence_queue = asyncio.Queue()
        self._running = False
        
        logger.info("PositionManager initialized (Write-Behind, Decimal Precision).")

    async def initialize(self):
        # Load state from DB
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_state_sync)
        self._running = True
        asyncio.create_task(self._persistence_loop())

    def _load_state_sync(self):
        session = self.SessionLocal()
        try:
            # Load Active Positions
            positions = session.query(Position).filter(
                Position.status.in_([PositionStatus.OPEN, PositionStatus.PENDING])
            ).all()
            for p in positions:
                session.expunge(p)
                self._position_cache[p.symbol] = p
            
            # Load Realized PnL
            # Note: In SQLite, sum() on strings might behave oddly if not cast, 
            # but we store as string. Better to load all closed and sum in python for safety or cast.
            # For simplicity/speed in this snippet, we assume 0.0 on restart or implement a separate PnL table.
            # Here we just reset realized PnL session tracking to 0 (historical is in DB).
            self._realized_pnl = Decimal(0) 
            
        finally:
            session.close()

    async def _persistence_loop(self):
        logger.info("Persistence loop started.")
        while self._running:
            try:
                task = await self._persistence_queue.get()
                await self._run_in_executor(task)
                self._persistence_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Persistence error", error=str(e))

    async def _run_in_executor(self, func):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func)

    # --- Public Interface (Decimal) ---

    async def get_open_position(self, symbol: str) -> Optional[Position]:
        pos = self._position_cache.get(symbol)
        if pos and pos.status == PositionStatus.OPEN:
            return pos
        return None

    async def get_all_active_positions(self) -> List[Position]:
        return list(self._position_cache.values())

    def get_portfolio_value(self, latest_prices: Dict[str, float], open_positions: List[Position]) -> float:
        # Returns float for compatibility with RiskManager (legacy parts) / Reporting
        # But calculates internally with Decimal
        unrealized_pnl = Decimal(0)
        for pos in open_positions:
            current_price = Arith.decimal(latest_prices.get(pos.symbol, pos.entry_price))
            qty = Arith.decimal(pos.quantity)
            entry = Arith.decimal(pos.entry_price)
            
            diff = (current_price - entry) * qty
            if pos.side == 'SELL':
                diff = -diff
            unrealized_pnl += diff
            
        total = self._initial_capital + self._realized_pnl + unrealized_pnl
        return float(total)

    async def create_pending_position(self, symbol: str, side: str, order_id: str, trade_id: str, decision_price: float, strategy_metadata: Optional[Dict[str, Any]] = None) -> Optional[Position]:
        if symbol in self._position_cache:
            return None

        new_pos = Position(
            symbol=symbol, side=side, quantity="0", entry_price="0",
            status=PositionStatus.PENDING, order_id=order_id, trade_id=trade_id,
            decision_price=str(decision_price), creation_timestamp=Clock.now(),
            strategy_metadata=json.dumps(strategy_metadata) if strategy_metadata else None
        )
        
        self._position_cache[symbol] = new_pos
        self._persistence_queue.put_nowait(lambda: self._persist_new(new_pos))
        return new_pos

    async def confirm_position_open(self, symbol: str, order_id: str, quantity: float, entry_price: float, stop_loss: float, take_profit: float, fees: float = 0.0) -> Optional[Position]:
        pos = self._position_cache.get(symbol)
        if not pos or pos.status != PositionStatus.PENDING:
            return None

        # Update In-Memory (Decimal)
        pos.status = PositionStatus.OPEN
        pos.quantity = str(Arith.decimal(quantity))
        pos.entry_price = str(Arith.decimal(entry_price))
        pos.stop_loss_price = str(Arith.decimal(stop_loss))
        pos.take_profit_price = str(Arith.decimal(take_profit))
        pos.fees = str(Arith.decimal(fees))
        pos.open_timestamp = Clock.now()
        
        # Persist
        # We pass a snapshot of data to avoid race conditions if pos changes before DB write
        snapshot = self._snapshot(pos)
        self._persistence_queue.put_nowait(lambda: self._persist_update(snapshot))
        return pos

    async def confirm_position_close(self, symbol: str, exit_price: float, filled_qty: float, fees: float, reason: str = "Filled") -> Optional[Position]:
        pos = self._position_cache.get(symbol)
        if not pos:
            return None

        exit_price_dec = Arith.decimal(exit_price)
        filled_qty_dec = Arith.decimal(filled_qty)
        fees_dec = Arith.decimal(fees)
        current_qty_dec = Arith.decimal(pos.quantity)
        entry_price_dec = Arith.decimal(pos.entry_price)

        # Calculate PnL
        gross_pnl = (exit_price_dec - entry_price_dec) * filled_qty_dec
        if pos.side == 'SELL':
            gross_pnl = -gross_pnl
        
        net_pnl = gross_pnl - fees_dec

        if filled_qty_dec >= (current_qty_dec * Decimal('0.999')):
            # Full Close
            pos.status = PositionStatus.CLOSED
            pos.close_price = str(exit_price_dec)
            pos.close_timestamp = Clock.now()
            pos.pnl = str(net_pnl)
            pos.exit_reason = reason
            pos.fees = str(Arith.decimal(pos.fees) + fees_dec)
            
            self._realized_pnl += net_pnl
            del self._position_cache[symbol]
            
            snapshot = self._snapshot(pos)
            self._persistence_queue.put_nowait(lambda: self._persist_update(snapshot))
            return pos
        else:
            # Partial Close (Not fully implemented in this snippet for brevity, but logic is similar)
            # Would update quantity and create a separate 'ClosedPosition' record
            return pos

    async def mark_position_failed(self, symbol: str, trade_id: str, reason: str):
        pos = self._position_cache.get(symbol)
        if pos and pos.trade_id == trade_id:
            pos.status = PositionStatus.FAILED
            pos.exit_reason = reason
            del self._position_cache[symbol]
            snapshot = self._snapshot(pos)
            self._persistence_queue.put_nowait(lambda: self._persist_update(snapshot))

    async def update_pending_order_id(self, symbol: str, old_id: str, new_id: str):
        pos = self._position_cache.get(symbol)
        if pos:
            pos.order_id = new_id
            snapshot = self._snapshot(pos)
            self._persistence_queue.put_nowait(lambda: self._persist_update(snapshot))

    async def register_closing_order(self, symbol: str, order_id: str):
        pos = self._position_cache.get(symbol)
        if pos:
            pos.closing_order_id = order_id
            snapshot = self._snapshot(pos)
            self._persistence_queue.put_nowait(lambda: self._persist_update(snapshot))

    # --- Persistence Helpers ---

    def _snapshot(self, pos: Position) -> Dict[str, Any]:
        """Creates a dict snapshot of the position for safe persistence."""
        return {c.name: getattr(pos, c.name) for c in pos.__table__.columns}

    def _persist_new(self, pos_obj: Position):
        session = self.SessionLocal()
        try:
            session.add(pos_obj)
            session.commit()
        except Exception as e:
            logger.error("DB Insert Failed", error=str(e))
            session.rollback()
        finally:
            session.close()

    def _persist_update(self, snapshot: Dict[str, Any]):
        session = self.SessionLocal()
        try:
            session.query(Position).filter(Position.id == snapshot['id']).update(snapshot)
            session.commit()
        except Exception as e:
            logger.error("DB Update Failed", error=str(e))
            session.rollback()
        finally:
            session.close()

    def close(self):
        self._running = False
        self.engine.dispose()
