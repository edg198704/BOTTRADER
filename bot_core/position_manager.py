import datetime
import asyncio
import enum
import json
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Enum as SQLAlchemyEnum, func, Boolean, cast, Date, text, inspect, event
from sqlalchemy.orm import sessionmaker, declarative_base

from bot_core.logger import get_logger
from bot_core.config import DatabaseConfig, RiskManagementConfig, BreakevenConfig
from bot_core.utils import Clock

if TYPE_CHECKING:
    from bot_core.monitoring import AlertSystem

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
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    status = Column(SQLAlchemyEnum(PositionStatus), default=PositionStatus.OPEN, nullable=False)
    order_id = Column(String, nullable=True)
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

class PositionManager:
    def __init__(self, config: DatabaseConfig, initial_capital: float, alert_system: Optional['AlertSystem'] = None):
        self.engine = create_engine(f'sqlite:///{config.path}')
        
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

        Base.metadata.create_all(self.engine)
        self._ensure_schema_updates()
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._initial_capital = initial_capital
        self._realized_pnl = 0.0
        self.alert_system = alert_system
        self._db_lock = asyncio.Lock()
        logger.info("PositionManager initialized and database table created (WAL mode enabled).")

    def _ensure_schema_updates(self):
        try:
            insp = inspect(self.engine)
            if insp.has_table('positions'):
                columns = [c['name'] for c in insp.get_columns('positions')]
                with self.engine.connect() as conn:
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
        except Exception as e:
            logger.warning(f"Schema update check failed: {e}")

    async def initialize(self):
        async with self._db_lock:
            self._realized_pnl = await self._run_in_executor(self._calculate_initial_realized_pnl_sync)
            await self._run_in_executor(self._initialize_portfolio_state_sync)
        logger.info("PositionManager state initialized.", realized_pnl=self._realized_pnl)

    async def _run_in_executor(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func, *args)

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
                logger.info("Initializing new PortfolioState in database.", initial_capital=self._initial_capital)
                state = PortfolioState(
                    id=1,
                    initial_capital=self._initial_capital,
                    peak_equity=self._initial_capital
                )
                session.add(state)
                session.commit()
        except Exception as e:
            logger.error("Failed to initialize portfolio state", error=str(e))
            session.rollback()
        finally:
            session.close()

    async def get_portfolio_state(self) -> Optional[Dict[str, float]]:
        async with self._db_lock:
            return await self._run_in_executor(self._get_portfolio_state_sync)

    def _get_portfolio_state_sync(self) -> Optional[Dict[str, float]]:
        session = self.SessionLocal()
        try:
            state = session.query(PortfolioState).filter(PortfolioState.id == 1).first()
            if state:
                return {'initial_capital': state.initial_capital, 'peak_equity': state.peak_equity}
            return None
        finally:
            session.close()

    async def update_portfolio_high_water_mark(self, new_peak: float):
        async with self._db_lock:
            await self._run_in_executor(self._update_portfolio_high_water_mark_sync, new_peak)

    def _update_portfolio_high_water_mark_sync(self, new_peak: float):
        session = self.SessionLocal()
        try:
            state = session.query(PortfolioState).filter(PortfolioState.id == 1).first()
            if state:
                if new_peak > state.peak_equity:
                    old_peak = state.peak_equity
                    state.peak_equity = new_peak
                    session.commit()
                    logger.info("Updated Portfolio High Water Mark", old_peak=old_peak, new_peak=new_peak)
            else:
                state = PortfolioState(id=1, initial_capital=self._initial_capital, peak_equity=new_peak)
                session.add(state)
                session.commit()
        except Exception as e:
            logger.error("Failed to update portfolio high water mark", error=str(e))
            session.rollback()
        finally:
            session.close()

    async def create_pending_position(self, symbol: str, side: str, order_id: str, trade_id: str, decision_price: float, strategy_metadata: Optional[Dict[str, Any]] = None) -> Optional[Position]:
        async with self._db_lock:
            return await self._run_in_executor(self._create_pending_position_sync, symbol, side, order_id, trade_id, decision_price, strategy_metadata)

    def _create_pending_position_sync(self, symbol: str, side: str, order_id: str, trade_id: str, decision_price: float, strategy_metadata: Optional[Dict[str, Any]]) -> Optional[Position]:
        session = self.SessionLocal()
        try:
            existing = session.query(Position).filter(
                Position.symbol == symbol, 
                Position.status.in_([PositionStatus.OPEN, PositionStatus.PENDING])
            ).first()
            
            if existing:
                logger.warning("Cannot create pending position: Active position already exists.", symbol=symbol, status=existing.status)
                return None

            metadata_json = json.dumps(strategy_metadata) if strategy_metadata else None

            new_position = Position(
                symbol=symbol,
                side=side,
                quantity=0.0,
                entry_price=0.0,
                status=PositionStatus.PENDING,
                order_id=order_id,
                trade_id=trade_id,
                decision_price=decision_price,
                creation_timestamp=Clock.now(),
                trailing_ref_price=0.0,
                trailing_stop_active=False,
                breakeven_active=False,
                strategy_metadata=metadata_json,
                fees=0.0
            )
            session.add(new_position)
            session.commit()
            session.refresh(new_position)
            logger.info("Created PENDING position", symbol=symbol, order_id=order_id, trade_id=trade_id, decision_price=decision_price)
            return new_position
        except Exception as e:
            logger.error("Failed to create pending position", symbol=symbol, error=str(e))
            session.rollback()
            return None
        finally:
            session.close()

    async def update_pending_order_id(self, symbol: str, old_order_id: str, new_order_id: str) -> bool:
        async with self._db_lock:
            return await self._run_in_executor(self._update_pending_order_id_sync, symbol, old_order_id, new_order_id)

    def _update_pending_order_id_sync(self, symbol: str, old_order_id: str, new_order_id: str) -> bool:
        session = self.SessionLocal()
        try:
            position = session.query(Position).filter(
                Position.symbol == symbol,
                Position.order_id == old_order_id,
                Position.status == PositionStatus.PENDING
            ).first()
            
            if position:
                position.order_id = new_order_id
                session.commit()
                logger.info("Updated PENDING position order ID", symbol=symbol, old_id=old_order_id, new_id=new_order_id)
                return True
            else:
                logger.warning("Could not find PENDING position to update order ID", symbol=symbol, old_id=old_order_id)
                return False
        except Exception as e:
            logger.error("Failed to update pending order ID", symbol=symbol, error=str(e))
            session.rollback()
            return False
        finally:
            session.close()

    async def confirm_position_open(self, symbol: str, order_id: str, quantity: float, entry_price: float, stop_loss: float, take_profit: float, fees: float = 0.0) -> Optional[Position]:
        async with self._db_lock:
            return await self._run_in_executor(self._confirm_position_open_sync, symbol, order_id, quantity, entry_price, stop_loss, take_profit, fees)

    def _confirm_position_open_sync(self, symbol: str, order_id: str, quantity: float, entry_price: float, stop_loss: float, take_profit: float, fees: float) -> Optional[Position]:
        session = self.SessionLocal()
        try:
            position = session.query(Position).filter(
                Position.symbol == symbol, 
                Position.order_id == order_id,
                Position.status == PositionStatus.PENDING
            ).first()

            if not position:
                logger.error("Cannot confirm position: No PENDING position found with matching Order ID.", symbol=symbol, order_id=order_id)
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
            logger.info("Confirmed position as OPEN", 
                        symbol=symbol, 
                        order_id=order_id, 
                        quantity=quantity, 
                        price=entry_price, 
                        fees=fees,
                        latency_ms=f"{latency:.2f}",
                        slippage_pct=f"{slippage:.4%}")
            
            if self.alert_system:
                asyncio.run_coroutine_threadsafe(self.alert_system.send_alert(
                    level='info',
                    message=f"🟢 Opened {position.side} position for {symbol}",
                    details={'symbol': symbol, 'side': position.side, 'quantity': quantity, 'entry_price': entry_price, 'fees': fees, 'slippage': f"{slippage:.4%}"}
                ), asyncio.get_running_loop()) return position
        except Exception as e:
            logger.error("Failed to confirm position open", symbol=symbol, error=str(e))
            session.rollback()
            return None
        finally:
            session.close()

    async def mark_position_failed(self, symbol: str, order_id: str, reason: str):
        async with self._db_lock:
            await self._run_in_executor(self._mark_position_failed_sync, symbol, order_id, reason)

    def _mark_position_failed_sync(self, symbol: str, order_id: str, reason: str):
        session = self.SessionLocal()
        try:
            position = session.query(Position).filter(
                Position.symbol == symbol, 
                Position.order_id == order_id,
                Position.status == PositionStatus.PENDING
            ).first()
            
            if position:
                position.status = PositionStatus.FAILED
                position.close_timestamp = Clock.now()
                position.exit_reason = reason
                session.commit()
                logger.info("Marked PENDING position as FAILED", symbol=symbol, order_id=order_id, reason=reason)
        except Exception as e:
            logger.error("Failed to mark position as failed", symbol=symbol, error=str(e))
            session.rollback()
        finally:
            session.close()

    async def get_pending_positions(self) -> List[Position]:
        async with self._db_lock:
            return await self._run_in_executor(self._get_pending_positions_sync)

    def _get_pending_positions_sync(self) -> List[Position]:
        session = self.SessionLocal()
        try:
            return session.query(Position).filter(Position.status == PositionStatus.PENDING).all()
        finally:
            session.close()

    async def close_position(self, symbol: str, close_price: float, reason: str = "Unknown", actual_filled_qty: Optional[float] = None, fees: float = 0.0) -> Optional[Position]:
        async with self._db_lock:
            position = await self._run_in_executor(self._close_position_sync, symbol, close_price, reason, actual_filled_qty, fees)
        if position:
            self._realized_pnl += position.pnl
        return position

    def _close_position_sync(self, symbol: str, close_price: float, reason: str, actual_filled_qty: Optional[float], fees: float) -> Optional[Position]:
        session = self.SessionLocal()
        try:
            position = session.query(Position).filter(Position.symbol == symbol, Position.status == PositionStatus.OPEN).first()
            if not position:
                logger.warning("No open position found to close for symbol", symbol=symbol)
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
            logger.info("Closed position", symbol=symbol, net_pnl=f"{net_pnl:.2f}", gross_pnl=f"{gross_pnl:.2f}", fees=f"{position.fees:.2f}", reason=reason)

            if self.alert_system:
                pnl_emoji = "🟢" if net_pnl >= 0 else "🔴"
                asyncio.run_coroutine_threadsafe(self.alert_system.send_alert(
                    level='info',
                    message=f"{pnl_emoji} Closed position for {symbol} due to {reason}",
                    details={'symbol': symbol, 'net_pnl': f'{net_pnl:.2f}', 'fees': f'{position.fees:.2f}', 'reason': reason, 'close_price': close_price}
                ), asyncio.get_running_loop())

            return position
        except Exception as e:
            logger.error("Failed to close position", symbol=symbol, error=str(e))
            session.rollback()
            return None
        finally:
            session.close()

    async def reduce_position(self, symbol: str, quantity: float, price: float, reason: str, fees: float = 0.0) -> Optional[Position]:
        async with self._db_lock:
            return await self._run_in_executor(self._reduce_position_sync, symbol, quantity, price, reason, fees)

    def _reduce_position_sync(self, symbol: str, quantity: float, price: float, reason: str, fees: float) -> Optional[Position]:
        session = self.SessionLocal()
        try:
            position = session.query(Position).filter(Position.symbol == symbol, Position.status == PositionStatus.OPEN).first()
            if not position:
                logger.warning("No open position found to reduce for symbol", symbol=symbol)
                return None

            if quantity >= (position.quantity * 0.999):
                logger.info("Reduction quantity covers entire position. Closing fully.", symbol=symbol, qty=quantity, pos_qty=position.quantity)
                session.close()
                return self._close_position_sync(symbol, price, reason, actual_filled_qty=quantity, fees=fees)

            gross_pnl = (price - position.entry_price) * quantity
            if position.side == 'SELL':
                gross_pnl = -gross_pnl

            chunk_ratio = quantity / position.quantity
            entry_fees_chunk = position.fees * chunk_ratio
            total_chunk_fees = entry_fees_chunk + fees
            net_pnl = gross_pnl - total_chunk_fees

            closed_chunk = Position(
                symbol=symbol,
                side=position.side,
                quantity=quantity,
                entry_price=position.entry_price,
                status=PositionStatus.CLOSED,
                order_id=position.order_id,
                trade_id=position.trade_id,
                stop_loss_price=position.stop_loss_price,
                take_profit_price=position.take_profit_price,
                open_timestamp=position.open_timestamp,
                close_timestamp=Clock.now(),
                close_price=price,
                pnl=net_pnl,
                fees=total_chunk_fees,
                trailing_ref_price=position.trailing_ref_price,
                trailing_stop_active=position.trailing_stop_active,
                breakeven_active=position.breakeven_active,
                strategy_metadata=position.strategy_metadata,
                exit_reason=reason,
                decision_price=position.decision_price,
                creation_timestamp=position.creation_timestamp,
                execution_latency_ms=position.execution_latency_ms,
                slippage_pct=position.slippage_pct
            )
            session.add(closed_chunk)

            position.quantity -= quantity
            position.fees -= entry_fees_chunk
            
            self._realized_pnl += net_pnl
            session.commit()
            session.refresh(position)
            
            logger.info("Reduced position", symbol=symbol, reduced_by=quantity, remaining=position.quantity, net_pnl=f"{net_pnl:.2f}", fees=f"{total_chunk_fees:.2f}", reason=reason)

            if self.alert_system:
                pnl_emoji = "🟢" if net_pnl >= 0 else "🔴"
                asyncio.run_coroutine_threadsafe(self.alert_system.send_alert(
                    level='info',
                    message=f"{pnl_emoji} Partially closed {symbol} ({reason})",
                    details={'symbol': symbol, 'reduced_qty': quantity, 'remaining': position.quantity, 'net_pnl': f'{net_pnl:.2f}'}
                ), asyncio.get_running_loop())

            return position
        except Exception as e:
            logger.error("Failed to reduce position", symbol=symbol, error=str(e))
            session.rollback()
            return None
        finally:
            session.close()

    async def update_position_stop_loss(self, pos: Position, new_stop_price: Optional[float], new_ref_price: Optional[float] = None, activate_trailing: bool = False) -> Position:
        async with self._db_lock:
            return await self._run_in_executor(self._update_position_stop_loss_sync, pos, new_stop_price, new_ref_price, activate_trailing)

    def _update_position_stop_loss_sync(self, pos: Position, new_stop_price: Optional[float], new_ref_price: Optional[float], activate_trailing: bool) -> Position:
        session = self.SessionLocal()
        try:
            pos = session.merge(pos)
            updated = False
            
            if new_stop_price is not None and new_stop_price != pos.stop_loss_price:
                pos.stop_loss_price = new_stop_price
                updated = True
            
            if new_ref_price is not None and new_ref_price != pos.trailing_ref_price:
                pos.trailing_ref_price = new_ref_price
                updated = True
                
            if activate_trailing and not pos.trailing_stop_active:
                pos.trailing_stop_active = True
                updated = True

            if updated:
                session.commit()
                session.refresh(pos)
                logger.debug("Updated position stop loss", symbol=pos.symbol, new_sl=pos.stop_loss_price, active=pos.trailing_stop_active)
            
            return pos
        except Exception as e:
            logger.error("Failed to update position stop loss", symbol=pos.symbol, error=str(e))
            session.rollback()
            return pos
        finally:
            session.close()

    async def manage_breakeven_stop(self, pos: Position, current_price: float, be_config: BreakevenConfig) -> Position:
        async with self._db_lock:
            return await self._run_in_executor(self._manage_breakeven_stop_sync, pos, current_price, be_config)

    def _manage_breakeven_stop_sync(self, pos: Position, current_price: float, be_config: BreakevenConfig) -> Position:
        if not be_config.enabled or pos.breakeven_active:
            return pos
            
        session = self.SessionLocal()
        try:
            pos = session.merge(pos)
            updated = False
            
            if pos.side == 'BUY':
                activation_price = pos.entry_price * (1 + be_config.activation_pct)
                if current_price >= activation_price:
                    new_sl = pos.entry_price * (1 + be_config.buffer_pct)
                    if new_sl > pos.stop_loss_price:
                        pos.stop_loss_price = new_sl
                        pos.breakeven_active = True
                        updated = True
                        
            elif pos.side == 'SELL':
                activation_price = pos.entry_price * (1 - be_config.activation_pct)
                if current_price <= activation_price:
                    new_sl = pos.entry_price * (1 - be_config.buffer_pct)
                    if new_sl < pos.stop_loss_price:
                        pos.stop_loss_price = new_sl
                        pos.breakeven_active = True
                        updated = True
            
            if updated:
                session.commit()
                session.refresh(pos)
                logger.info("Breakeven stop activated", symbol=pos.symbol, new_sl=pos.stop_loss_price)
            
            return pos
        except Exception as e:
            logger.error("Failed to manage breakeven stop", symbol=pos.symbol, error=str(e))
            session.rollback()
            return pos
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

    async def get_open_position(self, symbol: str) -> Optional[Position]:
        async with self._db_lock:
            return await self._run_in_executor(self._get_open_position_sync, symbol)

    def _get_open_position_sync(self, symbol: str) -> Optional[Position]:
        session = self.SessionLocal()
        try:
            return session.query(Position).filter(Position.symbol == symbol, Position.status == PositionStatus.OPEN).first()
        finally:
            session.close()

    async def get_all_open_positions(self) -> List[Position]:
        async with self._db_lock:
            return await self._run_in_executor(self._get_all_open_positions_sync)

    def _get_all_open_positions_sync(self) -> List[Position]:
        session = self.SessionLocal()
        try:
            return session.query(Position).filter(Position.status == PositionStatus.OPEN).all()
        finally:
            session.close()

    async def get_all_active_positions(self) -> List[Position]:
        async with self._db_lock:
            return await self._run_in_executor(self._get_all_active_positions_sync)

    def _get_all_active_positions_sync(self) -> List[Position]:
        session = self.SessionLocal()
        try:
            return session.query(Position).filter(
                Position.status.in_([PositionStatus.OPEN, PositionStatus.PENDING])
            ).all()
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

    async def get_aggregated_open_positions(self) -> Dict[str, float]:
        async with self._db_lock:
            return await self._run_in_executor(self._get_aggregated_open_positions_sync)

    def _get_aggregated_open_positions_sync(self) -> Dict[str, float]:
        session = self.SessionLocal()
        try:
            results = session.query(Position.symbol, func.sum(Position.quantity))\
                .filter(Position.status == PositionStatus.OPEN)\
                .group_by(Position.symbol).all()
            return {r[0]: r[1] for r in results}
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
