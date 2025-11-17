import logging
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, inspect, Text, func
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError

from bot_core.order_manager import FillEvent

logger = logging.getLogger(__name__)

Base = declarative_base()

class Position(Base):
    __tablename__ = 'positions'

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False, index=True)
    side = Column(String, nullable=False) # 'BUY' or 'SELL'
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float)
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    status = Column(String, nullable=False, default='OPEN', index=True) # 'OPEN', 'CLOSED'
    open_time = Column(DateTime, default=datetime.utcnow)
    close_time = Column(DateTime)
    stop_loss = Column(Float)
    take_profit_levels = Column(Text) # Storing as JSON string

    def __repr__(self):
        return (
            f"<Position(id={self.id}, symbol='{self.symbol}', side='{self.side}', "
            f"quantity={self.quantity}, entry_price={self.entry_price}, status='{self.status}')>"
        )

class PositionManager:
    def __init__(self, db_path: str):
        self.engine = create_engine(f'sqlite:///{db_path}')
        self._init_db()
        self.Session = sessionmaker(bind=self.engine)
        logger.info(f"PositionManager initialized with DB: {db_path}")

    def _init_db(self):
        Base.metadata.create_all(self.engine)
        logger.info("Database tables checked and/or created.")

    def _get_session(self):
        return self.Session()

    def update_from_fill(self, fill: FillEvent):
        """The single entry point to update position state from a fill event."""
        with self._get_session() as session:
            try:
                position = session.query(Position).filter_by(symbol=fill.symbol, status='OPEN').first()

                if not position:
                    # This must be an opening fill
                    if fill.metadata.get("intent") == "OPEN":
                        self._create_new_position(session, fill)
                    else:
                        logger.warning(f"Received a fill for {fill.symbol} with no open position and no 'OPEN' intent.")
                else:
                    # Handle fill for an existing position
                    if position.side == fill.side.upper():
                        self._increase_position(session, position, fill)
                    else:
                        self._decrease_or_close_position(session, position, fill)
                
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Database error processing fill for {fill.symbol}: {e}", exc_info=True)
            except Exception as e:
                session.rollback()
                logger.error(f"Unexpected error processing fill for {fill.symbol}: {e}", exc_info=True)

    def _create_new_position(self, session, fill: FillEvent):
        """Creates and persists a new position from an opening fill."""
        position = Position(
            symbol=fill.symbol,
            side=fill.side.upper(),
            quantity=fill.quantity,
            entry_price=fill.price,
            current_price=fill.price,
            status='OPEN',
            stop_loss=fill.metadata.get('stop_loss'),
            take_profit_levels=json.dumps(fill.metadata.get('take_profit_levels', []))
        )
        session.add(position)
        logger.info(f"Opened new position from fill: {position}")

    def _increase_position(self, session, position: Position, fill: FillEvent):
        """Increases an existing position and recalculates the average entry price."""
        logger.info(f"Increasing position for {position.symbol} by {fill.quantity} at {fill.price}")
        total_cost_before = position.quantity * position.entry_price
        additional_cost = fill.quantity * fill.price
        new_total_quantity = position.quantity + fill.quantity

        position.entry_price = (total_cost_before + additional_cost) / new_total_quantity
        position.quantity = new_total_quantity
        # Update risk levels if provided in new fill metadata
        if 'stop_loss' in fill.metadata:
            position.stop_loss = fill.metadata['stop_loss']
        logger.info(f"Position increased. New avg price: {position.entry_price}, new quantity: {position.quantity}")

    def _decrease_or_close_position(self, session, position: Position, fill: FillEvent):
        """Decreases or fully closes an existing position."""
        close_quantity = fill.quantity
        if close_quantity >= position.quantity:
            # Full close
            logger.info(f"Closing full position for {position.symbol} ({position.quantity}) with fill of {close_quantity}")
            if position.side == 'BUY':
                position.realized_pnl = (fill.price - position.entry_price) * position.quantity
            else: # SELL
                position.realized_pnl = (position.entry_price - fill.price) * position.quantity
            position.close_time = datetime.utcnow()
            position.status = 'CLOSED'
            position.current_price = fill.price
            position.unrealized_pnl = 0.0
            logger.info(f"Closed position {position.id}. Realized PnL: {position.realized_pnl:.2f}")
        else:
            # Partial close
            logger.info(f"Partially closing position for {position.symbol} by {close_quantity}")
            realized_pnl_on_portion = 0
            if position.side == 'BUY':
                realized_pnl_on_portion = (fill.price - position.entry_price) * close_quantity
            else: # SELL
                real_pnl_on_portion = (position.entry_price - fill.price) * close_quantity
            
            position.quantity -= close_quantity
            position.realized_pnl += realized_pnl_on_portion
            logger.info(f"Partial close on {position.id}. Realized PnL: {realized_pnl_on_portion:.2f}. Remaining qty: {position.quantity}")

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        with self._get_session() as session:
            try:
                query = session.query(Position).filter_by(status='OPEN')
                if symbol:
                    query = query.filter_by(symbol=symbol)
                return query.all()
            except SQLAlchemyError as e:
                logger.error(f"Error getting open positions: {e}")
                return []

    def get_position_by_id(self, position_id: int) -> Optional[Position]:
        """Retrieves a single open position by its ID."""
        with self._get_session() as session:
            try:
                return session.query(Position).filter_by(id=position_id, status='OPEN').first()
            except SQLAlchemyError as e:
                logger.error(f"Error getting position by id {position_id}: {e}")
                return None

    def update_position_pnl(self, position_id: int, current_price: float) -> Optional[Position]:
        with self._get_session() as session:
            try:
                position = session.query(Position).filter_by(id=position_id, status='OPEN').first()
                if position:
                    position.current_price = current_price
                    if position.side == 'BUY':
                        position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    else: # SELL
                        position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
                    session.commit()
                    return position
                return None
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Error updating PnL for position {position_id}: {e}")
                return None

    def update_position_risk(self, position_id: int, new_stop_loss: float) -> Optional[Position]:
        with self._get_session() as session:
            try:
                position = session.query(Position).filter_by(id=position_id, status='OPEN').first()
                if position:
                    position.stop_loss = new_stop_loss
                    session.commit()
                    logger.info(f"Updated stop loss for position {position_id} to {new_stop_loss}")
                    return position
                return None
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Error updating risk for position {position_id}: {e}")
                return None

    def get_total_unrealized_pnl(self) -> float:
        with self._get_session() as session:
            try:
                total_pnl = session.query(func.sum(Position.unrealized_pnl)).filter_by(status='OPEN').scalar()
                return total_pnl or 0.0
            except SQLAlchemyError as e:
                logger.error(f"Error calculating total unrealized PnL: {e}")
                return 0.0

    def get_daily_realized_pnl(self, date: Optional[datetime] = None) -> float:
        with self._get_session() as session:
            try:
                if date is None:
                    date = datetime.utcnow()
                start_of_day = datetime(date.year, date.month, date.day)
                
                daily_pnl = session.query(func.sum(Position.realized_pnl)).filter(
                    Position.status == 'CLOSED',
                    Position.close_time >= start_of_day
                ).scalar()
                return daily_pnl or 0.0
            except SQLAlchemyError as e:
                logger.error(f"Error calculating daily realized PnL: {e}")
                return 0.0

    def close(self):
        """Gracefully disposes of the database engine connection pool."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection pool disposed.")
