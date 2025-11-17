import logging
from datetime import datetime
from typing import Optional, List
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, inspect
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError

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
        inspector = inspect(self.engine)
        if not inspector.has_table('positions'):
            Base.metadata.create_all(self.engine)
            logger.info("Database table 'positions' created.")

    def _get_session(self):
        return self.Session()

    def add_position(self, symbol: str, side: str, quantity: float, entry_price: float) -> Optional[Position]:
        with self._get_session() as session:
            try:
                existing_position = session.query(Position).filter_by(symbol=symbol, status='OPEN').first()
                if existing_position:
                    logger.warning(f"Attempted to open a new position for {symbol} but one already exists.")
                    return None

                position = Position(
                    symbol=symbol,
                    side=side.upper(),
                    quantity=quantity,
                    entry_price=entry_price,
                    current_price=entry_price,
                    unrealized_pnl=0.0,
                    status='OPEN'
                )
                session.add(position)
                session.commit()
                session.refresh(position)
                logger.info(f"Added new position: {position}")
                return position
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Error adding position: {e}")
                return None

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

    def close_position(self, position_id: int, close_price: float) -> Optional[Position]:
        with self._get_session() as session:
            try:
                position = session.query(Position).filter_by(id=position_id, status='OPEN').first()
                if position:
                    if position.side == 'BUY':
                        position.realized_pnl = (close_price - position.entry_price) * position.quantity
                    else: # SELL
                        position.realized_pnl = (position.entry_price - close_price) * position.quantity
                    position.close_time = datetime.utcnow()
                    position.status = 'CLOSED'
                    position.current_price = close_price
                    position.unrealized_pnl = 0.0
                    session.commit()
                    logger.info(f"Closed position {position_id}. Realized PnL: {position.realized_pnl:.2f}")
                    return position
                return None
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Error closing position {position_id}: {e}")
                return None

    def get_total_unrealized_pnl(self) -> float:
        with self._get_session() as session:
            try:
                total_pnl = session.query(Float).filter(Position.status == 'OPEN').with_entities(Position.unrealized_pnl).sum()
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
                
                daily_pnl = session.query(Float).filter(
                    Position.status == 'CLOSED',
                    Position.close_time >= start_of_day
                ).with_entities(Position.realized_pnl).sum()
                return daily_pnl or 0.0
            except SQLAlchemyError as e:
                logger.error(f"Error calculating daily realized PnL: {e}")
                return 0.0
