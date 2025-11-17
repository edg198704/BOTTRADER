# bot_core/position_manager.py
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

Base = declarative_base()

class Position(Base):
    __tablename__ = 'positions'

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False) # 'BUY' or 'SELL'
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=True)
    unrealized_pnl = Column(Float, nullable=True)
    realized_pnl = Column(Float, nullable=True, default=0.0)
    status = Column(String, nullable=False, default='OPEN') # 'OPEN', 'CLOSED'
    open_time = Column(DateTime, default=datetime.utcnow)
    close_time = Column(DateTime, nullable=True)

    def __repr__(self):
        return (f"<Position(id={self.id}, symbol='{self.symbol}', side='{self.side}', "
                f"quantity={self.quantity}, entry_price={self.entry_price}, status='{self.status}')>")

class PositionManager:
    def __init__(self, db_path: str):
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        logger.info(f"PositionManager initialized with DB: {db_path}")

    def _get_session(self):
        return self.Session()

    def add_position(self, symbol: str, side: str, quantity: float, entry_price: float) -> Optional[Position]:
        session = self._get_session()
        try:
            position = Position(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                current_price=entry_price,
                unrealized_pnl=0.0,
                status='OPEN'
            )
            session.add(position)
            session.commit()
            logger.info(f"Added new position: {position}")
            return position
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error adding position: {e}")
            return None
        finally:
            session.close()

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        session = self._get_session()
        try:
            query = session.query(Position).filter_by(status='OPEN')
            if symbol:
                query = query.filter_by(symbol=symbol)
            return query.all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting open positions: {e}")
            return []
        finally:
            session.close()

    def update_position_pnl(self, position_id: int, current_price: float) -> Optional[Position]:
        session = self._get_session()
        try:
            position = session.query(Position).filter_by(id=position_id, status='OPEN').first()
            if position:
                position.current_price = current_price
                if position.side == 'BUY':
                    position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                else: # SELL
                    position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
                session.commit()
                logger.debug(f"Updated PnL for position {position_id}: {position.unrealized_pnl:.2f}")
                return position
            return None
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error updating PnL for position {position_id}: {e}")
            return None
        finally:
            session.close()

    def close_position(self, position_id: int, close_price: float) -> Optional[Position]:
        session = self._get_session()
        try:
            position = session.query(Position).filter_by(id=position_id, status='OPEN').first()
            if position:
                if position.side == 'BUY':
                    position.realized_pnl = (close_price - position.entry_price) * position.quantity
                else: # SELL
                    position.realized_pnl = (position.entry_price - close_price) * position.quantity
                position.close_time = datetime.utcnow()
                position.status = 'CLOSED'
                position.current_price = close_price # Final current price
                position.unrealized_pnl = 0.0 # Realized PnL takes over
                session.commit()
                logger.info(f"Closed position {position_id}. Realized PnL: {position.realized_pnl:.2f}")
                return position
            return None
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error closing position {position_id}: {e}")
            return None
        finally:
            session.close()

    def get_total_realized_pnl(self) -> float:
        session = self._get_session()
        try:
            total_pnl = session.query(
                (Position.realized_pnl)
            ).filter_by(status='CLOSED').sum()
            return total_pnl if total_pnl is not None else 0.0
        except SQLAlchemyError as e:
            logger.error(f"Error calculating total realized PnL: {e}")
            return 0.0
        finally:
            session.close()

    def get_daily_realized_pnl(self, date: Optional[datetime] = None) -> float:
        session = self._get_session()
        try:
            if date is None:
                date = datetime.utcnow()
            start_of_day = datetime(date.year, date.month, date.day)
            end_of_day = start_of_day.replace(hour=23, minute=59, second=59, microsecond=999999)

            daily_pnl = session.query(
                (Position.realized_pnl)
            ).filter(
                Position.status == 'CLOSED',
                Position.close_time >= start_of_day,
                Position.close_time <= end_of_day
            ).sum()
            return daily_pnl if daily_pnl is not None else 0.0
        except SQLAlchemyError as e:
            logger.error(f"Error calculating daily realized PnL: {e}")
            return 0.0
        finally:
            session.close()
