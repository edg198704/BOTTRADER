import sqlalchemy
from sqlalchemy import create_engine, Column, String, Float, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime, timezone
from typing import List, Optional, Dict

from bot_core.logger import get_logger

logger = get_logger(__name__)

Base = declarative_base()

class Position(Base):
    __tablename__ = 'positions'

    id = Column(String, primary_key=True)
    symbol = Column(String, nullable=False, index=True)
    side = Column(String, nullable=False) # 'BUY' or 'SELL'
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    pnl = Column(Float, default=0.0)
    is_open = Column(Boolean, default=True, index=True)
    opened_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    closed_at = Column(DateTime)

    def __repr__(self):
        return f"<Position(id='{self.id}', symbol='{self.symbol}', side='{self.side}', quantity={self.quantity})>"

class PositionManager:
    def __init__(self, db_path: str):
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        logger.info("PositionManager initialized", db_path=db_path)

    def _execute_in_session(self, func):
        session = self.Session()
        try:
            result = func(session)
            session.commit()
            return result
        except SQLAlchemyError as e:
            session.rollback()
            logger.error("Database transaction failed", error=str(e))
            raise
        finally:
            session.close()

    def open_position(self, symbol: str, side: str, quantity: float, entry_price: float) -> Position:
        pos_id = f"{symbol}-{int(datetime.now(timezone.utc).timestamp())}"
        new_position = Position(
            id=pos_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price
        )
        def _open(session):
            session.add(new_position)
            logger.info("Opening position", symbol=symbol, side=side, quantity=quantity, price=entry_price)
            return new_position
        return self._execute_in_session(_open)

    def close_position(self, symbol: str, exit_price: float) -> Optional[Position]:
        def _close(session):
            position = session.query(Position).filter_by(symbol=symbol, is_open=True).first()
            if not position:
                logger.warning("Attempted to close a non-existent open position", symbol=symbol)
                return None
            
            pnl_per_unit = exit_price - position.entry_price
            if position.side == 'SELL':
                pnl_per_unit = -pnl_per_unit
            
            position.exit_price = exit_price
            position.pnl = pnl_per_unit * position.quantity
            position.is_open = False
            position.closed_at = datetime.now(timezone.utc)
            logger.info("Closing position", symbol=symbol, exit_price=exit_price, pnl=position.pnl)
            return position
        return self._execute_in_session(_close)

    def get_open_position(self, symbol: str) -> Optional[Position]:
        def _get(session):
            return session.query(Position).filter_by(symbol=symbol, is_open=True).first()
        return self._execute_in_session(_get)

    def get_all_open_positions(self) -> List[Position]:
        def _get_all(session):
            return session.query(Position).filter_by(is_open=True).all()
        return self._execute_in_session(_get_all)

    def get_portfolio_value(self, latest_prices: Dict[str, float], initial_capital: float) -> float:
        def _calculate(session):
            closed_positions = session.query(Position).filter_by(is_open=False).all()
            realized_pnl = sum(pos.pnl for pos in closed_positions)

            open_positions = session.query(Position).filter_by(is_open=True).all()
            unrealized_pnl = 0.0
            for pos in open_positions:
                current_price = latest_prices.get(pos.symbol)
                if current_price:
                    pnl_per_unit = current_price - pos.entry_price
                    if pos.side == 'SELL':
                        pnl_per_unit = -pnl_per_unit
                    unrealized_pnl += pnl_per_unit * pos.quantity
            
            return initial_capital + realized_pnl + unrealized_pnl
        return self._execute_in_session(_calculate)

    def close(self):
        self.engine.dispose()
        logger.info("PositionManager database connection closed.")
