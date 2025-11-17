import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict

from sqlalchemy import create_engine, Column, String, Float, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError

from bot_core.logger import get_logger
from bot_core.config import DatabaseConfig

logger = get_logger(__name__)
Base = declarative_base()

class Position(Base):
    __tablename__ = 'positions'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False) # 'BUY' or 'SELL'
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    entry_time = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    exit_price = Column(Float)
    exit_time = Column(DateTime)
    is_open = Column(Boolean, default=True, index=True)
    pnl = Column(Float, default=0.0)

class PositionManager:
    def __init__(self, config: DatabaseConfig):
        self.engine = create_engine(f'sqlite:///{config.path}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self._open_positions_cache: Dict[str, Position] = {}
        self._load_open_positions()
        logger.info("PositionManager initialized", db_path=config.path)

    def _load_open_positions(self):
        with self.Session() as session:
            open_positions = session.query(Position).filter_by(is_open=True).all()
            for pos in open_positions:
                self._open_positions_cache[pos.symbol] = pos
        logger.info("Loaded open positions from database", count=len(self._open_positions_cache))

    def open_position(self, symbol: str, side: str, quantity: float, entry_price: float) -> Optional[Position]:
        if symbol in self._open_positions_cache:
            logger.warning("Attempted to open a position that already exists", symbol=symbol)
            return None
        
        with self.Session() as session:
            try:
                new_position = Position(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    entry_price=entry_price
                )
                session.add(new_position)
                session.commit()
                session.refresh(new_position)
                self._open_positions_cache[symbol] = new_position
                logger.info("Position opened and saved", symbol=symbol, side=side, quantity=quantity, price=entry_price)
                return new_position
            except SQLAlchemyError as e:
                logger.error("Failed to open position in database", symbol=symbol, error=str(e))
                session.rollback()
                return None

    def close_position(self, symbol: str, exit_price: float) -> Optional[Position]:
        if symbol not in self._open_positions_cache:
            logger.warning("Attempted to close a non-existent position", symbol=symbol)
            return None

        with self.Session() as session:
            try:
                position = session.query(Position).filter_by(symbol=symbol, is_open=True).one_or_none()
                if not position:
                    # Already closed by another process, clean cache
                    del self._open_positions_cache[symbol]
                    return None

                if position.side.upper() == 'BUY':
                    pnl = (exit_price - position.entry_price) * position.quantity
                else:
                    pnl = (position.entry_price - exit_price) * position.quantity
                
                position.exit_price = exit_price
                position.exit_time = datetime.now(timezone.utc)
                position.is_open = False
                position.pnl = pnl
                
                session.commit()
                del self._open_positions_cache[symbol]
                logger.info("Position closed and saved", symbol=symbol, exit_price=exit_price, pnl=pnl)
                return position
            except SQLAlchemyError as e:
                logger.error("Failed to close position in database", symbol=symbol, error=str(e))
                session.rollback()
                return None

    def get_open_position(self, symbol: str) -> Optional[Position]:
        return self._open_positions_cache.get(symbol)

    def get_all_open_positions(self) -> List[Position]:
        return list(self._open_positions_cache.values())

    def get_portfolio_value(self, current_prices: Dict[str, float], initial_capital: float) -> float:
        cash = initial_capital # This is a simplification; a real system would track cash separately
        position_value = 0.0
        realized_pnl = 0.0

        with self.Session() as session:
            closed_positions = session.query(Position).filter_by(is_open=False).all()
            realized_pnl = sum(pos.pnl for pos in closed_positions)

        for pos in self.get_all_open_positions():
            current_price = current_prices.get(pos.symbol, pos.entry_price)
            if pos.side.upper() == 'BUY':
                position_value += pos.quantity * current_price
            else: # SELL
                position_value += (pos.entry_price - (current_price - pos.entry_price)) * pos.quantity
        
        return cash + realized_pnl + position_value - initial_capital # Simplified unrealized pnl calculation

    def close(self):
        self.engine.dispose()
        logger.info("PositionManager database connection closed.")
