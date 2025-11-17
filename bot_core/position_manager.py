from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, inspect
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime, timezone
from typing import Dict, Optional, List

from bot_core.logger import get_logger
from bot_core.config import DatabaseConfig

logger = get_logger(__name__)
Base = declarative_base()

class Position(Base):
    __tablename__ = 'positions'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False, index=True)
    side = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    entry_time = Column(DateTime, nullable=False)
    exit_price = Column(Float)
    exit_time = Column(DateTime)
    pnl = Column(Float)
    status = Column(String, default='OPEN', index=True)

class PositionManager:
    def __init__(self, config: DatabaseConfig):
        self.engine = create_engine(f'sqlite:///{config.path}')
        self.Session = sessionmaker(bind=self.engine)
        self._create_table_if_not_exists()
        self._open_positions: Dict[str, Position] = self._load_open_positions()
        logger.info("PositionManager initialized", db_path=config.path, open_positions=len(self._open_positions))

    def _create_table_if_not_exists(self):
        inspector = inspect(self.engine)
        if not inspector.has_table('positions'):
            Base.metadata.create_all(self.engine)
            logger.info("Created 'positions' table in the database.")

    def _load_open_positions(self) -> Dict[str, Position]:
        session = self.Session()
        try:
            open_positions = session.query(Position).filter_by(status='OPEN').all()
            return {p.symbol: p for p in open_positions}
        finally:
            session.close()

    def open_position(self, symbol: str, side: str, quantity: float, entry_price: float) -> Optional[Position]:
        if symbol in self._open_positions:
            logger.warning("Attempted to open an already open position", symbol=symbol)
            return None

        session = self.Session()
        try:
            position = Position(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                entry_time=datetime.now(timezone.utc)
            )
            session.add(position)
            session.commit()
            self._open_positions[symbol] = position
            logger.info("Position opened", symbol=symbol, side=side, quantity=quantity, price=entry_price)
            return position
        except SQLAlchemyError as e:
            logger.error("Failed to open position in DB", symbol=symbol, error=str(e))
            session.rollback()
            return None
        finally:
            session.close()

    def close_position(self, symbol: str, exit_price: float) -> Optional[Position]:
        if symbol not in self._open_positions:
            logger.warning("Attempted to close a non-existent position", symbol=symbol)
            return None

        session = self.Session()
        try:
            position = self._open_positions.pop(symbol)
            session.query(Position).filter_by(id=position.id).update({
                'status': 'CLOSED',
                'exit_price': exit_price,
                'exit_time': datetime.now(timezone.utc),
                'pnl': (exit_price - position.entry_price) * position.quantity if position.side == 'BUY' else (position.entry_price - exit_price) * position.quantity
            })
            session.commit()
            logger.info("Position closed", symbol=symbol, exit_price=exit_price, pnl=position.pnl)
            return position
        except SQLAlchemyError as e:
            logger.error("Failed to close position in DB", symbol=symbol, error=str(e))
            session.rollback()
            self._open_positions[symbol] = position # Re-add to in-memory cache on failure
            return None
        finally:
            session.close()

    def get_open_position(self, symbol: str) -> Optional[Position]:
        return self._open_positions.get(symbol)

    def get_all_open_positions(self) -> Dict[str, Position]:
        return self._open_positions.copy()

    def get_portfolio_value(self, latest_prices: Dict[str, float], initial_capital: float) -> float:
        # This is a simplified equity calculation. A real system would track cash balance.
        realized_pnl = self.get_total_realized_pnl()
        unrealized_pnl = 0.0
        for symbol, position in self._open_positions.items():
            current_price = latest_prices.get(symbol, position.entry_price)
            if position.side == 'BUY':
                unrealized_pnl += (current_price - position.entry_price) * position.quantity
            else:
                unrealized_pnl += (position.entry_price - current_price) * position.quantity
        return initial_capital + realized_pnl + unrealized_pnl

    def get_total_realized_pnl(self) -> float:
        session = self.Session()
        try:
            total_pnl = session.query(func.sum(Position.pnl)).filter(Position.status == 'CLOSED').scalar()
            return total_pnl or 0.0
        finally:
            session.close()

    def close(self):
        self.engine.dispose()
        logger.info("PositionManager database connection closed.")
