import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, func
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime, timezone
from typing import List, Optional, Dict

from bot_core.logger import get_logger
from bot_core.config import DatabaseConfig

logger = get_logger(__name__)
Base = declarative_base()

class Position(Base):
    __tablename__ = 'positions'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False) # 'BUY' or 'SELL'
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    entry_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    exit_price = Column(Float)
    exit_time = Column(DateTime)
    pnl = Column(Float, default=0.0)
    status = Column(String, default='OPEN') # 'OPEN' or 'CLOSED'

class PositionManager:
    """Manages the state of all trading positions using an SQLite database via SQLAlchemy."""
    def __init__(self, config: DatabaseConfig):
        self.engine = create_engine(f'sqlite:///{config.path}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        logger.info("PositionManager initialized", db_path=config.path)

    def open_position(self, symbol: str, side: str, quantity: float, entry_price: float):
        session = self.Session()
        try:
            existing_position = session.query(Position).filter_by(symbol=symbol, status='OPEN').first()
            if existing_position:
                logger.warning("Attempted to open an already open position", symbol=symbol)
                return

            new_position = Position(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price
            )
            session.add(new_position)
            session.commit()
            logger.info("Position opened", symbol=symbol, side=side, quantity=quantity, price=entry_price)
        except Exception as e:
            logger.error("Failed to open position", symbol=symbol, error=str(e))
            session.rollback()
        finally:
            session.close()

    def close_position(self, symbol: str, exit_price: float):
        session = self.Session()
        try:
            position = session.query(Position).filter_by(symbol=symbol, status='OPEN').first()
            if not position:
                logger.warning("Attempted to close a non-existent position", symbol=symbol)
                return

            if position.side.upper() == 'BUY':
                pnl = (exit_price - position.entry_price) * position.quantity
            else: # SELL
                pnl = (position.entry_price - exit_price) * position.quantity

            position.exit_price = exit_price
            position.exit_time = datetime.now(timezone.utc)
            position.pnl = pnl
            position.status = 'CLOSED'
            session.commit()
            logger.info("Position closed", symbol=symbol, exit_price=exit_price, pnl=pnl)
        except Exception as e:
            logger.error("Failed to close position", symbol=symbol, error=str(e))
            session.rollback()
        finally:
            session.close()

    def get_open_position(self, symbol: str) -> Optional[Position]:
        session = self.Session()
        try:
            return session.query(Position).filter_by(symbol=symbol, status='OPEN').first()
        finally:
            session.close()

    def get_all_open_positions(self) -> List[Position]:
        session = self.Session()
        try:
            return session.query(Position).filter_by(status='OPEN').all()
        finally:
            session.close()

    def get_portfolio_value(self, latest_prices: Dict[str, float], initial_capital: float) -> float:
        session = self.Session()
        try:
            # Calculate realized PnL from closed positions
            realized_pnl = session.query(func.sum(Position.pnl)).filter_by(status='CLOSED').scalar() or 0.0
            
            # Calculate unrealized PnL from open positions
            unrealized_pnl = 0.0
            open_positions = self.get_all_open_positions()
            for pos in open_positions:
                current_price = latest_prices.get(pos.symbol, pos.entry_price)
                if pos.side.upper() == 'BUY':
                    unrealized_pnl += (current_price - pos.entry_price) * pos.quantity
                else:
                    unrealized_pnl += (pos.entry_price - current_price) * pos.quantity

            return initial_capital + realized_pnl + unrealized_pnl
        finally:
            session.close()

    def close(self):
        self.engine.dispose()
        logger.info("PositionManager database connection closed.")
