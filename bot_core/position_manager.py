import datetime
from typing import List, Optional, Dict

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Enum as SQLAlchemyEnum, func, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base

from bot_core.logger import get_logger
from bot_core.config import DatabaseConfig

logger = get_logger(__name__)
Base = declarative_base()

class PositionStatus(SQLAlchemyEnum):
    OPEN = 'OPEN'
    CLOSED = 'CLOSED'

class Position(Base):
    __tablename__ = 'positions'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False) # 'BUY' or 'SELL'
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    status = Column(String, default='OPEN', nullable=False)
    stop_loss_price = Column(Float, nullable=True)
    take_profit_price = Column(Float, nullable=True)
    open_timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    close_timestamp = Column(DateTime, nullable=True)
    close_price = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    # trailing_ref_price tracks peak price for longs, and trough price for shorts
    trailing_ref_price = Column(Float, nullable=True)
    trailing_stop_active = Column(Boolean, default=False, nullable=False)

class PositionManager:
    def __init__(self, config: DatabaseConfig):
        self.engine = create_engine(f'sqlite:///{config.path}')
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        logger.info("PositionManager initialized and database table created.")

    def open_position(self, symbol: str, side: str, quantity: float, entry_price: float, stop_loss: float, take_profit: float) -> Optional[Position]:
        session = self.SessionLocal()
        try:
            if self.get_open_position(symbol):
                logger.warning("Attempted to open a position for a symbol that already has one.", symbol=symbol)
                return None

            new_position = Position(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                stop_loss_price=stop_loss,
                take_profit_price=take_profit,
                status='OPEN',
                trailing_ref_price=entry_price, # Initialize reference price to entry price
                trailing_stop_active=False
            )
            session.add(new_position)
            session.commit()
            session.refresh(new_position)
            logger.info("Opened new position", symbol=symbol, side=side, quantity=quantity, entry_price=entry_price, sl=stop_loss, tp=take_profit)
            return new_position
        except Exception as e:
            logger.error("Failed to open position", symbol=symbol, error=str(e))
            session.rollback()
            return None
        finally:
            session.close()

    def close_position(self, symbol: str, close_price: float, reason: str = "Unknown") -> Optional[Position]:
        session = self.SessionLocal()
        try:
            position = session.query(Position).filter(Position.symbol == symbol, Position.status == 'OPEN').first()
            if not position:
                logger.warning("No open position found to close for symbol", symbol=symbol)
                return None

            pnl = (close_price - position.entry_price) * position.quantity
            if position.side == 'SELL':
                pnl = -pnl

            position.status = 'CLOSED'
            position.close_price = close_price
            position.close_timestamp = datetime.datetime.utcnow()
            position.pnl = pnl
            session.commit()
            session.refresh(position)
            logger.info("Closed position", symbol=symbol, pnl=f"{pnl:.2f}", reason=reason)
            return position
        except Exception as e:
            logger.error("Failed to close position", symbol=symbol, error=str(e))
            session.rollback()
            return None
        finally:
            session.close()

    def update_trailing_stop(self, symbol: str, new_stop_loss: float, new_ref_price: float, is_active: bool) -> Optional[Position]:
        session = self.SessionLocal()
        try:
            position = session.query(Position).filter(Position.symbol == symbol, Position.status == 'OPEN').first()
            if not position:
                logger.warning("No open position found to update trailing stop for", symbol=symbol)
                return None

            position.stop_loss_price = new_stop_loss
            position.trailing_ref_price = new_ref_price
            position.trailing_stop_active = is_active
            
            session.commit()
            session.refresh(position)
            logger.debug("Updated trailing stop for position", symbol=symbol, new_sl=new_stop_loss, ref_price=new_ref_price)
            return position
        except Exception as e:
            logger.error("Failed to update trailing stop", symbol=symbol, error=str(e))
            session.rollback()
            return None
        finally:
            session.close()

    def get_open_position(self, symbol: str) -> Optional[Position]:
        session = self.SessionLocal()
        try:
            return session.query(Position).filter(Position.symbol == symbol, Position.status == 'OPEN').first()
        finally:
            session.close()

    def get_all_open_positions(self) -> List[Position]:
        session = self.SessionLocal()
        try:
            return session.query(Position).filter(Position.status == 'OPEN').all()
        finally:
            session.close()

    def get_portfolio_value(self, latest_prices: Dict[str, float], initial_capital: float, open_positions: List[Position]) -> float:
        session = self.SessionLocal()
        try:
            # Calculate realized PnL from closed positions
            closed_pnl_query = session.query(func.sum(Position.pnl)).filter(Position.status == 'CLOSED')
            realized_pnl = closed_pnl_query.scalar() or 0.0
            
            # Calculate unrealized PnL from open positions passed as argument
            unrealized_pnl = 0.0
            for pos in open_positions:
                current_price = latest_prices.get(pos.symbol, pos.entry_price)
                pnl = (current_price - pos.entry_price) * pos.quantity
                if pos.side == 'SELL':
                    pnl = -pnl
                unrealized_pnl += pnl

            return initial_capital + realized_pnl + unrealized_pnl
        finally:
            session.close()

    def close(self):
        self.engine.dispose()
        logger.info("PositionManager database connection closed.")
