import datetime
import asyncio
from typing import List, Optional, Dict, TYPE_CHECKING

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Enum as SQLAlchemyEnum, func, Boolean, cast, Date
from sqlalchemy.orm import sessionmaker, declarative_base

from bot_core.logger import get_logger
from bot_core.config import DatabaseConfig, RiskManagementConfig

if TYPE_CHECKING:
    from bot_core.monitoring import AlertSystem

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
    def __init__(self, config: DatabaseConfig, initial_capital: float, alert_system: Optional['AlertSystem'] = None):
        self.engine = create_engine(f'sqlite:///{config.path}')
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._initial_capital = initial_capital
        self._realized_pnl = 0.0
        self.alert_system = alert_system
        logger.info("PositionManager initialized and database table created.")

    async def initialize(self):
        """Calculates initial realized PnL from the database to populate in-memory state."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._calculate_initial_realized_pnl)
        logger.info("PositionManager state initialized with historical PnL.", realized_pnl=self._realized_pnl)

    def _calculate_initial_realized_pnl(self):
        """Synchronous method to query the database for historical PnL."""
        session = self.SessionLocal()
        try:
            realized_pnl = session.query(func.sum(Position.pnl)).filter(Position.status == 'CLOSED').scalar()
            self._realized_pnl = realized_pnl or 0.0
        finally:
            session.close()

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
            
            if self.alert_system:
                asyncio.create_task(self.alert_system.send_alert(
                    level='info',
                    message=f"🟢 Opened {side} position for {symbol}",
                    details={'symbol': symbol, 'side': side, 'quantity': quantity, 'entry_price': entry_price}
                ))

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

            # Update in-memory realized PnL for high-performance portfolio tracking
            self._realized_pnl += pnl

            position.status = 'CLOSED'
            position.close_price = close_price
            position.close_timestamp = datetime.datetime.utcnow()
            position.pnl = pnl
            session.commit()
            session.refresh(position)
            logger.info("Closed position", symbol=symbol, pnl=f"{pnl:.2f}", reason=reason, total_realized_pnl=self._realized_pnl)

            if self.alert_system:
                pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                asyncio.create_task(self.alert_system.send_alert(
                    level='info',
                    message=f"{pnl_emoji} Closed position for {symbol} due to {reason}",
                    details={'symbol': symbol, 'pnl': f'{pnl:.2f}', 'reason': reason, 'close_price': close_price}
                ))

            return position
        except Exception as e:
            logger.error("Failed to close position", symbol=symbol, error=str(e))
            session.rollback()
            return None
        finally:
            session.close()

    def manage_trailing_stop(self, pos: Position, current_price: float, rm_config: RiskManagementConfig) -> Position:
        """
        Checks and updates the trailing stop for a position based on the current price.
        This method encapsulates the logic and database transaction.
        """
        session = self.SessionLocal()
        try:
            session.add(pos) # Re-attach the object to a new session
            original_stop_loss = pos.stop_loss_price

            if pos.side == 'BUY':
                # Activate TSL if price crosses activation threshold
                if not pos.trailing_stop_active:
                    activation_price = pos.entry_price * (1 + rm_config.trailing_stop_activation_pct)
                    if current_price >= activation_price:
                        pos.trailing_stop_active = True
                        logger.info("Trailing stop activated for LONG", symbol=pos.symbol, price=current_price, activation_price=activation_price)
                
                # Update peak price reference
                pos.trailing_ref_price = max(pos.trailing_ref_price, current_price)

                # If TSL is active, calculate new stop loss and update if it's higher
                if pos.trailing_stop_active:
                    new_stop_price = pos.trailing_ref_price * (1 - rm_config.trailing_stop_pct)
                    pos.stop_loss_price = max(pos.stop_loss_price, new_stop_price)

            elif pos.side == 'SELL':
                # Activate TSL
                if not pos.trailing_stop_active:
                    activation_price = pos.entry_price * (1 - rm_config.trailing_stop_activation_pct)
                    if current_price <= activation_price:
                        pos.trailing_stop_active = True
                        logger.info("Trailing stop activated for SHORT", symbol=pos.symbol, price=current_price, activation_price=activation_price)

                # Update trough price reference
                pos.trailing_ref_price = min(pos.trailing_ref_price, current_price)

                # If TSL is active, calculate new stop loss and update if it's lower
                if pos.trailing_stop_active:
                    new_stop_price = pos.trailing_ref_price * (1 + rm_config.trailing_stop_pct)
                    pos.stop_loss_price = min(pos.stop_loss_price, new_stop_price)

            if session.is_modified(pos):
                logger.debug("Updating trailing stop for position", symbol=pos.symbol, new_sl=pos.stop_loss_price, old_sl=original_stop_loss)
                session.commit()
                session.refresh(pos)
            
            return pos
        except Exception as e:
            logger.error("Failed to manage trailing stop", symbol=pos.symbol, error=str(e))
            session.rollback()
            return pos # Return original object on error
        finally:
            session.close()

    async def get_daily_realized_pnl(self) -> float:
        """Calculates the realized PnL for the current UTC day."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._calculate_daily_pnl_sync)

    def _calculate_daily_pnl_sync(self) -> float:
        """Synchronous method to query the database for today's PnL."""
        session = self.SessionLocal()
        try:
            today_utc = datetime.datetime.utcnow().date()
            daily_pnl = session.query(func.sum(Position.pnl)).filter(
                Position.status == 'CLOSED',
                cast(Position.close_timestamp, Date) == today_utc
            ).scalar()
            return daily_pnl or 0.0
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

    def get_portfolio_value(self, latest_prices: Dict[str, float], open_positions: List[Position]) -> float:
        """Calculates total portfolio equity using in-memory PnL and current market prices."""
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
