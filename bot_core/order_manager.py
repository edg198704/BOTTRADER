import asyncio
import logging
import random
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field
import time

from bot_core.exchange_api import ExchangeAPI

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"

@dataclass
class Order:
    client_order_id: str
    exchange_order_id: Optional[str] = None
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FillEvent:
    symbol: str
    side: str
    quantity: float
    price: float
    order_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class OrderManager:
    def __init__(self, exchange_api: ExchangeAPI):
        self.exchange_api = exchange_api
        self._orders: Dict[str, Order] = {}
        self._fill_queue = asyncio.Queue()
        self._tracking_task: Optional[asyncio.Task] = None
        self.running = False
        logger.info("OrderManager initialized.")

    async def start(self):
        """Starts the order tracking loop."""
        if self.running:
            logger.warning("OrderManager is already running.")
            return
        self.running = True
        self._tracking_task = asyncio.create_task(self._track_orders_loop())
        logger.info("OrderManager started.")

    async def stop(self):
        """Stops the order tracking loop."""
        self.running = False
        if self._tracking_task:
            self._tracking_task.cancel()
            try:
                await self._tracking_task
            except asyncio.CancelledError:
                pass
        logger.info("OrderManager stopped.")

    async def submit_order(self, symbol: str, side: str, order_type: str, quantity: float, price: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None) -> Order:
        """Submits an order to the exchange and starts tracking it."""
        client_order_id = f"bot_{int(time.time() * 1000)}_{random.randint(100,999)}"
        order = Order(
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            metadata=metadata or {}
        )
        self._orders[client_order_id] = order

        try:
            order_response = await self.exchange_api.place_order(symbol, side, order_type, quantity, price)
            
            if order_response and order_response.get('orderId'):
                order.exchange_order_id = order_response['orderId']
                order.status = OrderStatus.OPEN
                logger.info(f"Order submitted successfully: {order}")
            else:
                order.status = OrderStatus.REJECTED
                order.error_message = order_response.get('error', 'Submission failed, no order ID returned.')
                logger.error(f"Order submission failed: {order}")

        except Exception as e:
            order.status = OrderStatus.ERROR
            order.error_message = str(e)
            logger.error(f"Exception placing order {client_order_id}: {e}", exc_info=True)
        
        order.updated_at = time.time()
        return order

    async def get_fill_events(self) -> List[FillEvent]:
        """Retrieves all available fill events from the queue."""
        fills = []
        while not self._fill_queue.empty():
            fills.append(await self._fill_queue.get())
        return fills

    async def _track_orders_loop(self):
        """Periodically checks the status of open orders."""
        while self.running:
            try:
                await asyncio.sleep(10) # Check every 10 seconds
                open_orders = [o for o in self._orders.values() if o.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]]
                
                if not open_orders:
                    continue

                logger.debug(f"Tracking {len(open_orders)} open orders.")
                for order in open_orders:
                    await self._check_order_status(order)

            except asyncio.CancelledError:
                logger.info("Order tracking loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in order tracking loop: {e}", exc_info=True)

    async def _check_order_status(self, order: Order):
        """Fetches the latest status of a single order from the exchange."""
        try:
            if not order.exchange_order_id:
                logger.warning(f"Cannot track order {order.client_order_id} without exchange_order_id.")
                return

            status_response = await self.exchange_api.fetch_order(order.exchange_order_id, order.symbol)
            
            if not status_response:
                return

            new_status_str = status_response.get('status', 'ERROR').upper()
            new_status = OrderStatus[new_status_str] if new_status_str in OrderStatus.__members__ else OrderStatus.ERROR
            filled_qty = float(status_response.get('filled', 0.0))
            
            if new_status != order.status or filled_qty > order.filled_quantity:
                logger.info(f"Order {order.client_order_id} status changed: {order.status.value} -> {new_status.value}, Filled: {filled_qty}/{order.quantity}")
                
                fill_amount = filled_qty - order.filled_quantity
                if fill_amount > 0:
                    fill_price = float(status_response.get('average', order.price or 0.0))
                    await self._handle_fill(order, fill_amount, fill_price)

                order.status = new_status
                order.filled_quantity = filled_qty
                order.average_fill_price = float(status_response.get('average', order.average_fill_price))
                order.updated_at = time.time()

        except Exception as e:
            logger.error(f"Failed to check status for order {order.client_order_id}: {e}")
            order.status = OrderStatus.ERROR
            order.error_message = str(e)

    async def _handle_fill(self, order: Order, fill_quantity: float, fill_price: float):
        """Processes a fill and adds it to the queue."""
        fill = FillEvent(
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=fill_price,
            order_id=order.client_order_id,
            metadata=order.metadata
        )
        await self._fill_queue.put(fill)
        logger.info(f"Fill event generated: {fill}")
