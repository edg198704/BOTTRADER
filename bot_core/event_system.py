import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Callable, Awaitable, Type, Any
from bot_core.logger import get_logger

logger = get_logger(__name__)

@dataclass
class Event:
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class MarketDataEvent(Event):
    symbol: str
    data: Any  # Carries the DataFrame or Tick data

@dataclass
class SignalEvent(Event):
    signal: Any  # Carries the TradeSignal object

class EventBus:
    """
    Asynchronous Event Bus for decoupling system components.
    """
    def __init__(self):
        self._subscribers: Dict[Type[Event], List[Callable[[Event], Awaitable[None]]]] = {}

    def subscribe(self, event_type: Type[Event], handler: Callable[[Event], Awaitable[None]]):
        """Registers an async handler for a specific event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed handler to {event_type.__name__}")

    async def publish(self, event: Event):
        """
        Publishes an event to all subscribers.
        Waits for all handlers to complete to ensure data consistency (Backpressure).
        """
        event_type = type(event)
        if event_type in self._subscribers:
            handlers = self._subscribers[event_type]
            if handlers:
                # Execute all handlers concurrently
                await asyncio.gather(*[h(event) for h in handlers], return_exceptions=True)
