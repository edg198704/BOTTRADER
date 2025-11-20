import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Callable, Awaitable, Type, Any, Union
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
    Supports both blocking (await) and non-blocking (fire-and-forget) publishing.
    """
    def __init__(self):
        self._subscribers: Dict[Type[Event], List[Callable[[Event], Awaitable[None]]]] = {}

    def subscribe(self, event_type: Type[Event], handler: Callable[[Event], Awaitable[None]]):
        """Registers an async handler for a specific event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed handler {handler.__name__} to {event_type.__name__}")

    async def publish(self, event: Event, wait: bool = False):
        """
        Publishes an event to all subscribers.
        
        Args:
            event: The event object to publish.
            wait: If True, awaits all handlers (blocking). 
                  If False, schedules handlers as background tasks (non-blocking).
        """
        event_type = type(event)
        if event_type in self._subscribers:
            handlers = self._subscribers[event_type]
            if not handlers:
                return

            if wait:
                # Execute all handlers concurrently and wait for completion
                await asyncio.gather(*[self._safe_execute(h, event) for h in handlers], return_exceptions=True)
            else:
                # Fire and forget: Schedule tasks on the loop
                for handler in handlers:
                    asyncio.create_task(self._safe_execute(handler, event))

    async def _safe_execute(self, handler: Callable[[Event], Awaitable[None]], event: Event):
        """Executes a handler with exception isolation."""
        try:
            await handler(event)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error in event handler {handler.__name__} for {type(event).__name__}", error=str(e), exc_info=True)
