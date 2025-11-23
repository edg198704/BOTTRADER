docker-compose exec trading_bot python -c "
import os

content = '''import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Callable, Awaitable, Type, Any, Union, Optional
from bot_core.logger import get_logger
from bot_core.common import TradeSignal
from bot_core.position_manager import Position

logger = get_logger(__name__)


class Event:
    \"\"\"Base event class without dataclass\"\"\"
    def __init__(self, timestamp: Optional[datetime] = None):
        self.timestamp = timestamp or datetime.now(timezone.utc)


class MarketDataEvent(Event):
    \"\"\"Market data event\"\"\"
    def __init__(self, symbol: str, data: Any, timestamp: Optional[datetime] = None):
        super().__init__(timestamp)
        self.symbol = symbol
        self.data = data


class SignalEvent(Event):
    \"\"\"Signal event\"\"\"
    def __init__(self, signal: TradeSignal, timestamp: Optional[datetime] = None):
        super().__init__(timestamp)
        self.signal = signal


class TradeCompletedEvent(Event):
    \"\"\"Trade completed event\"\"\"
    def __init__(self, position: Position, timestamp: Optional[datetime] = None):
        super().__init__(timestamp)
        self.position = position


class EventBus:
    \"\"\"
    Asynchronous Event Bus for decoupling system components.
    Supports both blocking (await) and non-blocking (fire-and-forget) publishing.
    \"\"\"
    def __init__(self):
        self._subscribers: Dict[Type[Event], List[Callable[[Event], Awaitable[None]]]] = {}
    
    def subscribe(self, event_type: Type[Event], handler: Callable[[Event], Awaitable[None]]):
        \"\"\"Registers an async handler for a specific event type.\"\"\"
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        logger.debug(f\"Subscribed handler {handler.__name__} to {event_type.__name__}\")
    
    async def publish(self, event: Event, wait: bool = False):
        \"\"\"
        Publishes an event to all subscribers.
        
        Args:
            event: The event object to publish.
            wait: If True, awaits all handlers (blocking). 
                  If False, schedules handlers as background tasks (non-blocking).
        \"\"\"
        event_type = type(event)
        if event_type in self._subscribers:
            handlers = self._subscribers[event_type]
            if not handlers:
                return
            if wait:
                await asyncio.gather(*[self._safe_execute(h, event) for h in handlers], return_exceptions=True)
            else:
                for handler in handlers:
                    asyncio.create_task(self._safe_execute(handler, event))
    
    async def _safe_execute(self, handler: Callable[[Event], Awaitable[None]], event: Event):
        \"\"\"Executes a handler with exception isolation.\"\"\"
        try:
            await handler(event)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f\"Error in event handler {handler.__name__} for {type(event).__name__}\", error=str(e), exc_info=True)
'''

with open('/app/bot_core/event_system.py', 'w') as f:
    f.write(content)

print('File updated successfully')
"
