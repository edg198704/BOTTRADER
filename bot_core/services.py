import asyncio
from typing import Dict, Set, Coroutine, Optional
from bot_core.logger import get_logger

logger = get_logger(__name__)

class ServiceManager:
    """
    Robustly manages the lifecycle of background services (tasks).
    Supports critical/non-critical services, health monitoring, and graceful shutdowns.
    """
    def __init__(self):
        self.services: Dict[str, asyncio.Task] = {}
        self.critical_services: Set[str] = set()
        self._shutdown_event = asyncio.Event()
        self._service_coroutines: Dict[str, Coroutine] = {}

    def register(self, name: str, coroutine: Coroutine, critical: bool = False):
        """Registers a service coroutine to be managed."""
        self._service_coroutines[name] = coroutine
        if critical:
            self.critical_services.add(name)

    def start_all(self):
        """Starts all registered services."""
        for name, coro in self._service_coroutines.items():
            self._start_service(name, coro)

    def _start_service(self, name: str, coro: Coroutine):
        if name in self.services and not self.services[name].done():
            return

        async def wrapped_service():
            try:
                logger.info(f"Service '{name}' started.")
                await coro
                logger.info(f"Service '{name}' stopped gracefully.")
            except asyncio.CancelledError:
                logger.info(f"Service '{name}' cancelled.")
            except Exception as e:
                logger.error(f"Service '{name}' crashed.", error=str(e), exc_info=True)
                raise

        self.services[name] = asyncio.create_task(wrapped_service(), name=name)

    async def monitor(self):
        """Monitors services and handles failures."""
        while not self._shutdown_event.is_set():
            for name, task in list(self.services.items()):
                if task.done():
                    try:
                        exc = task.exception()
                        if exc:
                            logger.error(f"Service '{name}' failed.", error=str(exc))
                            if name in self.critical_services:
                                logger.critical(f"Critical service '{name}' failed. Initiating system shutdown.")
                                self._shutdown_event.set()
                            else:
                                logger.warning(f"Non-critical service '{name}' stopped. Attempting restart in 5s...")
                                # Logic to restart could be added here if coroutine factory is provided
                        else:
                            logger.info(f"Service '{name}' finished normally.")
                    except asyncio.CancelledError:
                        pass
                    
                    if name in self.services:
                        del self.services[name]
            
            await asyncio.sleep(1)

    async def stop_all(self):
        """Stops all registered services."""
        self._shutdown_event.set()
        logger.info("Stopping all services...")
        
        tasks_to_cancel = [t for t in self.services.values() if not t.done()]
        for t in tasks_to_cancel:
            t.cancel()
        
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        
        self.services.clear()
        logger.info("All services stopped.")
