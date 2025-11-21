import asyncio
import time
import os
from datetime import datetime, timezone
from typing import Dict, Any, Callable, Optional, Awaitable
import psutil

from bot_core.logger import get_logger

# Safe import for InfluxDB
try:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import ASYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False

logger = get_logger(__name__)

class InfluxDBMetrics:
    """Handles writing performance and health metrics to InfluxDB."""
    def __init__(self, url: str, token: str, org: str, bucket: str):
        self.enabled = False
        if not INFLUXDB_AVAILABLE:
            logger.warning("InfluxDB client library not installed. Metrics disabled.")
            return
        if all([url, token, org, bucket]):
            try:
                self.client = InfluxDBClient(url=url, token=token, org=org)
                self.write_api = self.client.write_api(write_options=ASYNCHRONOUS)
                self.bucket = bucket
                self.enabled = True
                logger.info("InfluxDB metrics enabled.", url=url)
            except Exception as e:
                logger.error("InfluxDB initialization failed.", error=str(e))
        else:
            logger.info("InfluxDB credentials not fully provided. Metrics disabled.")

    async def write_metric(self, measurement: str, fields: Dict[str, Any], tags: Dict[str, str] = None):
        if not self.enabled:
            return
        try:
            point = Point(measurement).time(datetime.now(timezone.utc))
            if tags:
                for key, value in tags.items():
                    point.tag(key, value)
            for key, value in fields.items():
                point.field(key, value)
            
            self.write_api.write(bucket=self.bucket, record=point)
        except Exception as e:
            logger.error("Failed to write metric to InfluxDB", measurement=measurement, error=str(e))

    async def close(self):
        if self.enabled and self.client:
            self.write_api.close()
            self.client.close()
            logger.info("InfluxDB connection closed.")

class AlertSystem:
    """A simple system for dispatching alerts."""
    def __init__(self):
        self.handlers: list[Callable] = []

    def register_handler(self, handler: Callable):
        self.handlers.append(handler)

    async def send_alert(self, level: str, message: str, **kwargs):
        alert_data = {'level': level, 'message': message, 'details': kwargs}
        logger.info("Sending alert", **alert_data)
        for handler in self.handlers:
            try:
                await handler(alert_data)
            except Exception as e:
                logger.error("Alert handler failed", error=str(e))

class HealthChecker:
    """Monitors the bot's system health (CPU, memory)."""
    def __init__(self):
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())
        logger.info("HealthChecker initialized.")

    def get_health_status(self) -> Dict[str, Any]:
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        cpu_percent = self.process.cpu_percent(interval=0.1)
        uptime_seconds = time.time() - self.start_time

        status = "healthy"
        issues = []
        if memory_mb > 2000: # 2GB threshold
            status = "warning"
            issues.append(f"High memory usage: {memory_mb:.2f} MB")
        if cpu_percent > 90:
            status = "warning"
            issues.append(f"High CPU usage: {cpu_percent:.2f}%")

        return {
            'status': status,
            'memory_mb': memory_mb,
            'cpu_percent': cpu_percent,
            'uptime_seconds': uptime_seconds,
            'issues': issues
        }

class SystemCircuitBreaker:
    """Tracks global error rates and triggers a system halt if thresholds are exceeded."""
    def __init__(self, max_errors_per_minute: int = 10):
        self.max_errors = max_errors_per_minute
        self.error_timestamps = []
        self.tripped = False

    def record_error(self):
        now = time.time()
        self.error_timestamps.append(now)
        self._cleanup(now)
        if len(self.error_timestamps) > self.max_errors:
            self.tripped = True
            return True
        return False

    def _cleanup(self, now: float):
        # Remove errors older than 1 minute
        self.error_timestamps = [t for t in self.error_timestamps if now - t < 60]

class Watchdog:
    """
    Proactive system monitor. Tracks symbol heartbeats, system health, and manages the circuit breaker.
    """
    def __init__(self, 
                 symbols: list[str], 
                 alert_system: AlertSystem, 
                 stop_callback: Callable[[], Awaitable[None]],
                 health_checker: HealthChecker,
                 timeout_seconds: int = 300):
        self.symbols = symbols
        self.alert_system = alert_system
        self.stop_callback = stop_callback
        self.health_checker = health_checker
        self.timeout_seconds = timeout_seconds
        
        self._heartbeats: Dict[str, float] = {s: time.time() for s in symbols}
        self._circuit_breaker = SystemCircuitBreaker()
        self.running = False

    def register_heartbeat(self, symbol: str):
        self._heartbeats[symbol] = time.time()

    def record_error(self, source: str):
        if self._circuit_breaker.record_error():
            logger.critical("System Circuit Breaker Tripped! Too many errors.", source=source)
            asyncio.create_task(self._emergency_stop())

    async def _emergency_stop(self):
        await self.alert_system.send_alert("CRITICAL", "System Circuit Breaker Tripped. Shutting down.")
        await self.stop_callback()

    async def run(self):
        self.running = True
        logger.info("Watchdog service started.")
        while self.running:
            try:
                now = time.time()
                # 1. Check Heartbeats
                for symbol in self.symbols:
                    last_beat = self._heartbeats.get(symbol, 0)
                    if (now - last_beat) > self.timeout_seconds:
                        logger.warning("Watchdog Alert: Symbol stalled.", symbol=symbol, last_beat=last_beat)
                        await self.alert_system.send_alert('warning', f"Watchdog: No data for {symbol} in {self.timeout_seconds}s.")
                
                # 2. Check System Health
                health = self.health_checker.get_health_status()
                if health['status'] != 'healthy':
                    logger.warning("System Health Warning", issues=health['issues'])

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Watchdog loop error", error=str(e))
            
            await asyncio.sleep(60)

    async def stop(self):
        self.running = False
        logger.info("Watchdog service stopped.")
