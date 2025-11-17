import asyncio
import time
import os
from datetime import datetime, timezone
from collections import deque
from typing import Dict, Any, Callable

import psutil

from bot_core.logger import get_logger

# Safe import for InfluxDB
try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
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
            logger.debug("Metric written to InfluxDB", measurement=measurement)
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
