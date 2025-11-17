import logging
import logging.handlers
import sys
import os
import uuid
from typing import Optional

# A context variable to hold the correlation ID for the current async task
try:
    from contextvars import ContextVar
    _correlation_id_ctx = ContextVar('correlation_id', default=None)
except ImportError:
    _correlation_id_ctx = None

class CorrelationIdFilter(logging.Filter):
    """Injects the correlation_id into the log record."""
    def filter(self, record):
        if _correlation_id_ctx:
            record.correlation_id = _correlation_id_ctx.get()
        else:
            record.correlation_id = 'N/A'
        return True

class JsonFormatter(logging.Formatter):
    """Formats log records as JSON strings."""
    def format(self, record):
        log_object = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, 'correlation_id', 'N/A'),
        }

        if hasattr(record, 'extra_info'):
            log_object.update(record.extra_info)

        if record.exc_info:
            log_object['exc_info'] = self.formatException(record.exc_info)

        try:
            import json
            return json.dumps(log_object)
        except (TypeError, ValueError):
            return str(log_object)

class StructuredLoggerAdapter(logging.LoggerAdapter):
    """An adapter to easily log with structured key-value pairs."""
    def process(self, msg, kwargs):
        extra = kwargs.pop('extra', {})
        extra_info = {}
        sensitive_keys = {'api_key', 'secret', 'password', 'token'}
        for key, value in kwargs.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                extra_info[key] = "[REDACTED]"
            else:
                extra_info[key] = value
        
        extra['extra_info'] = extra_info
        return msg, {'extra': extra}

def get_logger(name: str) -> StructuredLoggerAdapter:
    """Returns a configured logger adapter for structured logging."""
    logger = logging.getLogger(name)
    return StructuredLoggerAdapter(logger, {})

def setup_logging(level: str = "INFO", file_path: Optional[str] = None, use_json: bool = True):
    """Configures the root logger for the application."""
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level.upper())

    if use_json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s'
        )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(CorrelationIdFilter())
    root_logger.addHandler(console_handler)

    if file_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            file_path, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(CorrelationIdFilter())
        root_logger.addHandler(file_handler)

    if _correlation_id_ctx:
        _correlation_id_ctx.set(str(uuid.uuid4()))

    logging.info(f"Logging configured. Level: {level}, File: {file_path}, JSON: {use_json}")

def set_correlation_id(cid: Optional[str] = None):
    """Sets the correlation ID for the current context."""
    if _correlation_id_ctx:
        _correlation_id_ctx.set(cid or str(uuid.uuid4()))
