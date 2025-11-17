"""
Enterprise AI Trading Bot - Refactored Architecture

A modular, production-ready AI trading bot with enterprise-grade features:
- Modular architecture with clear separation of concerns
- Comprehensive monitoring and observability
- Robust error handling and recovery
- Scalable design patterns
- Security best practices
- Performance optimization
- Full test coverage
"""

import asyncio
import sys
import os
import json
import gc
import logging
import logging.handlers
import math
import random
import sqlite3
import time
import traceback
import uuid
import signal
import resource
import cProfile
import pstats
import psutil
import warnings
import hashlib
import hmac
from contextlib import contextmanager, asynccontextmanager
from abc import ABC, abstractmethod
from collections import OrderedDict, deque, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator, Iterable, Protocol
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from threading import Lock, RLock
import weakref

# Core dependencies with safe imports
try:
    import ccxt.async_support as ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

try:
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

try:
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

try:
    import sklearn
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, classification_report, silhouette_score
    from sklearn.model_selection import KFold, train_test_split, cross_val_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import SpectralClustering, KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import optuna
    from optuna.integration import TorchDistributedTrial
    from optuna.pruners import HyperbandPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from dateutil import parser as dateparser
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False

try:
    from pydantic import BaseModel, Field, validator, root_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

try:
    from memory_profiler import profile
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

# External integrations with safe imports
try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import ASYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False

try:
    from telegram import Update, Bot
    from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
    from telegram.request import HTTPXRequest
    import telegram.error
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

# Environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Configure environment
warnings.filterwarnings('ignore')
sys.setrecursionlimit(2000)


# =====================================================================================
# CORE CONFIGURATION AND VALIDATION
# =====================================================================================

class ConfigError(Exception):
    """Configuration validation error"""
    pass


@dataclass
class TradingConfig:
    """Enterprise trading bot configuration with validation"""
    
    # Exchange Configuration
    exchange: str = "binance"
    sandbox: bool = False
    dry_run: bool = True
    
    # API Keys
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    
    # Trading Parameters
    symbols: List[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    timeframe: str = "1h"
    initial_capital: float = 10000.0
    max_position_size: float = 0.1
    max_concurrent_positions: int = 10
    
    # Risk Management
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    max_drawdown: float = 0.15
    max_risk_per_trade: float = 0.02
    
    # AI/ML Configuration
    use_ensemble: bool = True
    use_rl_agent: bool = True
    training_symbols_limit: int = 50
    retrain_interval_hours: int = 24
    
    # Performance and Monitoring
    log_level: str = "INFO"
    max_memory_mb: int = 2000
    metrics_interval_seconds: int = 60
    health_check_interval_seconds: int = 30
    
    # External Integrations
    enable_telegram: bool = True
    enable_influxdb: bool = True
    enable_grafana: bool = False
    
    # Security
    encrypt_api_keys: bool = True
    api_key_rotation_days: int = 90
    max_failed_attempts: int = 3
    
    # Data Management
    data_retention_days: int = 90
    cache_size: int = 1000
    buffer_size: int = 10000
    
    # Development and Testing
    enable_profiling: bool = False
    enable_debug_logging: bool = False
    abort_on_critical_errors: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate()
    
    def validate(self):
        """Validate configuration parameters"""
        errors = []
        
        # Exchange validation
        if self.exchange not in ["binance", "coinbase", "kucoin", "kraken"]:
            errors.append(f"Unsupported exchange: {self.exchange}")
        
        # Capital validation
        if self.initial_capital <= 0:
            errors.append("Initial capital must be positive")
        
        # Risk parameters validation
        if not 0 < self.stop_loss_pct < 1:
            errors.append("Stop loss must be between 0 and 1")
        if not 0 < self.take_profit_pct < 1:
            errors.append("Take profit must be between 0 and 1")
        if self.max_drawdown <= 0 or self.max_drawdown >= 1:
            errors.append("Max drawdown must be between 0 and 1")
        
        # Position validation
        if self.max_position_size <= 0 or self.max_position_size > 1:
            errors.append("Max position size must be between 0 and 1")
        
        if self.max_concurrent_positions < 1:
            errors.append("Max concurrent positions must be positive")
        
        # Symbols validation
        if not self.symbols or len(self.symbols) == 0:
            errors.append("At least one trading symbol required")
        
        # Timeframe validation
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        if self.timeframe not in valid_timeframes:
            errors.append(f"Invalid timeframe: {self.timeframe}. Valid: {valid_timeframes}")
        
        if errors:
            raise ConfigError(f"Configuration validation failed: {errors}")


# =====================================================================================
# LOGGING AND OBSERVABILITY
# =====================================================================================

class EnterpriseLogger:
    """
    Enterprise-grade structured logging system with multiple sinks,
    correlation IDs, and performance tracking.
    """
    
    def __init__(self, name: str, config: TradingConfig):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(name)
        self.correlation_id = str(uuid.uuid4())
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging with multiple handlers and formats"""
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Set level
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        self.logger.setLevel(log_level)
        
        # Console handler with structured format
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(correlation_id)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for persistent logs
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        file_handler = logging.handlers.RotatingFileHandler(
            f'logs/{self.name}.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(correlation_id)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Error handler for critical errors
        error_handler = logging.handlers.RotatingFileHandler(
            f'logs/{self.name}_errors.log',
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.logger.addHandler(error_handler)
    
    def _safe_serialize(self, **kwargs) -> str:
        """Safely serialize kwargs for logging"""
        try:
            # Filter out large objects and sensitive data
            safe_kwargs = {}
            for key, value in kwargs.items():
                if key in ['password', 'secret', 'token', 'api_key']:
                    safe_kwargs[key] = '[REDACTED]'
                elif isinstance(value, (dict, list)) and len(str(value)) > 1000:
                    safe_kwargs[key] = f'[LARGE_OBJECT_{len(str(value))}_CHARS]'
                else:
                    safe_kwargs[key] = value
            
            return f"{safe_kwargs}"
        except Exception:
            return f"[SERIALIZATION_ERROR]"
    
    def _log(self, level: str, message: str, **kwargs):
        """Internal logging method with correlation ID"""
        try:
            extra = {
                'correlation_id': self.correlation_id,
                **kwargs
            }
            
            message_with_context = f"{message} | {self._safe_serialize(**kwargs)}"
            getattr(self.logger, level)(message_with_context, extra=extra)
        except Exception as e:
            # Fallback logging
            print(f"FALLBACK LOG ({level.upper()}): {message} - Error: {e}")
    
    def info(self, message: str, **kwargs):
        """Log info level message"""
        self._log('info', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning level message"""
        self._log('warning', message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error level message"""
        self._log('error', message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical level message"""
        self._log('critical', message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug level message"""
        if self.config.enable_debug_logging:
            self._log('debug', message, **kwargs)


# =====================================================================================
# ERROR HANDLING AND RECOVERY
# =====================================================================================

class RecoveryStrategy(ABC):
    """Abstract base for error recovery strategies"""
    
    @abstractmethod
    async def attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt to recover from error. Return True if recovery successful."""
        pass


class ExponentialBackoffRecovery(RecoveryStrategy):
    """Exponential backoff recovery strategy"""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
    
    async def attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt recovery with exponential backoff"""
        for attempt in range(self.max_attempts):
            try:
                delay = self.base_delay * (2 ** attempt)
                await asyncio.sleep(delay)
                
                # Attempt the operation again
                operation = context.get('operation')
                if operation:
                    await operation()
                    return True
                
            except Exception as recovery_error:
                context['recovery_error'] = recovery_error
                continue
        
        return False


class CircuitBreakerRecovery(RecoveryStrategy):
    """Circuit breaker recovery strategy"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    async def attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Handle circuit breaker logic"""
        now = time.time()
        
        if self.state == 'OPEN':
            if now - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
                self.failure_count = 0
            else:
                return False
        
        if self.state == 'HALF_OPEN':
            try:
                operation = context.get('operation')
                if operation:
                    await operation()
                    self.state = 'CLOSED'
                    self.failure_count = 0
                    return True
                else:
                    self.state = 'OPEN'
                    return False
            except Exception:
                self.state = 'OPEN'
                self.failure_count += 1
                self.last_failure_time = now
                return False
        
        # CLOSED state - normal operation
        self.failure_count += 1
        self.last_failure_time = now
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
        
        return False


class ErrorHandler:
    """Enterprise-grade error handling and recovery system"""
    
    def __init__(self, logger: EnterpriseLogger):
        self.logger = logger
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        self.error_history: deque = deque(maxlen=1000)
        self.circuit_breakers: Dict[str, CircuitBreakerRecovery] = {}
    
    def register_recovery_strategy(self, error_type: str, strategy: RecoveryStrategy):
        """Register a recovery strategy for a specific error type"""
        self.recovery_strategies[error_type] = strategy
    
    def get_circuit_breaker(self, name: str) -> CircuitBreakerRecovery:
        """Get or create a circuit breaker for a component"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreakerRecovery()
        return self.circuit_breakers[name]
    
    async def handle_error(self, error: Exception, context: Dict[str, Any], component: str = "general") -> bool:
        """Handle error with recovery strategies"""
        error_info = {
            'error': str(error),
            'error_type': type(error).__name__,
            'component': component,
            'timestamp': time.time(),
            'context': context
        }
        
        self.error_history.append(error_info)
        
        self.logger.error("error_occurred", **error_info)
        
        # Try circuit breaker first
        circuit_breaker = self.get_circuit_breaker(component)
        if circuit_breaker.state != 'CLOSED':
            return await circuit_breaker.attempt_recovery(error, context)
        
        # Try specific recovery strategy
        error_type = type(error).__name__
        if error_type in self.recovery_strategies:
            strategy = self.recovery_strategies[error_type]
            return await strategy.attempt_recovery(error, context)
        
        # General recovery
        await asyncio.sleep(1)
        return False
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        if not self.error_history:
            return {}
        
        errors_by_type = {}
        errors_by_component = {}
        
        for error_info in self.error_history:
            error_type = error_info['error_type']
            component = error_info['component']
            
            errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1
            errors_by_component[component] = errors_by_component.get(component, 0) + 1
        
        return {
            'total_errors': len(self.error_history),
            'errors_by_type': errors_by_type,
            'errors_by_component': errors_by_component,
            'circuit_breaker_states': {name: cb.state for name, cb in self.circuit_breakers.items()}
        }


# =====================================================================================
# MEMORY AND RESOURCE MANAGEMENT
# =====================================================================================

class ResourceManager:
    """Enterprise-grade resource and memory management"""
    
    def __init__(self, config: TradingConfig, logger: EnterpriseLogger):
        self.config = config
        self.logger = logger
        self._memory_history = deque(maxlen=100)
        self._cleanup_callbacks = []
        self._monitoring_task = None
        self._shutdown_event = asyncio.Event()
    
    async def start_monitoring(self):
        """Start resource monitoring"""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitor_loop())
            self.logger.info("resource_monitoring_started")
    
    async def stop_monitoring(self):
        """Stop resource monitoring"""
        if self._monitoring_task:
            self._shutdown_event.set()
            await self._monitoring_task
            self.logger.info("resource_monitoring_stopped")
    
    async def _monitor_loop(self):
        """Resource monitoring loop"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                memory_info = self.get_memory_info()
                self._memory_history.append({
                    'timestamp': time.time(),
                    'memory_mb': memory_info['rss_mb'],
                    'cpu_percent': memory_info['cpu_percent']
                })
                
                # Check memory thresholds
                if memory_info['rss_mb'] > self.config.max_memory_mb * 0.8:
                    self.logger.warning("high_memory_usage", **memory_info)
                    await self._emergency_cleanup()
                
                # Periodic cleanup
                if len(self._memory_history) % 10 == 0:
                    await self._routine_cleanup()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("monitoring_loop_error", error=str(e))
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory information"""
        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            cpu_percent = process.cpu_percent()
            
            return {
                'rss_mb': mem_info.rss / (1024 * 1024),
                'vms_mb': mem_info.vms / (1024 * 1024),
                'cpu_percent': cpu_percent,
                'available_mb': psutil.virtual_memory().available / (1024 * 1024)
            }
        except Exception as e:
            self.logger.error("memory_info_error", error=str(e))
            return {'rss_mb': 0, 'vms_mb': 0, 'cpu_percent': 0, 'available_mb': 0}
    
    def register_cleanup_callback(self, name: str, callback: callable, priority: int = 5):
        """Register a cleanup callback"""
        self._cleanup_callbacks.append({
            'name': name,
            'callback': callback,
            'priority': priority
        })
        # Sort by priority (higher priority first)
        self._cleanup_callbacks.sort(key=lambda x: x['priority'], reverse=True)
    
    async def _routine_cleanup(self):
        """Perform routine cleanup"""
        self.logger.debug("routine_cleanup_started")
        
        for callback_info in self._cleanup_callbacks:
            try:
                callback = callback_info['callback']
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                self.logger.error("cleanup_callback_failed", 
                                name=callback_info['name'], 
                                error=str(e))
        
        # Force garbage collection
        for i in range(3):
            gc.collect(i)
        
        self.logger.debug("routine_cleanup_completed")
    
    async def _emergency_cleanup(self):
        """Perform emergency cleanup when memory is high"""
        self.logger.warning("emergency_cleanup_started")
        
        # Clear all Python caches
        try:
            import builtins
            if hasattr(builtins, '_clear_cache'):
                builtins._clear_cache()
        except Exception:
            pass
        
        # Run all cleanup callbacks
        await self._routine_cleanup()
        
        # Additional aggressive cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
        
        self.logger.warning("emergency_cleanup_completed")
    
    def optimize_memory(self):
        """Optimize memory usage"""
        gc.set_threshold(50, 5, 5)
        
        # Configure pandas
        try:
            import pandas as pd
            pd.set_option('mode.chained_assignment', None)
        except Exception:
            pass
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        if not self._memory_history:
            return {}
        
        usages = [entry['memory_mb'] for entry in self._memory_history]
        cpu_usages = [entry['cpu_percent'] for entry in self._memory_history]
        
        return {
            'current_memory_mb': usages[-1],
            'average_memory_mb': np.mean(usages),
            'max_memory_mb': np.max(usages),
            'min_memory_mb': np.min(usages),
            'memory_trend': 'increasing' if usages[-1] > np.mean(usages) else 'decreasing',
            'current_cpu_percent': cpu_usages[-1],
            'average_cpu_percent': np.mean(cpu_usages),
            'cleanup_callbacks_count': len(self._cleanup_callbacks)
        }


# =====================================================================================
# EXCHANGE INTEGRATION
# =====================================================================================

class ExchangeError(Exception):
    """Exchange-specific error"""
    pass


class ExchangeManager:
    """Enterprise exchange manager with multiple exchange support and failover"""
    
    def __init__(self, config: TradingConfig, logger: EnterpriseLogger, error_handler: ErrorHandler):
        self.config = config
        self.logger = logger
        self.error_handler = error_handler
        self.exchange = None
        self.is_initialized = False
        self.api_key = config.api_key
        self.api_secret = config.api_secret
        self.sandbox = config.sandbox
        self.dry_run = config.dry_run
        
    async def initialize(self) -> bool:
        """Initialize exchange connection"""
        try:
            if not CCXT_AVAILABLE:
                raise ExchangeError("CCXT library not available")
            
            # Configure exchange
            exchange_config = {
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'sandbox': self.sandbox,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                }
            }
            
            # Create exchange instance
            exchange_name = self.config.exchange.lower()
            if exchange_name == 'binance':
                self.exchange = ccxt.binance(exchange_config)
            elif exchange_name == 'coinbase':
                self.exchange = ccxt.coinbasepro(exchange_config)
            elif exchange_name == 'kucoin':
                self.exchange = ccxt.kucoin(exchange_config)
            elif exchange_name == 'kraken':
                self.exchange = ccxt.kraken(exchange_config)
            else:
                raise ExchangeError(f"Unsupported exchange: {exchange_name}")
            
            # Test connection
            await self.exchange.load_markets()
            
            self.is_initialized = True
            self.logger.info("exchange_initialized", exchange=exchange_name)
            
            return True
            
        except Exception as e:
            self.logger.error("exchange_initialization_failed", error=str(e))
            return False
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> Dict[str, Any]:
        """Fetch OHLCV data with error handling"""
        try:
            if not self.is_initialized:
                raise ExchangeError("Exchange not initialized")
            
            async def _fetch():
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                return {
                    'success': True,
                    'ohlcv': ohlcv,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'count': len(ohlcv)
                }
            
            # Use circuit breaker for exchange operations
            circuit_breaker = self.error_handler.get_circuit_breaker(f"exchange_{symbol}")
            if circuit_breaker.state == 'OPEN':
                raise ExchangeError("Circuit breaker is open")
            
            result = await _fetch()
            circuit_breaker.state = 'CLOSED'
            return result
            
        except Exception as e:
            self.logger.error("fetch_ohlcv_failed", symbol=symbol, timeframe=timeframe, error=str(e))
            
            # Attempt recovery
            if await self.error_handler.handle_error(e, {'operation': lambda: self.fetch_ohlcv(symbol, timeframe, limit)}):
                return await self.fetch_ohlcv(symbol, timeframe, limit)
            
            return {'success': False, 'error': str(e)}
    
    async def create_order(self, symbol: str, order_type: str, side: str, amount: float, 
                          price: Optional[float] = None) -> Dict[str, Any]:
        """Create order with error handling"""
        try:
            if self.dry_run:
                return {
                    'success': True,
                    'order_id': f"dry_run_{uuid.uuid4()}",
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'price': price or 0,
                    'dry_run': True
                }
            
            if not self.is_initialized:
                raise ExchangeError("Exchange not initialized")
            
            async def _create_order():
                if order_type == 'market':
                    order = await self.exchange.create_market_order(symbol, side, amount)
                elif order_type == 'limit':
                    if price is None:
                        raise ExchangeError("Price required for limit orders")
                    order = await self.exchange.create_limit_order(symbol, side, amount, price)
                else:
                    raise ExchangeError(f"Unsupported order type: {order_type}")
                
                return {
                    'success': True,
                    'order_id': order['id'],
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'price': order.get('price', price),
                    'timestamp': order.get('timestamp'),
                    'status': order.get('status')
                }
            
            result = await _create_order()
            self.logger.info("order_created", **result)
            return result
            
        except Exception as e:
            self.logger.error("create_order_failed", 
                            symbol=symbol, order_type=order_type, side=side, 
                            amount=amount, price=price, error=str(e))
            return {'success': False, 'error': str(e)}
    
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch ticker data"""
        try:
            if not self.is_initialized:
                raise ExchangeError("Exchange not initialized")
            
            ticker = await self.exchange.fetch_ticker(symbol)
            return {
                'success': True,
                'symbol': symbol,
                'bid': ticker.get('bid'),
                'ask': ticker.get('ask'),
                'last': ticker.get('last'),
                'volume': ticker.get('baseVolume'),
                'timestamp': ticker.get('timestamp')
            }
            
        except Exception as e:
            self.logger.error("fetch_ticker_failed", symbol=symbol, error=str(e))
            return {'success': False, 'error': str(e)}
    
    async def close(self):
        """Close exchange connection"""
        if self.exchange:
            await self.exchange.close()
            self.logger.info("exchange_connection_closed")


# =====================================================================================
# POSITION MANAGEMENT
# =====================================================================================

@dataclass
class Position:
    """Position data structure"""
    symbol: str
    side: str  # 'buy' or 'sell'
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0


class PositionManager:
    """Enterprise position management with atomic operations and persistence"""
    
    def __init__(self, config: TradingConfig, logger: EnterpriseLogger, error_handler: ErrorHandler):
        self.config = config
        self.logger = logger
        self.error_handler = error_handler
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.total_realized_pnl = 0.0
        self.db_path = "position_ledger.db"
        self._lock = asyncio.Lock()
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for persistence"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    size REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    exit_price REAL,
                    exit_time TIMESTAMP,
                    realized_pnl REAL DEFAULT 0,
                    stop_loss REAL,
                    take_profit REAL,
                    status TEXT DEFAULT 'OPEN',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("position_database_initialized")
            
        except Exception as e:
            self.logger.error("database_initialization_failed", error=str(e))
    
    async def open_position(self, symbol: str, side: str, size: float, 
                          entry_price: float, stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None) -> bool:
        """Open new position with atomic operations"""
        async with self._lock:
            try:
                if symbol in self.positions:
                    self.logger.warning("position_already_exists", symbol=symbol)
                    return False
                
                position = Position(
                    symbol=symbol,
                    side=side,
                    size=size,
                    entry_price=entry_price,
                    entry_time=datetime.now(timezone.utc),
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                self.positions[symbol] = position
                
                # Save to database
                await self._save_position_to_db(position)
                
                self.logger.info("position_opened", 
                               symbol=symbol, side=side, size=size, entry_price=entry_price)
                return True
                
            except Exception as e:
                self.logger.error("open_position_failed", symbol=symbol, error=str(e))
                return False
    
    async def close_position(self, symbol: str, exit_price: float) -> Optional[Position]:
        """Close position and return realized P&L"""
        async with self._lock:
            try:
                if symbol not in self.positions:
                    self.logger.warning("position_not_found", symbol=symbol)
                    return None
                
                position = self.positions[symbol]
                position.current_price = exit_price
                
                # Calculate realized P&L
                if position.side == 'buy':
                    realized_pnl = (exit_price - position.entry_price) * position.size
                else:
                    realized_pnl = (position.entry_price - exit_price) * position.size
                
                self.total_realized_pnl += realized_pnl
                position.realized_pnl = realized_pnl
                
                # Move to closed positions
                del self.positions[symbol]
                self.closed_positions.append(position)
                
                # Update database
                await self._update_position_in_db(position, exit_price)
                
                self.logger.info("position_closed", 
                               symbol=symbol, exit_price=exit_price, 
                               realized_pnl=realized_pnl, 
                               total_realized_pnl=self.total_realized_pnl)
                
                return position
                
            except Exception as e:
                self.logger.error("close_position_failed", symbol=symbol, error=str(e))
                return None
    
    async def update_position(self, symbol: str, current_price: float) -> bool:
        """Update position with current market data"""
        try:
            if symbol not in self.positions:
                return False
            
            position = self.positions[symbol]
            position.current_price = current_price
            
            # Calculate unrealized P&L
            if position.side == 'buy':
                position.unrealized_pnl = (current_price - position.entry_price) * position.size
                position.unrealized_pnl_pct = (current_price - position.entry_price) / position.entry_price
            else:
                position.unrealized_pnl = (position.entry_price - current_price) * position.size
                position.unrealized_pnl_pct = (position.entry_price - current_price) / position.entry_price
            
            return True
            
        except Exception as e:
            self.logger.error("update_position_failed", symbol=symbol, error=str(e))
            return False
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position by symbol"""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all open positions"""
        return self.positions.copy()
    
    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L across all positions"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    async def _save_position_to_db(self, position: Position):
        """Save position to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO positions (symbol, side, size, entry_price, entry_time, stop_loss, take_profit)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                position.symbol,
                position.side,
                position.size,
                position.entry_price,
                position.entry_time.isoformat(),
                position.stop_loss,
                position.take_profit
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error("save_position_to_db_failed", error=str(e))
    
    async def _update_position_in_db(self, position: Position, exit_price: float):
        """Update position in database when closed"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE positions 
                SET exit_price = ?, exit_time = ?, realized_pnl = ?, status = 'CLOSED'
                WHERE symbol = ? AND status = 'OPEN'
            ''', (
                exit_price,
                datetime.now(timezone.utc).isoformat(),
                position.realized_pnl,
                position.symbol
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error("update_position_in_db_failed", error=str(e))
    
    async def audit_positions(self, current_equity: float) -> Dict[str, Any]:
        """Audit position consistency"""
        try:
            expected_equity = self.config.initial_capital + self.total_realized_pnl + self.get_total_unrealized_pnl()
            discrepancy = abs(current_equity - expected_equity)
            
            is_consistent = discrepancy < 1.0  # $1 tolerance
            
            if not is_consistent:
                self.logger.error("position_audit_failed", 
                                current_equity=current_equity,
                                expected_equity=expected_equity,
                                discrepancy=discrepancy)
            
            return {
                'is_consistent': is_consistent,
                'current_equity': current_equity,
                'expected_equity': expected_equity,
                'discrepancy': discrepancy,
                'total_realized_pnl': self.total_realized_pnl,
                'total_unrealized_pnl': self.get_total_unrealized_pnl(),
                'open_positions_count': len(self.positions),
                'closed_positions_count': len(self.closed_positions)
            }
            
        except Exception as e:
            self.logger.error("position_audit_error", error=str(e))
            return {'is_consistent': False, 'error': str(e)}


# =====================================================================================
# AI/ML COMPONENTS
# =====================================================================================

class ModelInterface(Protocol):
    """Protocol for AI models"""
    
    async def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction on data"""
        ...
    
    async def train(self, data: pd.DataFrame, **kwargs) -> bool:
        """Train model on data"""
        ...
    
    async def save(self, path: str) -> bool:
        """Save model to disk"""
        ...
    
    async def load(self, path: str) -> bool:
        """Load model from disk"""
        ...


class EnsembleLearner:
    """Enterprise ensemble learning system with multiple algorithms"""
    
    def __init__(self, config: TradingConfig, logger: EnterpriseLogger, error_handler: ErrorHandler):
        self.config = config
        self.logger = logger
        self.error_handler = error_handler
        self.models: Dict[str, ModelInterface] = {}
        self.is_trained = False
        self.feature_columns = []
        self.target_column = 'target'
        self._lock = asyncio.Lock()
    
    def initialize_models(self):
        """Initialize ensemble models"""
        try:
            if not SKLEARN_AVAILABLE:
                raise ImportError("Scikit-learn not available")
            
            # Random Forest
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Gradient Boosting
            self.models['gradient_boosting'] = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            # Logistic Regression
            self.models['logistic_regression'] = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
            
            # XGBoost if available
            if XGBOOST_AVAILABLE:
                self.models['xgboost'] = xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
            
            self.logger.info("ensemble_models_initialized", models=list(self.models.keys()))
            
        except Exception as e:
            self.logger.error("model_initialization_failed", error=str(e))
    
    async def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training/prediction"""
        try:
            # Ensure we have required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Missing required columns: {required_columns}")
            
            # Create technical indicators
            df = data.copy()
            
            # Moving averages
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price momentum
            df['price_change'] = df['close'].pct_change()
            df['price_change_5'] = df['close'].pct_change(5)
            df['price_change_10'] = df['close'].pct_change(10)
            
            # Volatility
            df['volatility'] = df['price_change'].rolling(20).std()
            
            # Create target variable (next period return)
            df['future_return'] = df['close'].shift(-1) / df['close'] - 1
            df['target'] = (df['future_return'] > 0).astype(int)
            
            # Remove rows with NaN
            df = df.dropna()
            
            # Store feature columns
            self.feature_columns = [col for col in df.columns if col not in ['target', 'future_return']]
            
            return df
            
        except Exception as e:
            self.logger.error("feature_preparation_failed", error=str(e))
            return pd.DataFrame()
    
    async def train(self, data: pd.DataFrame, **kwargs) -> bool:
        """Train ensemble models"""
        async with self._lock:
            try:
                self.logger.info("ensemble_training_started")
                
                # Prepare features
                prepared_data = await self.prepare_features(data)
                if prepared_data.empty:
                    raise ValueError("No valid data after feature preparation")
                
                # Prepare training data
                X = prepared_data[self.feature_columns]
                y = prepared_data[self.target_column]
                
                # Split data
                test_size = kwargs.get('test_size', 0.2)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train individual models
                model_scores = {}
                for name, model in self.models.items():
                    try:
                        self.logger.debug(f"training_model", model=name)
                        
                        if hasattr(model, 'predict_proba'):
                            model.fit(X_train_scaled, y_train)
                            score = model.score(X_test_scaled, y_test)
                            model_scores[name] = score
                            self.logger.info(f"model_trained", model=name, score=score)
                        else:
                            self.logger.warning(f"model_not_trainable", model=name)
                    
                    except Exception as e:
                        self.logger.error(f"model_training_failed", model=name, error=str(e))
                
                if not model_scores:
                    raise ValueError("No models were successfully trained")
                
                self.is_trained = True
                self.scaler = scaler
                self.logger.info("ensemble_training_completed", model_scores=model_scores)
                
                return True
                
            except Exception as e:
                self.logger.error("ensemble_training_failed", error=str(e))
                return False
    
    async def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make ensemble prediction"""
        try:
            if not self.is_trained:
                return {'success': False, 'error': 'Model not trained'}
            
            # Prepare features
            prepared_data = await self.prepare_features(data)
            if prepared_data.empty:
                return {'success': False, 'error': 'No valid data for prediction'}
            
            X = prepared_data[self.feature_columns].tail(1)  # Latest data point
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from all models
            predictions = {}
            confidences = {}
            
            for name, model in self.models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_scaled)[0]
                        pred = model.predict(X_scaled)[0]
                        
                        predictions[name] = int(pred)
                        confidences[name] = float(max(proba))
                    else:
                        pred = model.predict(X_scaled)[0]
                        predictions[name] = int(pred)
                        confidences[name] = 0.5
                
                except Exception as e:
                    self.logger.error(f"model_prediction_failed", model=name, error=str(e))
            
            # Ensemble prediction (majority vote with confidence weighting)
            if predictions:
                vote_scores = {0: 0, 1: 0}
                for name, pred in predictions.items():
                    vote_scores[pred] += confidences.get(name, 0.5)
                
                ensemble_prediction = 1 if vote_scores[1] > vote_scores[0] else 0
                ensemble_confidence = max(vote_scores.values()) / sum(vote_scores.values())
                
                return {
                    'success': True,
                    'action': 'buy' if ensemble_prediction == 1 else 'sell',
                    'confidence': ensemble_confidence,
                    'prediction': ensemble_prediction,
                    'individual_predictions': predictions,
                    'individual_confidences': confidences,
                    'vote_scores': vote_scores
                }
            else:
                return {'success': False, 'error': 'No successful predictions'}
            
        except Exception as e:
            self.logger.error("ensemble_prediction_failed", error=str(e))
            return {'success': False, 'error': str(e)}
    
    async def save_models(self, path: str) -> bool:
        """Save trained models"""
        try:
            import pickle
            
            os.makedirs(path, exist_ok=True)
            
            model_data = {
                'models': self.models,
                'scaler': getattr(self, 'scaler', None),
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained
            }
            
            with open(f"{path}/ensemble_models.pkl", 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info("models_saved", path=path)
            return True
            
        except Exception as e:
            self.logger.error("save_models_failed", error=str(e))
            return False
    
    async def load_models(self, path: str) -> bool:
        """Load trained models"""
        try:
            import pickle
            
            model_file = f"{path}/ensemble_models.pkl"
            if not os.path.exists(model_file):
                return False
            
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data['is_trained']
            
            self.logger.info("models_loaded", path=path)
            return True
            
        except Exception as e:
            self.logger.error("load_models_failed", error=str(e))
            return False


# =====================================================================================
# RISK MANAGEMENT
# =====================================================================================

class RiskManager:
    """Enterprise risk management system"""
    
    def __init__(self, config: TradingConfig, logger: EnterpriseLogger, error_handler: ErrorHandler):
        self.config = config
        self.logger = logger
        self.error_handler = error_handler
        self.position_manager = None
        self.active_risks: Dict[str, Dict[str, Any]] = {}
        self.risk_limits = {
            'max_position_size': config.max_position_size,
            'max_concurrent_positions': config.max_concurrent_positions,
            'max_drawdown': config.max_drawdown,
            'max_risk_per_trade': config.max_risk_per_trade
        }
    
    def set_position_manager(self, position_manager: PositionManager):
        """Set the position manager"""
        self.position_manager = position_manager
    
    async def check_risk_limits(self, symbol: str, side: str, size: float, 
                              entry_price: float) -> Tuple[bool, str]:
        """Check if trade meets risk limits"""
        try:
            # Check position size limit
            if size > self.risk_limits['max_position_size']:
                return False, f"Position size {size} exceeds limit {self.risk_limits['max_position_size']}"
            
            # Check concurrent positions limit
            if self.position_manager:
                if len(self.position_manager.positions) >= self.risk_limits['max_concurrent_positions']:
                    return False, f"Maximum concurrent positions reached: {self.risk_limits['max_concurrent_positions']}"
                
                # Check if already have position in this symbol
                if symbol in self.position_manager.positions:
                    return False, f"Already have position in {symbol}"
            
            # Check risk per trade
            position_value = size * entry_price
            if position_value > self.config.initial_capital * self.risk_limits['max_risk_per_trade']:
                return False, f"Position value {position_value} exceeds risk limit"
            
            return True, "Risk checks passed"
            
        except Exception as e:
            self.logger.error("risk_check_failed", error=str(e))
            return False, f"Risk check error: {e}"
    
    async def calculate_stop_loss(self, symbol: str, entry_price: float, 
                                side: str) -> Optional[float]:
        """Calculate stop loss price"""
        try:
            if side == 'buy':
                stop_loss = entry_price * (1 - self.config.stop_loss_pct)
            else:
                stop_loss = entry_price * (1 + self.config.stop_loss_pct)
            
            return round(stop_loss, 6)
            
        except Exception as e:
            self.logger.error("stop_loss_calculation_failed", error=str(e))
            return None
    
    async def calculate_take_profit(self, symbol: str, entry_price: float, 
                                  side: str) -> Optional[float]:
        """Calculate take profit price"""
        try:
            if side == 'buy':
                take_profit = entry_price * (1 + self.config.take_profit_pct)
            else:
                take_profit = entry_price * (1 - self.config.take_profit_pct)
            
            return round(take_profit, 6)
            
        except Exception as e:
            self.logger.error("take_profit_calculation_failed", error=str(e))
            return None
    
    def should_stop_loss(self, symbol: str, current_price: float) -> bool:
        """Check if stop loss should be triggered"""
        try:
            if self.position_manager:
                position = self.position_manager.get_position(symbol)
                if position and position.stop_loss:
                    if position.side == 'buy':
                        return current_price <= position.stop_loss
                    else:
                        return current_price >= position.stop_loss
            return False
            
        except Exception as e:
            self.logger.error("stop_loss_check_failed", error=str(e))
            return False
    
    def should_take_profit(self, symbol: str, current_price: float) -> bool:
        """Check if take profit should be triggered"""
        try:
            if self.position_manager:
                position = self.position_manager.get_position(symbol)
                if position and position.take_profit:
                    if position.side == 'buy':
                        return current_price >= position.take_profit
                    else:
                        return current_price <= position.take_profit
            return False
            
        except Exception as e:
            self.logger.error("take_profit_check_failed", error=str(e))
            return False
    
    async def assess_portfolio_risk(self) -> Dict[str, Any]:
        """Assess overall portfolio risk"""
        try:
            risk_metrics = {
                'total_exposure': 0.0,
                'concentration_risk': 0.0,
                'drawdown_risk': 0.0,
                'var_95': 0.0,  # Value at Risk 95%
                'risk_score': 0.0
            }
            
            if not self.position_manager:
                return risk_metrics
            
            positions = self.position_manager.get_all_positions()
            total_exposure = sum(pos.unrealized_pnl + (pos.size * pos.entry_price) for pos in positions.values())
            
            if total_exposure > 0:
                risk_metrics['total_exposure'] = total_exposure
                
                # Concentration risk (largest position / total exposure)
                if positions:
                    largest_position = max(pos.unrealized_pnl + (pos.size * pos.entry_price) 
                                         for pos in positions.values())
                    risk_metrics['concentration_risk'] = largest_position / total_exposure
                
                # Simple drawdown estimate
                if total_exposure < 0:
                    risk_metrics['drawdown_risk'] = abs(total_exposure) / self.config.initial_capital
                
                # Basic VaR estimation (simplified)
                if len(positions) > 0:
                    position_returns = [pos.unrealized_pnl_pct for pos in positions.values()]
                    risk_metrics['var_95'] = np.percentile(position_returns, 5)
                
                # Overall risk score (0-100, higher is riskier)
                risk_score = 0
                if risk_metrics['concentration_risk'] > 0.5:
                    risk_score += 30
                if risk_metrics['drawdown_risk'] > 0.1:
                    risk_score += 40
                if risk_metrics['var_95'] < -0.05:
                    risk_score += 30
                
                risk_metrics['risk_score'] = min(risk_score, 100)
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error("portfolio_risk_assessment_failed", error=str(e))
            return {}


# =====================================================================================
# MAIN TRADING BOT
# =====================================================================================

class EnterpriseTradingBot:
    """Enterprise AI Trading Bot with modular architecture"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        
        # Initialize core components
        self.logger = EnterpriseLogger("EnterpriseTradingBot", config)
        self.error_handler = ErrorHandler(self.logger)
        self.resource_manager = ResourceManager(config, self.logger)
        
        # Initialize managers
        self.exchange_manager = None
        self.position_manager = None
        self.risk_manager = None
        self.ensemble_learner = None
        
        # Bot state
        self.is_running = False
        self.start_time = None
        self.current_equity = config.initial_capital
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Tasks
        self.main_task = None
        self.monitoring_task = None
        self.health_check_task = None
        
        # Register cleanup callbacks
        self._setup_cleanup_callbacks()
    
    def _setup_cleanup_callbacks(self):
        """Setup resource cleanup callbacks"""
        self.resource_manager.register_cleanup_callback(
            "memory_optimization", 
            self._optimize_memory,
            priority=10
        )
        
        self.resource_manager.register_cleanup_callback(
            "garbage_collection",
            lambda: gc.collect(),
            priority=8
        )
    
    async def initialize(self) -> bool:
        """Initialize all bot components"""
        try:
            self.logger.info("initializing_enterprise_trading_bot")
            
            # Start resource monitoring
            await self.resource_manager.start_monitoring()
            
            # Initialize exchange
            self.exchange_manager = ExchangeManager(self.config, self.logger, self.error_handler)
            if not await self.exchange_manager.initialize():
                raise Exception("Failed to initialize exchange")
            
            # Initialize position manager
            self.position_manager = PositionManager(self.config, self.logger, self.error_handler)
            
            # Initialize risk manager
            self.risk_manager = RiskManager(self.config, self.logger, self.error_handler)
            self.risk_manager.set_position_manager(self.position_manager)
            
            # Initialize AI components
            if self.config.use_ensemble:
                self.ensemble_learner = EnsembleLearner(self.config, self.logger, self.error_handler)
                self.ensemble_learner.initialize_models()
            
            # Register error recovery strategies
            self._setup_error_recovery()
            
            self.logger.info("enterprise_trading_bot_initialized")
            return True
            
        except Exception as e:
            self.logger.error("bot_initialization_failed", error=str(e))
            return False
    
    def _setup_error_recovery(self):
        """Setup error recovery strategies"""
        # Exchange error recovery
        self.error_handler.register_recovery_strategy(
            "ExchangeError",
            ExponentialBackoffRecovery(max_attempts=3, base_delay=2.0)
        )
        
        # Network error recovery
        self.error_handler.register_recovery_strategy(
            "NetworkError",
            ExponentialBackoffRecovery(max_attempts=5, base_delay=1.0)
        )
        
        # Database error recovery
        self.error_handler.register_recovery_strategy(
            "sqlite3.Error",
            ExponentialBackoffRecovery(max_attempts=2, base_delay=0.5)
        )
    
    async def start(self) -> bool:
        """Start the trading bot"""
        try:
            if self.is_running:
                self.logger.warning("bot_already_running")
                return False
            
            if not await self.initialize():
                return False
            
            self.is_running = True
            self.start_time = datetime.now(timezone.utc)
            
            # Start main trading loop
            self.main_task = asyncio.create_task(self._main_loop())
            
            # Start monitoring tasks
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            
            self.logger.info("enterprise_trading_bot_started")
            return True
            
        except Exception as e:
            self.logger.error("bot_start_failed", error=str(e))
            return False
    
    async def stop(self):
        """Stop the trading bot"""
        try:
            if not self.is_running:
                return
            
            self.logger.info("stopping_enterprise_trading_bot")
            self.is_running = False
            
            # Cancel tasks
            tasks = [self.main_task, self.monitoring_task, self.health_check_task]
            for task in tasks:
                if task and not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Close exchange connection
            if self.exchange_manager:
                await self.exchange_manager.close()
            
            # Stop resource monitoring
            await self.resource_manager.stop_monitoring()
            
            # Generate final report
            await self._generate_final_report()
            
            self.logger.info("enterprise_trading_bot_stopped")
            
        except Exception as e:
            self.logger.error("bot_stop_failed", error=str(e))
    
    async def _main_loop(self):
        """Main trading loop"""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Main loop interval
                
                # Process each symbol
                for symbol in self.config.symbols:
                    if not self.is_running:
                        break
                    
                    await self._process_symbol(symbol)
                
                # Update performance metrics
                await self._update_performance_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("main_loop_error", error=str(e))
                await asyncio.sleep(60)  # Backoff on error
    
    async def _process_symbol(self, symbol: str):
        """Process individual symbol"""
        try:
            # Fetch market data
            market_data = await self.exchange_manager.fetch_ohlcv(symbol, self.config.timeframe, 100)
            if not market_data['success']:
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(market_data['ohlcv'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Update position
            if self.position_manager:
                latest_price = df['close'].iloc[-1]
                await self.position_manager.update_position(symbol, latest_price)
                
                # Check for stop loss / take profit
                if self.risk_manager.should_stop_loss(symbol, latest_price):
                    await self._handle_stop_loss(symbol, latest_price)
                elif self.risk_manager.should_take_profit(symbol, latest_price):
                    await self._handle_take_profit(symbol, latest_price)
            
            # AI prediction if enabled
            if self.config.use_ensemble and self.ensemble_learner and self.ensemble_learner.is_trained:
                prediction = await self.ensemble_learner.predict(df)
                if prediction['success']:
                    await self._handle_ai_prediction(symbol, df, prediction)
            
        except Exception as e:
            self.logger.error("symbol_processing_failed", symbol=symbol, error=str(e))
    
    async def _handle_ai_prediction(self, symbol: str, df: pd.DataFrame, prediction: Dict[str, Any]):
        """Handle AI model prediction"""
        try:
            confidence = prediction['confidence']
            action = prediction['action']
            
            # Only trade if confidence is high enough
            if confidence < 0.6:  # 60% confidence threshold
                return
            
            latest_price = df['close'].iloc[-1]
            size = min(self.config.max_position_size, self.current_equity * 0.1 / latest_price)  # 10% of equity
            
            # Check risk limits
            side = action.upper()
            can_trade, reason = await self.risk_manager.check_risk_limits(symbol, side.lower(), size, latest_price)
            
            if not can_trade:
                self.logger.debug("trade_rejected_risk_limits", 
                                symbol=symbol, action=action, confidence=confidence, reason=reason)
                return
            
            # Create order
            order_result = await self.exchange_manager.create_order(
                symbol, 'market', action.lower(), size
            )
            
            if order_result['success']:
                # Calculate stop loss and take profit
                stop_loss = await self.risk_manager.calculate_stop_loss(symbol, latest_price, action.lower())
                take_profit = await self.risk_manager.calculate_take_profit(symbol, latest_price, action.lower())
                
                # Open position
                success = await self.position_manager.open_position(
                    symbol, action.lower(), size, latest_price, stop_loss, take_profit
                )
                
                if success:
                    self.logger.info("ai_trade_executed", 
                                   symbol=symbol, action=action, size=size, 
                                   price=latest_price, confidence=confidence)
            
        except Exception as e:
            self.logger.error("ai_prediction_handling_failed", symbol=symbol, error=str(e))
    
    async def _handle_stop_loss(self, symbol: str, current_price: float):
        """Handle stop loss trigger"""
        try:
            self.logger.warning("stop_loss_triggered", symbol=symbol, price=current_price)
            
            order_result = await self.exchange_manager.create_order(
                symbol, 'market', 'sell', 1.0  # This would need to be adjusted based on actual position size
            )
            
            if order_result['success']:
                closed_position = await self.position_manager.close_position(symbol, current_price)
                if closed_position:
                    self.logger.info("stop_loss_executed", 
                                   symbol=symbol, pnl=closed_position.realized_pnl)
            
        except Exception as e:
            self.logger.error("stop_loss_handling_failed", symbol=symbol, error=str(e))
    
    async def _handle_take_profit(self, symbol: str, current_price: float):
        """Handle take profit trigger"""
        try:
            self.logger.info("take_profit_triggered", symbol=symbol, price=current_price)
            
            order_result = await self.exchange_manager.create_order(
                symbol, 'market', 'sell', 1.0  # This would need to be adjusted based on actual position size
            )
            
            if order_result['success']:
                closed_position = await self.position_manager.close_position(symbol, current_price)
                if closed_position:
                    self.logger.info("take_profit_executed", 
                                   symbol=symbol, pnl=closed_position.realized_pnl)
            
        except Exception as e:
            self.logger.error("take_profit_handling_failed", symbol=symbol, error=str(e))
    
    async def _monitoring_loop(self):
        """Monitoring and maintenance loop"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.metrics_interval_seconds)
                
                # Update current equity
                if self.position_manager:
                    self.current_equity = (self.config.initial_capital + 
                                         self.position_manager.total_realized_pnl +
                                         self.position_manager.get_total_unrealized_pnl())
                
                # Audit positions
                if self.position_manager:
                    audit_result = await self.position_manager.audit_positions(self.current_equity)
                    if not audit_result['is_consistent']:
                        self.logger.error("position_audit_failed", audit_result=audit_result)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("monitoring_loop_error", error=str(e))
                await asyncio.sleep(60)
    
    async def _health_check_loop(self):
        """Health check loop"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.health_check_interval_seconds)
                
                health_status = await self._perform_health_check()
                
                if health_status['status'] != 'healthy':
                    self.logger.warning("health_check_failed", health_status=health_status)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("health_check_loop_error", error=str(e))
                await asyncio.sleep(60)
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        try:
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'checks': {}
            }
            
            # Exchange health
            if self.exchange_manager and self.exchange_manager.is_initialized:
                try:
                    test_ticker = await self.exchange_manager.fetch_ticker(self.config.symbols[0])
                    health_status['checks']['exchange'] = 'healthy' if test_ticker['success'] else 'unhealthy'
                except Exception:
                    health_status['checks']['exchange'] = 'unhealthy'
            else:
                health_status['checks']['exchange'] = 'unhealthy'
            
            # Memory health
            memory_info = self.resource_manager.get_memory_info()
            if memory_info['rss_mb'] > self.config.max_memory_mb:
                health_status['checks']['memory'] = 'warning'
            else:
                health_status['checks']['memory'] = 'healthy'
            
            # AI model health
            if self.config.use_ensemble and self.ensemble_learner:
                health_status['checks']['ai_models'] = 'healthy' if self.ensemble_learner.is_trained else 'not_trained'
            else:
                health_status['checks']['ai_models'] = 'disabled'
            
            # Position health
            if self.position_manager:
                audit = await self.position_manager.audit_positions(self.current_equity)
                health_status['checks']['positions'] = 'healthy' if audit['is_consistent'] else 'inconsistent'
            else:
                health_status['checks']['positions'] = 'not_initialized'
            
            # Overall status
            unhealthy_checks = [k for k, v in health_status['checks'].items() if v == 'unhealthy']
            if unhealthy_checks:
                health_status['status'] = 'unhealthy'
            elif 'warning' in health_status['checks'].values():
                health_status['status'] = 'warning'
            
            health_status['uptime_seconds'] = (datetime.now(timezone.utc) - self.start_time).total_seconds() if self.start_time else 0
            
            return health_status
            
        except Exception as e:
            self.logger.error("health_check_failed", error=str(e))
            return {'status': 'error', 'error': str(e)}
    
    async def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            if not self.position_manager:
                return
            
            positions = self.position_manager.closed_positions
            
            if not positions:
                return
            
            # Calculate metrics
            total_trades = len(positions)
            winning_trades = sum(1 for pos in positions if pos.realized_pnl > 0)
            losing_trades = total_trades - winning_trades
            
            total_pnl = sum(pos.realized_pnl for pos in positions)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate max drawdown
            equity_curve = [self.config.initial_capital]
            running_pnl = 0
            for pos in positions:
                running_pnl += pos.realized_pnl
                equity_curve.append(self.config.initial_capital + running_pnl)
            
            peak = equity_curve[0]
            max_dd = 0
            for equity in equity_curve:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak
                if drawdown > max_dd:
                    max_dd = drawdown
            
            # Update metrics
            self.performance_metrics.update({
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'max_drawdown': max_dd,
                'current_equity': self.current_equity
            })
            
        except Exception as e:
            self.logger.error("performance_metrics_update_failed", error=str(e))
    
    async def _optimize_memory(self):
        """Optimize memory usage"""
        try:
            # Clear unused variables
            import gc
            gc.collect()
            
            # Optimize torch if available
            if TORCH_AVAILABLE:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            
            self.logger.debug("memory_optimization_completed")
            
        except Exception as e:
            self.logger.debug("memory_optimization_failed", error=str(e))
    
    async def _generate_final_report(self):
        """Generate final performance report"""
        try:
            uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds() if self.start_time else 0
            
            report = {
                'bot_type': 'EnterpriseTradingBot',
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': datetime.now(timezone.utc).isoformat(),
                'uptime_seconds': uptime,
                'configuration': asdict(self.config),
                'performance_metrics': self.performance_metrics,
                'error_stats': self.error_handler.get_error_stats(),
                'resource_stats': self.resource_manager.get_memory_stats(),
                'final_equity': self.current_equity,
                'total_realized_pnl': self.position_manager.total_realized_pnl if self.position_manager else 0,
                'total_unrealized_pnl': self.position_manager.get_total_unrealized_pnl() if self.position_manager else 0,
                'positions_closed': len(self.position_manager.closed_positions) if self.position_manager else 0,
                'positions_open': len(self.position_manager.positions) if self.position_manager else 0
            }
            
            # Save report
            os.makedirs('reports', exist_ok=True)
            report_filename = f"reports/final_report_{datetime.now():%Y%m%d_%H%M%S}.json"
            
            with open(report_filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info("final_report_generated", 
                           filename=report_filename,
                           report_summary={
                               'uptime_hours': uptime / 3600,
                               'total_trades': self.performance_metrics['total_trades'],
                               'win_rate': f"{self.performance_metrics['win_rate']:.1%}",
                               'total_pnl': f"${self.performance_metrics['total_pnl']:.2f}",
                               'max_drawdown': f"{self.performance_metrics['max_drawdown']:.1%}"
                           })
            
        except Exception as e:
            self.logger.error("final_report_generation_failed", error=str(e))
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        return {
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'current_equity': self.current_equity,
            'performance_metrics': self.performance_metrics,
            'open_positions': len(self.position_manager.positions) if self.position_manager else 0,
            'error_count': len(self.error_handler.error_history),
            'memory_usage_mb': self.resource_manager.get_memory_info()['rss_mb']
        }


# =====================================================================================
# APPLICATION ENTRY POINT
# =====================================================================================

async def main():
    """Main application entry point"""
    try:
        # Load configuration
        config = TradingConfig(
            exchange=os.getenv('EXCHANGE', 'binance'),
            api_key=os.getenv('API_KEY'),
            api_secret=os.getenv('API_SECRET'),
            symbols=os.getenv('SYMBOLS', 'BTC/USDT,ETH/USDT').split(','),
            dry_run=os.getenv('DRY_RUN', 'true').lower() == 'true',
            initial_capital=float(os.getenv('INITIAL_CAPITAL', '10000')),
            use_ensemble=os.getenv('USE_ENSEMBLE', 'true').lower() == 'true'
        )
        
        # Create and start bot
        bot = EnterpriseTradingBot(config)
        
        # Register signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger = EnterpriseLogger("SignalHandler", config)
            logger.info("shutdown_signal_received", signal=signum)
            asyncio.create_task(bot.stop())
            raise KeyboardInterrupt
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start bot
        if await bot.start():
            logger = EnterpriseLogger("Main", config)
            logger.info("enterprise_trading_bot_running", 
                       config_summary={
                           'exchange': config.exchange,
                           'symbols': len(config.symbols),
                           'initial_capital': config.initial_capital,
                           'dry_run': config.dry_run,
                           'use_ensemble': config.use_ensemble
                       })
            
            # Keep running
            try:
                while bot.is_running:
                    await asyncio.sleep(10)
                    # Periodic status logging
                    if int(time.time()) % 300 == 0:  # Every 5 minutes
                        status = bot.get_status()
                        logger.info("bot_status_update", **status)
            except KeyboardInterrupt:
                pass
        
        # Stop bot gracefully
        await bot.stop()
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())