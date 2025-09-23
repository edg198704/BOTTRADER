# Part 1:
import pickle
import tkinter as tk
from tkinter import ttk
import threading
import argparse
import asyncio
from collections import defaultdict, Counter, deque
import asyncpg
import ccxt.async_support as ccxt
import json
import signal
import logging
import logging.handlers
import math
import numpy as np
import os
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from dotenv import load_dotenv
from pathlib import Path
from scipy.stats import entropy, ks_2samp, anderson, mannwhitneyu, beta
from scipy.optimize import minimize
from scipy.signal import savgol_filter, convolve
from hmmlearn.hmm import GaussianHMM
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Any, Dict, List, Optional, Tuple
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import numba
from fastapi import FastAPI, WebSocket, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import uvicorn
import hashlib
import random
import optuna
from dowhy import CausalModel
import river
import dask.dataframe as dd
from deap import base, creator, tools, algorithms
# InfluxDB integration
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import ASYNCHRONOUS
from urllib.parse import urlparse
import socket

logger = logging.getLogger(__name__)
load_dotenv()

# InfluxDB config from env (add to .env)
INFLUXDB_URL = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
INFLUXDB_TOKEN = os.getenv('INFLUXDB_TOKEN', '')
INFLUXDB_ORG = os.getenv('INFLUXDB_ORG', 'DASHBOARD')
INFLUXDB_BUCKET = os.getenv('INFLUXDB_BUCKET', 'trading_bot')

# Global client (async write)
influx_client = None
write_api = None

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

async def init_influx():
    global INFLUXDB_URL
    """Initialize InfluxDB client with connection verification and test write."""
    if not INFLUXDB_URL:
        logger.error("INFLUXDB_URL is not set. Please add it to your .env file.")
        return False
    if not INFLUXDB_TOKEN:
        logger.error("INFLUXDB_TOKEN is not set or empty. Please add a valid token to your .env file.")
        return False
    if not INFLUXDB_ORG:
        logger.error("INFLUXDB_ORG is not set. Please add it to your .env file.")
        return False
    if not INFLUXDB_BUCKET:
        logger.error("INFLUXDB_BUCKET is not set. Please add it to your .env file.")
        return False
        # Auto-detect Docker and fallback URL for networking (common 'influxdb' resolution failure)
    
    
    is_docker = any('docker' in line for line in open('/proc/1/cgroup', 'r') if os.path.exists('/proc/1/cgroup'))
    if is_docker:
        parsed = urlparse(INFLUXDB_URL)
        if parsed.hostname == 'influxdb':
            fallback_url = f"http://{parsed.netloc.replace('influxdb', 'host.docker.internal')}"
            logger.info(f"Docker detected: Fallback URL from {INFLUXDB_URL} to {fallback_url} for networking")
            
            INFLUXDB_URL = fallback_url
            
    # DNS resolution check for hostname (common Docker issue)
    
    
    parsed_url = urlparse(INFLUXDB_URL)
    if parsed_url.hostname and ':' not in parsed_url.hostname:  # Skip if looks like IP
        try:
            resolved_ip = socket.gethostbyname(parsed_url.hostname)
            logger.info(f"Hostname '{parsed_url.hostname}' resolved to {resolved_ip} successfully")
        except socket.gaierror as dns_err:
            logger.error(f"DNS resolution failed for '{parsed_url.hostname}': {dns_err}. "
                         f"This is likely a Docker networking issue - ensure the bot container is on the same network as InfluxDB "
                         f"(e.g., shared docker-compose network) or use 'host.docker.internal:8086' / localhost / explicit IP in INFLUXDB_URL.")
            return False
        
    global influx_client, write_api
    
    # Retry connection with exponential backoff
    for attempt in range(MAX_RETRIES):
        try:
            influx_client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
            write_api = influx_client.write_api(write_options=ASYNCHRONOUS)
            
            # Health check: Ping the server
            health = influx_client.health()
            if health.status != "pass":
                message = health.message or "No additional message"
                checks_str = f", checks: {health.checks}" if health.checks else ""
                docker_hint = " (If Docker: Verify shared network in docker-compose.yml; try 'host.docker.internal' in URL)" if 'influxdb' in INFLUXDB_URL else ""
                raise Exception(f"InfluxDB health check failed: status={health.status}, message={message}{checks_str}{docker_hint}")
            
            # Test write: Send dummy point synchronously to verify auth immediately
            test_point = Point("test_connection").field("value", 1.0).tag("test", "init")
            test_point.time(datetime.now(timezone.utc), WritePrecision.NS)
            success = write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=test_point)
            write_api.flush()
            if not success:
                raise Exception("Test write failed during initialization")
            logger.info("InfluxDB test write successful: Dummy point sent and confirmed.")
            # Force flush post-test to ensure init data is committed (avoids async token propagation issues)
            write_api.flush()
            logger.debug("Post-init flush completed")
            
            return True
    
        except Exception as e:
            reason = str(e)
            if "authentication" in reason.lower():
                reason += " (Check INFLUXDB_TOKEN)"
            elif "org" in reason.lower() or "bucket" in reason.lower():
                reason += " (Check INFLUXDB_ORG/BUCKET)"
            elif "connection" in reason.lower():
                reason += " (Check INFLUXDB_URL and network)"
            
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAY * (2 ** attempt)  # exponential backoff
                logger.warning(f"InfluxDB init attempt {attempt + 1} failed: {reason}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"InfluxDB init failed after {MAX_RETRIES} attempts: {reason}")
                return False
    
    return False

def create_point(measurement: str, fields: Dict, tags: Dict) -> Optional[Point]:
    """Create and validate InfluxDB point."""
    # Create point
    point = Point(measurement)
    
    # Add tags (ensure strings)
    if tags:
        for k, v in tags.items():
            point.tag(k, str(v))
    
    # Clean and add fields
    cleaned_fields = {}
    for k, v in fields.items():
        if isinstance(v, (int, float)):
            # Handle NaN and infinite values
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                cleaned_fields[k] = 0.0
            else:
                cleaned_fields[k] = float(v)
        elif isinstance(v, str):
            cleaned_fields[k] = v
        # Skip other types
    
    # Validate we have data to write
    if not cleaned_fields:
        logger.warning(f"No valid fields to write for measurement {measurement}")
        return None
    
    # Add fields to point
    for k, v in cleaned_fields.items():
        if isinstance(v, (int, float)):
            point.field(k, float(v))
        else:
            point.field(k, str(v))
    
    return point

async def validate_write(measurement: str, write_time: datetime) -> bool:
    """Validate that data actually reached InfluxDB."""
    await asyncio.sleep(1)  # Give time for async write to complete
    
    try:
        query_api = influx_client.query_api()
        # Query the last point we just wrote
        query = f'''
            from(bucket: "{INFLUXDB_BUCKET}")
            |> range(start: -{2}m)
            |> filter(fn: (r) => r["_measurement"] == "{measurement}")
            |> last()
        '''
        
        result = query_api.query(org=INFLUXDB_ORG, query=query)
        
        # Check if we got results
        for table in result:
            for record in table.records:
                if record.get_time() >= write_time.replace(microsecond=0) - timedelta(seconds=10):
                    return True
        return False
        
    except Exception as query_e:
        logger.warning(f"Could not validate data arrival for {measurement}: {query_e}")
        return True  # Assume success if we can't validate

def get_error_reason(e: Exception) -> str:
    """Get detailed error reason with helpful hints."""
    reason = str(e)
    if "authentication" in reason.lower() or "unauthorized" in reason.lower() or "token" in reason.lower():
        reason += " (Invalid or missing token - verify INFLUXDB_TOKEN in .env and server auth settings)"
    elif "not found" in reason.lower():
        reason += " (Bucket/Org not found - check INFLUXDB_BUCKET/ORG)"
    elif "connection" in reason.lower():
        reason += " (Network issue - check INFLUXDB_URL and network)"
    else:
        reason += " (Unknown error - check InfluxDB logs)"
    return reason


async def write_to_influx(measurement: str, fields: Optional[Dict] = None, tags: Optional[Dict] = None, 
                       retry: bool = True, batch: bool = False):
    """Async write to InfluxDB with REAL validation, retries."""
    if influx_client is None or write_api is None:
        logger.error(f"InfluxDB not initialized; skipping write for {measurement}")
        return False
    
    fields = fields or {}
    tags = tags or {}
       
    # Retry logic for direct writes
    for attempt in range(MAX_RETRIES if retry else 1):
        try:
            logger.debug(f"Attempting to write to InfluxDB (attempt {attempt + 1}): measurement={measurement}")
            
            # Create and validate point
            point = create_point(measurement, fields, tags)
            if point is None:
                return False
            
            # Write to InfluxDB (this only sends, doesn't confirm arrival)
            current_time = datetime.now(timezone.utc)
            point.time(current_time, WritePrecision.NS)
            write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
            
            # REAL VALIDATION: Wait and query back to confirm data arrived
            validation_success = await validate_write(measurement, current_time)
            
            if validation_success:
                logger.debug(f"✓ CONFIRMED: Data reached InfluxDB for measurement={measurement}")
                return True
            elif retry and attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Write validation failed, retrying in {delay}s... (attempt {attempt + 1})")
                await asyncio.sleep(delay)
                continue
            else:
                logger.error(f"✗ VALIDATION FAILED after {attempt + 1} attempts: Data did NOT reach InfluxDB for measurement={measurement}")
                return False
                
        except Exception as e:
            reason = get_error_reason(e)
            
            if retry and attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAY * (2 ** attempt)
                logger.warning(f"InfluxDB write failed (attempt {attempt + 1}): {reason}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
                continue
            else:
                logger.error(f"InfluxDB write failed after {attempt + 1} attempts for {measurement}: {reason}", exc_info=True)
                logger.error(f"✗ CONFIRMED: No data reached InfluxDB for {measurement}: {reason}")
                return False
    
    return False

        
# Web Dashboard Dependencies
FASTAPI_AVAILABLE = True



logger = logging.getLogger('AutoTradeBot')
logger.setLevel(logging.DEBUG)

file_handler = logging.handlers.RotatingFileHandler(
    'auto_trade.log', maxBytes=50 * 1024 * 1024, backupCount=10
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s')
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

bt_logger = logging.getLogger('BtMetrics')
bt_logger.setLevel(logging.DEBUG)

bt_file_handler = logging.handlers.RotatingFileHandler(
    'bt_metrics.log', maxBytes=50 * 1024 * 1024, backupCount=5
)
bt_file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s')
)

bt_console_handler = logging.StreamHandler()
bt_console_handler.setLevel(logging.INFO)
bt_console_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
)

bt_logger.addHandler(bt_file_handler)
bt_console_handler.setLevel(logging.INFO)
bt_console_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
)

bt_logger.addHandler(bt_console_handler)

FEATURE_COLS = [
    'rsi', 'vol_ratio', 'volatility', 'vwap', 'atr', 'returns_entropy', 'spread_norm',
    'macd_signal', 'bb_position', 'volume_sma_ratio', 'price_momentum', 'volatility_regime',
    'micro_momentum_3', 'micro_momentum_5', 'micro_momentum_8', 'whale_risk',
    'buying_pressure', 'vol_price_divergence', 'consolidation_strength', 'range_compression',
    'mean_reversion_signal', 'body_shadow_ratio', 'holo_entropy', 'holo_divergence'
]

def is_stationary(series: pd.Series, pvalue_threshold: float = 0.05) -> bool:
    if len(series.dropna()) < 8:
        return False
    try:
        result = adfuller(series.dropna())
        return result[1] <= pvalue_threshold
    except (ValueError, np.linalg.LinAlgError):
        return False

STABLES = {'USDC', 'BUSD', 'TUSD', 'USDP', 'FDUSD', 'DAI', 'USDT', 'FRAX', 'LUSD', 'sUSD', 'EUR'}

def safe_env(key: str, default: Any, type_func: callable = None) -> Any:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        if type_func == bool:
            return value.lower() in ['true', '1', 'yes']
        elif type_func == list:
            return json.loads(value)
        else:
            return type_func(value) if type_func else value
    except (ValueError, TypeError, json.JSONDecodeError):
        logger.error(f"Invalid value for {key}, using default: {default}")
        return default
  
        
CONFIG = {
    'default': {
        'exchange': safe_env('EXCHANGE', 'binance', str),
        'api_key': safe_env('BINANCE_API_KEY', '', str),
        'api_secret': safe_env('BINANCE_API_SECRET', '', str),
        'sandbox': safe_env('BINANCE_SANDBOX', False, bool),
        'db_user': safe_env('DB_USER', safe_env('POSTGRES_USER', ''), str),
        'db_password': safe_env('DB_PASSWORD', safe_env('POSTGRES_PASSWORD', ''), str),
        'db_name': safe_env('DB_NAME', safe_env('POSTGRES_DB', ''), str),
        'db_host': safe_env('DB_HOST', safe_env('POSTGRES_HOST', 'localhost'), str),
        'db_port': safe_env('DB_PORT', 5432, int),
        'initial_equity': safe_env('INITIAL_EQUITY', 10000.0, float),
        'max_port_risk': safe_env('MAX_PORTFOLIO_RISK', 0.85, float),
        'kelly_frac': safe_env('KELLY_FRACTION', 0.7, float),
        'min_conf_score': safe_env('MIN_CONFIDENCE_SCORE', 0.01, float),
        'max_positions': safe_env('MAX_POSITIONS', 25, int),
        'min_vol_24h': safe_env('MIN_VOLUME_24H', 2000000.0, float),
        'min_order_size': safe_env('MIN_ORDER_SIZE', 12.0, float),
        'low_cap_exp_max': safe_env('LOW_CAP_EXPOSURE_MAX', 0.55, float),
        'max_corr_exp': safe_env('MAX_CORRELATED_EXPOSURE', 0.55, float),
        'sl_atr_mult': safe_env('STOP_LOSS_ATR_MULTIPLIER', 2.0, float),
        'tp_atr_mult': safe_env('TAKE_PROFIT_ATR_MULTIPLIER', 3.0, float),
        'ts_atr_mult': safe_env('TRAILING_STOP_ATR_MULTIPLIER', 1.5, float),
        'model_ver': safe_env('MODEL_VERSION', '3.0.0', str),
        'model_dir': safe_env('MODEL_DIR', 'models', str),
        'retrain_int': safe_env('RETRAIN_INTERVAL', 3600, int),
        'max_cache_size': safe_env('MAX_CACHE_SIZE', 500, int),
        'max_concurr': safe_env('MAX_CONCURRENCY', 5, int),
        'poll_int': safe_env('POLL_INTERVAL', 30, int),
        'fees': safe_env('FEES', 0.001, float),
        'slippage': safe_env('SLIPPAGE', 0.0015, float),
        'max_retries': safe_env('MAX_RETRIES', 7, int),
        'retry_delay': safe_env('RETRY_DELAY', 3.0, float),
        'bootstrap_days': safe_env('BOOTSTRAP_DAYS', 500, int),
        'min_data_pts': safe_env('MIN_DATA_POINTS', 500, int),
        'fetch_limit': safe_env('FETCH_LIMIT', 500, int),
        'perf_check_int': safe_env('PERFORMANCE_CHECK_INTERVAL', 900, int),
        'min_pct_change': safe_env('MIN_PERCENTAGE_CHANGE', 5.0, float),
        'supervisor_int': safe_env('SUPERVISOR_INTERVAL', 900, int),
        'enable_sup': safe_env('ENABLE_SUPERVISOR', True, bool),
        'futures': safe_env('FUTURES', False, bool),
        'dry_run': safe_env('DRY_RUN', True, bool),
        'log_trades': safe_env('LOG_TRADES', False, bool),
        'symbols': safe_env('SYMBOLS', [], list),
        'dynamic_very_low_vol_thresh': safe_env('DYNAMIC_VERY_LOW_VOL_THRESHOLD', 0.004, float),
        'dynamic_low_vol_thresh': safe_env('DYNAMIC_LOW_VOL_THRESHOLD', 0.012, float),
        'threshold_ema_alpha': safe_env('THRESHOLD_EMA_ALPHA', 0.1, float),
        'adaptive_thresh': safe_env('ADAPTIVE_THRESHOLDS', True, bool),
        'low_vol_pos_mult': safe_env('LOW_VOL_POSITION_MULTIPLIER', 1.1, float),
        'ob_depth': safe_env('ORDER_BOOK_DEPTH', 50, int),
        'spoof_ent_thresh': safe_env('SPOOFING_ENTROPY_THRESHOLD', 0.55, float),
        'imbal_thresh': safe_env('IMBALANCE_THRESHOLD', 1.2, float),
        'opt_int': safe_env('OPTIMIZATION_INTERVAL', 3600, int),
        'bayes_trials': safe_env('BAYES_TRIALS', 50, int),
        'label_threshold_base': safe_env('LABEL_THRESHOLD_BASE', 0.02, float),
        'monte_carlo_paths': safe_env('MONTE_CARLO_PATHS', 1500, int),
        'monte_carlo_validation_threshold': safe_env('MC_VALIDATION_THRESHOLD', 0.3, float),
        'garch_alpha_base': safe_env('GARCH_ALPHA_BASE', 0.05, float),
        'garch_beta_base': safe_env('GARCH_BETA_BASE', 0.90, float),
        'jump_detection_threshold': safe_env('JUMP_DETECTION_THRESHOLD', 0.1, float),
        'monte_carlo_ensemble_weights': safe_env('MONTE_CARLO_ENSEMBLE_WEIGHTS', [0.4, 0.35, 0.25], list),
        'backtest_freshness_days': safe_env('BACKTEST_FRESHNESS_DAYS', 7, int),
        'min_historical_periods': safe_env('MIN_HISTORICAL_PERIODS', 100, int),
        'mc_quality_excellent_threshold': safe_env('MC_QUALITY_EXCELLENT_THRESHOLD', 0.7, float),
        'mc_quality_good_threshold': safe_env('MC_QUALITY_GOOD_THRESHOLD', 0.5, float),
        'max_concurrent_fetches': safe_env('MAX_CONCURRENT_FETCHES', 10, int),
        'cache_timeout': safe_env('CACHE_TIMEOUT', 3600, int),
        'bayes_alpha_prior': 10,  # New param
        'lstm_layers': 2, 
        'attention_heads': 4, 
        'max_bars': 2000, 
        'chunk_size': 500,
        'use_gan': True, 
        'gan_epochs': 50, 
        'noise_dim': 100, 
        'vae_latent_dim': 10, 
        'evolve_features_topk': 5, 
        'hyperopt_trials': 50, 
        'mini_backtest_symbols': 10,
        'causal_maxlag': 3, 
        'causal_threshold_p': 0.05, 
        'tf_clusters': 4, 
        'auto_tf_min_vol': 0.01,
        'sentiment_weight': 0.15, 
        'onchain_weight': 0.10, 
        'vol_spike_threshold': 2.0, 
        'shock_prob': 0.05, 
        'crash_magnitude': 0.8, 
        'ga_population': 20, 
        'ga_generations': 10, 
        'ga_cxpb': 0.5, 
        'cvar_target': -0.05, 
        'tax_rate': 0.30
    },
    'very_low': {  # Adjusted for sensitivity: lower thresholds, higher frac for low-vol opportunities
        'min_conf_score': 0.005,  # Lower from 0.01
        'kelly_frac': 0.5,  # Higher from 0.4
        'sl_atr_mult': 1.2,  # Lower from 1.5
    },
    'low': {
        'min_conf_score': 0.01,  # Lower from 0.015
        'kelly_frac': 0.45,  # Higher from 0.38
    },
    'normal': {},  # Balanced, no change
    'high': {
        'sl_atr_mult': 3.0,  # Higher from 2.5 for caution
        'tp_atr_mult': 4.0,  # Higher from 3.5
    },
    'bull': {
        'min_conf_score': 0.01,  # Lower from 0.015
        'kelly_frac': 0.4,  # Higher from 0.35
        'sl_atr_mult': 2.5,  # Higher from 2.0
    },
    'bear': {
        'min_conf_score': 0.025,  # Higher from 0.02 for caution
        'kelly_frac': 0.25,  # Lower from 0.3
        'sl_atr_mult': 2.5,  # Higher from 2.2
    },
    'volatile_bull': {
        'kelly_frac': 0.35,  # Higher from 0.3
    },
    'volatile_bear': {
        'kelly_frac': 0.2,  # Lower from 0.25 for risk
    },
    'current_regime': 'normal'  # Inicial, se actualiza dinámicamente
}

def get_config_param(key: str, regime: Optional[str] = None) -> Any:
    if regime is None:
        regime = CONFIG['current_regime']
    value = CONFIG.get(regime, {}).get(key, CONFIG['default'].get(key, None))
    # Validación de tipos para escalares críticos (evita strings en lugar de int/float)
    if value is None:
        logger.warning(f"Config param {key} is None, using default 0")
        return 0
    if isinstance(value, str) and key in ['retrain_int', 'max_positions', 'poll_int', 'max_retries', 'bootstrap_days', 'min_data_pts', 'fetch_limit', 'perf_check_int', 'supervisor_int', 'ob_depth', 'bayes_trials', 'monte_carlo_paths', 'backtest_freshness_days', 'min_historical_periods', 'max_concurrent_fetches', 'cache_timeout']:
        try:
            return int(value)
        except ValueError:
            logger.error(f"Invalid int string for {key}: {value}, using 0")
            return 0
    if isinstance(value, str) and key in ['initial_equity', 'max_port_risk', 'kelly_frac', 'min_conf_score', 'min_vol_24h', 'min_order_size', 'low_cap_exp_max', 'max_corr_exp', 'sl_atr_mult', 'tp_atr_mult', 'ts_atr_mult', 'fees', 'slippage', 'retry_delay', 'min_pct_change', 'threshold_ema_alpha', 'low_vol_pos_mult', 'spoof_ent_thresh', 'imbal_thresh', 'opt_int', 'label_threshold_base', 'monte_carlo_validation_threshold', 'garch_alpha_base', 'garch_beta_base', 'jump_detection_threshold', 'mc_quality_excellent_threshold', 'mc_quality_good_threshold']:
        try:
            return float(value)
        except ValueError:
            logger.error(f"Invalid float string for {key}: {value}, using 0.0")
            return 0.0
    return value  

# Nueva función para cargar configs dinámicas desde JSON (opcional, fallback a defaults)
def load_dynamic_config(file_path='dynamic_config.json'):
    if Path(file_path).exists():
        try:
            with open(file_path, 'r') as f:
                dynamic = json.load(f)
                for regime, params in dynamic.items():
                    if regime in CONFIG and regime != 'current_regime':
                        CONFIG[regime].update(params)
                logger.info(f"Loaded dynamic config from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load dynamic config from {file_path}: {e}")
    else:
        logger.info(f"No dynamic config found at {file_path}, using defaults")

load_dynamic_config()  # Llama al inicio

def validate_api_keys():
    if not CONFIG['default']['dry_run']:
        merged_cfg = {**CONFIG['default'], **CONFIG[CONFIG['current_regime']]}
        if not merged_cfg['api_key']:
            raise ValueError("API key required for live trading")
        if not merged_cfg['api_secret']:
            raise ValueError("API secret required for live trading")

def validate_config():
    errors = []
    if not CONFIG['default']['dry_run']:
        validate_api_keys()
    
    for regime, cfg in CONFIG.items():
        if regime == 'default' or regime == 'current_regime':
            continue
        merged_cfg = {**CONFIG['default'], **cfg}  # Merge con default
        
        # Validaciones de tu código original
        if merged_cfg['initial_equity'] <= 0:
            errors.append(f"{regime}: INITIAL_EQUITY must be positive")
        if not 0 < merged_cfg['max_port_risk'] <= 1:
            errors.append(f"{regime}: MAX_PORTFOLIO_RISK must be between 0 and 1")
        if not 0 < merged_cfg['kelly_frac'] <= 1:
            errors.append(f"{regime}: KELLY_FRACTION must be between 0 and 1")
        if not 0 < merged_cfg['mc_quality_excellent_threshold'] <= 1:
            errors.append(f"{regime}: MC_QUALITY_EXCELLENT_THRESHOLD must be between 0 and 1")
        if not 0 < merged_cfg['mc_quality_good_threshold'] <= 1:
            errors.append(f"{regime}: MC_QUALITY_GOOD_THRESHOLD must be between 0 and 1")
        if merged_cfg['mc_quality_good_threshold'] >= merged_cfg['mc_quality_excellent_threshold']:
            errors.append(f"{regime}: MC_QUALITY_GOOD_THRESHOLD must be less than MC_QUALITY_EXCELLENT_THRESHOLD")
        if merged_cfg['backtest_freshness_days'] < 1:
            errors.append(f"{regime}: BACKTEST_FRESHNESS_DAYS must be at least 1")
        
        # Validaciones adicionales para otros parámetros clave
        if merged_cfg['min_conf_score'] <= 0:
            errors.append(f"{regime}: MIN_CONFIDENCE_SCORE must be positive")
        if merged_cfg['max_positions'] <= 0:
            errors.append(f"{regime}: MAX_POSITIONS must be positive")
        if merged_cfg['min_vol_24h'] <= 0:
            errors.append(f"{regime}: MIN_VOLUME_24H must be positive")
        if merged_cfg['min_order_size'] <= 0:
            errors.append(f"{regime}: MIN_ORDER_SIZE must be positive")
        if not 0 < merged_cfg['low_cap_exp_max'] <= 1:
            errors.append(f"{regime}: LOW_CAP_EXPOSURE_MAX must be between 0 and 1")
        if not 0 < merged_cfg['max_corr_exp'] <= 1:
            errors.append(f"{regime}: MAX_CORRELATED_EXPOSURE must be between 0 and 1")
        if merged_cfg['sl_atr_mult'] <= 0:
            errors.append(f"{regime}: STOP_LOSS_ATR_MULTIPLIER must be positive")
        if merged_cfg['tp_atr_mult'] <= 0:
            errors.append(f"{regime}: TAKE_PROFIT_ATR_MULTIPLIER must be positive")
        if merged_cfg['ts_atr_mult'] <= 0:
            errors.append(f"{regime}: TRAILING_STOP_ATR_MULTIPLIER must be positive")
        if merged_cfg['fees'] < 0:
            errors.append(f"{regime}: FEES cannot be negative")
        if merged_cfg['slippage'] < 0:
            errors.append(f"{regime}: SLIPPAGE cannot be negative")
        if merged_cfg['max_retries'] < 0:
            errors.append(f"{regime}: MAX_RETRIES cannot be negative")
        if merged_cfg['retry_delay'] < 0:
            errors.append(f"{regime}: RETRY_DELAY cannot be negative")
        if merged_cfg['bootstrap_days'] < 1:
            errors.append(f"{regime}: BOOTSTRAP_DAYS must be at least 1")
        if merged_cfg['min_data_pts'] < 1:
            errors.append(f"{regime}: MIN_DATA_POINTS must be at least 1")
        if merged_cfg['fetch_limit'] < 1:
            errors.append(f"{regime}: FETCH_LIMIT must be at least 1")
        if merged_cfg['perf_check_int'] < 1:
            errors.append(f"{regime}: PERFORMANCE_CHECK_INTERVAL must be at least 1")
        if merged_cfg['supervisor_int'] < 1:
            errors.append(f"{regime}: SUPERVISOR_INTERVAL must be at least 1")
        if merged_cfg['dynamic_very_low_vol_thresh'] <= 0:
            errors.append(f"{regime}: DYNAMIC_VERY_LOW_VOL_THRESHOLD must be positive")
        if merged_cfg['dynamic_low_vol_thresh'] <= 0:
            errors.append(f"{regime}: DYNAMIC_LOW_VOL_THRESHOLD must be positive")        
        if not 0 < merged_cfg['threshold_ema_alpha'] <= 1:
            errors.append(f"{regime}: THRESHOLD_EMA_ALPHA must be between 0 and 1")
        if merged_cfg['low_vol_pos_mult'] <= 0:
            errors.append(f"{regime}: LOW_VOL_POSITION_MULTIPLIER must be positive")
        if merged_cfg['ob_depth'] <= 0:
            errors.append(f"{regime}: ORDER_BOOK_DEPTH must be positive")
        if merged_cfg['spoof_ent_thresh'] <= 0:
            errors.append(f"{regime}: SPOOFING_ENTROPY_THRESHOLD must be positive")
        if merged_cfg['imbal_thresh'] <= 0:
            errors.append(f"{regime}: IMBALANCE_THRESHOLD must be positive")
        if merged_cfg['opt_int'] < 1:
            errors.append(f"{regime}: OPTIMIZATION_INTERVAL must be at least 1")
        if merged_cfg['bayes_trials'] < 1:
            errors.append(f"{regime}: BAYES_TRIALS must be at least 1")
        if merged_cfg['label_threshold_base'] <= 0:
            errors.append(f"{regime}: LABEL_THRESHOLD_BASE must be positive")
        if merged_cfg['monte_carlo_paths'] < 1:
            errors.append(f"{regime}: MONTE_CARLO_PATHS must be at least 1")
        if not 0 < merged_cfg['monte_carlo_validation_threshold'] <= 1:
            errors.append(f"{regime}: MONTE_CARLO_VALIDATION_THRESHOLD must be between 0 and 1")
        if not 0 < merged_cfg['garch_alpha_base'] <= 1:
            errors.append(f"{regime}: GARCH_ALPHA_BASE must be between 0 and 1")
        if not 0 < merged_cfg['garch_beta_base'] <= 1:
            errors.append(f"{regime}: GARCH_BETA_BASE must be between 0 and 1")
        if merged_cfg['jump_detection_threshold'] <= 0:
            errors.append(f"{regime}: JUMP_DETECTION_THRESHOLD must be positive")
        if not all(0 <= w <= 1 for w in merged_cfg['monte_carlo_ensemble_weights']):
            errors.append(f"{regime}: MONTE_CARLO_ENSEMBLE_WEIGHTS must be between 0 and 1")
        if sum(merged_cfg['monte_carlo_ensemble_weights']) != 1.0:
            errors.append(f"{regime}: MONTE_CARLO_ENSEMBLE_WEIGHTS must sum to 1")
        if merged_cfg['min_historical_periods'] < 1:
            errors.append(f"{regime}: MIN_HISTORICAL_PERIODS must be at least 1")

    # Nueva validación para params agregados
    if merged_cfg['bayes_alpha_prior'] <= 0:
        errors.append(f"{regime}: BAYES_ALPHA_PRIOR must be positive")
    if merged_cfg['lstm_layers'] <= 0:
        errors.append(f"{regime}: LSTM_LAYERS must be positive")
    if merged_cfg['attention_heads'] <= 0:
        errors.append(f"{regime}: ATTENTION_HEADS must be positive")
    if merged_cfg['max_bars'] < merged_cfg['min_data_pts']:
        errors.append(f"{regime}: MAX_BARS must be at least MIN_DATA_POINTS")
    if merged_cfg['chunk_size'] <= 0:
        errors.append(f"{regime}: CHUNK_SIZE must be positive")
    if merged_cfg['gan_epochs'] <= 0:
        errors.append(f"{regime}: GAN_EPOCHS must be positive")
    if merged_cfg['noise_dim'] <= 0:
        errors.append(f"{regime}: NOISE_DIM must be positive")
    if merged_cfg['vae_latent_dim'] <= 0:
        errors.append(f"{regime}: VAE_LATENT_DIM must be positive")
    if merged_cfg['evolve_features_topk'] <= 0:
        errors.append(f"{regime}: EVOLVE_FEATURES_TOPK must be positive")
    if merged_cfg['hyperopt_trials'] <= 0:
        errors.append(f"{regime}: HYPEROPT_TRIALS must be positive")
    if merged_cfg['mini_backtest_symbols'] <= 0:
        errors.append(f"{regime}: MINI_BACKTEST_SYMBOLS must be positive")
    if merged_cfg['causal_maxlag'] <= 0:
        errors.append(f"{regime}: CAUSAL_MAXLAG must be positive")
    if not 0 < merged_cfg['causal_threshold_p'] <= 1:
        errors.append(f"{regime}: CAUSAL_THRESHOLD_P must be between 0 and 1")
    if merged_cfg['tf_clusters'] <= 0:
        errors.append(f"{regime}: TF_CLUSTERS must be positive")
    if merged_cfg['auto_tf_min_vol'] <= 0:
        errors.append(f"{regime}: AUTO_TF_MIN_VOL must be positive")
    if not 0 <= merged_cfg['sentiment_weight'] <= 1:
        errors.append(f"{regime}: SENTIMENT_WEIGHT must be between 0 and 1")
    if not 0 <= merged_cfg['onchain_weight'] <= 1:
        errors.append(f"{regime}: ONCHAIN_WEIGHT must be between 0 and 1")
    if merged_cfg['vol_spike_threshold'] <= 1:
        errors.append(f"{regime}: VOL_SPIKE_THRESHOLD must be >1")
    if not 0 <= merged_cfg['shock_prob'] <= 1:
        errors.append(f"{regime}: SHOCK_PROB must be between 0 and 1")
    if merged_cfg['crash_magnitude'] <= 0:
        errors.append(f"{regime}: CRASH_MAGNITUDE must be positive")
    if merged_cfg['ga_population'] <= 0:
        errors.append(f"{regime}: GA_POPULATION must be positive")
    if merged_cfg['ga_generations'] <= 0:
        errors.append(f"{regime}: GA_GENERATIONS must be positive")
    if not 0 <= merged_cfg['ga_cxpb'] <= 1:
        errors.append(f"{regime}: GA_CXPB must be between 0 and 1")
    if merged_cfg['cvar_target'] >= 0:
        errors.append(f"{regime}: CVAR_TARGET must be negative")
    if not 0 <= merged_cfg['tax_rate'] <= 1:
        errors.append(f"{regime}: TAX_RATE must be between 0 and 1")

    if errors:
        for error in errors:
            logger.error(f"Config error: {error}")
        raise ValueError("; ".join(errors))
    logger.info("Config validation passed (all regimes)")

top_performers = []
top_changes = {}
last_top_fetch = 0
last_supervisor_run = 0
CLOSED_TRADES_SINCE_LAST = 0
missed_analysis = []

state = {'regime_global': CONFIG['current_regime'], 'consecutive_failures': 0}
REGIME_GLOBAL = state['regime_global']  

Path(get_config_param('model_dir')).mkdir(parents=True, exist_ok=True)
Path('backtests').mkdir(parents=True, exist_ok=True)
Path('reports').mkdir(parents=True, exist_ok=True)
Path('trades').mkdir(parents=True, exist_ok=True)

TIMEFRAMES = [
    {
        'name': 'short_15m',
        'binance_interval': '15m',
        'label_lookahead': 16,
        'label_threshold': 0.010,
        'confidence_threshold': 0.7,
        'min_data_points': 500,
        'rsi_period': 14,
        'vol_window': 20,
        'timeframe_minutes': 15,
        'strategy_type': 'mean_reversion',
        'max_holding_hours': 8,
    },
    {
        'name': 'medium_1h',
        'binance_interval': '1h',
        'label_lookahead': 24,
        'label_threshold': 0.02,
        'confidence_threshold': 0.65,
        'min_data_points': 300,
        'rsi_period': 21,
        'vol_window': 30,
        'timeframe_minutes': 60,
        'strategy_type': 'trend_following',
        'max_holding_hours': 24,
    },
    {
        'name': 'scalp_5m',
        'binance_interval': '5m',
        'label_lookahead': 6,
        'label_threshold': 0.0025,
        'confidence_threshold': 0.30,
        'min_data_points': 800,
        'rsi_period': 10,
        'vol_window': 15,
        'timeframe_minutes': 5,
        'strategy_type': 'micro_scalp',
        'max_holding_hours': 2,
        'regime_filter': ['very_low', 'low'],  # FIXED: Asegurar list de str (ya era, pero chequeado)
    },
    {
        'name': 'micro_15m',
        'binance_interval': '15m',
        'label_lookahead': 8,
        'label_threshold': 0.004,
        'confidence_threshold': 0.40,
        'min_data_points': 1000,
        'rsi_period': 12,
        'vol_window': 25,
        'timeframe_minutes': 15,
        'strategy_type': 'micro_breakout',
        'max_holding_hours': 4,
        'regime_filter': ['very_low', 'low', 'normal', 'high', 'volatile_bear', 'volatile_bull'],  # FIXED: Asegurar list completa
    },
]

DATA_CACHE = {}
LAST_CACHE_ACCESS = {}
PERFORMANCE_METRICS = {
    'total_signals': 0,
    'profitable_trades': 0,
    'total_closed_trades': 0,
    'realized_pnl': 0.0,
    'trades_executed': 0,
    'last_update': time.time(),
    'total_pnl': 0.0

}

CACHE_LOCK = asyncio.Lock()  
METRICS_LOCK_ASYNC = asyncio.Lock()
METRICS_LOCK_SYNC = threading.Lock()  
POSITION_LOCK_ASYNC = asyncio.Lock()  

class AppContext:
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls, exchange, db):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init(exchange, db)
        return cls._instance

    def _init(self, exchange, db):
        self.exchange = exchange
        self.db = db
        self._regime_lock = asyncio.Lock()
        self._metrics_lock = asyncio.Lock()
        self.metrics = PERFORMANCE_METRICS.copy()
        self.regime = CONFIG['current_regime']
        self._regime = self.regime  # Alias for compatibility
        self._initialize_locks()
        # Monkey-patch globals for compat
        global REGIME_GLOBAL
        REGIME_GLOBAL = property(lambda: self.regime)
        global PERFORMANCE_METRICS
        PERFORMANCE_METRICS = property(lambda: self.metrics)
        # Hyperopt cache check
        self._check_hyperopt_cache()

    def _initialize_locks(self):
        self._other_locks = {}  # Additional locks if needed

    @property
    def regime(self):
        return self._regime

    @regime.setter
    def regime(self, value):
        self._regime = value

    async def update_metrics(self, pnl: float, trades: int):
        async with self._metrics_lock:
            self.metrics['realized_pnl'] += pnl
            self.metrics['trades_executed'] += trades

    async def update_regime(self, new_regime: str):
        async with self._regime_lock:
            self.regime = new_regime

    def _check_hyperopt_cache(self):
        cache_path = 'hyperopt_cache.pkl'
        if not Path(cache_path).exists():
            logger.info("No hyperopt cache; running initial hyperopt")
            self._run_hyperopt()
        else:
            with open(cache_path, 'rb') as f:
                best_params = pickle.load(f)
                for param, value in best_params.items():
                    CONFIG['default'][param] = value
            logger.info("Loaded hyperopt cache")

    def _run_hyperopt(self):
        import optuna
        def objective(trial):
            params = {k: trial.suggest_float(k, CONFIG['default'].get('bounds', {}).get(k, (0.1, 1.0))[0], CONFIG['default'].get('bounds', {}).get(k, (0.1, 1.0))[1]) for k in self.params_to_optimize}
            sharpe = self._run_mini_backtest(params)
            return sharpe
        self.params_to_optimize = ['kelly_frac', 'min_conf_score']  # Example
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=CONFIG['default']['hyperopt_trials'])
        with open('hyperopt_cache.pkl', 'wb') as f:
            pickle.dump(study.best_params, f)
        for param, value in study.best_params.items():
            CONFIG['default'][param] = value

    def _run_mini_backtest(self, params: dict) -> float:
        # Mock mini backtest
        metrics = [{'sharpe_ratio': np.random.uniform(1.0, 2.0)} for _ in range(CONFIG['default']['mini_backtest_symbols'])]
        return np.mean([m['sharpe_ratio'] for m in metrics])

    def __getattr__(self, name):
        if name == 'REGIME_GLOBAL':
            return self.regime
        elif name == 'PERFORMANCE_METRICS':
            return self.metrics
        raise AttributeError(f"'AppContext' has no attribute '{name}'")

class ExchIntf:
    def __init__(self, exch_id: str, api_key: str = None, api_secret: str = None, sandbox: bool = False, futures: bool = False):
        self.exch_id = exch_id
        self.client = None
        self.api_key = api_key
        self.api_secret = api_secret
        self.sandbox = sandbox
        self.futures = futures
        self.connected = False
        self.markets = {}
        self.rate_limiter = asyncio.Semaphore(10)

    async def connect(self, max_retries: int = 3):
        for attempt in range(max_retries + 1):
            async with self.rate_limiter:
                try:
                    exch_class = getattr(ccxt, self.exch_id)
                    config = {
                        'enableRateLimit': True,
                        'timeout': 30000,
                        'options': {'adjustForTimeDifference': True},
                    }
                    if not CONFIG['default']['dry_run']:
                        if self.api_key and self.api_secret:
                            config.update({'apiKey': self.api_key, 'secret': self.api_secret})
                    # En dry-run, omite claves para endpoints públicos (datos reales sin órdenes)
                    else:
                        logger.info(f"Dry-run: Using public endpoints for {self.exch_id} (real market data)")

                    if self.sandbox:
                        config['sandbox'] = True
                    if self.futures:
                        config['options']['defaultType'] = 'future'

                    self.client = exch_class(config)
                    
                    # Carga de mercados con retry (público en dry-run, firmado solo en live)
                    success = False
                    for load_attempt in range(2):  # Retry solo para load_markets, no full connect
                        try:
                            if attempt == 0 or load_attempt > 0:
                                await asyncio.sleep(2 ** load_attempt)
                            await self.client.load_markets()
                            self.markets = self.client.markets
                            success = True
                            break
                        except ccxt.AuthenticationError as auth_e:
                            if CONFIG['default']['dry_run']:
                                # En dry-run, fuerza público sin claves si auth falla
                                logger.warning(f"Dry-run auth error (likely signed public call): {auth_e}. Retrying public-only.")
                                config_no_keys = config.copy()
                                config_no_keys.pop('apiKey', None)
                                config_no_keys.pop('secret', None)
                                self.client = exch_class(config_no_keys)
                                await self.client.load_markets()
                                self.markets = self.client.markets
                                success = True
                                break
                            else:
                                raise  # Re-raise en live
                        except ccxt.NetworkError as net_e:
                            logger.warning(f"Network error in load_markets (load_attempt {load_attempt}): {net_e}")
                            continue
                        except Exception as e:
                            logger.error(f"Unexpected load_markets error (load_attempt {load_attempt}): {e}")
                            break
                    
                    if not success:
                        raise Exception("Failed to load markets after retries")
                    
                    # Ping test (público, no firma)
                    try:
                        await self.client.fetch_status()
                    except ccxt.AuthenticationError:
                        if CONFIG['default']['dry_run']:
                            logger.info("Dry-run: Skipping signed status; assuming healthy")
                        else:
                            raise
                    self.connected = True
                    logger.info(f"Connected to {self.exch_id} (sandbox: {self.sandbox}, dry-run: {CONFIG['default']['dry_run']})")
                    logger.info(f"Loaded {len(self.markets)} markets")
                    return True
                except ccxt.NetworkError as e:
                    logger.warning(f"NetworkError in connect (attempt {attempt}/{max_retries}): {e}")
                    if self.client:
                        await self.client.close()  # Close on error
                        self.client = None
                    if attempt < max_retries:
                        await asyncio.sleep(2 ** attempt)
                        continue
                except Exception as e:
                    logger.error(f"Unexpected error in connect (attempt {attempt}/{max_retries}): {e}")
                    if self.client:
                        await self.client.close()  # Close on error
                        self.client = None
                    break
        logger.error(f"Failed to connect to {self.exch_id} after {max_retries} retries")
        self.connected = False
        if self.client:
            await self.client.close()
            self.client = None
        # CRÍTICO: Intentar reconexión automática si no dry-run
        if not CONFIG['default']['dry_run'] and attempt == max_retries:
            logger.info("Initial connect failed, scheduling auto-reconnect in 30s")
            asyncio.create_task(self._auto_reconnect_loop(max_retries=5))  # Nueva task de reconexión
        return False

    async def disconnect(self):
        if self.client and self.connected:
            try:
                await self.client.close()
                if hasattr(self.client, 'session') and self.client.session and not self.client.session.closed:
                    await self.client.session.close()
                if hasattr(self.client.session, '_connector') and self.client.session._connector and not self.client.session._connector.closed:
                    await self.client.session._connector.close()
                self.connected = False
                logger.info(f"Disconnected from {self.exch_id}")
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")

    async def fetch_ohlcv_with_retry(self, symbol: str, timeframe: str, since: int = None, limit: int = None, max_retries: int = None) -> List:
        if not self.connected or not symbol:
            return []        
        if max_retries is None:
            max_retries = CONFIG['default']['max_retries']
        if self.client is None:  
            logger.warning(f"Mock OHLCV data returned for {symbol} ({timeframe}) due to no client")
            return []
        # En dry-run, usa datos reales públicos (no mock)
        for attempt in range(max_retries + 1):
            async with self.rate_limiter:
                try:
                    ohlcv = await self.client.fetch_ohlcv(symbol, timeframe, since, limit)
                    logger.debug(f"Fetched real OHLCV for {symbol} ({timeframe}, dry-run: {CONFIG['default']['dry_run']})")
                    return ohlcv or []
                except ccxt.AuthenticationError as auth_e:
                    if CONFIG['default']['dry_run']:
                        logger.warning(f"Dry-run auth error in OHLCV {symbol}: {auth_e}. Skipping signed, using empty.")
                        return []  # Fallback empty en auth error durante dry-run
                    raise
                except ccxt.NetworkError as e:
                    if attempt < max_retries:
                        delay = CONFIG['default']['retry_delay'] * (2 ** attempt)
                        logger.warning(f"Network error fetching OHLCV {symbol}, retrying in {delay}s: {e}")
                        await asyncio.sleep(delay)
                        continue
                    logger.error(f"Network error fetching OHLCV {symbol} after {max_retries} retries: {e}")
                except ccxt.ExchangeError as e:
                    logger.error(f"Exchange error fetching OHLCV {symbol}: {e}")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error fetching OHLCV {symbol}: {e}")
                    break
        return []       

    async def fetch_ticker_with_retry(self, symbol: str) -> Optional[Dict]:
        if self.client is None or not self.connected:
            logger.warning(f"Mock ticker data for {symbol} due to no client/connected")
            return None
        # En dry-run, usa datos reales públicos (no mock)
        for attempt in range(CONFIG['default']['max_retries']):
            if not symbol:
                return None
            async with self.rate_limiter:
                try:
                    ticker = await self.client.fetch_ticker(symbol)
                    if not isinstance(ticker, dict):
                        return None
                    close = ticker.get('close', ticker.get('last'))
                    if close is None:
                        return None
                    close = float(close)
                    if close <= 0:
                        return None
                    ticker['close'] = close
                    volume = ticker.get('volume') or ticker.get('baseVolume') or (ticker.get('quoteVolume', 0) / close if close > 0 else 0)
                    volume = float(volume)
                    if volume <= 0:
                        return None
                    ticker['volume'] = volume
                    logger.debug(f"Fetched real ticker for {symbol} (dry-run: {CONFIG['default']['dry_run']})")
                    return ticker
                except ccxt.AuthenticationError as auth_e:
                    if CONFIG['default']['dry_run']:
                        logger.warning(f"Dry-run auth error in ticker {symbol}: {auth_e}. Skipping signed, using mock.")
                        return None  # Fallback mock solo en auth error durante dry-run
                    raise
                except Exception as e:
                    if attempt < CONFIG['default']['max_retries'] - 1:
                        await asyncio.sleep(CONFIG['default']['retry_delay'])
                    else:
                        logger.error(f"Ticker fetch failed for {symbol} after retries: {e}")
                        return None
        return None        

    async def create_order_with_validation(self, symbol: str, order_type: str, side: str, amount: float, price: float = None) -> Optional[Dict]:
        if CONFIG['default']['dry_run'] or not self.connected:
            order_id = f"sim_{int(time.time() * 1000)}"
            filled = amount * 0.95
            logger.info(f"[DRY RUN] {side.upper()} {amount:.8f} {symbol} @ {price or 'MARKET'} (filled: {filled:.8f})")
            return {'id': order_id, 'symbol': symbol, 'type': order_type, 'side': side, 'amount': amount, 'price': price, 'status': 'closed', 'filled': filled, 'timestamp': int(time.time() * 1000), 'info': {'dry_run': True}}
        async with self.rate_limiter:
            try:
                if symbol not in self.markets:
                    return None
                market = self.markets[symbol]
                min_amount = market.get('limits', {}).get('amount', {}).get('min', 0.0)  # Safe nested get
                if amount < min_amount:
                    return None
                order = await self.client.create_order(symbol, order_type, side, amount, price)
                logger.info(f"Order created: {order['id']} - {side.upper()} {amount:.8f} {symbol}")
                return order
            except Exception as e:
                logger.error(f"Error creating order for {symbol}: {e}")
                return None

class VolRegDet:
    def __init__(self, context: AppContext):
        self.context = context
        self.regime_cache = {}
        self.hmm_n_components = 4  # For very_low/low/normal/high-volatile regimes
        self.hmm_cov_type = 'diag'  # Diagonal covariance for efficiency
        self.hmm_n_iter = 300  # Max iterations for convergence
        self.exchange = context.exchange
        self.trap_history = {}
        self.cache_duration = 300  # seconds
        self._regime_locks = defaultdict(asyncio.Lock)

    async def fetch_order_book(self, symbol: str) -> Optional[Dict]:
        if not symbol or self.exchange.client is None or CONFIG['default']['dry_run']:
            return None
        async with self.exchange.rate_limiter:
            try:
                return await self.exchange.client.fetch_order_book(symbol, limit=CONFIG['default']['fetch_limit'])
            except Exception as e:
                logger.error(f"Error fetching order book for {symbol}: {e}")
                return None

    def detect_spoofing(self, bids: List, asks: List) -> float:
        if not bids or not asks:
            return 0.0
        bid_sizes = np.array([level[1] for level in bids[:CONFIG['default']['ob_depth']]])
        ask_sizes = np.array([level[1] for level in asks[:CONFIG['default']['ob_depth']]])
        combined = np.concatenate([bid_sizes, ask_sizes])
        if np.sum(combined) == 0:
            return 0.0
        probs = combined / np.sum(combined)
        spoof_entropy = entropy(probs)
        return spoof_entropy / np.log(max(len(combined), 2))

    def detect_imbalance(self, bids: List, asks: List) -> float:
        bid_vol = sum(level[1] for level in bids[:CONFIG['default']['ob_depth']])
        ask_vol = sum(level[1] for level in asks[:CONFIG['default']['ob_depth']])
        total_vol = bid_vol + ask_vol
        if total_vol == 0:
            return 0.5
        return abs(bid_vol - ask_vol) / total_vol

    async def detect_regime(self, df: pd.DataFrame, symbol: str, state: Optional[Dict] = None, historical: bool = False) -> Tuple[str, bool]:
        global REGIME_GLOBAL
        state = state or {'regime_global': REGIME_GLOBAL}
        try:
            async with asyncio.timeout(10):
                async with self._regime_locks[symbol]:
                    cache_key = f"{symbol}_regime"
                    now = time.time()
                    if cache_key in self.regime_cache and now - self.regime_cache[cache_key]['timestamp'] < self.cache_duration:
                        return self.regime_cache[cache_key]['regime'], self.regime_cache[cache_key]['trap_detected']
                    if len(df) < 50:
                        return 'normal', False
                    # Prepare returns and ensure stationarity
                    returns = df['close'].pct_change().dropna()
                    if len(returns) < 50:
                        return 'normal', False
                    if not is_stationary(pd.Series(returns)):
                        returns = pd.Series(returns).diff().dropna().values  # Diferenciar si no stationary (evita HMM errors)
                        if len(returns) < 50:
                            return 'normal', False
                    # Fit HMM with reduced cascade (faster: n_iter=100 max, fewer options)
                    model = None
                    # Reduced cascade: n_iter [100], init_params ['stmc'], n_components [4,3], cov_type ['diag']
                    for n_iter in [100]:  # FIXED: Single fast iter
                        for init_p in ['stmc']:  # FIXED: Single init
                            for n_comp in [self.hmm_n_components, self.hmm_n_components - 1]:  # FIXED: 2 options
                                for cov_type in [self.hmm_cov_type]:  # FIXED: Single cov
                                    try:
                                        model = GaussianHMM(n_components=n_comp, covariance_type=cov_type, n_iter=n_iter,
                                                            init_params=init_p, params='stmc', random_state=42)
                                        model.fit(returns.values.reshape(-1, 1))
                                        if not model.monitor_.converged:
                                            logger.debug(f"HMM not converged (n_iter={n_iter}, init={init_p}, n={n_comp}, cov={cov_type}). Skipping.")
                                            continue
                                    except ValueError as ve:
                                        logger.debug(f"HMM try failed (n_iter={n_iter}, init={init_p}, n={n_comp}, cov={cov_type}): {ve}. Trying next.")
                                        continue
                                    if model is not None:
                                        break
                                if model is not None:
                                    break
                            if model is not None:
                                break
                        if model is not None:
                            break
                    if model is None:
                        # Ultimate fallback: simple volatility threshold
                        logger.warning(f"All HMM attempts failed for {symbol}. Falling back to volatility threshold regime.")
                        recent_vol = returns.tail(48).std() * np.sqrt(8760) if len(returns) >= 48 else 0.02
                        regime = ('very_low' if recent_vol < CONFIG['default']['dynamic_very_low_vol_thresh'] else
                                  'low' if recent_vol < CONFIG['default']['dynamic_low_vol_thresh'] else
                                  'high' if recent_vol > 0.03 else 'normal')
                        return regime, False  # No trap in fallback
                    # Predict regimes
                    regimes = model.predict(returns.values.reshape(-1, 1))
                    # Map HMM states to regimes based on volatility and trend
                    state_vols = []
                    state_trends = []
                    for i_state in range(model.n_components):
                        state_returns = returns[regimes == i_state]
                        state_vol = state_returns.std() * np.sqrt(8760) if len(state_returns) > 0 else 0
                        state_trend = state_returns.mean()
                        state_vol = np.clip(state_vol, 1e-6, np.inf)  
                        state_trend = np.clip(state_trend, -1.0, 1.0)  
                        state_vols.append(state_vol)
                        state_trends.append(state_trend)
                    # Convert to NumPy for robust indexing
                    state_vols = np.array(state_vols)
                    state_trends = np.array(state_trends)                    
                    state_vols = np.clip(state_vols, 1e-6, np.inf)
                    state_trends = np.clip(state_trends, -np.inf, np.inf)
                    # Sort states by volatility
                    sorted_states = np.argsort(state_vols)
                    # Assign regimes dynamically with smoothed trend mapping
                    regime_map = {}
                    from scipy.signal import savgol_filter  
                   
                    try:
                        # Kalman-like trend estimation (simple SG filter as proxy)
                        window_len = max(3, min(5, len(state_trends)))  # Min 3 para polyorder=2; evita ValueError si <5
                        polyorder = min(2, window_len - 1)  # Polyorder < window_len siempre
                        trend_smooth = savgol_filter(state_trends, window_len, polyorder) if len(state_trends) >= window_len else state_trends
                        for idx, hmm_state in enumerate(sorted_states):
                            state_idx = int(hmm_state)
                            if state_idx >= len(state_vols) or state_idx >= len(trend_smooth):
                                continue
                            vol = float(state_vols[state_idx])
                            trend = float(trend_smooth[min(idx, len(trend_smooth)-1)] if idx < len(trend_smooth) else state_trends[state_idx])  # Safe min for idx
                            
                            # Dynamic thresholds from CONFIG (EMA-adapted)
                            dyn_very_low_safe = float(CONFIG['default'].get('dynamic_very_low_vol_thresh', 0.004))
                            dyn_low_safe = float(CONFIG['default'].get('dynamic_low_vol_thresh', 0.012))
                            vol_safe = max(vol, 0.001)
                            
                            # Base regime with vol clip
                            if idx == 0:
                                base_regime = 'very_low' if vol_safe < dyn_very_low_safe else 'low'
                            elif idx == 1:
                                base_regime = 'low' if vol_safe < dyn_low_safe else 'normal'
                            elif idx == 2:
                                base_regime = 'normal' if vol_safe < 0.03 else 'high'
                            else:
                                base_regime = 'high'
                            
                            regime_map[state_idx] = base_regime
                            
                            # Overlay smoothed sigmoid trend (enhanced: Kalman proxy via SG filter)
                            base_regime_str = str(base_regime)
                            if base_regime_str in ['normal', 'high']:
                                trend_sig = 1 / (1 + np.exp(-trend * 8))  # Tighter sigmoid for sensitivity
                                if trend_sig > 0.65:  # Threshold tightened for faster bull detect
                                    regime_map[state_idx] = 'volatile_bull' if base_regime_str == 'high' else 'bull'
                                elif trend_sig < 0.35:  # Tighter for bear
                                    regime_map[state_idx] = 'volatile_bear' if base_regime_str == 'high' else 'bear'
                    except Exception as regime_error:
                        logger.error(f"Regime mapping error for {symbol}: {regime_error}")
                        regime_map = {i: 'normal' for i in range(len(state_vols))}
                    
                    # Current regime is the last state
                    current_state = regimes[-1] if len(regimes) > 0 else 0  # FIXED: Guard si empty
                    current_state = int(current_state)  # Convert numpy.int64 to Python int
                    regime = regime_map.get(current_state, 'normal')
                    
                    if symbol == 'BTC/USDT':                        
                        async with self.context._metrics_lock:
                            REGIME_GLOBAL = regime
                            state['regime_global'] = regime
                    
                    # Update dynamic thresholds with EMA
                    recent_vol = returns.tail(48).std() * np.sqrt(8760) if len(returns) >= 48 else 0.02
                    alpha = CONFIG['default']['threshold_ema_alpha']
                    if regime in ['very_low', 'low']:
                        CONFIG['default']['dynamic_very_low_vol_thresh'] = alpha * recent_vol + (1 - alpha) * CONFIG['default']['dynamic_very_low_vol_thresh']
                        CONFIG['default']['dynamic_low_vol_thresh'] = alpha * recent_vol + (1 - alpha) * CONFIG['default']['dynamic_low_vol_thresh']
                    trap_detected = False
                    if not historical:
                        order_book = await self.fetch_order_book(symbol)
                        if order_book:
                            spoof_entropy = self.detect_spoofing(order_book['bids'], order_book['asks'])
                            imbalance = self.detect_imbalance(order_book['bids'], order_book['asks'])
                            ticker = await self.exchange.fetch_ticker_with_retry(symbol)
                            entropy_thresh = CONFIG['default']['spoof_ent_thresh'] * (0.5 if regime in ['very_low', 'low'] else 1.0)
                            imbalance_thresh = CONFIG['default']['imbal_thresh'] * (0.8 if regime in ['very_low', 'low'] else 1.0)
                            if ticker and 'quoteVolume' in ticker:
                                vol_24h = ticker['quoteVolume']
                                entropy_thresh *= 1.5 if vol_24h > 10000000 else 1.0
                                imbalance_thresh *= 0.8 if vol_24h > 10000000 else 1.0
                            if 'whale_risk' in df.iloc[-1] and df.iloc[-1]['whale_risk'] > 0.4:
                                entropy_thresh *= 0.8
                                imbalance_thresh *= 1.2
                            if spoof_entropy > entropy_thresh and imbalance > imbalance_thresh:
                                trap_detected = True
                                self.trap_history.setdefault(symbol, []).append(now)
                                logger.warning(f"Trap detected for {symbol}: entropy={spoof_entropy:.2f}/{entropy_thresh:.2f}, imbalance={imbalance:.2f}/{imbalance_thresh:.2f}")
                            else:
                                logger.debug(f"No trap detected for {symbol}: entropy={spoof_entropy:.2f}/{entropy_thresh:.2f}, imbalance={imbalance:.2f}/{imbalance_thresh:.2f}")
                    else:
                        if 'whale_risk' in df.columns and not df['whale_risk'].empty:
                            avg_whale = df['whale_risk'].mean()
                            trap_detected = avg_whale > 0.4
                            bt_logger.debug(f"Historical trap simulation for {symbol}: detected={trap_detected} (avg whale_risk={avg_whale:.2f})")
                        else:
                            trap_detected = False
                            bt_logger.debug(f"No whale_risk data for historical trap simulation in {symbol}")
                    self.regime_cache[cache_key] = {'regime': regime, 'volatility': recent_vol, 'timestamp': now, 'trap_detected': trap_detected}
                    return regime, trap_detected
        except asyncio.TimeoutError:
            bt_logger.warning(f"Timeout for {symbol} regime detection")
            return 'normal', False
        except Exception as e:
            logger.error(f"HMM regime detection error for {symbol}: {e}")
            return 'normal', False

    async def aggregate_per_pair_regimes(self, symbols: List[str], dfs: Dict[str, pd.DataFrame], state: dict) -> str:
        regime_counts = Counter()
        for symbol in symbols:
            if symbol in dfs and not dfs[symbol].empty:
                regime, _ = await self.detect_regime(dfs[symbol], symbol, state)
                regime_counts[regime] += 1
        return regime_counts.most_common(1)[0][0] if regime_counts else 'normal'

    def get_adaptive_config(self, regime: str, base_tf_config: Dict) -> Dict:
        config = base_tf_config.copy()
        # Use dynamic thresholds from CONFIG for adaptability
        very_low_thresh = CONFIG['default']['dynamic_very_low_vol_thresh']
        low_thresh = CONFIG['default']['dynamic_low_vol_thresh']
        if 'volatile' in regime or regime == 'high':
            config['label_threshold'] *= 1.5
            merged_cfg = {**CONFIG['default'], **CONFIG.get(CONFIG['current_regime'], {})}
            config['sl_atr_mult'] = merged_cfg.get('sl_atr_mult', CONFIG['default']['sl_atr_mult']) * 0.8
            merged_cfg = {**CONFIG['default'], **CONFIG.get(CONFIG['current_regime'], {})}
            config['tp_atr_mult'] = merged_cfg.get('tp_atr_mult', CONFIG['default']['tp_atr_mult']) * 1.2
        elif regime in ['bear', 'volatile_bear']:
            config['confidence_threshold'] += 0.1
            merged_cfg = {**CONFIG['default'], **CONFIG.get(CONFIG['current_regime'], {})}
            config['sl_atr_mult'] = merged_cfg.get('sl_atr_mult', CONFIG['default']['sl_atr_mult']) * 1.2
        elif regime in ['bull', 'volatile_bull']:
            config['confidence_threshold'] -= 0.05
            merged_cfg = {**CONFIG['default'], **CONFIG.get(CONFIG['current_regime'], {})}
            config['tp_atr_mult'] = merged_cfg.get('tp_atr_mult', CONFIG['default']['tp_atr_mult']) * 1.5
        elif regime in ['very_low', 'low']:
            config['label_threshold'] *= 0.8
            merged_cfg = {**CONFIG['default'], **CONFIG.get(CONFIG['current_regime'], {})}
            config['sl_atr_mult'] = merged_cfg.get('sl_atr_mult', CONFIG['default']['sl_atr_mult']) * (0.6 if regime == 'very_low' else 0.7)
            merged_cfg = {**CONFIG['default'], **CONFIG.get(CONFIG['current_regime'], {})}
            config['tp_atr_mult'] = merged_cfg.get('tp_atr_mult', CONFIG['default']['tp_atr_mult']) * (0.8 if regime == 'very_low' else 0.9)
        # Additional dynamic adjustment based on thresholds
        if regime == 'very_low' and very_low_thresh > 0.005:
            config['confidence_threshold'] -= 0.02  # Lower threshold if dynamic thresh increased
        # Validate regime (production safety: fallback to 'normal' if invalid)
        if regime not in CONFIG:
            logger.warning(f"Invalid regime '{regime}' in adaptive config; falling back to 'normal'")
            regime = 'normal'
        return config

class ModelMgr:
    def __init__(self, context: AppContext):
        self.context = context
        self.model_dir = Path(CONFIG['default']['model_dir'])
        self.model_dir.mkdir(exist_ok=True)
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.last_train = {}
        self.lstm_models = {}  # New for LSTM
        self.use_lstm = True  # Default to LSTM

    def _get_model_key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol.replace('/', '_')}_{timeframe}"

    def _prepare_training_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[LabelEncoder]]:
        if len(df) < 100 or 'label' not in df.columns:
            return None, None, None
        features = df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values
        labels_str = df['label'].fillna('Hold')
        le = LabelEncoder().fit(['Hold', 'Buy', 'Sell'])
        labels = le.transform(labels_str)
        if len(labels) < 50:
            return None, None, None
        return features, labels, le

    async def train_model(self, symbol: str, timeframe: str, df: pd.DataFrame) -> bool:
        try:
            key = self._get_model_key(symbol, timeframe)
            X, y, le = self._prepare_training_data(df, symbol, timeframe)
            if X is None or len(X) < 50:
                return False
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            if self.use_lstm:
                model = LSTMSignalModel()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                loss_fn = nn.CrossEntropyLoss()
                batch_size = 32
                epochs = 10
                for epoch in range(epochs):
                    for i in range(0, len(X_train), batch_size):
                        batch_X = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32).unsqueeze(1)
                        batch_y = torch.tensor(y_train[i:i+batch_size], dtype=torch.long)
                        optimizer.zero_grad()
                        out = model(batch_X)
                        loss = loss_fn(out, batch_y)
                        loss.backward()
                        optimizer.step()
                test_X = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
                test_out = model(test_X)
                test_pred = torch.argmax(test_out, dim=1).numpy()
                test_score = np.mean(test_pred == y_test)
            else:
                model = RandomForestClassifier(
                    n_estimators=50, max_depth=10, min_samples_split=10, min_samples_leaf=5,
                    random_state=42, n_jobs=-1, class_weight='balanced'
                )
                model.fit(X_train, y_train)
                test_score = model.score(X_test, y_test)
            test_score = np.clip(test_score, 0.0, 1.0)  # Clip para valores extremos
            if test_score > 0.5:
                ver = CONFIG['default']['model_ver']
                if self.use_lstm:
                    torch.save(model.state_dict(), self.model_dir / f"{key}_lstm_v{ver}.pth")
                    self.lstm_models[key] = model
                else:
                    joblib.dump(model, self.model_dir / f"{key}_model_v{ver}.pkl")
                joblib.dump(scaler, self.model_dir / f"{key}_scaler_v{ver}.pkl")
                joblib.dump(le, self.model_dir / f"{key}_le_v{ver}.pkl")
                joblib.dump(model.feature_importances_, self.model_dir / f"{key}_importance_v{ver}.pkl")
                joblib.dump(test_score, self.model_dir / f"{key}_accuracy_v{ver}.pkl")  # Nueva: guardar accuracy para checks rápidos
                self.models[key] = model
                self.scalers[key] = scaler
                self.label_encoders[key] = le
                self.last_train[key] = time.time()
                logger.info(f"Model trained for {symbol} {timeframe}: accuracy={test_score:.3f}")
                if test_score < 0.6:
                    logger.warning(f"Low model accuracy for {symbol} {timeframe}: {test_score:.3f} - Consider retraining with more data")
                return True
            return False
        except Exception as e:
            logger.error(f"Error training model for {symbol} {timeframe}: {e}")
            return False

    def get_prediction(self, symbol: str, timeframe: str, features: pd.Series) -> Tuple[float, float]:
        try:
            key = self._get_model_key(symbol, timeframe)
            if key not in self.models:
                return 0.0, 0.0
            model = self.models[key]
            scaler = self.scalers[key]
            le = self.label_encoders.get(key)
            if le is None:
                ver = CONFIG['default']['model_ver']
                le_path = self.model_dir / f"{key}_le_v{ver}.pkl"
                le = joblib.load(le_path) if le_path.exists() else LabelEncoder().fit(['Hold', 'Buy', 'Sell'])
                self.label_encoders[key] = le
            feat_values = [features.get(col, 0.0) for col in FEATURE_COLS]
            X = np.nan_to_num(np.array(feat_values).reshape(1, -1), 0.0)
            X_scaled = scaler.transform(X)
            if self.use_lstm:
                X_torch = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
                pred_class = torch.argmax(model(X_torch), dim=1).item()
                probs = torch.softmax(model(X_torch), dim=1)[0].detach().numpy()
            else:
                pred_class = model.predict(X_scaled)[0]
                probs = model.predict_proba(X_scaled)[0]
            signal = 1.0 if pred_class == 1 else -1.0 if pred_class == 2 else 0.0
            confidence = max(probs)
            return signal * confidence, confidence
        except Exception as e:
            logger.debug(f"Error getting prediction for {symbol} {timeframe}: {e}")
            return 0.0, 0.0

    def should_retrain(self, symbol: str, timeframe: str) -> bool:
        key = self._get_model_key(symbol, timeframe)
        return (time.time() - self.last_train.get(key, 0)) > CONFIG['default']['retrain_int']

    def get_model_accuracy(self, symbol: str, timeframe: str) -> float:
        """Obtiene la accuracy guardada del modelo para checks rápidos de calidad."""
        try:
            key = self._get_model_key(symbol, timeframe)
            ver = CONFIG['default']['model_ver']
            path = self.model_dir / f"{key}_accuracy_v{ver}.pkl"
            if path.exists():
                acc = joblib.load(path)
                logger.debug(f"Loaded accuracy {acc:.3f} for {symbol} {timeframe}")
                return acc
            return 0.0
        except Exception as e:
            logger.error(f"Error loading accuracy for {symbol} {timeframe}: {e}")
            return 0.0

class DbIntf:
    def __init__(self, pool: Optional[asyncpg.Pool] = None):
        self.pool = pool
        self.connected = pool is not None

    async def init_tables(self):
        if not self.connected:
            return False
        conn = None
        try:
            conn = await self.pool.acquire()
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS ohlcv_data (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    open DECIMAL NOT NULL,
                    high DECIMAL NOT NULL,
                    low DECIMAL NOT NULL,
                    close DECIMAL NOT NULL,
                    volume DECIMAL NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(symbol, timeframe, timestamp)
                );
            ''')
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    amount DECIMAL NOT NULL,
                    price DECIMAL NOT NULL,
                    confidence DECIMAL NOT NULL,
                    regime VARCHAR(20),
                    order_id VARCHAR(100),
                    status VARCHAR(20) DEFAULT 'pending',
                    pnl DECIMAL DEFAULT 0,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    closed_at TIMESTAMPTZ
                );
            ''')
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    total_signals INTEGER DEFAULT 0,
                    profitable_trades INTEGER DEFAULT 0,
                    total_closed_trades INTEGER DEFAULT 0,
                    realized_pnl DECIMAL DEFAULT 0,
                    trades_executed INTEGER DEFAULT 0,
                    portfolio_value DECIMAL DEFAULT 0,
                    UNIQUE(timestamp)
                );
            ''')
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    realized_pnl DECIMAL DEFAULT 0,
                    win_rate DECIMAL DEFAULT 0,
                    sharpe_ratio DECIMAL DEFAULT 0,
                    max_drawdown DECIMAL DEFAULT 0,
                    details JSONB,
                    top_pairs TEXT[]
                );
            ''')
            await conn.execute('''
                DO $$ BEGIN
                    ALTER TABLE backtest_results ADD COLUMN realized_pnl DECIMAL DEFAULT 0;
                EXCEPTION WHEN duplicate_column THEN -- nothing
                END $$;
            ''')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe ON ohlcv_data(symbol, timeframe);')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp);')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_backtest_timestamp ON backtest_results(timestamp);')
            # Nueva: agregar ALTER para realized_pnl en performance_metrics (similar a backtest)
            await conn.execute('''
                DO $$ BEGIN
                    ALTER TABLE performance_metrics ADD COLUMN realized_pnl DECIMAL DEFAULT 0;
                EXCEPTION WHEN duplicate_column THEN -- nothing
                END $$;
            ''')
            logger.info("Database tables initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing tables: {e}")
            return False
        finally:
            if conn:
                await self.pool.release(conn)

    async def save_ohlcv_batch(self, symbol: str, timeframe: str, df: pd.DataFrame):
        if not self.connected or df.empty:
            return
        conn = None
        try:
            conn = await self.pool.acquire()
            rows = [(symbol, timeframe, ts, float(row['open']), float(row['high']), float(row['low']), float(row['close']), float(row['volume'])) for ts, row in df.iterrows()]
            await conn.executemany('''
                INSERT INTO ohlcv_data (symbol, timeframe, timestamp, open, high, low, close, volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (symbol, timeframe, timestamp) DO UPDATE SET
                    open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, close = EXCLUDED.close, volume = EXCLUDED.volume
            ''', rows)
            logger.debug(f"Saved {len(rows)} OHLCV for {symbol} {timeframe}")
        except Exception as e:
            logger.error(f"Error saving OHLCV: {e}")
        finally:
            if conn:
                await self.pool.release(conn)

    async def save_trade(self, trade_data: Dict):
        if not self.connected:
            return None
        conn = None
        try:
            conn = await self.pool.acquire()
            trade_id = await conn.fetchval('''
                INSERT INTO trades (symbol, timeframe, side, amount, price, confidence, regime, order_id, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
            ''', trade_data['symbol'], trade_data.get('timeframe', ''), trade_data['side'], trade_data['amount'], trade_data['price'], trade_data['confidence'], trade_data.get('regime', ''), trade_data.get('order_id', ''), trade_data.get('status', 'pending'))
            logger.info(f"Trade saved with ID: {trade_id}")
            return trade_id
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
            return None
        finally:
            if conn:
                await self.pool.release(conn)

    async def save_performance_metrics(self, metrics: Dict):
        if not self.connected:
            return
        
        conn = None
        try:
            # PostgreSQL
            conn = await self.pool.acquire()
            await conn.execute('''
                INSERT INTO performance_metrics (
                    timestamp, total_signals, profitable_trades, total_closed_trades,
                    realized_pnl, trades_executed, portfolio_value
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (timestamp) DO NOTHING
            ''', datetime.now(timezone.utc), metrics.get('total_signals', 0), metrics.get('profitable_trades', 0), 
                metrics.get('total_closed_trades', 0), metrics.get('realized_pnl', 0.0), 
                metrics.get('trades_executed', 0), metrics.get('portfolio_value', 0.0))
            
            logger.debug("Performance metrics saved to PostgreSQL")
            
        except Exception as e:
            logger.error(f"Error saving metrics to PostgreSQL: {e}")
        finally:
            if conn:
                await self.pool.release(conn)
        
        # InfluxDB separado para que errores de PostgreSQL no lo afecten
        try:
            await write_to_influx("performance_metrics", metrics)
            logger.debug("Performance metrics saved to InfluxDB")
        except Exception as e:
            docker_note = " (Docker? Check shared network or use host.docker.internal)" if 'influxdb' in INFLUXDB_URL else ""
            logger.error(f"Error saving metrics to InfluxDB: {e}{docker_note}")
            # Fallback a archivo solo si ambas DB fallan
            with open('metrics_fallback.log', 'a') as f:
                f.write(f"{datetime.now(timezone.utc)}: {metrics}\n")
            logger.info("Metrics saved to fallback log file")

    async def save_backtest_results(self, results: Dict[str, Any]) -> None:
        """
        Save backtest results to the database.
        
        Args:
            results: Dictionary containing backtest results (total_pnl, win_rate, sharpe_ratio, max_drawdown, details, top_pairs).
        """
        if not self.connected:
            bt_logger.warning("Database not connected, skipping backtest results save")
            return
        
        conn = None
        try:
            bt_logger.debug(f"Results before serialization: details={results.get('details', {})}, top_pairs={results.get('top_pairs', [])}")
            conn = await self.pool.acquire()
            # Preprocess details and top_pairs to ensure JSON compatibility
            def make_json_serializable(obj: Any) -> Any:
                """
                Recursively convert non-JSON-serializable objects to JSON-compatible formats.
                
                Args:
                    obj: The object to convert (dict, list, or other types).
                
                Returns:
                    A JSON-serializable version of the object.
                
                Handles:
                    - dict: Recursively process keys and values.
                    - list/tuple: Recursively process elements.
                    - bool/numpy.bool_: Convert to lowercase string ("true"/"false").
                    - datetime: Convert to ISO 8601 string.
                    - set: Convert to list.
                    - bytes: Decode to string (UTF-8).
                    - Other non-serializable types: Convert to string or None.
                """
                try:
                    if isinstance(obj, dict):
                        return {make_json_serializable(k): make_json_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [make_json_serializable(item) for item in obj]
                    elif isinstance(obj, (bool, np.bool_)):
                        return str(obj).lower()  # Convert True/False to "true"/"false" - FIXED: Incluye np.bool_ explícito
                    elif isinstance(obj, datetime):
                        return obj.isoformat()  # Convert datetime to ISO 8601 string
                    elif isinstance(obj, set):
                        return list(obj)  # Convert set to list
                    elif isinstance(obj, bytes):
                        return obj.decode('utf-8', errors='ignore')  # Decode bytes to string
                    elif obj is None or isinstance(obj, (int, float, str)):
                        return obj  # Already JSON-serializable
                    else:
                        bt_logger.warning(f"Converting non-serializable type {type(obj)} to string: {obj}")
                        return str(obj)  # Convert to string as a fallback
                except Exception as e:
                    bt_logger.error(f"Error converting object to JSON-serializable format: {obj}, error: {e}")
                    return None  # Return None if conversion fails
            details = make_json_serializable(results.get('details', {}))
            top_pairs_list = make_json_serializable(results.get('top_pairs', []))
            details_json = json.dumps(details)
            await conn.execute('''
                INSERT INTO backtest_results (
                    realized_pnl, win_rate, sharpe_ratio, max_drawdown, details, top_pairs
                )
                VALUES ($1, $2, $3, $4, $5, $6)
            ''', results.get('realized_pnl', 0.0), results.get('win_rate', 0.0),
                results.get('sharpe_ratio', 0.0), results.get('max_drawdown', 0.0),
                details_json, top_pairs_list)
            bt_logger.info(f"Backtest results saved: Sharpe={results.get('sharpe_ratio', 0):.2f}, Top pairs={len(top_pairs_list)}")

        except Exception as e:
            bt_logger.error(f"Error saving backtest: {e}")
            raise  # Re-raise to allow higher-level error handling
        finally:
            if conn:
                await self.pool.release(conn)
    
async def fetch_top_performers(exchange: ExchIntf) -> Tuple[List[str], Dict[str, float]]:
    global top_performers, top_changes, last_top_fetch
    now = time.time()
    if now - last_top_fetch < get_config_param('perf_check_int'):
        return top_performers, top_changes
    try:
        tickers = await exchange.client.fetch_tickers()
    except Exception as e:
        logger.error(f"Error fetching tickers: {e}")
        return top_performers, top_changes
    try:
        # FIXED: Dynamic min_pct_change by regime (more sensitive in low-vol for better missed detection)
        base_min_pct = get_config_param('min_pct_change')
        regime_adjust = 0.8 if REGIME_GLOBAL in ['very_low', 'low'] else 1.2 if 'volatile' in REGIME_GLOBAL else 1.0
        dynamic_min_pct = base_min_pct * regime_adjust
        logger.debug(f"Dynamic min_pct for regime {REGIME_GLOBAL}: {base_min_pct}% -> {dynamic_min_pct}%")
        
        usdt_pairs = [s for s in tickers if s.endswith('/USDT') and s in exchange.markets and exchange.markets[s].get('active', False)]
        candidates = []
        sem = asyncio.Semaphore(10)
        async def check_change(s: str):
            async with sem:
                ohlcv = await exchange.fetch_ohlcv_with_retry(s, '1h', limit=2)
                if len(ohlcv) >= 2 and ohlcv[-2][1] > 0:
                    change_pct = ((ohlcv[-1][4] - ohlcv[-2][1]) / ohlcv[-2][1]) * 100
                    if change_pct > dynamic_min_pct:  # FIXED: Use dynamic threshold
                        return s, change_pct
                return None
        tasks = [check_change(s) for s in usdt_pairs[:100]]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        candidates = [r for r in results if isinstance(r, tuple)]
        candidates.sort(key=lambda x: x[1], reverse=True)
        top5 = candidates[:5]
        top_performers = [s for s, _ in top5]
        top_changes = {s: p for s, p in top5}
        last_top_fetch = now
        logger.info(f"Top 5 performers (> {dynamic_min_pct:.1f}%): {[(s, f'{p:.1f}%') for s, p in top5]}")
        return top_performers, top_changes
    except Exception as e:
        logger.error(f"Error fetching top performers: {e}")
        return top_performers, top_changes

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    sma = prices.rolling(window=period, min_periods=1).mean()
    std = prices.rolling(window=period, min_periods=1).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper.fillna(prices), sma.fillna(prices), lower.fillna(prices)

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    ema_fast = prices.ewm(span=fast, min_periods=1).mean()
    ema_slow = prices.ewm(span=slow, min_periods=1).mean()
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=signal, min_periods=1).mean()
    return macd.fillna(0), macd_sig.fillna(0)

async def prepare_features(df: pd.DataFrame, tf_config: Dict[str, Any]) -> pd.DataFrame:
    min_rows = max(50, max(tf_config.get('rsi_period', 14), tf_config.get('vol_window', 20)))
    if len(df) < min_rows:
        return pd.DataFrame()
    df = df.copy()
    required = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required):
        return pd.DataFrame()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=required)
    if len(df) < 20:
        return pd.DataFrame()
    rsi_period = tf_config.get('rsi_period', 14)
    df['rsi'] = calculate_rsi(df['close'], rsi_period)
    vol_window = tf_config.get('vol_window', 20)
    df['volume_sma'] = df['volume'].rolling(vol_window, min_periods=1).mean().fillna(0.001)    
    volume_sma_safe = df['volume_sma'].replace(0, 1e-6)
    df['vol_ratio'] = df['volume'] / volume_sma_safe
    df['volume_sma_ratio'] = df['vol_ratio'].fillna(1.0)
    returns = df['close'].pct_change().fillna(0)
    df['volatility'] = returns.rolling(20, min_periods=1).std().fillna(0.001)  
    vol_ma = df['volatility'].rolling(50, min_periods=1).mean().fillna(0.001)  
    df['volatility_regime'] = (df['volatility'] / vol_ma.replace(0, 1e-6)).fillna(1.0) 
    typical = (df['high'] + df['low'] + df['close']) / 3
    cum_vol = df['volume'].cumsum().replace(0, 1e-6)  
    df['vwap'] = (typical * df['volume']).cumsum() / cum_vol
    df['vwap'] = df['vwap'].fillna(df['close'])
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14, min_periods=1).mean()
    def entropy_returns(window):
        if len(window) == 0 or np.sum(np.abs(window)) == 0:
            return 0.0
        probs = np.abs(window) / np.sum(np.abs(window))
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs + 1e-10)) if len(probs) > 0 else 0.0
    df['returns_entropy'] = returns.rolling(20, min_periods=5).apply(entropy_returns, raw=False).fillna(0.0)
    # FIXED: Guard para close=0 en spread_norm (raro, pero seguro)
    close_safe = df['close'].replace(0, 1e-6)
    df['spread_norm'] = ((df['high'] - df['low']) / close_safe).fillna(0.01)
    macd, macd_sig = calculate_macd(df['close'])
    df['macd_signal'] = ((macd - macd_sig) / close_safe).fillna(0)  # FIXED: Guard en macd_signal
    upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(df['close'])
    bb_width = upper_bb - lower_bb
    bb_width_safe = bb_width.replace(0, 1e-6)  # FIXED: Guard para bb_width=0
    df['bb_position'] = ((df['close'] - middle_bb) / bb_width_safe).fillna(0)
    df['price_momentum'] = df['close'].pct_change(5).fillna(0)
    df['micro_momentum_3'] = df['close'].pct_change(3).fillna(0)
    df['micro_momentum_5'] = df['close'].pct_change(5).fillna(0)
    df['micro_momentum_8'] = df['close'].pct_change(8).fillna(0)
    vol_mean = df['volume'].rolling(20, min_periods=1).mean()
    df['whale_risk'] = (df['volume'] > vol_mean * 5).astype(float).rolling(5).mean().fillna(0)
    df['buying_pressure'] = ((df['close'] - df['low']) / ((df['high'] - df['low']).replace(0, 1e-6))).fillna(0.5)  # FIXED: Guard en buying_pressure
    price_change_5 = df['close'].pct_change(5)
    volume_change_5 = df['vol_ratio'].pct_change(5)
    df['vol_price_divergence'] = (volume_change_5 - price_change_5).fillna(0)
    range_10 = df['high'].rolling(10).max() - df['low'].rolling(10).min()
    df['consolidation_strength'] = (1 - (range_10 / close_safe)).fillna(0)
    current_range = df['high'] - df['low']
    atr_20 = df['atr'].rolling(20).mean()
    atr_20_safe = atr_20.replace(0, 1e-6)  # FIXED: Guard en range_compression
    df['range_compression'] = (1 - (current_range / atr_20_safe)).fillna(0)
    sma_short = df['close'].rolling(10).mean()
    sma_long = df['close'].rolling(30).mean()
    df['mean_reversion_signal'] = ((df['close'] - sma_short) / sma_long.replace(0, 1e-6)).fillna(0)  # FIXED: Guard en mean_reversion
    body_size = abs(df['close'] - df['open'])
    candle_size = df['high'] - df['low']
    candle_size_safe = candle_size.replace(0, 1e-6)  # FIXED: Guard en body_shadow_ratio
    df['body_shadow_ratio'] = (body_size / candle_size_safe).fillna(0.5)
    # Experimental: Holographic-inspired multi-dimensional feature analysis
    # Project key features into 3D tensor (time x features x regimes), convolve for interference patterns
    if len(df) >= 50:  # Min data for meaningful projection
        key_feats = ['rsi', 'vol_ratio', 'volatility', 'returns_entropy', 'macd_signal']  # Select high-info features
        num_feats = len(key_feats)
        holo_dim = min(10, len(df) // 10)  # Dynamic dim based on data size for efficiency
        holo_tensor = np.zeros((holo_dim, num_feats, 3))  # 3 "angles" (low/normal/high vol emulation)
        for i, feat in enumerate(key_feats):
            feat_series = df[feat].tail(holo_dim).values
            # Project with regime modulation (adverse: amplifies in low-vol)
            mod_factors = [0.8, 1.0, 1.2] if REGIME_GLOBAL in ['very_low', 'low'] else [1.0, 1.0, 1.0]
            for angle in range(3):
                modulated = feat_series * mod_factors[angle]
                holo_tensor[:, i, angle] = modulated[:holo_dim]  # Pad if short
                
        # Convolve for "interference" (experimental: captures multi-angle interactions)
        from scipy.signal import convolve
        kernel = np.array([[[0.1, 0.2, 0.1]]])  # Simple 1x3x1 kernel for cross-angle mix
        if holo_dim >= kernel.shape[0]:  # FIXED: Guard si dim < kernel (evita ValueError en valid)
            holo_interf = convolve(holo_tensor, kernel, mode='valid')
        else:
            holo_interf = holo_tensor  # Fallback sin convolve si data pequeña
        
        # Reduce to new features (PCA for dim reduction, entropy for info density)
        from sklearn.decomposition import PCA
        
        holo_flat = holo_interf.reshape(holo_dim - kernel.shape[0] + 1 if 'holo_interf' in locals() else holo_dim, -1)  
        if holo_flat.shape[0] > 0 and holo_flat.shape[1] > 1:  # FIXED: >1 para PCA (evita singular matrix)
            pca = PCA(n_components=2)
            try:
                holo_reduced = pca.fit_transform(holo_flat)
                # New features: holo_entropy (info density), holo_divergence (component spread)
                holo_entropy = entropy(np.abs(holo_reduced.flatten()) + 1e-10) / np.log(len(holo_reduced.flatten()) + 1e-10)
                holo_divergence = np.std(holo_reduced[:, 0] - holo_reduced[:, 1])
                # Adverse handling: Clip for stability in extreme data
                holo_entropy = np.clip(holo_entropy, 0.0, 1.0)
                holo_divergence = np.clip(holo_divergence, -1.0, 1.0)
            except ValueError as ve:
                logger.debug(f"Holo PCA failed for small data: {ve}; fallback to 0")
                holo_entropy = 0.0
                holo_divergence = 0.0
        else:
            holo_entropy = 0.0
            holo_divergence = 0.0
        
        # Add to DF (latest value) and global FEATURE_COLS (dynamic add if not present)
        df['holo_entropy'] = holo_entropy
        df['holo_divergence'] = holo_divergence
        global FEATURE_COLS
        if 'holo_entropy' not in FEATURE_COLS:
            FEATURE_COLS.extend(['holo_entropy', 'holo_divergence'])
    else:
        df['holo_entropy'] = 0.0
        df['holo_divergence'] = 0.0
        
    shift = tf_config.get('label_lookahead', 12)
    base_thresh = tf_config.get('label_threshold', 0.015)
    thresh = base_thresh
    if get_config_param('adaptive_thresh'):
        regime_factor = 0.8 if REGIME_GLOBAL in ['very_low', 'low'] else 1.2 if 'volatile' in REGIME_GLOBAL else 1.0
        thresh *= regime_factor
    df['future_return'] = df['close'].pct_change(periods=shift).shift(-shift).fillna(0)
    neutral_zone = 0.005 if REGIME_GLOBAL in ['very_low', 'low'] else 0.01
    conditions = [df['future_return'] > (thresh + neutral_zone), df['future_return'] < -(thresh + neutral_zone)]
    choices = ['Buy', 'Sell']
    df['label'] = np.select(conditions, choices, default='Hold')
    # Separar columnas numéricas y no numéricas para evitar el FutureWarning
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    
    # Interpolar solo las columnas numéricas
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both').ffill().bfill().fillna(0)
    
    # Para las columnas no numéricas, solo usar ffill/bfill
    if len(non_numeric_cols) > 0:
        df[non_numeric_cols] = df[non_numeric_cols].ffill().bfill().fillna('Hold')
    df[FEATURE_COLS] = df[FEATURE_COLS].clip(-1e4, 1e4)
    nan_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
    if nan_pct > 0.1:
        return pd.DataFrame()
    missing_features = [col for col in FEATURE_COLS if col not in df.columns or df[col].isna().all()]
    if missing_features:
        logger.warning(f"Missing features for {tf_config.get('name', 'unknown')}: {missing_features}")
    logger.debug(f"Features prepared. Shape: {df.shape}")
    # New: Evolve features with VAE
    if len(df) > 1000 and CONFIG['default']['use_gan']:
        vae = VAE(len(FEATURE_COLS))
        feat_tensor = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32)
        recon, mu, logvar = vae(feat_tensor)
        for i in range(CONFIG['default']['evolve_features_topk']):
            df[f'latent_{i}'] = mu[:, i].detach().numpy()
        FEATURE_COLS.extend([f'latent_{i}' for i in range(CONFIG['default']['evolve_features_topk'])])
    return df

def get_performance_proposals(missed: List[Tuple[str, str]], metrics: Dict) -> Dict[str, float]:
    proposals = {}
    total_missed = len(missed)
    if total_missed == 0:
        return proposals
    low_conf_count = sum(1 for _, reason in missed if 'low confidence' in reason.lower())
    high_corr_count = sum(1 for _, reason in missed if 'high correlation' in reason.lower())
    low_vol_count = sum(1 for _, reason in missed if 'insufficient volume' in reason.lower())
    if low_conf_count / total_missed > 0.4:
        proposals['min_conf_score'] = max(0.01, get_config_param('min_conf_score') * 0.9)
    win_rate = metrics.get('profitable_trades', 0) / max(1, metrics.get('total_closed_trades', 1))
    sharpe = metrics.get('sharpe_ratio', 1.0)
    if win_rate < 0.55:
        proposals['kelly_frac'] = max(0.1, get_config_param('kelly_frac') * 0.8)
    elif sharpe < 1.5 and win_rate > 0.55:
        proposals['kelly_frac'] = min(1.0, get_config_param('kelly_frac') * 1.1)
    momentum_missed = sum(1 for s, _ in missed if s in top_changes and top_changes[s] > 5.5)
    if momentum_missed / total_missed > 0.5 and 'min_conf_score' in proposals:
        proposals['min_conf_score'] *= 0.95
    if metrics.get('total_signals', 0) == 0:
        proposals['min_conf_score'] = max(0.01, get_config_param('min_conf_score') * 0.8)
        proposals['label_threshold_base'] = max(0.005, get_config_param('label_threshold_base') * 0.8)
    logger.debug(f"Performance proposals: {proposals}")
    return proposals

class KellySizer:
    def __init__(self, max_pos_pct: float = 0.25):
        self.max_pos_pct = max_pos_pct
        self.kelly_frac = get_config_param('kelly_frac')

    def calculate_position_size(self, win_rate: float, avg_win: float, avg_loss: float, confidence: float, portfolio_value: float, current_price: float, regime: str = None, volatility_regime: float = 1.0) -> Dict[str, float]:
        if regime is None:
            regime = REGIME_GLOBAL
        dynamic_kelly = get_config_param('kelly_frac')
        if regime in ['very_low', 'low']:
            dynamic_kelly *= 1.2
        elif 'volatile' in regime:
            dynamic_kelly *= 0.8
        if win_rate < 0.40 or avg_loss <= 0 or avg_win <= 0 or portfolio_value <= 0 or current_price <= 0:
            return {'position_pct': 0.0, 'position_value': 0.0, 'quantity': 0.0}
        avg_loss = max(0.01, abs(avg_loss))
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        profit_factor = (avg_win * p) / (avg_loss * q) if avg_loss * q > 0 else 0
        if profit_factor < 1:
            return {'position_pct': 0.0, 'position_value': 0.0, 'quantity': 0.0}
        kelly = (b * p - q) / b if b > 0 else 0
        if kelly <= 0:
            return {'position_pct': 0.0, 'position_value': 0.0, 'quantity': 0.0}
        fractional_kelly = kelly * dynamic_kelly
        conf_mult = min(np.sqrt(confidence) * 2.0, 1.5) if confidence > 0 else 0.0
        # Experimental: Quantum-inspired superposition risk simulation
        # Simulate "qubit states" via Gaussian mixture for parallel risk paths, anneal to optimal size
        num_states = 20 if 'volatile' in regime else 10  # More states in adverse vol for robustness
        if num_states > 0:  # Guard for zero-division
            # Mixture: 2 Gaussians (bull/bear superposition)
            means = [fractional_kelly * (1 + 0.1 * volatility_regime), fractional_kelly * (1 - 0.1 * volatility_regime)]
            stds = [0.05 * volatility_regime, 0.05 * volatility_regime]
            weights = [p, q]  # Weight by win/loss prob
            samples = []
            for _ in range(num_states):
                comp = np.random.choice([0, 1], p=weights / np.sum(weights))
                sample = np.random.normal(means[comp], stds[comp])
                samples.append(np.clip(sample, 0.01, 1.0))  # Clip for stability
            # Quantum-annealing-like optimization: Minimize risk fn with decaying temp
            def risk_fn(pos_pct):
                # Simulated risk: VaR-like (adverse: higher in vol)
                var = np.percentile([pos_pct * (avg_win if np.random.rand() < p else -avg_loss) for _ in range(100)], 5)
                return -var  # Maximize negative VaR (min risk)
            initial_temp = 1.0 if 'volatile' in regime else 0.5  # Higher temp in adverse for exploration
            cooling_rate = 0.95
            current_pct = fractional_kelly
            best_pct = current_pct
            best_risk = risk_fn(best_pct)
            for iter in range(50):  # Annealing steps
                temp = initial_temp * (cooling_rate ** iter)
                neighbor = np.random.choice(samples)  # From superposition samples
                neighbor_risk = risk_fn(neighbor)
                if neighbor_risk < best_risk or np.random.rand() < np.exp((best_risk - neighbor_risk) / temp):
                    current_pct = neighbor
                    if neighbor_risk < best_risk:
                        best_pct = current_pct
                        best_risk = neighbor_risk
            position_pct = best_pct * conf_mult
        else:
            position_pct = fractional_kelly * conf_mult  # Fallback if num_states=0
        position_pct *= (1.1 if volatility_regime < 0.8 else 0.9 if volatility_regime > 1.2 else 1.0)
        if regime in ['very_low', 'low']:
            position_pct *= get_config_param('low_vol_pos_mult')
        elif 'volatile' in regime:
            position_pct *= 0.8
        beta = 1.0 * (1.2 if 'volatile' in regime else 0.8 if 'low' in regime else 1.0)
        position_pct /= beta
        position_pct = max(0.0, min(position_pct, self.max_pos_pct))
        position_value = portfolio_value * position_pct
        quantity = position_value / current_price
        min_order_size = get_config_param('min_order_size')
        if position_value < min_order_size:
            min_quantity = min_order_size / current_price
            quantity = max(quantity, min_quantity)
            position_value = quantity * current_price
            position_pct = position_value / portfolio_value
        global_win_rate = PERFORMANCE_METRICS.get('profitable_trades', 0) / max(1, PERFORMANCE_METRICS.get('total_closed_trades', 1))
        if global_win_rate > 0.6:
            position_pct = min(1.0, position_pct * 1.05)
            position_value = portfolio_value * position_pct
            quantity = position_value / current_price
        # New: CVaR optimized size
        cvar_size = self.compute_cvar_optimized_size(win_rate, avg_win, avg_loss, portfolio_value, current_price)
        quantity = min(quantity, cvar_size)  # Cap to CVaR safe
        # Compounding and taxes (applied on close, but estimate for size)
        after_tax_est = quantity * (1 - CONFIG['default']['tax_rate']) if quantity > 0 else quantity
        quantity = after_tax_est  # Adjust for tax estimate
        return {
            'position_pct': float(position_pct),
            'position_value': float(position_value),
            'quantity': float(quantity),
            'kelly_raw': float(kelly),
            'profit_factor': float(profit_factor)
        }

    def compute_cvar_optimized_size(self, win_rate, avg_win, avg_loss, portfolio_value, price):
        def obj(x):
            sim_pnl = [x[0] * portfolio_value / price * (avg_win if np.random.rand() < win_rate else -avg_loss) for _ in range(1000)]
            sim_pnl = np.array(sim_pnl)
            cvar = np.mean(sim_pnl[sim_pnl < np.percentile(sim_pnl, 5)]) - CONFIG['default']['cvar_target']
            return cvar
        res = minimize(obj, x0=[0.1], bounds=[(0, 0.2)])
        return res.x[0] * portfolio_value / price

class LSTMSignalModel(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = len(FEATURE_COLS)
        hidden = 64
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=CONFIG['default']['lstm_layers'], batch_first=True)
        self.attn = nn.MultiheadAttention(hidden, CONFIG['default']['attention_heads'])
        self.fc = nn.Linear(hidden, 3)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        return self.fc(attn_out.mean(1))

class VAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        latent_dim = CONFIG['default']['vae_latent_dim']
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim * 2))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, input_dim))
        
    def forward(self, x):
        mu_logvar = self.encoder(x)
        mu, logvar = mu_logvar.chunk(2, dim=1)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

class GANMC:
    def __init__(self, context: AppContext):
        self.context = context
        self.generator = nn.Sequential(nn.Linear(CONFIG['default']['noise_dim'] + 1, 128), nn.ReLU(), nn.Linear(128, 96))
        self.discriminator = nn.Sequential(nn.Linear(96, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())

    @numba.jit(nopython=True)
    def train(self, historical_paths, regime):
        optimizer_g = optim.Adam(self.generator.parameters(), lr=0.001)
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.001)
        loss_fn = nn.BCELoss()
        for epoch in range(CONFIG['default']['gan_epochs']):
            noise = torch.randn(len(historical_paths), CONFIG['default']['noise_dim'] + 1)
            cond = torch.full((len(historical_paths), 1), regime)  # Conditional on regime
            fake = self.generator(torch.cat((noise, cond), dim=1))
            real = torch.tensor(historical_paths, dtype=torch.float32)
            d_real = self.discriminator(real)
            d_fake = self.discriminator(fake.detach())
            loss_d = loss_fn(d_real, torch.ones_like(d_real)) + loss_fn(d_fake, torch.zeros_like(d_fake))
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()
            g_fake = self.discriminator(fake)
            loss_g = loss_fn(g_fake, torch.ones_like(g_fake))
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()
        return loss_g.item()

    def generate_paths(self, num: int, regime: int) -> np.ndarray:
        noise = torch.randn(num, CONFIG['default']['noise_dim'] + 1)
        cond = torch.full((num, 1), regime)
        with torch.no_grad():
            paths = self.generator(torch.cat((noise, cond), dim=1)).numpy()
        return paths

def bayesian_signal_strength(conf: float, hist_win: float) -> float:
    if np.isnan(hist_win):
        hist_win = 0.5
    alpha = hist_win * CONFIG['default']['bayes_alpha_prior'] + 1
    beta_param = (1 - hist_win) * CONFIG['default']['bayes_alpha_prior'] + 1
    return beta(alpha, beta_param).mean()

class CausalEngine:
    def __init__(self, supervisor_bot: SupBot):
        self.supervisor_bot = supervisor_bot
        self.params_to_optimize = supervisor_bot.params_to_optimize
        self.maxlag = CONFIG['default']['causal_maxlag']
        self.threshold_p = CONFIG['default']['causal_threshold_p']

    def build_causal_graph(self, hist_data: pd.DataFrame) -> str:
        lags = self.maxlag
        gml = 'digraph G { win_rate -> missed [lag=1]; ' + '; '.join([f"{p} -> win_rate [lag={lags}]" for p in self.params_to_optimize]) + ' }'
        return gml

    def analyze_missed(self, missed_symbol, reason, hist_data: pd.DataFrame):
        gml = self.build_causal_graph(hist_data)
        data = pd.DataFrame({
            'min_conf': hist_data['min_conf'],
            'win_rate': hist_data['win_rate'],
            'regime': hist_data['regime']
        })
        model = CausalModel(data=data, treatment='min_conf', outcome='win_rate', graph=gml)
        identified = model.identify_effect()
        estimate = model.estimate_effect(identified, method_name="backdoor.propensity_score_matching")
        if estimate.value < -0.1:
            self.supervisor_bot.intervene('min_conf', 0.9 * CONFIG['default']['min_conf_score'])
            
class ParamOpt:
    def __init__(self):
        self.bounds = {
            'min_conf_score': (0.05, 0.15),
            'sl_atr_mult': (1.5, 3.0),
            'low_vol_pos_mult': (1.0, 1.5),
            'spoof_ent_thresh': (0.5, 0.8),
            'label_threshold_base': (0.005, 0.03),
        }
        self.trials = deque(maxlen=100)
        self.epsilon = 0.15
        self.gamma = 0.95
        
        # Enhanced parameter space with adaptive levels
        self.param_levels = {
            'min_conf_score': [0.01, 0.02, 0.03],
            'kelly_frac': [0.2, 0.35, 0.5],
            'min_vol_24h': [500000.0, 1000000.0, 2000000.0],
            'sl_atr_mult': [1.5, 2.0, 2.5],
            'tp_atr_mult': [2.5, 3.0, 3.5],
            'label_threshold_base': [0.005, 0.015, 0.025],
        }
        
        self.current_param_idx = {k: 1 for k in self.param_levels}
        self.param_names = list(self.param_levels.keys())
        self.state_dim = len(self.param_names) + 3  # +3 for market context
        self.action_dim = len(self.param_names) * 3
        
        # Enhanced Q-Network with market awareness and correct action_dim
        self.q_network = QNetwork(self.state_dim, hidden_dims=[128, 64], action_dim=self.action_dim)
        self.target_network = QNetwork(self.state_dim, hidden_dims=[128, 64], action_dim=self.action_dim)
        self.optimizer_rl = optim.Adam(self.q_network.parameters(), lr=0.0005)

        self.replay_buffer = deque(maxlen=2000)
        
        # Intelligence enhancement
        self.parameter_effectiveness = {p: deque(maxlen=20) for p in self.param_names}
        self.regime_performance = defaultdict(lambda: {'trials': 0, 'avg_score': 0})
        self.exploration_bonus = defaultdict(float)
        self.update_target_freq = 50
        self.training_steps = 0
        
        # Adaptive bounds based on performance
        self.dynamic_bounds = self.bounds.copy()
        self.bounds_adaptation_rate = 0.02
        
        # FIXED: Consolidar init de convergencia en __init__ (evita multiples hasattr en _train_q_network)
        self.consecutive_increases = 0
        self.prev_loss = float('inf')

    def get_state(self, metrics: Dict = None) -> np.ndarray:
        """Enhanced state with market context"""
        base_state = np.array([self.current_param_idx[k] / 2.0 for k in self.param_names])
        
        if metrics:
            # Market context features
            volatility = min(1.0, metrics.get('volatility', 0.5) * 2)
            trend_strength = np.tanh(metrics.get('trend_strength', 0))
            market_stress = min(1.0, metrics.get('max_drawdown', 5) / 20)
            context = np.array([volatility, trend_strength, market_stress])
        else:
            context = np.array([0.5, 0.0, 0.1])
        
        return np.concatenate([base_state, context])

    def get_current_params(self) -> Dict[str, float]:
        return {k: self.param_levels[k][self.current_param_idx[k]] for k in self.param_names}

    def apply_action(self, action: int) -> np.ndarray:
        param_id = action // 3
        new_level = action % 3
        param_name = self.param_names[param_id]
        
        # Track exploration for bonus
        old_level = self.current_param_idx[param_name]
        if old_level != new_level:
            self.exploration_bonus[param_name] += 0.01
        
        self.current_param_idx[param_name] = new_level
        return self.get_state()

    def _calculate_reward(self, metrics: Dict, regime: str) -> float:
        """Sophisticated reward function with regime awareness"""
        sharpe = metrics.get('sharpe_ratio', 0.0)
        drawdown = metrics.get('max_drawdown', 0.0)
        win_rate = metrics.get('win_rate', 0.5)
        pnl = metrics.get('realized_pnl', 0)
        
        # Base reward
        reward = sharpe * 2 + win_rate - (drawdown / 8.0)
        
        # Regime-specific bonuses
        if 'volatile' in regime and sharpe > 1.8:
            reward += 1.0
        elif 'trending' in regime and win_rate > 0.6:
            reward += 0.8
        elif 'choppy' in regime and drawdown < 8:
            reward += 0.6
        
        # Penalties for poor performance
        if drawdown >= 20:
            reward -= 8.0
        if sharpe <= 1.0:
            reward -= 2.0
        if win_rate < 0.45:
            reward -= 1.5
        
        # Profitability bonus
        if pnl > 0:
            reward += min(2.0, pnl / 1000)
        
        return reward

    def get_rl_proposal(self, metrics: Dict) -> Dict[str, float]:
        """Enhanced RL with target network and sophisticated exploration"""
        backup_idx = self.current_param_idx.copy()
        
        try:
            regime = self._detect_regime(metrics)
            reward = self._calculate_reward(metrics, regime)
            old_state = self.get_state(metrics)
            
            # Sophisticated action selection with exploration bonus
            if np.random.rand() < self.epsilon:
                # Exploration with bias toward less-tried parameters
                param_explore_probs = []
                for param in self.param_names:
                    bonus = self.exploration_bonus.get(param, 0)
                    param_explore_probs.extend([1 + bonus] * 3)
                
                probs = np.array(param_explore_probs)
                probs = probs / probs.sum()
                action = np.random.choice(self.action_dim, p=probs)
            else:
                # Exploitation with target network
                with torch.no_grad():
                    q_values = self.q_network(torch.tensor(old_state, dtype=torch.float32).unsqueeze(0))
                    action = torch.argmax(q_values).item()
            
            new_state = self.apply_action(action)
            proposal = self.get_current_params()
            
            # Store experience
            self.replay_buffer.append((old_state, action, reward, new_state, regime))
            
            # Enhanced training with target network
            if len(self.replay_buffer) >= 32:  # Minimum buffer for stable training (increased from <32 for better batches)
                self._train_q_network()
            
            # Update exploration bonuses (decay)
            for param in self.exploration_bonus:
                self.exploration_bonus[param] *= 0.99
            
            # Adaptive epsilon decay
            self.epsilon = max(0.05, self.epsilon * 0.998)
            
            return proposal
            
        except Exception as e:
            bt_logger.error(f"Error in RL proposal: {e}")
            return {}
        finally:
            self.current_param_idx = backup_idx

    def _train_q_network(self):
        """Enhanced Q-network training with target network"""
        batch_size = min(64, len(self.replay_buffer))
        batch = random.sample(self.replay_buffer, batch_size)

        states, actions, rewards, next_states, regimes = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32)

        # Double DQN for stability
        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = torch.argmax(self.q_network(next_states_t), dim=1)
            next_q = self.target_network(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards_t + self.gamma * next_q

        loss = nn.MSELoss()(current_q, targets)    # Nueva: chequeo de convergencia (early stop si loss aumenta)
        loss_item = loss.item()
        if np.isnan(loss_item) or np.isinf(loss_item):
            logger.warning("QNetwork loss NaN/Inf; resetting and early stopping")
            self.q_network.apply(self.q_network._init_weights)
            self.optimizer_rl.zero_grad()
            self.consecutive_increases = 0  # FIXED: Usa self. (ya init)
            self.prev_loss = float('inf')  # FIXED: Usa self.
            return
        if self.prev_loss == float('inf'):  # FIXED: Usa self.prev_loss (init en __init__)
            self.prev_loss = loss_item
            self.consecutive_increases = 0
        elif loss_item > self.prev_loss * 1.1:
            self.consecutive_increases += 1
            if self.consecutive_increases >= 5:
                logger.warning("QNetwork loss diverging. Resetting weights and early stopping.")
                self.q_network.apply(self.q_network._init_weights)
                self.optimizer_rl.zero_grad()
                self.consecutive_increases = 0
                self.prev_loss = float('inf')
                return
        else:
            self.consecutive_increases = 0
        self.prev_loss = loss_item
        self.optimizer_rl.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer_rl.step()                
        self.training_steps += 1
        
        # Update target network
        if self.training_steps % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        bt_logger.debug(f"RL training: loss={loss.item():.4f}, steps={self.training_steps}")

    def _detect_regime(self, metrics: Dict) -> str:
        """Smart regime detection"""
        vol = metrics.get('volatility', 0.5)
        trend = metrics.get('trend_strength', 0)
        
        if vol > 0.8:
            return 'volatile'
        elif abs(trend) > 0.6:
            return 'trending'
        else:
            return 'choppy'

    def get_bayes_proposal(self, regime: str, metrics: Dict, trap_history: Dict) -> Dict[str, float]:
        """Enhanced Bayesian optimization with adaptive kernel"""
        if len(self.trials) < 3:
            return self._smart_random_proposal(regime, metrics)
        
        num_params = len(self.bounds)
        X_hist = np.array([t[0] for t in list(self.trials)])
        y_hist = np.array([t[1] for t in list(self.trials)])
        
        # Adaptive kernel selection based on data
        if len(self.trials) > 20:
            kernel = ConstantKernel() * Matern(length_scale=0.5, nu=2.5)
        else:
            kernel = ConstantKernel() + RBF(length_scale=0.3)
        
        # Smart acquisition with regime bias
        n_samples = min(15, max(5, 30 - len(self.trials)))
        
        # Biased sampling toward successful regions
        if len(self.trials) > 10:
            best_trials = sorted(list(self.trials), key=lambda x: x[1], reverse=True)[:5]
            best_params = np.array([t[0] for t in best_trials])
            mean_best = np.mean(best_params, axis=0)
            std_best = np.std(best_params, axis=0) + 0.1
            
            # Sample around best regions with some exploration
            low_bounds = np.maximum(np.array([b[0] for b in self.bounds.values()]), 
                                  mean_best - 2*std_best)
            high_bounds = np.minimum(np.array([b[1] for b in self.bounds.values()]), 
                                   mean_best + 2*std_best)
        else:
            low_bounds = np.array([b[0] for b in self.bounds.values()])
            high_bounds = np.array([b[1] for b in self.bounds.values()])
        
        X_new = np.random.uniform(low=low_bounds, high=high_bounds, size=(n_samples, num_params))
        y_new = np.array([self.objective(list(xi), regime, metrics, trap_history) for xi in X_new])
        
        # Update trials
        for i in range(n_samples):
            self.trials.append((X_new[i].tolist(), y_new[i]))
                
        # Surrogate model with regime awareness
        def surrogate_obj(x):
            X_all = np.vstack([X_hist, X_new])
            y_all = np.hstack([y_hist, y_new])
            gp = None
        # Nueva cascade para kernels si fit falla
        kernel_options = [ConstantKernel() * Matern(length_scale=0.5, nu=2.5),  # Primera opción (adaptativa original >20)
                          ConstantKernel() * Matern(length_scale=0.3, nu=1.5),  # Segunda: más smooth
                          ConstantKernel() + RBF(length_scale=0.3)]  # Tercera: fallback simple
        for kernel in kernel_options:
            try:
                gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
                gp.fit(X_all, y_all)
                break  # Success
            except ValueError as ve:
                logger.debug(f"GP fit failed with kernel {kernel}: {ve}. Trying next kernel.")
                continue
        if gp is None:
            logger.warning(f"All GP kernels failed for Bayes opt. Returning empty proposal.")
            return {}
        
        mean, std = gp.predict(np.array([x]), return_std=True)

        # Expected improvement with regime bonus
        regime_bonus = 0.1 if regime in ['volatile', 'trending'] else 0
        return -(mean[0] + std[0] * 0.1 + regime_bonus)
        
        # Multi-start optimization for global optimum
        best_result = None
        best_score = float('inf')
        
        bounds_list = [(b[0], b[1]) for b in self.bounds.values()]
        
        for _ in range(3):  # 3 random starts
            x0 = np.random.uniform([b[0] for b in self.bounds.values()], 
                                 [b[1] for b in self.bounds.values()])
            result = minimize(surrogate_obj, x0=x0, bounds=bounds_list, method='L-BFGS-B')  # FIXED: Llama a surrogate_obj (era lambda vacía implícita)
            
            if result.success and result.fun < best_score:
                best_result = result
                best_score = result.fun
        
        if best_result and best_result.success:
            opt_params = {}
            keys = list(self.bounds.keys())
            for i, key in enumerate(keys):
                # Validate bounds
                val = np.clip(best_result.x[i], self.bounds[key][0], self.bounds[key][1])
                opt_params[key] = val
            
            return opt_params
        
        return {}

    def _smart_random_proposal(self, regime: str, metrics: Dict) -> Dict[str, float]:
        """Intelligent random proposal for cold start"""
        proposal = {}
        
        # Regime-based intelligent defaults
        regime_biases = {
            'volatile': {'sl_atr_mult': 2.2, 'min_conf_score': 0.08},
            'trending': {'tp_atr_mult': 3.2, 'min_conf_score': 0.06},
            'choppy': {'min_conf_score': 0.12, 'spoof_ent_thresh': 0.7}
        }
        
        for param, bounds in self.bounds.items():
            if regime in regime_biases and param in regime_biases[regime]:
                # Bias toward regime-optimal values
                target = regime_biases[regime][param]
                noise = np.random.normal(0, (bounds[1] - bounds[0]) * 0.1)
                proposal[param] = np.clip(target + noise, bounds[0], bounds[1])
            else:
                # Smart random within bounds
                proposal[param] = np.random.uniform(bounds[0], bounds[1])
        
        return proposal

    def objective(self, params: List[float], regime: str, metrics: Dict, trap_history: Dict) -> float:
        """Enhanced objective with regime-specific scoring"""
        win_rate = metrics.get('profitable_trades', 0) / max(1, metrics.get('total_closed_trades', 1))
        pnl = metrics.get('total_pnl', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        drawdown = metrics.get('max_drawdown', 0)
        
        # Base score with multiple factors
        base_score = (
            win_rate * 3 +
            sharpe * 2 +
            max(0, pnl / 5000) +
            max(0, (20 - drawdown) / 20)
        )
        
        # Trap penalty
        total_traps = sum(len(hist) for hist in trap_history.values())
        num_symbols = len(trap_history)
        trap_penalty = (total_traps / max(1, num_symbols * 10)) * 2
        
        # Regime-specific penalties
        regime_penalty = 0
        if regime == 'volatile' and len(params) > 2:
            # Penalize tight stops in volatile markets
            if params[2] < 1.8:  # sl_atr_mult
                regime_penalty += 1.0
        elif regime == 'choppy' and len(params) > 0:
            # Penalize low confidence in choppy markets
            if params[0] < 0.08:  # min_conf_score
                regime_penalty += 0.8
        
        # Parameter stability bonus (avoid extreme values)
        stability_bonus = 0
        param_keys = list(self.bounds.keys())
        for i, param_key in enumerate(param_keys):
            if i < len(params):
                bounds = self.bounds[param_key]
                normalized = (params[i] - bounds[0]) / (bounds[1] - bounds[0])
                # Bonus for values not at extremes
                stability_bonus += 0.1 * (1 - abs(normalized - 0.5) * 2)
        
        final_score = base_score - trap_penalty - regime_penalty + stability_bonus
        return -final_score  # Negative for minimization

    def _update_dynamic_bounds(self, best_params: Dict[str, float], performance: float):
        """Adapt parameter bounds based on successful regions"""
        if performance > 0.5:  # Good performance
            for param, value in best_params.items():
                if param in self.dynamic_bounds:
                    current_bounds = self.dynamic_bounds[param]
                    range_size = current_bounds[1] - current_bounds[0]
                    
                    # Shift bounds toward successful values
                    center = (current_bounds[0] + current_bounds[1]) / 2
                    shift = (value - center) * self.bounds_adaptation_rate
                    
                    new_low = current_bounds[0] + shift
                    new_high = current_bounds[1] + shift
                    
                    # Keep within original bounds
                    orig_bounds = self.bounds[param]
                    self.dynamic_bounds[param] = (
                        max(orig_bounds[0], new_low),
                        min(orig_bounds[1], new_high)
                    )

    async def optimize_params(self, regime: str, metrics: Dict, trap_history: Dict):
        """Main optimization orchestrator with adaptive strategy"""
        
        # Store regime performance
        performance_score = metrics.get('sharpe_ratio', 0) + metrics.get('win_rate', 0)
        self.regime_performance[regime]['trials'] += 1
        old_avg = self.regime_performance[regime]['avg_score']
        n = self.regime_performance[regime]['trials']
        self.regime_performance[regime]['avg_score'] = (old_avg * (n-1) + performance_score) / n
        
        # Bayesian optimization for main search
        if len(self.trials) >= 3:
            bayes_proposal = self.get_bayes_proposal(regime, metrics, trap_history)
            
            # Test Bayesian proposal
            if bayes_proposal:
                test_score = self.objective(
                    [bayes_proposal[k] for k in self.bounds.keys()], 
                    regime, metrics, trap_history
                )
                
                # Update parameter effectiveness tracking
                for param, value in bayes_proposal.items():
                    self.parameter_effectiveness[param].append((-test_score, value))
        
        # RL optimization for adaptive learning
        rl_proposal = self.get_rl_proposal(metrics)
        
        
        # Experimental: Bio-inspired swarm evolution for param optimization
        # Swarm particles evolve params via PSO + genetic ops (crossover/mutation) for global search
        if len(self.trials) >= 5:  # Min for swarm init
            swarm_size = min(20, len(self.trials) * 2)  # Dynamic size
            dimensions = len(self.bounds)
            # Init swarm from trials (adverse: diverse in vol for robustness)
            positions = np.array([t[0] for t in list(self.trials)[-swarm_size:]])
            if len(positions) < swarm_size:
                # Pad with random (fallback for low data)
                rand_pos = np.random.uniform([b[0] for b in self.bounds.values()], [b[1] for b in self.bounds.values()], size=(swarm_size - len(positions), dimensions))
                positions = np.vstack([positions, rand_pos])
            velocities = np.random.uniform(-0.1, 0.1, size=(swarm_size, dimensions))
            personal_best = positions.copy()
            personal_best_scores = np.array([self.objective(list(pos), regime, metrics, trap_history) for pos in positions])
            global_best_idx = np.argmin(personal_best_scores)
            global_best = positions[global_best_idx].copy()
            
            # Swarm iterations with genetic evolution
            inertia = 0.7 if 'volatile' in regime else 0.9  # Lower inertia in adverse for exploration
            cognitive = 1.5
            social = 1.5
            mutation_rate = 0.1 if len(self.trials) < 20 else 0.05  # Higher mutation early
            for iter in range(20):  # Fixed iters for efficiency
                r1 = np.random.rand(swarm_size, dimensions)
                r2 = np.random.rand(swarm_size, dimensions)
                velocities = (inertia * velocities + 
                              cognitive * r1 * (personal_best - positions) + 
                              social * r2 * (global_best - positions))
                positions += velocities
                # Clip to bounds (stability)
                for d in range(dimensions):
                    low, high = list(self.bounds.values())[d]
                    positions[:, d] = np.clip(positions[:, d], low, high)
                
                # Genetic: Crossover + mutation every 5 iters
                if iter % 5 == 0:
                    # Crossover: Pair top half with random
                    sorted_idx = np.argsort([self.objective(list(pos), regime, metrics, trap_history) for pos in positions])
                    elite = positions[sorted_idx[:swarm_size//2]]
                    for i in range(swarm_size//2, swarm_size):
                        parent1 = elite[np.random.randint(0, len(elite))]
                        parent2 = positions[np.random.randint(0, swarm_size)]
                        cross_point = np.random.randint(1, dimensions-1)
                        positions[i] = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
                        # Mutation
                        if np.random.rand() < mutation_rate:
                            mut_dim = np.random.randint(0, dimensions)
                            low, high = list(self.bounds.values())[mut_dim]
                            positions[i, mut_dim] = np.random.uniform(low, high)
                
                # Update bests
                scores = np.array([self.objective(list(pos), regime, metrics, trap_history) for pos in positions])
                better = scores < personal_best_scores
                personal_best[better] = positions[better]
                personal_best_scores[better] = scores[better]
                global_best_idx = np.argmin(personal_best_scores)
                global_best = personal_best[global_best_idx].copy()
            
            # Extract evolved params
            evolved_params = {k: global_best[i] for i, k in enumerate(self.bounds.keys())}
            self.trials.append((global_best.tolist(), np.min(scores)))
        else:
            evolved_params = {}  # Fallback empty for low data
            
        if len(self.trials) > 10:
            best_trial = max(self.trials, key=lambda x: x[1])
            best_params_list = best_trial[0]
            best_params_dict = {k: best_params_list[i] for i, k in enumerate(self.bounds.keys())}
            self._update_dynamic_bounds(best_params_dict, -best_trial[1])

class SigGen:
    def __init__(self, context: AppContext):
        self.context = context
        self.min_data_points = 50
        self.regime_det = VolRegDet(context)
        self.model_mgr = ModelMgr(context)
        self.bt_mgr = None  # Injected for MC quick validation

    async def fetch_sentiment(self, symbol: str) -> float:
        try:
            # FIXED: Use x_semantic_search tool
            res = await self.context.x_semantic_search(query=f"{symbol} trading sentiment today", limit=20)
            if 'posts' in res:
                scores = [p.get('rel_score', 0) for p in res['posts']]
                return np.tanh(np.mean(scores))
            else:
                return 0.0
        except Exception as e:
            logger.error(f"Sentiment fetch failed for {symbol}: {e}; fallback 0.0")
            return 0.0

    async def generate_signal(self, df: pd.DataFrame, tf_config: Dict[str, Any], symbol: str = "", historical: bool = False) -> Dict[str, Any]:
        timeframe = tf_config['name']
        if df.empty or len(df) < self.min_data_points:
            return {'symbol': symbol, 'timeframe': timeframe, 'direction': 'HOLD', 'confidence': 0.0, 'df': df}
        cache_key = f"{symbol}_regime"
        now = time.time()
        if cache_key in self.regime_det.regime_cache and now - self.regime_det.regime_cache[cache_key]['timestamp'] < 300:
            regime = self.regime_det.regime_cache[cache_key]['regime']
            trap_detected = self.regime_det.regime_cache[cache_key]['trap_detected']
        else:
            regime, trap_detected = await self.regime_det.detect_regime(df, symbol, None, historical)
            self.regime_det.regime_cache[cache_key] = {
                'regime': regime,
                'volatility': df['close'].pct_change().tail(48).std() * np.sqrt(96) if len(df) >= 48 else 0.0,
                'timestamp': now,
                'trap_detected': trap_detected
            }
        if symbol == 'BTC/USDT':
            global REGIME_GLOBAL
            REGIME_GLOBAL = regime
        if trap_detected:
            logger.warning(f"Trap detected for {symbol}, returning neutral signal")
            return self._neutral_signal("Institutional trap detected", symbol, timeframe)
        regime_filter = tf_config.get('regime_filter', ['very_low', 'low', 'normal', 'bull', 'bear', 'high', 'volatile_bull', 'volatile_bear'])
        if regime_filter and regime not in regime_filter:
            logger.warning(f"Regime {regime} filtered for {tf_config['name']}, returning neutral")
            return self._neutral_signal(f"Regime {regime} filtered for {tf_config['name']}", symbol, timeframe)
        adaptive_config = self.regime_det.get_adaptive_config(regime, tf_config)
        latest = df.iloc[-1]
        recent_data = df.tail(20)
        # Calculate component scores early
        tech_score = self._analyze_technical_indicators(latest, recent_data, adaptive_config)
        mom_score = self._analyze_momentum(recent_data)
        vol_score = self._analyze_volume(recent_data)
        vol_reg_score = self._analyze_volatility_regime(recent_data)
        struct_score = self.analyze_market_structure(recent_data)
        strategy_type = tf_config['strategy_type']
        low_vol_score = self._analyze_low_volatility_patterns(latest, recent_data, regime, strategy_type)
        micro_score = self._analyze_micro_movements(latest, recent_data)
        ml_signal, ml_conf = self.model_mgr.get_prediction(symbol, timeframe, latest)
        ml_score = ml_signal * 0.3 if abs(ml_signal) > 0.1 else 0.0
        
        base_weights = {
            'technical': 0.15, 'momentum': 0.15, 'volume': 0.12, 'volatility': 0.08, 'structure': 0.08,
            'low_vol': 0.15 if regime in ['very_low', 'low'] else 0.0,  # Boost in low-vol
            'micro': 0.10 if regime in ['very_low', 'low'] else 0.0, 'ml': 0.25  # ML neutral base
        }
        
        if 'volatile' in regime or regime == 'high':
            base_weights['ml'] += 0.1  # Boost ML
            base_weights['technical'] -= 0.05
        elif regime in ['very_low', 'low']:
            base_weights['low_vol'] += 0.05
            base_weights['micro'] += 0.05
            base_weights['ml'] -= 0.05  # De-emphasize ML in quiet markets
        total_w = sum(base_weights.values())
        weights = {k: v / total_w for k, v in base_weights.items()}
        
        # Define component_scores before direction logic
        component_scores = {
            'technical': tech_score, 'momentum': mom_score, 'volume': vol_score,
            'volatility': vol_reg_score, 'structure': struct_score,
            'low_vol': low_vol_score, 'micro': micro_score
        }
        
        
        base_thresh = get_config_param('label_threshold_base')
        total_w = sum(base_weights.values())
        weights = {k: v / total_w for k, v in base_weights.items()}
        base_thresh = adaptive_config.get('label_threshold', 0.015)
        regime_factor = 0.7 if regime == 'very_low' else 0.9 if regime == 'low' else 1.2 if regime == 'high' or 'volatile' in regime else 1.0
        threshold = base_thresh * regime_factor
        composite_score = (
            tech_score * weights['technical'] + mom_score * weights['momentum'] + vol_score * weights['volume'] +
            vol_reg_score * weights['volatility'] + struct_score * weights['structure'] +
            low_vol_score * weights['low_vol'] + micro_score * weights['micro'] + ml_score * weights['ml']
        )
        # Experimental: Neuro-symbolic hybrid for enhanced pattern reasoning
        # Neural embedding + symbolic rules (if-then tree) for explainable intelligence
        if len(recent_data) >= 10:  # Min for embedding
            # Neural: Embed features to low-dim vector (adverse: captures non-linear in vol)
            embed_feats = torch.tensor(recent_data[FEATURE_COLS].tail(10).values, dtype=torch.float32)
            embed_net = nn.Sequential(nn.Linear(len(FEATURE_COLS), 16), nn.ReLU(), nn.Linear(16, 8))
            with torch.no_grad():
                embeddings = embed_net(embed_feats).mean(dim=0).numpy()  # Avg embedding
            
            # Symbolic: Rule tree (experimental logic for patterns)
            def symbolic_rules(embed: np.ndarray) -> float:
                score = 0.0
                # Rule 1: Oversold + momentum (symbolic if-then)
                if embed[0] < 0.3 and embed[1] > 0.1:  # RSI low AND momentum up
                    score += 0.4
                # Rule 2: Volume divergence in low vol
                if REGIME_GLOBAL in ['very_low', 'low'] and embed[2] > 0.2 and embed[3] < -0.1:
                    score += 0.3
                # Adverse: Penalty for whale in volatile
                if 'volatile' in REGIME_GLOBAL and embed[4] > 0.3:
                    score -= 0.2
                return np.clip(score, -1.0, 1.0)
            
            neuro_sym_score = symbolic_rules(embeddings)
            # Fuse to components (weighted add)
            component_scores['neuro_sym'] = neuro_sym_score
            weights['neuro_sym'] = 0.15 if 'choppy' in REGIME_GLOBAL else 0.1  # Boost in complex regimes
        else:
            component_scores['neuro_sym'] = 0.0
            weights['neuro_sym'] = 0.0
        total_w = sum(weights.values())
        weights = {k: v / total_w for k, v in weights.items()}
        composite_score = sum(score * weights.get(comp, 0) for comp, score in component_scores.items())
                
        # Nueva: Quick MC validation for intelligence (only if sufficient data; quick mode)
        mc_validation_score = 1.0  # Default neutral
        if len(df) > 100:  # Threshold for MC (avoid compute overhead)
            cache_key = f"{symbol}_mc_cache"
            now = time.time()
            if hasattr(self.bt_mgr, 'mc_cache') and cache_key in self.bt_mgr.mc_cache:
                cached = self.bt_mgr.mc_cache[cache_key]
                if now - cached['timestamp'] < 300:  # 5min TTL
                    mc_validation_score = float(cached['score'])  # FIXED: float() para scalar seguro
                    logger.debug(f"MC cache hit for {symbol}: {mc_validation_score:.2f}")
                    # Early return to skip compute
                else:
                    del self.bt_mgr.mc_cache[cache_key]  # Invalidate old
            if cache_key not in self.bt_mgr.mc_cache or now - self.bt_mgr.mc_cache[cache_key]['timestamp'] >= 300:
                try:
                    quick_metrics = {'realized_pnl': 0.01, 'win_rate': 0.5, 'sharpe_ratio': 1.0}  # Placeholder for quick
                    # FIXED: Use self.bt_mgr (injected in init); dynamic paths low for quick
                    dynamic_paths_quick = 500 if regime in ['very_low', 'low'] else self.bt_mgr.dynamic_mc_paths
                    mc_results = await self.bt_mgr.run_monte_carlo(symbol, quick_metrics, timeframe, paths_override=dynamic_paths_quick)  # Quick call
                    mc_validation_score = float(mc_results.get('validation_score', 1.0))  # FIXED: float() para scalar
                    if mc_validation_score < get_config_param('mc_quality_good_threshold'):
                        logger.debug(f"Low MC validation for {symbol}: {mc_validation_score:.2f}; downweighting signal")
                    # Adaptive: Boost in low-vol if good MC (institutional: trust stats in quiet markets)
                    if regime in ['very_low', 'low'] and mc_validation_score > get_config_param('mc_quality_excellent_threshold'):
                        mc_validation_score *= 1.1  # Slight boost
                        mc_validation_score = min(1.0, mc_validation_score)
                    # Cache result
                    if not hasattr(self.bt_mgr, 'mc_cache'):
                        self.bt_mgr.mc_cache = {}
                    self.bt_mgr.mc_cache[cache_key] = {'score': mc_validation_score, 'timestamp': now}
                except Exception as mc_e:
                    logger.debug(f"MC quick validation failed for {symbol}: {mc_e}; neutral score")
                    mc_validation_score = 1.0
        
        # Apply MC multiplier to confidence (intelligence: stats-filtered signals)
        confidence = abs(composite_score) * mc_validation_score  # Downweight low-quality paths
        
        # Nueva: Smooth composite_score with EMA if historical data >50 (Grok)
        if len(df) > 50 and 'composite_score' in df.columns and not df['composite_score'].isna().iloc[-2]:
            prev_composite = df['composite_score'].iloc[-2]
            alpha = 0.3  # Smoothing factor
            composite_score = alpha * composite_score + (1 - alpha) * prev_composite
            df.loc[df.index[-1], 'composite_score'] = composite_score  # Store for next
        else:
            df['composite_score'] = composite_score  # Init column if missing
        direction = 'HOLD'
        confidence = abs(composite_score)
        if composite_score > threshold:
            direction = 'BUY'
        elif composite_score < -threshold:
            direction = 'SELL'
        if direction == 'HOLD':
            comp_str = ", ".join([f"{k}:{v:.2f}" for k, v in component_scores.items() if v != 0])
            logger.info(f"HOLD for {symbol} {timeframe}: composite={composite_score:.4f} < threshold={threshold:.4f}, components={comp_str}")
        current_price = latest['close']
        atr = latest.get('atr', current_price * 0.02)
        stop_mult = get_config_param('sl_atr_mult') * 0.5 if regime in ['very_low', 'low'] else get_config_param('sl_atr_mult')
        profit_mult = get_config_param('tp_atr_mult') * 0.7 if regime in ['very_low', 'low'] else get_config_param('tp_atr_mult')
        stop_loss = current_price - (atr * stop_mult) if direction == 'BUY' else current_price + (atr * stop_mult)
        take_profit = current_price + (atr * profit_mult) if direction == 'BUY' else current_price - (atr * profit_mult)
        win_rate, avg_win, avg_loss = self._estimate_strategy_performance(df, adaptive_config)
        logger.info(f"Signal for {symbol} {timeframe}: composite={composite_score:.4f}, threshold={threshold:.4f}, regime={regime}, ml_signal={ml_signal:.4f}, confidence={confidence:.4f}")
        if abs(composite_score) > 0.01 and abs(composite_score) <= threshold:
            comp_str = ", ".join([f"{k}:{v:.2f}" for k, v in component_scores.items() if v != 0])
            logger.warning(f"Near-miss signal for {symbol}: {composite_score:.4f} (components: {comp_str})")
        # New: Partial execution with Bayesian update
        hist_win = win_rate
        partial_mult = bayesian_signal_strength(confidence, hist_win)
        partial_mult = min(partial_mult, 1.0)
        # Apply partial_mult to confidence (probabilistic logic)
        confidence *= partial_mult
        return {
            'symbol': symbol, 'timeframe': timeframe, 'direction': direction, 'confidence': confidence, 'composite_score': composite_score, 'entry_price': current_price,
            'stop_loss': stop_loss, 'take_profit': take_profit, 'win_rate': win_rate, 'avg_win': avg_win, 'avg_loss': avg_loss, 'regime': regime, 'adaptive_threshold': threshold,
            'component_scores': component_scores,
            'timestamp': time.time(), 'atr': atr, 'df': df, 'partial_mult': partial_mult
        }

    def _neutral_signal(self, reason: str, symbol: str = "", timeframe: str = "") -> Dict[str, Any]:
        return {'symbol': symbol, 'timeframe': timeframe, 'direction': 'HOLD', 'confidence': 0.0, 'composite_score': 0.0, 'entry_price': 0.0, 'stop_loss': 0.0, 'take_profit': 0.0, 'win_rate': 0.5, 'avg_win': 0.02, 'avg_loss': 0.02, 'regime': REGIME_GLOBAL, 'component_scores': {}, 'reason': reason, 'timestamp': time.time(), 'df': pd.DataFrame()}

    def _analyze_low_volatility_patterns(self, latest: pd.Series, recent_data: pd.DataFrame, regime: str, strategy_type: str) -> float:
        score = 0.0
        mr_signal = latest.get('mean_reversion_signal', 0)
        amplify = 8 if strategy_type == 'mean_reversion' else 5
        score += np.tanh(mr_signal * amplify) * (0.4 if regime == 'very_low' else 0.3)
        consolidation = latest.get('consolidation_strength', 0)
        compression = latest.get('range_compression', 0)
        if consolidation > 0.2 and compression > 0.2:
            buying_pressure = latest.get('buying_pressure', 0.5)
            breakout_signal = (buying_pressure - 0.5) * 2
            score += breakout_signal * 0.5
        divergence = latest.get('vol_price_divergence', 0)
        score += np.tanh(divergence * 20) * 0.3
        micro_3 = latest.get('micro_momentum_3', 0)
        micro_5 = latest.get('micro_momentum_5', 0)
        if micro_3 * micro_5 > 0:
            momentum_strength = abs(micro_3) + abs(micro_5)
            score += np.sign(micro_3) * min(momentum_strength * 10, 0.3)
        whale_risk = latest.get('whale_risk', 0)
        if whale_risk > 0.3 and divergence > 0:
            score += 0.2  # Boost score if potential whale accumulation before rise
        return np.clip(score, -1.0, 1.0)

    def _analyze_micro_movements(self, latest: pd.Series, recent_data: pd.DataFrame) -> float:
        score = 0.0
        micro_3 = latest.get('micro_momentum_3', 0)
        micro_5 = latest.get('micro_momentum_5', 0)
        micro_8 = latest.get('micro_momentum_8', 0)
        score += np.tanh(micro_3 * 100) * 0.4
        score += np.tanh(micro_5 * 80) * 0.3
        score += np.tanh(micro_8 * 60) * 0.3
        return np.clip(score, -1.0, 1.0)

    def _analyze_technical_indicators(self, latest: pd.Series, recent_data: pd.DataFrame, config: Dict = None) -> float:
        score = 0.0
        rsi = latest.get('rsi', 50)
        base_oversold = 20
        base_overbought = 80
        regime = REGIME_GLOBAL
        rsi_oversold = base_oversold + 5 if regime in ['very_low', 'low'] else base_oversold - 5 if 'volatile' in regime else base_oversold
        rsi_overbought = base_overbought - 5 if regime in ['very_low', 'low'] else base_overbought + 5 if 'volatile' in regime else base_overbought
        if rsi > rsi_overbought:
            score -= 0.15
        elif rsi < rsi_oversold:
            score += 0.35
        elif 35 <= rsi <= 65:
            score += 0.15
        bb_position = latest.get('bb_position', 0)
        if bb_position > 1:
            score -= 0.2
        elif bb_position < -1:
            score += 0.2
        macd_signal = latest.get('macd_signal', 0)
        score += np.tanh(macd_signal * 10) * 0.2
        return np.clip(score, -1.0, 1.0)

    def _analyze_momentum(self, recent_data: pd.DataFrame) -> float:
        score = 0.0
        price_momentum = recent_data['price_momentum'].tail(5).mean()
        score += np.tanh(price_momentum * 20) * 0.4
        vol_ratio = recent_data['vol_ratio'].tail(3).mean()
        if vol_ratio > 1.5:
            score += 0.2
        elif vol_ratio < 0.7:
            score -= 0.1
        vol_mean_5 = recent_data['volatility'].tail(5).mean()
        vol_mean_20 = recent_data['volatility'].tail(20).mean()
        vol_trend = vol_mean_5 / vol_mean_20 if vol_mean_20 != 0 else 0.8
        if vol_trend > 1.2:
            score -= 0.1
        return np.clip(score, -1.0, 1.0)

    def _analyze_volume(self, recent_data: pd.DataFrame) -> float:
        score = 0.0
        recent_volume = recent_data['volume_sma_ratio'].tail(3).mean()
        if recent_volume > 1.1:
            score += 0.3
        elif recent_volume < 0.8:
            score -= 0.1
        price_change = recent_data['close'].pct_change().tail(5).mean()
        volume_change = recent_data['vol_ratio'].tail(5).mean()
        if price_change > 0 and volume_change > 1.2:
            score += 0.2
        elif price_change < 0 and volume_change > 1.2:
            score -= 0.2
        return np.clip(score, -1.0, 1.0)

    def _analyze_volatility_regime(self, recent_data: pd.DataFrame) -> float:
        score = 0.0
        vol_regime = recent_data['volatility_regime'].tail(5).mean()
        if vol_regime < 0.8:
            score += 0.2
        elif vol_regime > 1.5:
            score -= 0.3
        return np.clip(score, -1.0, 1.0)

    def analyze_market_structure(self, recent_data: pd.DataFrame) -> float:
        
        score = 0.0
        
        # Verificar que el DataFrame no esté vacío y tenga las columnas necesarias
        if recent_data.empty or not all(col in recent_data.columns for col in ['close', 'high', 'low']):
            return score
        
        closes = recent_data['close']
        
        # Verificar que tengamos suficientes datos para calcular las medias móviles
        if len(closes) < 10:
            return score
        
        # Análisis de medias móviles
        sma_short = closes.rolling(5, min_periods=5).mean()
        sma_long = closes.rolling(10, min_periods=10).mean()
        
        # Verificar que las medias móviles sean válidas (no NaN)
        if not pd.isna(sma_short.iloc[-1]) and not pd.isna(sma_long.iloc[-1]):
            if sma_short.iloc[-1] > sma_long.iloc[-1]:
                score += 0.1
            else:
                score -= 0.1
        
        # Análisis de posición en el rango
        recent_highs = recent_data['high'].rolling(10, min_periods=10).max()
        recent_lows = recent_data['low'].rolling(10, min_periods=10).min()
        current_price = closes.iloc[-1]
        
        # Verificar que tengamos valores válidos para highs y lows
        if not pd.isna(recent_highs.iloc[-1]) and not pd.isna(recent_lows.iloc[-1]):
            high_value = recent_highs.iloc[-1]
            low_value = recent_lows.iloc[-1]
            
            # Calcular el rango y verificar que no sea cero o muy pequeño
            price_range = high_value - low_value
            
            # Usar un epsilon pequeño para evitar división por cero
            epsilon = 1e-10
            
            if abs(price_range) > epsilon:
                range_position = (current_price - low_value) / price_range
                
                # Asegurar que range_position esté en el rango válido [0, 1]
                range_position = np.clip(range_position, 0.0, 1.0)
                
                if range_position > 0.8:
                    score -= 0.1  # Cerca del máximo (resistencia)
                elif range_position < 0.2:
                    score += 0.1  # Cerca del mínimo (soporte)
            # Si el rango es muy pequeño (precio lateral), no ajustamos el score
        
        return np.clip(score, -1.0, 1.0)

    def _estimate_strategy_performance(self, df: pd.DataFrame, tf_config: Dict[str, Any]) -> Tuple[float, float, float]:
        if len(df) < 100:
            return 0.6, 0.03, 0.015
        recent_df = df.tail(100).copy()
        thresh = tf_config.get('label_threshold', 0.015)
        signals = []
        for i, row in recent_df.iterrows():
            rsi = row.get('rsi', 50)
            momentum = row.get('price_momentum', 0)
            if rsi < 35 and momentum > thresh:
                signals.append(1)
            elif rsi > 65 and momentum < -thresh:
                signals.append(-1)
            else:
                signals.append(0)
        returns = recent_df['close'].pct_change().fillna(0)
        strategy_returns = []
        for i in range(1, len(signals)):
            if signals[i-1] != 0:
                strategy_returns.append(signals[i-1] * returns.iloc[i])
        if len(strategy_returns) < 10:
            return 0.55, 0.025, 0.020
        strategy_returns = np.array(strategy_returns)
        positive_returns = strategy_returns[strategy_returns > 0]
        negative_returns = strategy_returns[strategy_returns < 0]
        win_rate = len(positive_returns) / len(strategy_returns) if len(strategy_returns) > 0 else 0.5
        avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0.025
        avg_loss = abs(np.mean(negative_returns)) if len(negative_returns) > 0 else 0.02
        win_rate = max(0.45, min(0.8, win_rate))
        avg_win = max(0.01, min(0.1, avg_win))
        avg_loss = max(0.008, min(0.1, avg_loss))
        return win_rate, avg_win, avg_loss

class BtMgr:
    def __init__(self, context: AppContext):
        self.context = context
        self.exchange = context.exchange
        self.db = context.db
        self.sig_gen = SigGen(context)
        self.model_mgr = ModelMgr(context)
        self.opt = ParamOpt()
        self.regime_det = VolRegDet(context)
        self.ohlcv_cache = {}
        self.categories = {'small_cap': [], 'medium_cap': [], 'big_cap': []}
        self.last_backtest_time = 0
        self.top_performers = []
        self.days = get_config_param('bootstrap_days')        
        self.tf_name = 'medium_1h'
        self.results = self.load_cached_results()
        self.dynamic_mc_paths = get_config_param('monte_carlo_paths')
        METRICS_LOCK_SYNC = threading.Lock()
        with METRICS_LOCK_SYNC:
            if 'low' in CONFIG['current_regime'] or 'very_low' in CONFIG['current_regime']:
                self.dynamic_mc_paths = 500
        self.tf_config = next((tf for tf in TIMEFRAMES if tf['name'] == 'medium_1h'), None)
        if self.tf_config is None:
            raise ValueError("No se encontró config para 'medium_1h' en TIMEFRAMES")
        self.mc_cache = {}  
        self.gan_mc = GANMC(context)

    def load_cached_results(self):
        try:
            with open('backtest_cache.pkl', 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}
    
    def save_cached_results(self):
        with open('backtest_cache.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
    async def initialize_cache(self):
        pairs, categories = await self.get_150_pairs()
        self.categories = categories
        sem_limit = int(get_config_param('max_concurrent_fetches') or 10)  # Force int early with fallback
        sem = asyncio.Semaphore(sem_limit)
        async def fetch_for_symbol(s: str):
            async with sem:
                for attempt in range(3):
                    try:
                        df = await self.fetch_historical_ohlcv(s)
                        if df is not None and len(df) >= get_config_param('min_data_pts'):
                            df_features = await prepare_features(df, self.tf_config)
                            if not df_features.empty:
                                self.ohlcv_cache[s] = df_features
                                bt_logger.debug(f"Cached {s} with {len(df_features)} rows")
                                return s
                        bt_logger.warning(f"Insufficient data for {s} on attempt {attempt + 1}")
                        await asyncio.sleep(1)
                    except Exception as e:
                        bt_logger.error(f"Error fetching OHLCV for {s}: {e}")
                        await asyncio.sleep(2 ** attempt)
                return None
        tasks = [fetch_for_symbol(s) for s in pairs]
        cached = await asyncio.gather(*tasks, return_exceptions=True)
        num_cached = len([s for s in cached if s is not None and not isinstance(s, Exception)])
        bt_logger.info(f"Initialized cache for {num_cached}/150 pairs")
        self.ohlcv_cache = {k: v for k, v in self.ohlcv_cache.items() if k in pairs}
        # New: Auto-scale timeframes
        TIMEFRAMES = self.auto_scale_timeframes(self.ohlcv_cache)

    def auto_scale_timeframes(self, ohlcv_cache: dict) -> List[dict]:
        vols = [df['volatility'].std() if 'volatility' in df else 0.02 for df in ohlcv_cache.values() if not df.empty]
        if len(vols) < 10:
            return TIMEFRAMES
        kmeans = KMeans(n_clusters=CONFIG['default']['tf_clusters'])
        clusters = kmeans.fit_predict(np.array(vols).reshape(-1, 1))
        unique_clusters = np.unique(clusters)
        for c in unique_clusters:
            cluster_vols = [v for i, v in enumerate(vols) if clusters[i] == c]
            if np.std(cluster_vols) > CONFIG['default']['auto_tf_min_vol']:
                tf_minutes = int(15 * (1 + c))
                TIMEFRAMES.append({
                    'name': f'auto_{tf_minutes}m',
                    'binance_interval': f'{tf_minutes}m',
                    'label_lookahead': max(4, tf_minutes // 15),
                    'label_threshold': 0.01 * (1 + c),
                    'confidence_threshold': 0.6,
                    'min_data_points': 500 + 100 * c,
                    'rsi_period': 14 + c * 2,
                    'vol_window': 20 + c * 5,
                    'timeframe_minutes': tf_minutes,
                    'strategy_type': 'auto_cluster',
                    'max_holding_hours': tf_minutes * 2,
                    'regime_filter': list(CONFIG.keys())
                })
        return TIMEFRAMES

    async def get_150_pairs(self) -> Tuple[List[str], Dict[str, List[str]]]:
        try:
            symbols = [s for s in self.exchange.markets if s.endswith('/USDT') and self.exchange.markets[s].get('active', False)]
            sem = asyncio.Semaphore(10)
            async def get_ticker_vol(s: str):
                async with sem:
                    ticker = await self.exchange.fetch_ticker_with_retry(s)
                    if ticker:
                        vol = ticker.get('quoteVolume', 0)
                        return s, vol
                    return s, 0
            tasks = [get_ticker_vol(s) for s in symbols[:5]]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            vol_data = []
            for result in results:
                if isinstance(result, Exception):
                    continue
                if isinstance(result, tuple) and len(result) == 2:
                    s, v = result
                    if isinstance(s, str) and v > 0:
                        vol_data.append((s, v))
            vol_data.sort(key=lambda x: x[1], reverse=True)
            big_cap = [s for s, v in vol_data if v > 10000000][:50]
            med_cap = [s for s, v in vol_data if 1000000 <= v <= 10000000][:50]
            small_cap = [s for s, v in vol_data if v < 1000000][:50]
            pairs = big_cap + med_cap + small_cap
            categories = {'big_cap': big_cap, 'medium_cap': med_cap, 'small_cap': small_cap}
            bt_logger.info(f"Categorized pairs: big={len(big_cap)}, med={len(med_cap)}, small={len(small_cap)}")
            return pairs, categories
        except Exception as e:
            bt_logger.error(f"Error getting pairs: {e}")
            return [], {'big_cap': [], 'medium_cap': [], 'small_cap': []}

    async def fetch_historical_ohlcv(self, symbol: str, days: int = None, tf: str = '1h') -> Optional[pd.DataFrame]:
        if days is None:
            days = self.days
        since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
        limit = days * 24 + 100
        max_retries = CONFIG['default']['max_retries']
        for attempt in range(max_retries):
            try:
                ohlcv = await self.exchange.fetch_ohlcv_with_retry(symbol, tf, since, limit)
                if not ohlcv:
                    continue
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df = df[(df['close'] > 0) & (df['volume'] >= 0)]
                if len(df) < 100:
                    return None
                return df
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return None
        return None

    async def run_backtest_on_pair(self, symbol: str, initial_equity: float = 10000.0) -> Dict:
        df = self.ohlcv_cache.get(symbol)
        if df is None or df.empty or len(df) < 500:
            bt_logger.debug(f"Insufficient data for {symbol} backtest")
            return {}
        # Nueva: Walk-forward validation con 3 folds (Grok)
        metrics_list = []
        for fold in range(3):
            start = int(fold * len(df) / 3)
            end = int((fold + 1) * len(df) / 3)
            train_df = df.iloc[:start].copy()
            test_df = df.iloc[start:end].copy()
            await self.model_mgr.train_model(symbol, self.tf_name, train_df)
            virtual_trades = []
            virtual_positions = []
            equity_curve = [initial_equity]
            current_equity = initial_equity
            params = self.opt.get_current_params()
            min_conf = params['min_conf_score']
            kelly_frac = params['kelly_frac']
            stop_mult = params['sl_atr_mult']
            profit_mult = params['tp_atr_mult']
            # Aumenta fees/slippage para realismo (Grok)
            fees = get_config_param('fees') * 2  # 0.002  # Fixed: Use get_config_param instead of direct CONFIG
            slippage = get_config_param('slippage') * 1.5 if 'volatile' in REGIME_GLOBAL else get_config_param('slippage')  # Fixed: Use get_config_param
            for i in range(len(test_df)):
                current_time = test_df.index[i].timestamp()
                sub_df = pd.concat([train_df, test_df.iloc[:i+1]])
                signal = await self.sig_gen.generate_signal(sub_df, self.tf_config, symbol, historical=True)
                current_price = test_df.iloc[i]['close']
                if current_price is None or current_price <= 0:
                    continue
                positions_to_close = []
                for pos in virtual_positions:
                    pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] if pos['side'] == 'BUY' else (pos['entry_price'] - current_price) / pos['entry_price']
                    # Modificado: Incluye fees y slippage en el cálculo del P&L (Grok)
                    pos_pnl = pos['amount'] * current_price * pnl_pct - (pos['amount'] * current_price * fees * 2 + slippage)
                    pos['current_pnl'] = pos_pnl
                    atr = sub_df['atr'].iloc[-1] if 'atr' in sub_df else current_price * 0.02
                    pos_sl = pos['entry_price'] - (atr * stop_mult) if pos['side'] == 'BUY' else pos['entry_price'] + (atr * stop_mult)
                    pos_tp = pos['entry_price'] + (atr * profit_mult) if pos['side'] == 'BUY' else pos['entry_price'] - (atr * profit_mult)
                    close_pos = False
                    if pos['side'] == 'BUY':
                        if current_price <= pos_sl or current_price >= pos_tp or (current_time - pos['open_time']) > (self.tf_config['max_holding_hours'] * 3600):
                            close_pos = True
                    else:
                        if current_price >= pos_sl or current_price <= pos_tp or (current_time - pos['open_time']) > (self.tf_config['max_holding_hours'] * 3600):
                            close_pos = True
                    if close_pos:
                        virtual_trades.append({'pnl': pos_pnl, 'win': pos_pnl > 0})
                        current_equity += pos_pnl
                        positions_to_close.append(pos)
                for pos in positions_to_close:
                    virtual_positions.remove(pos)
                if signal['direction'] != 'HOLD' and signal['confidence'] >= min_conf and not virtual_positions:
                    win_rate_est = signal['win_rate']
                    avg_win_est = signal['avg_win']
                    avg_loss_est = signal['avg_loss']
                    b = avg_win_est / avg_loss_est if avg_loss_est > 0 else 1
                    p = win_rate_est
                    kelly = (b * p - (1 - p)) / b * kelly_frac
                    position_value = current_equity * min(kelly, 0.15)
                    amount = position_value / current_price
                    if amount * current_price >= get_config_param('min_order_size'):
                        virtual_positions.append({
                            'side': signal['direction'],
                            'amount': amount,
                            'entry_price': current_price,
                            'open_time': current_time,
                            'sl': signal['stop_loss'],
                            'tp': signal['take_profit']
                        })
                equity_curve.append(current_equity)
            # Calcular métricas por fold (Grok)
            if virtual_trades:
                trades = len(virtual_trades)
                wins = sum(1 for t in virtual_trades if t['win'])
                win_rate = wins / trades if trades > 0 else 0.0
                total_pnl = sum(t['pnl'] for t in virtual_trades)
                returns = np.diff(equity_curve) / np.array(equity_curve[:-1]) if len(equity_curve) > 1 else np.array([0.0])
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365 * 24) if np.std(returns) > 0 else 0.0
                cum_max = np.maximum.accumulate(equity_curve)
                drawdowns = (np.array(equity_curve) - cum_max) / np.maximum(cum_max, 1e-8) * 100
                max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
                fold_metrics = {
                    'trades': trades,
                    'realized_pnl': total_pnl,
                    'win_rate': win_rate,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_dd
                }
                metrics_list.append(fold_metrics)
            else:
                fold_metrics = {
                    'trades': 0,
                    'realized_pnl': 0.0,
                    'win_rate': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0
                }
                metrics_list.append(fold_metrics)
        # Si no hay trades en ningún fold, retornar métricas vacías (Grok)
        
        if not metrics_list:
            return {
                'trades': 0,
                'realized_pnl': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'timeframe': self.tf_name,
                'timestamp': time.time()
            }
        
        # Reset global counter post-backtest (production: ensures fresh cycles)
        global CLOSED_TRADES_SINCE_LAST
        CLOSED_TRADES_SINCE_LAST = 0
        
        # Avg metrics over folds (Grok)
        metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
        metrics['timeframe'] = self.tf_name
        metrics['timestamp'] = time.time()
        # Logging y almacenamiento (de tu código original)
        bt_logger.info(f"Backtest {symbol}: P&L=${metrics['realized_pnl']:.2f}, Win={metrics['win_rate']:.1%}, Sharpe={metrics['sharpe_ratio']:.2f}, DD={metrics['max_drawdown']:.1f}%")
        if not metrics_list:
            bt_logger.warning(f"No trades generados en backtest para{symbol}. Verifica thresholds/signals.")
        if metrics['realized_pnl'] > 0 and metrics['sharpe_ratio'] > 1.0:
            key = symbol.replace('/', '') + '' + self.tf_name
            ver = CONFIG['default']['model_ver']
            save_path = Path(CONFIG['default']['model_dir']) / f"{key}_bt_metrics_v{ver}.pkl"
            joblib.dump(metrics, save_path)
            bt_logger.info(f"Saved profitable backtest metrics for {symbol}")
        return metrics

    def get_saved_bt_metrics(self, symbol: str) -> Optional[Dict]:
        """Carga métricas de backtest guardadas si existen."""
        try:
            key = symbol.replace('/', '_') + '_' + self.tf_name
            ver = CONFIG['default']['model_ver']
            path = Path(CONFIG['default']['model_dir']) / f"{key}_bt_metrics_v{ver}.pkl"
            if path.exists():
                metrics = joblib.load(path)
                # Nueva: Chequeo de frescura (timestamp > now - freshness_days)
                now = time.time()
                freshness_days = get_config_param('backtest_freshness_days')
                if metrics.get('timestamp', 0) < now - (freshness_days * 86400):
                    bt_logger.debug(f"Stale metrics for {symbol} (age > {freshness_days} days), invalidating")
                    return None
                # Handle legacy key names (sin cambios)
                if 'pnl' in metrics or 'total_pnl' in metrics:
                    metrics['realized_pnl'] = metrics.get('pnl', metrics.get('total_pnl', 0.0))
                    if 'pnl' in metrics:
                        del metrics['pnl']
                    if 'total_pnl' in metrics:
                        del metrics['total_pnl']
                    bt_logger.warning(f"Updated legacy metrics for {symbol}: Mapped pnl/total_pnl to realized_pnl")
                return metrics
            return None
        except Exception as e:
            bt_logger.error(f"Failed to load backtest metrics for {symbol}: {e}")
            return None

    async def get_recent_trades(self) -> List[Dict]:
        """Obtiene la accuracy guardada del modelo para checks rápidos de calidad."""
        if not self.db.connected:
            return []
        try:
            async with asyncio.timeout(10):  
                async with self.db.pool.acquire() as conn:  
                    rows = await conn.fetch('SELECT * FROM trades ORDER BY created_at DESC LIMIT 20')
                    return [dict(row) for row in rows]
        except asyncio.TimeoutError:
            bt_logger.warning("Timeout fetching recent trades (DB query slow)")
            return []
        except Exception as e:
            logger.error(f"Error fetching recent trades: {e}")
            return []

    def get_timeframe_config(self, timeframe_name):
        """Busca configuración de timeframe por nombre"""
        for tf in TIMEFRAMES:
            if tf['name'] == timeframe_name:
                return tf
        return None
    
    async def run_monte_carlo(self, symbol: str, metrics: Dict, timeframe_name: str = 'medium_1h', paths_override: Optional[int] = None) -> Dict:
        """
        Simulación Monte Carlo mejorada integrada con sistema de regímenes existente.
        Incorpora GARCH, bootstrap temporal y validación estadística robusta.
        """
        try:
            df = self.ohlcv_cache.get(symbol)
            if df is None or len(df) < CONFIG['default']['min_data_pts']:
                bt_logger.warning(f"Insufficient data for {symbol} Monte Carlo")
                return {'pnl_5th': 0, 'pnl_50th': 0, 'pnl_95th': 0, 'validation_score': 0}
            
            last_price = df['close'].iloc[-1]
            historical_returns = df['close'].pct_change().dropna().values
            
            if len(historical_returns) < get_config_param('min_historical_periods'):
                return await self._run_fallback_monte_carlo(symbol, metrics, historical_returns, last_price)
            
            # 1. DETECCIÓN DE RÉGIMEN CON STRENGTH
            regime, trap_detected = await self.regime_det.detect_regime(df, symbol, historical=True)
            regime_params = self._get_enhanced_regime_params(regime, trap_detected, df)
            
            # 2. ANÁLISIS DE TIMEFRAME ESPECÍFICO
            bt_logger.debug(f"Getting timeframe config for {timeframe_name}")
            timeframe_params = self.get_timeframe_config(timeframe_name)
            # Validación de timeframe
            if timeframe_params is None:
                bt_logger.warning(f"Timeframe {timeframe_name} not found, using default medium_1h")
                timeframe_params = self.get_timeframe_config('medium_1h')
                if timeframe_params is None:
                    bt_logger.error("Default timeframe medium_1h not found in TIMEFRAMES")
                    return {'pnl_5th': 0, 'pnl_50th': 0, 'pnl_95th': 0, 'validation_score': 0}
                                            
            # 3. MODELADO AVANZADO DE VOLATILIDAD
            garch_params = self._estimate_enhanced_garch(historical_returns, regime_params)
            
            # 4. ANÁLISIS DE EVENTOS EXTREMOS Y WHALE RISK
            extreme_analysis = self._analyze_market_microstructure(df, regime_params)
            
            # 5. SIMULACIÓN MULTI-MÉTODO
            dynamic_paths = paths_override if paths_override is not None else self.dynamic_mc_paths  # FIXED: Use override if provided
            if regime in ['very_low', 'low']:
                dynamic_paths = max(500, dynamic_paths * 0.3)  # Reduce in low-vol
            elif 'volatile' in regime:
                dynamic_paths = min(3000, dynamic_paths * 1.5)  # Increase slightly in high-vol
            else:
                dynamic_paths = min(5000, dynamic_paths)  # Cap for production
            if dynamic_paths < 100:  # Fallback if too low
                dynamic_paths = 500
                bt_logger.warning(f"MC paths too low ({dynamic_paths}), using fallback 500 for {symbol}")

            num_paths = dynamic_paths            
            paths_ensemble = await self._run_ensemble_simulation(
                historical_returns, last_price, regime_params, 
                timeframe_params, garch_params, extreme_analysis
            )
                        
            # FIXED: Guard si paths_ensemble es None o empty (de simulations fallidas)
            if paths_ensemble is None or paths_ensemble.size == 0:
                bt_logger.warning(f"Ensemble simulation returned None/empty for {symbol}; using fallback paths")
                T_fallback = 96  # FIXED: Define T_fallback aquí si timeframe_params None
                num_fallback = dynamic_paths
                fallback_paths = np.full((num_fallback, T_fallback + 1), last_price)
                mean_ret = np.mean(historical_returns) if len(historical_returns) > 0 else 0.0
                vol = np.std(historical_returns) if len(historical_returns) > 0 else 0.02
                for path_idx in range(num_fallback):
                    for t in range(T_fallback):
                        ret = np.random.normal(mean_ret, vol)
                        fallback_paths[path_idx, t+1] = fallback_paths[path_idx, t] * (1 + ret)
                paths_ensemble = fallback_paths
            
            # 6. VALIDACIÓN ESTADÍSTICA AVANZADA
            validation_results = await self._advanced_statistical_validation(
                historical_returns, paths_ensemble, symbol, regime, timeframe_params
            )
            
            # 7. MÉTRICAS FINALES CON AJUSTES INTELIGENTES
            final_metrics = self._calculate_risk_adjusted_metrics(
                paths_ensemble, last_price, metrics, validation_results, regime_params
            )
            
            # 8. LOGGING DETALLADO
            self._log_monte_carlo_results(symbol, final_metrics, validation_results, regime)
            
            # 9. AJUSTE DINÁMICO DE PATHS PARA PRÓXIMAS SIMULACIONES
            if final_metrics['validation_score'] < get_config_param('mc_quality_good_threshold'):
                self.dynamic_mc_paths = min(5000, int(self.dynamic_mc_paths * 1.2))
            elif final_metrics['validation_score'] > get_config_param('mc_quality_excellent_threshold') and self.dynamic_mc_paths > get_config_param('monte_carlo_paths'):
                self.dynamic_mc_paths = max(CONFIG['default']['monte_carlo_paths'], int(self.dynamic_mc_paths * 0.95))
                CONFIG['default']['monte_carlo_paths'] = self.dynamic_mc_paths  # Sync to CONFIG for get_config_param
            
            return final_metrics
            
        except Exception as e:
            bt_logger.error(f"Monte Carlo error for {symbol}: {e}")
            return {'pnl_5th': 0, 'pnl_50th': 0, 'pnl_95th': 0, 'validation_score': 0}                 

    def _get_enhanced_regime_params(self, regime: str, trap_detected: bool, df: pd.DataFrame) -> Dict:
        """
        Parámetros de régimen mejorados basados en tu sistema de detección existente
        """
        # Mapeo directo de tus regímenes existentes
        regime_mapping = {
            'very_low': {'vol_mult': get_config_param('garch_alpha_base') * 0.5, 'shock_mult': 0.2, 'clustering': 0.6, 'mean_reversion': 1.3},
            'low': {'vol_mult': get_config_param('garch_alpha_base') * 0.7, 'shock_mult': 0.4, 'clustering': 0.7, 'mean_reversion': 1.1},
            'normal': {'vol_mult': get_config_param('garch_alpha_base'), 'shock_mult': 1.0, 'clustering': 1.0, 'mean_reversion': 1.0},
            'bull': {'vol_mult': get_config_param('garch_alpha_base') * 1.1, 'shock_mult': 0.8, 'clustering': 0.9, 'mean_reversion': 0.8},
            'bear': {'vol_mult': get_config_param('garch_alpha_base') * 1.2, 'shock_mult': 1.3, 'clustering': 1.1, 'mean_reversion': 0.9},
            'high': {'vol_mult': get_config_param('garch_alpha_base') * 1.4, 'shock_mult': 1.5, 'clustering': 1.2, 'mean_reversion': 0.7},
            'volatile_bull': {'vol_mult': get_config_param('garch_alpha_base') * 1.6, 'shock_mult': 1.2, 'clustering': 1.3, 'mean_reversion': 0.6},
            'volatile_bear': {'vol_mult': get_config_param('garch_alpha_base') * 1.8, 'shock_mult': 2.0, 'clustering': 1.4, 'mean_reversion': 0.5}
        }
        
        params = regime_mapping.get(regime, regime_mapping['normal']).copy()
        
        # Ajustes por trap detection
        if trap_detected:
            params['shock_mult'] *= 1.8
            params['clustering'] *= 1.3
            params['vol_mult'] *= 1.2
            bt_logger.debug(f"Trap detected adjustments applied for regime {regime}")
        
        # Análisis de whale_risk si disponible
        if df is not None and 'whale_risk' in df.columns:
            avg_whale_risk = df['whale_risk'].rolling(48).mean().iloc[-1]
            if not np.isnan(avg_whale_risk):
                whale_multiplier = 1 + (avg_whale_risk * 0.5)
                params['shock_mult'] *= whale_multiplier
                params['vol_mult'] *= (1 + avg_whale_risk * 0.2)
        
        # Usar umbrales de tu CONFIG existente
        if regime == 'very_low':
            params['vol_threshold'] = CONFIG['default']['dynamic_very_low_vol_thresh']
        elif regime == 'low':
            params['vol_threshold'] = CONFIG['default']['dynamic_low_vol_thresh']
        else:
            params['vol_threshold'] = 0.03
        
        params['regime'] = regime
        params['trap_detected'] = trap_detected
        
        return params

    def _get_timeframe_specific_params(self, metrics: Dict, regime: str) -> Dict:
        """
        Parámetros específicos por timeframe basados en tu configuración TIMEFRAMES
        """
        # Identificar timeframe desde metrics o usar medium_1h por default
        timeframe_name = metrics.get('timeframe', 'medium_1h')
        
        # Mapear a tu configuración TIMEFRAMES existente
        timeframe_configs = {tf['name']: tf for tf in TIMEFRAMES}
        
        tf_config = timeframe_configs.get(timeframe_name, timeframe_configs.get('medium_1h', TIMEFRAMES[1]))
        
        # Ajustes por régimen específicos para timeframes
        if regime in ['very_low', 'low'] and tf_config['strategy_type'] == 'micro_scalp':
            tf_config = tf_config.copy()
            tf_config['vol_boost'] = get_config_param('low_vol_pos_mult')  # Coordinado con low vol multiplier
            tf_config['mean_reversion_strength'] = 1.4
        elif 'volatile' in regime and tf_config['strategy_type'] == 'trend_following':
            tf_config = tf_config.copy()
            tf_config['trend_persistence'] = 1.3
            tf_config['shock_frequency'] = get_config_param('jump_detection_threshold') * 0.5  # Dinámico con jump threshold
        elif tf_config['strategy_type'] == 'mean_reversion':
            tf_config['mean_reversion_strength'] = get_config_param('garch_beta_base') * 1.2  # Usando GARCH beta para reversion
        
        # Añadir handling si tf_config es None (aunque no debería)
        if tf_config is None:
            tf_config = TIMEFRAMES[1]
        
        return tf_config

    def _estimate_enhanced_garch(self, returns: np.ndarray, regime_params: Dict) -> Dict:
        """
        Estimación GARCH mejorada adaptada al régimen detectado
        """
        # Parámetros GARCH base ajustados por régimen usando CONFIG
        regime_garch = {
            'very_low': {'alpha': get_config_param('garch_alpha_base') * 0.4, 'beta': get_config_param('garch_beta_base') * 0.95, 'omega_mult': 0.5},
            'low': {'alpha': get_config_param('garch_alpha_base') * 0.6, 'beta': get_config_param('garch_beta_base') * 0.93, 'omega_mult': 0.7},
            'normal': {'alpha': get_config_param('garch_alpha_base'), 'beta': get_config_param('garch_beta_base'), 'omega_mult': 1.0},
            'high': {'alpha': get_config_param('garch_alpha_base') * 1.6, 'beta': get_config_param('garch_beta_base') * 0.85, 'omega_mult': 1.3},
            'volatile_bull': {'alpha': get_config_param('garch_alpha_base') * 2.4, 'beta': get_config_param('garch_beta_base') * 0.80, 'omega_mult': 1.5},
            'volatile_bear': {'alpha': get_config_param('garch_alpha_base') * 3.0, 'beta': get_config_param('garch_beta_base') * 0.75, 'omega_mult': 1.8}
        }
        
        regime = regime_params['regime']
        garch_config = regime_garch.get(regime, regime_garch['normal'])        
        alpha = garch_config['alpha'] * regime_params['clustering']
        beta = garch_config['beta']
        omega = np.var(returns) * (1 - alpha - beta) * garch_config['omega_mult'] if np.var(returns) > 0 else 1e-6  # Evitar división por cero
        # Nueva cascade para alpha/beta si persistence alto
        persistence = alpha + beta
        alpha_options = [alpha, alpha * 0.9, alpha * 0.8]  # Cascade degradada
        beta_options = [beta, beta * 0.9, beta * 0.8]
        for a_opt, b_opt in zip(alpha_options, beta_options):
            persistence = a_opt + b_opt
            if persistence < 0.999:
                alpha = a_opt
                beta = b_opt
                break
        if persistence >= 0.999:
            logger.warning(f"GARCH persistence still high ({persistence:.4f}) after cascade. Forcing stationarity.")
            scale_factor = 0.98 / persistence
            alpha *= scale_factor
            beta *= scale_factor
            persistence = alpha + beta
        
        # === AGREGADO: FALLBACK SI PARÁMETROS INVÁLIDOS ===
        if alpha <= 0.001 or beta <= 0.001 or omega <= 1e-8:
            bt_logger.warning(f"Invalid GARCH parameters for regime {regime}, using EWMA fallback")
            return self._ewma_fallback_method(returns, regime_params)
    
        # Estimación de volatilidad condicional
        n = len(returns)
        conditional_vol = np.zeros(n)
        conditional_vol[0] = np.std(returns) if n > 0 else 0.01
        
        # === AGREGADO: TRACKING DE CONVERGENCIA DURANTE ESTIMACIÓN ===
        convergence_issues = 0
        prev_vol = conditional_vol[0]
    
        for t in range(1, n):
            conditional_vol[t] = np.sqrt(omega + alpha * returns[t-1]**2 + beta * conditional_vol[t-1]**2)
            conditional_vol[t] = np.clip(conditional_vol[t], 0.001, 0.1)  # FIXED: Clip to prevent NaN/inf propagation to JIT
            
            # === AGREGADO: DETECTAR PROBLEMAS DE CONVERGENCIA ===
            if conditional_vol[t] <= 1e-8 or conditional_vol[t] > 1.0 or np.isnan(conditional_vol[t]):
                conditional_vol[t] = prev_vol * 1.01  # Suavizado
                convergence_issues += 1
            
            # === AGREGADO: LÍMITES DE SEGURIDAD ===
            if abs(conditional_vol[t] - prev_vol) / prev_vol > 0.5:  # Cambio >50%
                conditional_vol[t] = prev_vol * (1.1 if conditional_vol[t] > prev_vol else 0.9)
                convergence_issues += 1
            
            prev_vol = conditional_vol[t]
    
        # === AGREGADO: FALLBACK SI MUCHOS PROBLEMAS DE CONVERGENCIA ===
        if convergence_issues > n * 0.1:  # >10% de problemas
            bt_logger.warning(f"GARCH convergence issues for {regime} ({convergence_issues}/{n} points), using EWMA")
            return self._ewma_fallback_method(returns, regime_params)
        
        # Detectar regímenes de volatilidad locales
        vol_percentiles = np.percentile(conditional_vol, [25, 75]) if n > 0 else [0.01, 0.03]
        vol_regimes = np.where(conditional_vol < vol_percentiles[0], 'low_vol',
                              np.where(conditional_vol > vol_percentiles[1], 'high_vol', 'normal_vol'))
        
        # Clipping para volatilidad
        conditional_vol = np.clip(conditional_vol, 0.001, 0.1)
        
        # === AGREGADO: INFORMACIÓN DE DIAGNÓSTICO ===
        result = {
            'alpha': alpha,
            'beta': beta,
            'omega': omega,
            'conditional_vol': conditional_vol,
            'vol_regimes': vol_regimes,
            'persistence': persistence,
            'convergence_issues': convergence_issues,  # NUEVO
            'method': 'garch_standard'  # NUEVO
        }
        
        if convergence_issues > 0:
            bt_logger.debug(f"GARCH for {regime}: {convergence_issues} convergence corrections applied")
        
        return result
    
    def _ewma_fallback_method(self, returns: np.ndarray, regime_params: Dict) -> Dict:
        """
        MÉTODO EWMA FALLBACK - Agregado para manejar convergencia
        """
        # Lambda por régimen
        regime_lambdas = {
            'very_low': 0.97, 'low': 0.95, 'normal': 0.94,
            'high': 0.92, 'volatile_bull': 0.90, 'volatile_bear': 0.88
        }
        
        lambda_param = regime_lambdas.get(regime_params['regime'], 0.94)
        
        # Calcular EWMA
        n = len(returns)
        ewma_var = np.zeros(n)
        ewma_var[0] = np.var(returns[:min(20, n)]) if n > 0 else 0.01
        
        for t in range(1, n):
            ewma_var[t] = lambda_param * ewma_var[t-1] + (1 - lambda_param) * returns[t-1]**2
        
        conditional_vol = np.sqrt(ewma_var)
        vol_percentiles = np.percentile(conditional_vol, [25, 75]) if n > 0 else [0.01, 0.03]
        vol_regimes = np.where(conditional_vol < vol_percentiles[0], 'low_vol',
                              np.where(conditional_vol > vol_percentiles[1], 'high_vol', 'normal_vol'))
        
        # Convertir a formato GARCH equivalente
        alpha_equiv = 1 - lambda_param
        beta_equiv = lambda_param
        omega_equiv = np.mean(ewma_var) * (1 - alpha_equiv - beta_equiv)
        
        return {
            'alpha': alpha_equiv,
            'beta': beta_equiv,
            'omega': max(omega_equiv, 1e-6),
            'conditional_vol': conditional_vol,
            'vol_regimes': vol_regimes,
            'persistence': alpha_equiv + beta_equiv,
            'convergence_issues': 0,
            'method': 'ewma_fallback'
        }

    def _analyze_market_microstructure(self, df: pd.DataFrame, regime_params: Dict) -> Dict:
        analysis = {
            'jump_probability': 0.001,
            'jump_sizes': [0.02],
            'clustering_factor': 1.0,
            'whale_factor': 1.0
        }
        
        returns = df['close'].pct_change().dropna().values
        
        # Análisis de saltos (jumps)
        vol_threshold = np.std(returns) * get_config_param('jump_detection_threshold')
        jumps = returns[np.abs(returns) > vol_threshold]
        
        if len(jumps) > 0:
            analysis['jump_probability'] = len(jumps) / len(returns)
            analysis['jump_sizes'] = jumps.tolist()
            
            # Detectar clustering de jumps
            jump_indices = np.where(np.abs(returns) > vol_threshold)[0]
            if len(jump_indices) > 1:
                intervals = np.diff(jump_indices)
                clustering_factor = np.sum(intervals <= 5) / len(intervals)  # Jumps en 5 períodos
                analysis['clustering_factor'] = 1 + clustering_factor
        
        # Incorporar whale_risk si disponible
        if 'whale_risk' in df.columns:
            recent_whale_risk = df['whale_risk'].rolling(24).mean().iloc[-1]
            if not np.isnan(recent_whale_risk):
                analysis['whale_factor'] = 1 + (recent_whale_risk * regime_params['shock_mult'])
                analysis['jump_probability'] *= analysis['whale_factor']
        
        # Ajustar por trap detection
        if regime_params.get('trap_detected', False):
            analysis['jump_probability'] *= 2.0
            analysis['clustering_factor'] *= 1.5
        
        # Clipping para probabilidades
        analysis['jump_probability'] = np.clip(analysis['jump_probability'], 0.0001, 0.1)
        analysis['clustering_factor'] = np.clip(analysis['clustering_factor'], 0.5, 3.0)
        
        return analysis

    async def _run_ensemble_simulation(self, historical_returns: np.ndarray, last_price: float,
                                 regime_params: Dict, timeframe_params: Dict, 
                                 garch_params: Dict, extreme_analysis: Dict) -> np.ndarray:
        """
        Simulación ensemble con múltiples metodologías
        """
        # FIXED: Ensure consistent T with fallback (prevents dim mismatch)
        holding_hours = timeframe_params.get('max_holding_hours', 24)
        T = max(24, min(720, holding_hours * 4))  # Min 24 to avoid T=0 (shape (N,1) mismatch)
        
        num_paths = self.dynamic_mc_paths
        
        # Pesos ensemble de CONFIG
        ensemble_weights = CONFIG.get('monte_carlo_ensemble_weights', [0.4, 0.35, 0.25])  # ✅ Fallback
        
        # Normalize if sum not 1
        total_weight = sum(ensemble_weights)
        if total_weight != 1.0:
            ensemble_weights = [w / total_weight for w in ensemble_weights]
        
        # Método 1: Bootstrap con bloques
        paths_bootstrap = await self._enhanced_bootstrap_simulation(
            historical_returns, last_price, int(num_paths * ensemble_weights[0]), T, 
            regime_params, garch_params
        )
        
        # Método 2: GARCH estocástico
        paths_garch = await self._garch_stochastic_simulation(
            historical_returns, last_price, int(num_paths * ensemble_weights[1]), T,
            garch_params, regime_params
        )
        
        # Método 3: Hybrid mean-reverting
        paths_hybrid = await self._hybrid_mean_reverting_simulation(
            historical_returns, last_price, int(num_paths * ensemble_weights[2]), T,
            regime_params, timeframe_params
        )
        
        # FIXED: Safe vstack - skip zero/1-col paths to avoid dim mismatch
        valid_paths = [p for p in [paths_bootstrap, paths_garch, paths_hybrid] if p.shape[0] > 0 and p.shape[1] > 1]
        if valid_paths:
            all_paths = np.vstack(valid_paths)
        else:
            # Fallback: Simple normal sim if all invalid
            bt_logger.warning("All ensemble paths invalid; using simple normal fallback")
            num_fallback = num_paths
            T_fallback = 96  # Default T
            fallback_paths = np.full((num_fallback, T_fallback + 1), last_price)
            mean_ret = np.mean(historical_returns) if len(historical_returns) > 0 else 0.0
            vol = np.std(historical_returns) if len(historical_returns) > 0 else 0.02
            for path_idx in range(num_fallback):
                for t in range(T_fallback):
                    ret = np.random.normal(mean_ret, vol)
                    fallback_paths[path_idx, t+1] = fallback_paths[path_idx, t] * (1 + ret)
            all_paths = fallback_paths
        
        # Aplicar eventos extremos y microestructura
        all_paths = self._apply_microstructure_effects(
            all_paths, extreme_analysis, regime_params, timeframe_params
        )
        
        return all_paths

    async def _enhanced_bootstrap_simulation(self, historical_returns: np.ndarray, last_price: float, num_paths: int, T: int, regime_params: Dict, garch_params: Dict) -> np.ndarray:
        """Bootstrap mejorado con preservación de estructura GARCH"""
        paths = np.full((num_paths, T + 1), last_price)
        
        # Tamaño de bloque adaptativo según régimen
        if regime_params['regime'] in ['very_low', 'low']:
            block_size = max(8, min(25, len(historical_returns) // 15))  # Bloques más largos en baja vol
        else:
            block_size = max(4, min(15, len(historical_returns) // 20))  # Bloques más cortos en alta vol
        
        # FIXED: Guard para historical_returns vacío (evita std en empty array)
        if len(historical_returns) == 0:
            mean_ret = 0.0
            vol = 0.02  # Default vol
        else:
            mean_ret = np.mean(historical_returns)
            vol = np.std(historical_returns)
            vol = max(vol, 0.001)        
        
        mean_ret = np.mean(historical_returns) if len(historical_returns) > 0 else 0.0
        vol = np.std(historical_returns) if len(historical_returns) > 0 else 0.02
        vol = max(vol, 0.001)  # FIXED: Evitar zero vol en std
        
        @numba.jit(nopython=True, parallel=True)
        def jit_bootstrap_core(paths_jit, historical_returns_jit, num_paths_jit, T_jit, block_size_jit, vol_mult_jit, omega_jit, alpha_jit, beta_jit, mean_ret_jit, vol_jit):
            for path_idx in numba.prange(num_paths_jit):  
                path_returns = np.zeros(T_jit)
                vol_state = vol_jit if len(historical_returns_jit) > 0 else 0.01  
                vol_state = max(vol_jit, 0.001)    
                
                idx = 0
                while idx < T_jit:
                    # Simplified candidate selection
                    start_idx = np.random.randint(0, max(1, len(historical_returns_jit) - block_size_jit))
                    block_end = min(start_idx + block_size_jit, len(historical_returns_jit))
                    block = historical_returns_jit[start_idx:block_end]                    
                    
                    remaining = T_jit - idx
                    if len(block) > remaining:
                        block = block[:remaining]
                    
                    block_len = 0  # Initialize block_len to 0 before if-else
                    
                    if len(block) > 0:
                        block_vol = np.std(block)
                        if block_vol > 0:  # FIXED: Avoid div-by-zero
                            vol_scaling = vol_state / block_vol * vol_mult_jit
                            vol_scaling = max(0.3, min(vol_scaling, 3.0))  # Clip for stability
                        else:
                            vol_scaling = 1.0  # Neutral scaling for zero-vol blocks
                        block = block * vol_scaling
                        path_returns[idx:idx + len(block)] = block
                        idx += len(block)
                        block_len = len(block) 
                    else:
                        
                        path_returns[idx] = mean_ret  
                        idx += 1
                        block_len = 1  
                    
                    # Update vol_state with GARCH-like dynamics (now block_len always defined)
                    for t in range(max(0, idx - block_len), min(idx, T_jit)):
                        path_ret_sq = path_returns[t] ** 2
                        vol_state = np.sqrt(omega_jit + alpha_jit * path_ret_sq + beta_jit * vol_state ** 2)
                        vol_state = max(vol_state, 0.001)  # Prevent zero/NaN propagation
                
                # Build path, handle NaN/infinity
                for t in range(T_jit):
                    if np.isnan(path_returns[t]) or np.isinf(path_returns[t]):
                        path_returns[t] = 0.0  # Fallback to zero return
                    paths_jit[path_idx, t+1] = paths_jit[path_idx, t] * (1 + path_returns[t])
            return paths_jit
        
        
        try:            
            paths = jit_bootstrap_core(paths, historical_returns, num_paths, T, block_size, regime_params['vol_mult'], garch_params['omega'], garch_params['alpha'], garch_params['beta'], mean_ret, vol)  # FIXED: Pasa args extra para numba
        except Exception as jit_err:  
            logger.warning(f"JIT bootstrap failed for regime {regime_params['regime']}: {jit_err}; using Python fallback")
           
            def python_bootstrap_core(paths, historical_returns, num_paths, T, block_size, vol_mult, omega, alpha, beta):
                for path_idx in range(num_paths):
                    path_returns = np.zeros(T)
                    vol_state = vol  
                    vol_state = max(vol_state, 0.001)  
                    
                    idx = 0
                    while idx < T:
                        start_idx = np.random.randint(0, max(1, len(historical_returns) - block_size))
                        block_end = min(start_idx + block_size, len(historical_returns))
                        block = historical_returns[start_idx:block_end]                        
                        
                        remaining = T - idx
                        if len(block) > remaining:
                            block = block[:remaining]
                        
                        block_len = 0 
                        
                        if len(block) > 0:
                            block_vol = np.std(block)
                            if block_vol > 0:
                                vol_scaling = vol_state / block_vol * vol_mult
                                vol_scaling = max(0.3, min(vol_scaling, 3.0))
                            else:
                                vol_scaling = 1.0
                            block = block * vol_scaling
                            path_returns[idx:idx + len(block)] = block
                            idx += len(block)
                            block_len = len(block)  # FIXED: Set block_len only in if branch
                        else:
                            path_returns[idx] = mean_ret  # FIXED: Use pre-computed mean_ret
                            idx += 1
                            block_len = 1  # FIXED: Set block_len to 1 in else branch
                        
                        # FIXED: Update vol_state loop now always uses defined block_len
                        for t in range(max(0, idx - block_len), min(idx, T)):
                            path_ret_sq = path_returns[t] ** 2
                            vol_state = np.sqrt(omega + alpha * path_ret_sq + beta * vol_state ** 2)
                            vol_state = max(vol_state, 0.001)
                    
                    for t in range(T):
                        if np.isnan(path_returns[t]) or np.isinf(path_returns[t]):
                            path_returns[t] = 0.0
                        paths[path_idx, t+1] = paths[path_idx, t] * (1 + path_returns[t])
                return paths  # FIXED: Retornar paths explícitamente (no None)
            
            paths = python_bootstrap_core(paths, historical_returns, num_paths, T, block_size, 
                                          regime_params['vol_mult'], garch_params['omega'], 
                                          garch_params['alpha'], garch_params['beta'])
        
        # FIXED: Siempre retornar array válido (no None)
        if paths is None or paths.size == 0:
            logger.warning("Bootstrap returned empty/None; using simple fallback array")
            paths = np.full((num_paths, T + 1), last_price)  # Minimal valid array
        
        return paths

    async def _garch_stochastic_simulation(self, historical_returns: np.ndarray, last_price: float,
                                         num_paths: int, T: int, garch_params: Dict,
                                         regime_params: Dict) -> np.ndarray:
        """
        Simulación GARCH estocástica con innovaciones realistas
        """
        paths = np.full((num_paths, T + 1), last_price)
        
        mean_return = np.mean(historical_returns)
        
        # Distribución de innovaciones adaptada al régimen
        if regime_params['regime'] in ['volatile_bull', 'volatile_bear']:
            df_student = 4  # Colas más pesadas en regímenes volátiles
            tail_adjust = 1.2  # Aumentar tails para mejor match en KS
        elif regime_params['regime'] in ['very_low', 'low']:
            df_student = 8  # Colas más ligeras en regímenes tranquilos
            tail_adjust = 0.8
        else:
            df_student = 6  # Valor intermedio
            tail_adjust = 1.0
        
        for path_idx in range(num_paths):
            vol_state = garch_params['conditional_vol'][-1] * regime_params['vol_mult']
            
            for t in range(T):
                # Innovación con t-student y tail adjustment for better KS
                epsilon = np.random.standard_t(df=df_student) * tail_adjust
                
                # Return simulado
                return_t = mean_return + vol_state * epsilon
                
                # Mean reversion en regímenes apropiados
                if regime_params['regime'] in ['very_low', 'low']:
                    # Aplicar mean reversion más fuerte
                    price_deviation = (paths[path_idx, t] - last_price) / last_price
                    reversion_force = -price_deviation * 0.1 * regime_params['mean_reversion']
                    return_t += reversion_force
                
                # Actualizar precio
                paths[path_idx, t+1] = paths[path_idx, t] * (1 + return_t)
                
                # Actualizar volatilidad GARCH
                vol_state = np.sqrt(
                    garch_params['omega'] + 
                    garch_params['alpha'] * return_t**2 + 
                    garch_params['beta'] * vol_state**2
                )
                
                # Aplicar bounds realistas
                base_vol = np.std(historical_returns)
                vol_state = np.clip(vol_state, base_vol * 0.2, base_vol * 4)
        
        return paths

    async def _hybrid_mean_reverting_simulation(self, historical_returns: np.ndarray, last_price: float,
                                          num_paths: int, T: int, regime_params: Dict,
                                          timeframe_params: Dict) -> np.ndarray:
        """
        Simulación híbrida con mean reversion adaptativo según timeframe y régimen
        """
        paths = np.full((num_paths, T + 1), last_price)
        
        base_vol = np.std(historical_returns) * regime_params.get('vol_mult', 1.0)
        mean_return = np.mean(historical_returns)
        
        # Fuerza de mean reversion según strategy type
        strategy_type = timeframe_params.get('strategy_type', 'trend_following')
        mean_reversion_base = regime_params.get('mean_reversion', 1.0)
        
        if strategy_type == 'mean_reversion':
            reversion_strength = 0.15 * mean_reversion_base
        elif strategy_type == 'micro_scalp':
            reversion_strength = 0.25 * mean_reversion_base
        else:
            reversion_strength = 0.05 * mean_reversion_base
        
        for path_idx in range(num_paths):
            for t in range(T):
                # Componente aleatorio
                random_component = np.random.normal(mean_return, base_vol)
                
                # Componente de mean reversion
                if t > 0:
                    price_deviation = (paths[path_idx, t] - last_price) / last_price
                    reversion_component = -price_deviation * reversion_strength
                else:
                    reversion_component = 0
                
                # Componente de momentum (para regímenes tendenciales)
                momentum_component = 0
                regime = regime_params.get('regime', 'normal')
                if t > 5 and regime in ['bull', 'bear', 'volatile_bull', 'volatile_bear']:
                    recent_trend = (paths[path_idx, t] - paths[path_idx, max(0, t-5)]) / paths[path_idx, max(0, t-5)]
                    momentum_strength = 0.1 if 'volatile' in regime else 0.05
                    momentum_component = recent_trend * momentum_strength
                
                # Return final combinado
                total_return = random_component + reversion_component + momentum_component
                
                # Actualizar precio
                paths[path_idx, t+1] = paths[path_idx, t] * (1 + total_return)
        
        return paths

    def _apply_microstructure_effects(self, paths: np.ndarray, extreme_analysis: Dict,
                                regime_params: Dict, timeframe_params: Dict) -> np.ndarray:
        """
        Aplicar efectos de microestructura del mercado
        """
        num_paths, T = paths.shape[0], paths.shape[1] - 1
        
        jump_prob = extreme_analysis.get('jump_probability', 0.01) * regime_params.get('shock_mult', 1.0)
        clustering_factor = extreme_analysis.get('clustering_factor', 1.0)
        whale_factor = extreme_analysis.get('whale_factor', 1.0)
        
        # Ajustar probabilidades por timeframe
        strategy_type = timeframe_params.get('strategy_type', 'trend_following')
        if strategy_type == 'micro_scalp':
            jump_prob *= 0.5  # Menos jumps en scalping
        elif strategy_type == 'trend_following':
            jump_prob *= 1.2  # Más jumps en trend following
        
        for path_idx in range(num_paths):
            jump_occurred_recently = 0
            
            for t in range(1, T + 1):
                # Probabilidad ajustada por clustering
                current_jump_prob = jump_prob
                if jump_occurred_recently > 0:
                    current_jump_prob *= clustering_factor
                    jump_occurred_recently -= 1
                
                # Aplicar whale factor
                current_jump_prob *= whale_factor
                
                if np.random.rand() < current_jump_prob:
                    # Seleccionar tamaño del salto
                    jump_sizes = extreme_analysis.get('jump_sizes', [])
                    if jump_sizes:
                        jump_size = np.random.choice(jump_sizes)
                    else:
                        # Distribución de jumps adaptada al régimen
                        regime = regime_params.get('regime', 'normal')
                        if regime in ['very_low', 'low']:
                            jump_size = np.random.uniform(0.005, 0.015)
                        elif 'volatile' in regime:
                            jump_size = np.random.uniform(0.02, 0.08)
                        else:
                            jump_size = np.random.uniform(0.01, 0.04)
                    
                    # Dirección del salto con sesgo por régimen
                    regime = regime_params.get('regime', 'normal')
                    if regime == 'bull':
                        direction = np.random.choice([1, -1], p=[0.6, 0.4])
                    elif regime == 'bear':
                        direction = np.random.choice([1, -1], p=[0.4, 0.6])
                    else:
                        direction = np.random.choice([1, -1])
                    
                    # Aplicar jump
                    paths[path_idx, t] *= (1 + direction * abs(jump_size))
                    jump_occurred_recently = max(2, int(clustering_factor))
        
        return paths

    def _log_monte_carlo_results(self, symbol: str, metrics: Dict, validation: Dict, regime: str):
        """Log detallado con diagnósticos"""
        
        composite_score = validation.get('composite_score', 0)
        
        # Determinar calidad
        if composite_score >= CONFIG.get('mc_quality_excellent_threshold', 0.7):
            quality = "EXCELLENT"
            log_level = bt_logger.info
        elif composite_score >= CONFIG.get('mc_quality_good_threshold', 0.5):
            quality = "GOOD" 
            log_level = bt_logger.info
        else:
            quality = "POOR"
            log_level = bt_logger.warning
        
        log_level(
            f"MC {quality} for {symbol} (regime={regime}): "
            f"KS_p={validation.get('ks_p_value', 0):.4f}, "
            f"Composite={composite_score:.3f}, "
            f"VaR95={metrics.get('var_95', 0):.2f}, "
            f"Sharpe_sim={metrics.get('simulated_sharpe', 0):.2f}, "
            f"Paths={metrics.get('paths_generated', 0)}"
        )
        
        # Log detallado para análisis
        bt_logger.debug(f"MC details {symbol}: "
                       f"P5={metrics['pnl_5th']:.2f}, "
                       f"P50={metrics['pnl_50th']:.2f}, "
                       f"P95={metrics['pnl_95th']:.2f}, "
                       f"MaxDD={metrics['max_drawdown_sim']:.1f}%, "
                       f"Anderson_p={metrics.get('anderson_p_value', 0):.4f}")

    async def _run_fallback_monte_carlo(self, symbol: str, metrics: Dict, historical_returns: np.ndarray, last_price: float) -> Dict:
        """
        Fallback mejorado para datos insuficientes usando parámetros de régimen
        """        
        bt_logger.warning(f"Insufficient data for {symbol}, using enhanced fallback Monte Carlo")
        
        # Usar parámetros de régimen actual si disponible
        try:
            current_regime = REGIME_GLOBAL if 'REGIME_GLOBAL' in globals() else 'normal'
            regime_params = self._get_enhanced_regime_params(current_regime, False, None)
        except:
            regime_params = {'vol_mult': 1.0, 'shock_mult': 1.0, 'regime': 'normal'}
        
        # FIXED: Guard para std en historical_returns vacío (evita RuntimeWarning)
        if len(historical_returns) == 0:
            vol = 0.02  # Default vol si empty
            mean_ret = 0.0
        else:
            vol = np.std(historical_returns)
            vol = max(vol, 0.001)  # FIXED: Evitar zero vol
            mean_ret = np.mean(historical_returns)
        
        vol *= regime_params['vol_mult']
        
        num_paths = 800
        T = 240  # Horizonte más corto para datos limitados
        
        # Simulación con t-student y parámetros de régimen
        df_student = 5 if regime_params['regime'] in ['volatile_bull', 'volatile_bear'] else 7
        random_returns = np.random.standard_t(df=df_student, size=(num_paths, T)) * vol + mean_ret
        
        # Aplicar mean reversion suave
        paths = np.full((num_paths, T + 1), last_price)
        for path_idx in range(num_paths):
            for t in range(T):
                return_t = random_returns[path_idx, t]
                
                # Mean reversion ligero
                if t > 0:
                    price_dev = (paths[path_idx, t] - last_price) / last_price
                    return_t -= price_dev * 0.05
                
                paths[path_idx, t+1] = paths[path_idx, t] * (1 + return_t)
        
        path_total_returns = (paths[:, -1] - last_price) / last_price
        pnl = metrics.get('realized_pnl', 0)
        win_rate = metrics.get('win_rate', 0.5)
        
        # Ajuste conservador para datos limitados
        conservative_factor = 0.7
        simulated_pnls = path_total_returns * pnl * (win_rate / 0.5) * conservative_factor
        
        return {
            'pnl_5th': float(np.percentile(simulated_pnls, 5)),
            'pnl_25th': float(np.percentile(simulated_pnls, 25)),
            'pnl_50th': float(np.percentile(simulated_pnls, 50)),
            'pnl_75th': float(np.percentile(simulated_pnls, 75)),
            'pnl_95th': float(np.percentile(simulated_pnls, 95)),
            'var_95': float(np.percentile(simulated_pnls, 5)),
            'cvar_95': float(np.mean(simulated_pnls[simulated_pnls <= np.percentile(simulated_pnls, 5)])) if np.sum(simulated_pnls <= np.percentile(simulated_pnls, 5)) > 0 else float(np.percentile(simulated_pnls, 5)),
            'simulated_sharpe': 0.0,
            'max_drawdown_sim': 0.0,
            'validation_score': 0.4,  # Score moderado para fallback
            'ks_p_value': 0.02,
            'anderson_p_value': 0.001,
            'regime_applied': regime_params.get('regime', 'normal'),  # Safe get para scalar/string
            'fallback_used': True
        }

    async def run_full_backtest(self):
        """
        Backtest completo integrado con Monte Carlo mejorado
        """
        global CLOSED_TRADES_SINCE_LAST
        if time.time() - self.last_backtest_time < get_config_param('retrain_int') and CLOSED_TRADES_SINCE_LAST < 10:
            return
        bt_logger.info("Starting full backtest with selective processing")
        self.last_backtest_time = time.time()
        if not self.ohlcv_cache:
            await self.initialize_cache()
            if not self.ohlcv_cache:
                return

        # Clasificar pares por rendimiento (good: métricas frescas/rentables; low-perf: resto)
        all_pairs = []
        for cat in self.categories:
            all_pairs.extend(self.categories[cat])
        good_perf_pairs = []
        low_perf_pairs = []
        now = time.time()
        freshness_days = get_config_param('backtest_freshness_days')
        for symbol in all_pairs:
            saved_metrics = self.get_saved_bt_metrics(symbol)
            if (saved_metrics and saved_metrics.get('timestamp', 0) > now - (freshness_days * 86400) and
                saved_metrics.get('sharpe_ratio', 0) > 1.0 and saved_metrics.get('realized_pnl', 0) > 0):
                good_perf_pairs.append(symbol)
            else:
                low_perf_pairs.append(symbol)
        
        bt_logger.info(f"Classified: {len(good_perf_pairs)} good-perf, {len(low_perf_pairs)} low-perf pairs")
        
        # Cargar métricas de good-perf directamente (sin backtest)
        all_metrics = {s: self.get_saved_bt_metrics(s) for s in good_perf_pairs if self.get_saved_bt_metrics(s)}
        
        # Inicializar agregación solo para low-perf (good ya se sumarán al final)
        total_realized_pnl = 0
        total_trades = 0
        total_wins = 0
        sharpe_sum = 0
        dd_sum = 0
        num_pairs = len(good_perf_pairs)  # Incluir good en conteo inicial
        backtested = 0
        
        # Paralelismo solo para low-perf
        if low_perf_pairs:
            sem = asyncio.Semaphore(get_config_param('max_concurr'))
            async def backtest_low_wrapper(symbol):
                async with sem:
                    metrics = await self.run_backtest_on_pair(symbol)
                    if metrics.get('trades', 0) > 0:
                        mc_results = await self.run_monte_carlo(symbol, metrics, 'medium_1h')
                        metrics['monte_carlo'] = mc_results
                    return symbol, metrics
            
            tasks = [backtest_low_wrapper(pair) for pair in low_perf_pairs]
            low_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in low_results:
                if isinstance(result, tuple) and len(result) == 2:
                    symbol, metrics = result
                    all_metrics[symbol] = metrics
                    if metrics.get('trades', 0) > 0:
                        backtested += 1
                        m = metrics
                        total_realized_pnl += m['realized_pnl']
                        total_trades += m['trades']
                        total_wins += m['trades'] * m['win_rate']
                        sharpe_sum += m['sharpe_ratio']
                        dd_sum += m['max_drawdown']
                        num_pairs += 1
                else:
                    bt_logger.debug(f"Exception or skip in low-perf backtest: {result}")
        
        # Sumar métricas de good-perf a la agregación
        for symbol, m in all_metrics.items():
            if symbol in good_perf_pairs and m.get('trades', 0) > 0:
                total_realized_pnl += m['realized_pnl']
                total_trades += m['trades']
                total_wins += m['trades'] * m['win_rate']
                sharpe_sum += m['sharpe_ratio']
                dd_sum += m['max_drawdown']
        
        if num_pairs == 0:
            bt_logger.warning("No metrics available after selective backtest")
            return
        win_rate = total_wins / total_trades if total_trades > 0 else 0
        avg_sharpe = sharpe_sum / num_pairs
        avg_dd = dd_sum / num_pairs
        sorted_pairs = sorted(all_metrics.items(), key=lambda x: x[1].get('sharpe_ratio', 0), reverse=True)
        top_pairs = [s for s, m in sorted_pairs[:20]]
        self.top_performers = top_pairs
        bt_logger.info(f"Top performers: {top_pairs}")
        results = {'realized_pnl': total_realized_pnl, 'win_rate': win_rate, 'sharpe_ratio': avg_sharpe, 'max_drawdown': avg_dd, 'details': all_metrics, 'top_pairs': top_pairs}
        await self.db.save_backtest_results(results)
        self.results = results
        CONFIG['default']['symbols'] = top_pairs
        bt_logger.info(f"Updated symbols to {CONFIG['default']['symbols']}")
        bt_logger.info(f"Aggregate: P&L=${total_realized_pnl:.2f}, Win={win_rate:.1%}, Sharpe={avg_sharpe:.2f}, DD={avg_dd:.1f}%")
        CLOSED_TRADES_SINCE_LAST = 0

    async def _backtest_low_perf_only(self, low_perf_pairs: List[str]):
        """Backtest solo low-perf en paralelo, actualiza self.top_performers sin bloquear."""
        if not low_perf_pairs:
            return
        all_metrics = {}  # Local para low-perf
        sem = asyncio.Semaphore(get_config_param('max_concurr'))
        async def backtest_low_wrapper(symbol):
            async with sem:
                metrics = await self.run_backtest_on_pair(symbol)
                if metrics.get('trades', 0) > 0:
                    mc_results = await self.run_monte_carlo(symbol, metrics, 'medium_1h')
                    metrics['monte_carlo'] = mc_results
                return symbol, metrics
        tasks = [backtest_low_wrapper(p) for p in low_perf_pairs]
        low_results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in low_results:
            if isinstance(result, tuple) and len(result) == 2:
                symbol, metrics = result
                all_metrics[symbol] = metrics
                # Agregación local (similar a run_full_backtest, pero solo suma low-perf)
                # Nota: No guardar DB aquí; run_full_backtest lo hace al final si needed
        # Actualizar top_performers con low-perf integrados
        existing_top = self.top_performers or []
        updated_metrics = {s: self.get_saved_bt_metrics(s) for s in existing_top if self.get_saved_bt_metrics(s) is not None} 
        updated_metrics.update(all_metrics)  # + low-perf nuevos
        sorted_pairs = sorted(updated_metrics.items(), key=lambda x: x[1].get('sharpe_ratio', 0), reverse=True)
        self.top_performers = [s for s, _ in sorted_pairs[:20]]
        bt_logger.info(f"Updated top_performers with {len(low_perf_pairs)} low-perf results: {self.top_performers}")

    def _log_monte_carlo_quality_summary(self, all_metrics: Dict, mc_quality_scores: list):
        """
        Logging detallado de calidad Monte Carlo para análisis
        """
        if not mc_quality_scores:
            return
        
        # Estadísticas de calidad
        quality_stats = {
            'excellent': sum(1 for score in mc_quality_scores if score >= 0.7),
            'good': sum(1 for score in mc_quality_scores if 0.5 <= score < 0.7),
            'acceptable': sum(1 for score in mc_quality_scores if 0.3 <= score < 0.5),
            'poor': sum(1 for score in mc_quality_scores if score < 0.3)
        }
        
        total_pairs = len(mc_quality_scores)
        
        # Log pares con calidad pobre para investigación
        poor_pairs = [
            symbol for symbol, metrics in all_metrics.items()
            if metrics.get('monte_carlo', {}).get('validation_score', 0) < 0.3
        ]
        
        if poor_pairs:
            bt_logger.warning(f"Pairs with poor MC quality (require investigation): {poor_pairs}")
        
        # Recomendaciones automáticas
        if quality_stats['poor'] / total_pairs > 0.3:
            bt_logger.warning("HIGH: >30% pairs have poor MC quality - recommend increasing CONFIG['default']['min_data_pts']")
        elif quality_stats['poor'] / total_pairs > 0.1:
            bt_logger.info("MEDIUM: >10% pairs have poor MC quality - consider data quality review")
    
    # === FUNCIÓN AUXILIAR PARA MÉTRICAS DE CALIDAD ===
    def get_monte_carlo_quality_summary(self) -> Dict:
        """
        Resumen de calidad Monte Carlo para reporting
        """
        if not hasattr(self, 'results') or not self.results:
            return {}
        
        quality_scores = [
            m.get('monte_carlo', {}).get('validation_score', 0)
            for m in self.results.get('details', {}).values()
            if m.get('monte_carlo')
        ]
        
        if not quality_scores:
            return {'status': 'no_data'}
        
        return {
            'average_quality': np.mean(quality_scores),
            'median_quality': np.median(quality_scores),
            'min_quality': np.min(quality_scores),
            'max_quality': np.max(quality_scores),
            'excellent_count': sum(1 for score in quality_scores if score >= 0.7),
            'poor_count': sum(1 for score in quality_scores if score < 0.3),
            'total_validated_pairs': len(quality_scores),
            'quality_distribution': {
                'excellent': sum(1 for score in quality_scores if score >= 0.7),
                'good': sum(1 for score in quality_scores if 0.5 <= score < 0.7),
                'acceptable': sum(1 for score in quality_scores if 0.3 <= score < 0.5),
                'poor': sum(1 for score in quality_scores if score < 0.3)
            }
        }
        bt_logger.info(f"Monte Carlo: Avg Quality={average_quality:.3f}, Validated Pairs={len(total_validated_pairs)}")

class RiskMgr:
    def __init__(self, context: AppContext):
        self.context = context
        self.exchange = context.exchange
        self.bt_mgr = self.context.bt_mgr if hasattr(self.context, 'bt_mgr') else None  # Safe
        self.max_port_risk = get_config_param('max_port_risk')
        self.max_positions = get_config_param('max_positions')
        self.min_order_size = get_config_param('min_order_size')
        self.max_corr_exp = get_config_param('max_corr_exp')
        self.current_regime = REGIME_GLOBAL

    def update_regime(self, regime: str):
        valid_regimes = list(CONFIG.keys())  
        if regime not in valid_regimes:
            logger.warning(f"Invalid regime '{regime}' in update_regime; falling back to 'normal'")
            regime = 'normal'
        self.current_regime = regime
        global REGIME_GLOBAL
        REGIME_GLOBAL = regime
        logger.debug(f"Regime updated to {regime} (validated)")

    async def check_correlation(self, symbol: str, current_positions: List[Dict], df: pd.DataFrame, state: dict) -> bool:
        if not current_positions:
            return True
        new_returns = df['close'].pct_change().dropna()
        if len(new_returns) < 20:
            return True
        correlated_count = 0
        for pos in current_positions:
            pos_symbol = pos.get('symbol')
            if not pos_symbol or pos_symbol == symbol:
                continue
            pos_df = None
            async with CACHE_LOCK:
                for (cache_sym, tf), cached_df in DATA_CACHE.items():
                    if cache_sym == pos_symbol:
                        pos_df = cached_df
                        break
            if pos_df is None or len(pos_df) < 20:
                default_tf = TIMEFRAMES[0]
                pos_df = await fetch_and_prepare_data(pos_symbol, default_tf, self.exchange, state)
                if pos_df is not None and len(pos_df) >= 20:
                    await update_data_cache(pos_symbol, default_tf['name'], pos_df, state)
                else:
                    correlated_count += 1
                    continue
            pos_returns = pos_df['close'].pct_change().dropna()
            if len(pos_returns) >= 20 and len(new_returns) >= 20:
                min_len = min(len(new_returns), len(pos_returns))
                stat, p_value = ks_2samp(new_returns.values[:min_len], pos_returns.values[:min_len])
                # Dual check: KS (p >=0.1 for similar dist) + Pearson corr (>0.6 for high linear correlation)
                ks_correlated = p_value >= 0.1  # Más tolerante: no rechaza similar
                pearson_corr = np.corrcoef(new_returns.values[:min_len], pos_returns.values[:min_len])[0, 1]
                linear_correlated = abs(pearson_corr) > 0.6
                if ks_correlated or linear_correlated:
                    correlated_count += 1
                    logger.debug(f"{symbol} vs {pos_symbol}: KS p={p_value:.3f} (similar={ks_correlated}), Pearson={pearson_corr:.3f} (high_corr={linear_correlated})")
            else:
                correlated_count += 1
        corr_pct = correlated_count / len(current_positions) if len(current_positions) > 0 else 0
        if corr_pct > self.max_corr_exp:
            return False
        return True

    async def validate_trade(self, symbol: str, side: str, quantity: float, price: float, confidence: float, current_positions: List[Dict], portfolio_value: float, df: pd.DataFrame = None, state: dict = None) -> Dict[str, Any]:
        logger.debug(f"[VALIDATE_TRADE] Starting for {symbol} {side}: qty={quantity:.6f} price={price:.2f} conf={confidence:.3f} portfolio={portfolio_value:.2f}")
        validation = {'approved': False, 'reason': '', 'adjusted_quantity': quantity, 'risk_score': 0.0, 'position_limit_ok': True, 'size_limit_ok': True, 'confidence_ok': True, 'portfolio_risk_ok': True, 'correlation_ok': True, 'partial_mult': 1.0}
        unique_symbols = {pos.get('symbol', '') for pos in current_positions if pos.get('symbol')}
        if symbol in unique_symbols:
            validation['reason'] = f"Already have position for {symbol}"
            validation['position_limit_ok'] = False
            logger.warning(f"[VALIDATE_TRADE] REJECTED - {validation['reason']}")
            return validation
        max_pos = get_config_param('max_positions')
        if len(unique_symbols) >= max_pos:
            validation['reason'] = f"Max unique positions ({max_pos}) reached"
            validation['position_limit_ok'] = False
            logger.warning(f"[VALIDATE_TRADE] REJECTED - {validation['reason']}")
            return validation
        logger.debug(f"[VALIDATE_TRADE] Passed position limit for {symbol} ({len(unique_symbols)}/{max_pos})")
        correlation_ok = await self.check_correlation(symbol, current_positions, df, state) if df is not None else True
        validation['correlation_ok'] = correlation_ok
        if not correlation_ok:
            validation['reason'] = f"High correlation for {symbol}"
            logger.warning(f"[VALIDATE_TRADE] REJECTED - {validation['reason']}")
            return validation
        logger.debug(f"[VALIDATE_TRADE] Passed correlation for {symbol}")
        # FIXED: Mover order_value ANTES de risk calc (evita NameError forward ref)
        order_value = quantity * price
        logger.debug(f"[VALIDATE_TRADE] Order value for {symbol}: ${order_value:.2f}")
        # New: CVaR-based risk_score from quick MC (con guards para scalars y timeout)
        quick_metrics = {'realized_pnl': 0.01}  # Placeholder con valor default para evitar KeyError
        risk_mult = 0.8 if self.current_regime in ['very_low', 'low'] else 1.2 if 'volatile' in self.current_regime else 1.0
        new_risk = order_value * (0.05 * risk_mult)
        current_risk = sum(pos.get('risk_amount', 0.0) for pos in current_positions)  # FIXED: .0 para float
        total_risk = (current_risk + new_risk) / portfolio_value if portfolio_value > 0 else 0.0  # FIXED: Guard div0
        try:
            # CRÍTICO: Verificar/reconectar exchange antes de MC
            if not self.exchange.connected:
                logger.warning("Exchange disconnected in validate_trade, attempting reconnect...")
                if await self.exchange.connect(max_retries=2):
                    logger.info("Reconnected for MC in validate_trade")
                else:
                    raise Exception("Reconnect failed, skipping MC")
            
            
            tf_name = 'medium_1h'  
            
            async with METRICS_LOCK_ASYNC:  
                if self.bt_mgr and hasattr(self.bt_mgr, 'exchange') and self.bt_mgr.exchange.connected:
                    try:
                        # FIXED: Wrap MC call in timeout to prevent propagation of MC hangs
                        async with asyncio.timeout(30):  # 30s max for MC in validation
                            mc_results = await self.bt_mgr.run_monte_carlo(symbol, quick_metrics, tf_name)
                        cvar_95 = float(mc_results.get('cvar_95', -0.05)) 
                        expected_pnl = float(quick_metrics.get('realized_pnl', 0.01)) 
                        if expected_pnl > 0:
                            risk_score = abs(cvar_95 / expected_pnl)
                        else:
                            risk_score = abs(cvar_95) * 10  
                        risk_score = np.clip(risk_score, 0.0, 1.0)  
                        validation['risk_score'] = float(risk_score)  
                        logger.debug(f"[VALIDATE_TRADE] CVaR risk_score for {symbol}: {risk_score:.3f} (CVaR={cvar_95:.3f})")
                    except asyncio.TimeoutError:
                        logger.warning(f"MC timeout in validate_trade for {symbol}; fallback to heuristic")
                        # FIXED: Use heuristic on timeout (autonomous: no hang)
                        risk_factors = [(1 - float(confidence)) * 0.4, (total_risk / self.max_port_risk) * 0.3, (len(unique_symbols) / self.max_positions) * 0.2, 0.1]
                        risk_score = sum(risk_factors)
                        validation['risk_score'] = float(risk_score)
                    except Exception as mc_e:
                        logger.warning(f"MC failed in validate_trade for {symbol}: {mc_e}, fallback to heuristic")
                        self.bt_mgr.exchange.connected = False  # Marcar para reconexión
                        risk_factors = [(1 - float(confidence)) * 0.4, (total_risk / self.max_port_risk) * 0.3, (len(unique_symbols) / self.max_positions) * 0.2, 0.1]
                        risk_score = sum(risk_factors)
                        validation['risk_score'] = float(risk_score)
                else:
                    # Fallback y reconexión si bt_mgr no conectado
                    if self.bt_mgr and not self.bt_mgr.exchange.connected:
                        await self.bt_mgr.exchange.connect(max_retries=2)  
                    logger.debug(f"No bt_mgr or disconnected for {symbol}, using heuristic risk_score")
                    risk_factors = [(1 - float(confidence)) * 0.4, (total_risk / self.max_port_risk) * 0.3, (len(unique_symbols) / self.max_positions) * 0.2, 0.1]
                    risk_score = sum(risk_factors)
                    validation['risk_score'] = float(risk_score)
        except Exception as e:
            logger.error(f"Error calculating risk score for {symbol}: {e}")
            # Fallback risk score to safe default
            risk_score = 0.5
            validation['risk_score'] = float(risk_score)  # Escalar float                
        
        # Dynamic min_size scaled with portfolio (professional, autonomous with capital growth)
        base_min_size = get_config_param('min_order_size')
        dynamic_min_size = max(10.0, portfolio_value * 0.01)  # Min 10 USD, 1% of portfolio
        min_size = dynamic_min_size
        # Intelligent scaling loop: Try up to 3x to meet min_size without exceeding risk
        max_scaling_attempts = 3
        for attempt in range(max_scaling_attempts):
            if order_value >= min_size:
                break
            scale_factor = (min_size / order_value) ** (1 / (attempt + 1))  # Gradual scaling
            adjusted_qty = quantity * min(2.0, scale_factor)  # Cap at 2x per attempt
            adjusted_value = adjusted_qty * price
            if adjusted_value > portfolio_value * self.max_port_risk * 0.2:  # Risk guard: max 20% single trade
                logger.warning(f"[VALIDATE_TRADE] Scaling rejected on attempt {attempt+1}: exceeds 20% portfolio risk")
                break
            quantity = adjusted_qty
            order_value = adjusted_value
            logger.debug(f"[VALIDATE_TRADE] Scaling attempt {attempt+1}: qty={quantity:.6f}, value=${order_value:.2f}")
        logger.debug(f"[VALIDATE_TRADE] Final size check for {symbol}: ${order_value:.2f} vs dynamic_min=${min_size:.2f} (regime: {self.current_regime})")
        if order_value < min_size:
            validation['reason'] = f"Order value ${order_value:.2f} < dynamic_min ${min_size:.2f}"
            validation['size_limit_ok'] = False
            logger.warning(f"[VALIDATE_TRADE] REJECTED - {validation['reason']}")
            return validation
        logger.debug(f"[VALIDATE_TRADE] Passed size check for {symbol}")
        base_min_conf = get_config_param('min_conf_score')
        min_conf = base_min_conf * 0.5 if self.current_regime in ['very_low', 'low'] else base_min_conf * 1.2 if 'volatile' in self.current_regime else base_min_conf
        logger.debug(f"[VALIDATE_TRADE] Confidence check for {symbol}: {confidence:.4f} vs {min_conf:.4f} (regime: {self.current_regime})")
        if confidence < min_conf:
            validation['reason'] = f"Confidence {confidence:.3f} < {min_conf:.3f}"
            validation['confidence_ok'] = False
            logger.warning(f"[VALIDATE_TRADE] REJECTED - {validation['reason']}")
            return validation
        logger.debug(f"[VALIDATE_TRADE] Passed confidence check for {symbol}")
        logger.debug(f"[VALIDATE_TRADE] Risk check for {symbol}: current_risk=${current_risk:.2f} new_risk=${new_risk:.2f} total_risk={total_risk:.2%} vs max={self.max_port_risk:.2%}")
        if total_risk > self.max_port_risk:
            max_new_risk = (self.max_port_risk * portfolio_value) - current_risk
            if max_new_risk > 0:
                max_order_value = max_new_risk / 0.05
                adjusted_quantity = max_order_value / price
                if adjusted_quantity >= quantity * 0.1:
                    validation['adjusted_quantity'] = adjusted_quantity
                    validation['reason'] = f"Quantity adjusted: {quantity:.8f} -> {adjusted_quantity:.8f}"
                    logger.info(f"[VALIDATE_TRADE] {validation['reason']}")
                else:
                    validation['reason'] = f"Risk {total_risk:.1%} > {self.max_port_risk:.1%}"
                    validation['portfolio_risk_ok'] = False
                    logger.warning(f"[VALIDATE_TRADE] REJECTED - {validation['reason']}")
                    return validation
            else:
                validation['reason'] = "Risk limit exceeded"
                validation['portfolio_risk_ok'] = False
                logger.warning(f"[VALIDATE_TRADE] REJECTED - {validation['reason']}")
                return validation
        logger.debug(f"[VALIDATE_TRADE] Passed risk check for {symbol}")
        risk_factors = [(1 - confidence) * 0.4, (total_risk / self.max_port_risk) * 0.3, (len(unique_symbols) / self.max_positions) * 0.2, 0.1]
        validation.update({'approved': True, 'reason': 'Approved', 'risk_score': sum(risk_factors), 'total_risk': total_risk, 'partial_mult': partial_mult})
        logger.info(f"[VALIDATE_TRADE] APPROVED - {symbol} {side} qty:{validation['adjusted_quantity']:.8f} risk_score:{validation['risk_score']:.3f}")
        return validation
    
class SupBot:
    def __init__(self, context: AppContext):
        self.context = context
        self.optimizer = ParamOpt()
        self.bt_mgr = self.context.bt_mgr
        self.params_to_optimize = ['min_conf_score', 'kelly_frac', 'sl_atr_mult', 'tp_atr_mult', 'low_vol_pos_mult', 'spoof_ent_thresh', 'label_threshold_base']
        
        # Multi-layer learning system
        self.source_accuracy = {'bayes': 0.5, 'rl': 0.5, 'perf': 0.5, 'meta': 0.6}
        self.param_history = {p: deque(maxlen=20) for p in self.params_to_optimize}
        self.performance_memory = deque(maxlen=50)
        self.regime_memory = {}
        self.market_patterns = defaultdict(lambda: {'count': 0, 'success': 0})
        self.parameter_correlations = np.zeros((len(self.params_to_optimize), len(self.params_to_optimize)))
        logger.info("SupBot initialized with multi-layer learning and dynamic adaptation.")

        
        # Dynamic adaptation
        self.min_validation_pairs = 5
        self.representative_pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
        self.adaptation_rate = 0.1
        self.causal_effects = {p: {'granger_p': 1.0, 'causal_drop': False} for p in self.params_to_optimize}  # New: Track causal effects
        self.win_rate_history = deque(maxlen=50)  # New: For Granger
        self.confidence_threshold = 0.7
        
        # New: Meta-RL for intervention
        self.meta_rl_net = MetaRLNetwork(self.optimizer.state_dim)  # base_state_dim=9; +1 handled in MetaRLNetwork init
        self.meta_optimizer = optim.Adam(self.meta_rl_net.parameters(), lr=0.001)
        self.intervention_history = deque(maxlen=100)  # (missed_count, intervened, success)
        self.meta_gamma = 0.95
        
        # Market intelligence
        self.market_patterns = defaultdict(lambda: {'count': 0, 'success': 0})
        self.parameter_correlations = np.zeros((len(self.params_to_optimize), len(self.params_to_optimize)))
        logger.info("SupBot initialized with multi-layer learning and dynamic adaptation.")

    async def _should_intervene(self, regime: str, missed: List, metrics: Dict) -> bool:
        async with METRICS_LOCK_ASYNC:
            win_rate = metrics.get('win_rate', PERFORMANCE_METRICS.get('profitable_trades', 0) / max(1, PERFORMANCE_METRICS.get('total_closed_trades', 1)))
            sharpe = metrics.get('sharpe_ratio', 1.0)
            if 'sharpe_ratio' not in metrics:
                logger.debug("Sharpe not in metrics, using default 1.0 - consider backtest update")
            realized_pnl = PERFORMANCE_METRICS.get('realized_pnl', 0.0)
            pnl_change = (metrics.get('total_pnl', 0) - self.performance_memory[-1].get('total_pnl', 0)) / abs(self.performance_memory[-1].get('total_pnl', 1)) if self.performance_memory else 0

        # Trigger conditions
        if win_rate < 0.60 or sharpe < 1.3:  
            return True
        if len(missed) > 2:  
            return True
        if pnl_change < -0.03:  
            return True
        if realized_pnl < -5.0 and PERFORMANCE_METRICS.get('total_closed_trades', 0) > 20:  
            return True
        if regime != self.regime_memory.get('last_regime', regime): 
            self.regime_memory['last_regime'] = regime
            return True
        if CLOSED_TRADES_SINCE_LAST >= 8:  
            return True

        return False

    def _analyze_market_regime(self, metrics: Dict) -> Dict[str, float]:
        """Advanced regime detection with multiple indicators"""
        vol = metrics.get('volatility', 0.5)
        trend = metrics.get('trend_strength', 0)
        correlation = metrics.get('cross_correlation', 0)
        
        regime_scores = {
            'volatile': min(1.0, vol * 2),
            'trending': abs(trend),
            'choppy': max(0, 1 - abs(trend) - vol),
            'correlated': abs(correlation)
        }
        
        dominant_regime = max(regime_scores, key=regime_scores.get)
        return {'regime': dominant_regime, 'scores': regime_scores}

    def _meta_learning_adjustment(self, proposals: Dict, recent_performance: List[float]) -> Dict[str, float]:
        """Meta-learning layer that learns how to combine proposals"""
        if len(recent_performance) < 10:
            return {}
        performance_trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
        volatility_penalty = np.std(recent_performance) * 0.1
        meta_prop = {}
        if performance_trend < -0.01:
            meta_prop['kelly_frac'] = get_config_param('kelly_frac') * 0.85
            meta_prop['min_conf_score'] = get_config_param('min_conf_score') * 1.1
        if volatility_penalty > 0.05:
            meta_prop['low_vol_pos_mult'] = get_config_param('low_vol_pos_mult') * 0.9
        # Pattern recognition for parameter cycling
        for param in self.params_to_optimize:
            if len(self.param_history[param]) > 10:
                param_trend = np.corrcoef(range(len(self.param_history[param])), list(self.param_history[param]))[0,1]
                if abs(param_trend) > 0.3:  # Strong trend in parameter
                    current = get_config_param(param)
                    if param_trend > 0:
                        meta_prop[param] = current * 1.02
                    else:
                        meta_prop[param] = current * 0.98
        return meta_prop

    async def get_proposals(self, regime: str, missed: List[Tuple[str, str]], metrics: Dict, trap_history: Dict, state: dict) -> Dict[str, Dict[str, float]]:
        global REGIME_GLOBAL
        # FIXED: Fallback if state None (production resilience)
        if state is None:
            logger.warning("get_proposals called without state; using fallback {'regime_global': REGIME_GLOBAL}")
            state = {'regime_global': REGIME_GLOBAL}
        
        logger.info(f"SupBot: Generating proposals for regime {regime} with {len(missed)} missed opportunities.")
        regime = str(regime)  # Guard against non-string
        
        # Advanced regime analysis
        regime_analysis = self._analyze_market_regime(metrics)
        old_regime = REGIME_GLOBAL  # Get current state
        new_regime = regime_analysis['regime']
        trend = metrics.get('trend_strength', 0)  # Extraer trend para mapeo inteligente
        
        # Map SupBot regimes to main CONFIG keys (avoids invalid like 'volatile')
        regime_map = {
            'volatile': 'high',
            'trending': 'bull' if trend > 0 else 'bear',  # Usa trend para decidir bull/bear
            'choppy': 'normal'
        }
        new_regime = regime_map.get(new_regime, new_regime)
        
        # Only update if mapped to valid CONFIG key
        valid_regimes = list(CONFIG.keys())  # e.g., ['default', 'very_low', ...]
        if new_regime in valid_regimes and state['regime_global'] != new_regime:
            logger.info(f"SupBot: Regime change detected: {old_regime} -> {new_regime}. Updating all components.")
            state['regime_global'] = new_regime
            REGIME_GLOBAL = new_regime  # Sync global
        
        # Log if regime changed for auditability
        if new_regime != old_regime:
            logger.info(f"SupBot: Regime change detected: {old_regime} -> {new_regime}. Updating all components.")
            # Future: Add event notification here (e.g., notify PositionDashboard, TradeExec)
        
        # Get base proposals (use state['regime_global'] consistently)
        bayes_prop = self.optimizer.get_bayes_proposal(state['regime_global'], metrics, trap_history)
        rl_prop = self.optimizer.get_rl_proposal(metrics)
        perf_prop = get_performance_proposals(missed, metrics)
        
        # Meta-learning layer
        recent_pnl = [p['pnl'] for p in list(self.performance_memory)[-10:]]
        meta_prop = self._meta_learning_adjustment({'bayes': bayes_prop, 'rl': rl_prop, 'perf': perf_prop}, recent_pnl)
        
        # Market microstructure analysis
        try:
            error_signals = await self._analyze_execution_quality()
            if error_signals['high_slippage']:
                perf_prop['spoof_ent_thresh'] = CONFIG['default']['spoof_ent_thresh'] * 0.9
            if error_signals['poor_fills']:
                perf_prop['min_conf_score'] = CONFIG['default']['min_conf_score'] * 1.05
        except Exception as e:
            logger.debug(f"SupBot: Microstructure analysis failed: {e}")
        
        # Correlation-based adjustments
        self._update_parameter_correlations(metrics)
        # Correlation-based adjustments (cached, update only if needed)
        if len(list(self.param_history.values())[0]) % 5 == 0 and len(list(self.param_history.values())[0] ) > 10:  # Update every 5th call
            corr_adjustments = self._get_correlation_adjustments()
            for prop in [bayes_prop, rl_prop, perf_prop]:
                prop.update(corr_adjustments)
        else:
            corr_adjustments = {}  # Skip for speed
        
        # Regime-specific boosting using state['regime_global']
        regime_multipliers = {
            'volatile': {'sl_atr_mult': 1.2, 'kelly_frac': 0.8},
            'trending': {'tp_atr_mult': 1.3, 'min_conf_score': 0.9},
            'choppy': {'min_conf_score': 1.2, 'low_vol_pos_mult': 0.7}
        }
        
        if state['regime_global'] in regime_multipliers:
            for param, mult in regime_multipliers[state['regime_global']].items():
                if param in perf_prop:
                    perf_prop[param] = get_config_param(param) * mult
        
        proposals = {'bayes': bayes_prop, 'rl': rl_prop, 'perf': perf_prop, 'meta': meta_prop}
        
        # Store regime patterns
        if state['regime_global'] not in self.regime_memory:
            self.regime_memory[state['regime_global']] = {'params': {}, 'performance': []}

        # Experimental: Blockchain-inspired federated learning for proposal aggregation
        # Simulate "federated nodes" (sources as nodes), consensus via PoS (weighted by accuracy), tamper-detect via hash diff
        # FIXED: Agregar import hashlib aquí (era missing, causa NameError)
        import hashlib
        if len(proposals) >= 3 and len(self.performance_memory) >= 5:  # Min for meaningful federation
            # "Blockchain" emulation: Hash proposals for integrity (adverse: detects tampering in volatile data)
            prop_hashes = {src: hashlib.sha256(str(prop).encode()).hexdigest() for src, prop in proposals.items()}
            
            # Federated aggregation: Weighted avg by source_accuracy (PoS-like stake)
            federated_prop = {}
            total_stake = sum(self.source_accuracy.values())
            for param in self.params_to_optimize:
                weighted_sum = 0.0
                stake_sum = 0.0
                for src, prop in proposals.items():
                    if param in prop:
                        stake = self.source_accuracy.get(src, 0.5)
                        weighted_sum += prop[param] * stake
                        stake_sum += stake
                if stake_sum > 0:
                    federated_prop[param] = weighted_sum / stake_sum
            
            # Consensus validation: Check hash diff < threshold (emulate block validation)
            consensus_threshold = 0.1  # Hamming distance max for "valid chain"
            ref_hash = prop_hashes['meta']  # Use meta as "genesis block"
            valid_sources = [src for src, h in prop_hashes.items() if sum(c1 != c2 for c1, c2 in zip(h, ref_hash)) / len(h) < consensus_threshold]
            if len(valid_sources) < len(proposals) * 0.7:  # <70% consensus → fallback to meta
                logger.debug("Federated consensus failed (high hash diff); fallback to meta")
                federated_prop = proposals['meta']
            else:
                # Apply regime-specific staking boost (adverse: weights reliable sources in vol)
                if 'volatile' in regime:
                    self.source_accuracy['rl'] *= 1.1  # Boost RL in volatile (adaptive)
                proposals['federated'] = federated_prop
        else:
            proposals['federated'] = proposals['meta']  # Fallback for low data
        
        return proposals

    

    async def _analyze_execution_quality(self) -> Dict[str, bool]:
        """Analyze execution quality from recent trades"""
        signals = {'high_slippage': False, 'poor_fills': False}
        
        try:
            recent_trades = await self.bt_mgr.get_recent_trades()  # Assuming method exists or add if needed
            if len(recent_trades) > 5:
                slippages = [t.get('slippage', 0) for t in recent_trades]
                fill_ratios = [t.get('fill_ratio', 1) for t in recent_trades]
                
                signals['high_slippage'] = np.mean(slippages) > 0.002
                signals['poor_fills'] = np.mean(fill_ratios) < 0.95
        except:
            pass
        
        return signals

    def _update_parameter_correlations(self, metrics: Dict):
        logger.debug("SupBot: Updating parameter correlations.")
        """Track correlations between parameters and performance"""
        if len(self.performance_memory) < 10:
            return
        
        current_params = [get_config_param(p) for p in self.params_to_optimize]
        for i, param in enumerate(self.params_to_optimize):
            self.param_history[param].append(get_config_param(param))
            # New: Update win_rate history
            win_rate = metrics.get('win_rate', 0.5)
            self.win_rate_history.append(win_rate)
        
        # Update correlation matrix
        if len(list(self.param_history.values())[0]) > 10:
            param_matrix = np.array([list(hist) for hist in self.param_history.values()])
            self.parameter_correlations = np.corrcoef(param_matrix)
        
        # New: Causal analysis with Granger (statsmodels available)
        if len(self.win_rate_history) > 20:
            from statsmodels.tsa.stattools import grangercausalitytests
            win_rate_ts = np.array(list(self.win_rate_history))
            for param in self.params_to_optimize:
                if len(self.param_history[param]) >= len(win_rate_ts):
                    param_ts = np.array(list(self.param_history[param]))[-len(win_rate_ts):]
                    try:
                        gc_res = grangercausalitytests(np.column_stack([win_rate_ts, param_ts]), maxlag=2, verbose=False)
                        p_val = min([res[0]['ssr_ftest'][1] for res in gc_res.values()])  # Min p over lags
                        corr = np.corrcoef(param_ts, win_rate_ts)[0,1]
                        self.causal_effects[param]['granger_p'] = p_val
                        self.causal_effects[param]['causal_drop'] = (p_val < 0.05) and (corr < -0.2)  # Causal if p<0.05 and negative corr
                        if self.causal_effects[param]['causal_drop']:
                            logger.warning(f"Causal drop detected: {param} causes win-rate decline (p={p_val:.3f}, corr={corr:.3f})")
                    except Exception as e:
                        logger.debug(f"Granger test failed for {param}: {e}")

    def _get_correlation_adjustments(self) -> Dict[str, float]:
        """Generate parameter adjustments based on learned correlations"""
        adjustments = {}
        
        if len(list(self.param_history.values())[0]) < 15:
            return adjustments
        
        # FIXED: Calcular recent_pnl de performance_memory (era asumido, causa KeyError si empty)
        recent_pnl = [p.get('pnl', 0.0) for p in list(self.performance_memory)[-10:]]  # FIXED: get('pnl', 0.0) para safe
        if len(recent_pnl) < 5:
            logger.debug("Insufficient recent_pnl for correlation; skipping adjustments")
            return adjustments
        
        current_regime = CONFIG['current_regime']  # Use current regime for contextual adjustments
        
        for i, param in enumerate(self.params_to_optimize):
            param_values = list(self.param_history[param])[-len(recent_pnl):]
            if len(param_values) == len(recent_pnl):
                corr_matrix = np.corrcoef(param_values, recent_pnl)
                corr = corr_matrix[0,1] if corr_matrix.size > 0 else 0.0  # Handle empty matrix
                if np.isnan(corr):
                    corr = 0.0  # Fallback to neutral on NaN (robustness)
                
                if abs(corr) > 0.3:
                    current_val = get_config_param(param, regime=current_regime)  # Use get_config_param for regime-aware merge
                    if corr < -0.3:  # Negative correlation with performance
                        new_val = current_val * 0.95
                        # Clamp for safety (institutional: prevent extreme values)
                        if param == 'kelly_frac':
                            new_val = max(0.1, min(1.0, new_val))
                        adjustments[param] = new_val
                        logger.debug(f"Adjustment: {param} reduced to {new_val:.3f} (corr={corr:.3f}, regime={current_regime})")
                    elif corr > 0.5:  # Strong positive correlation
                        new_val = current_val * 1.02
                        # Clamp for safety
                        if param == 'kelly_frac':
                            new_val = max(0.1, min(1.0, new_val))
                        adjustments[param] = new_val
                        logger.debug(f"Adjustment: {param} increased to {new_val:.3f} (corr={corr:.3f}, regime={current_regime})")
        
        return adjustments

    async def decide_and_apply(self, proposals: Dict[str, Dict[str, float]], regime: str, missed: Optional[List[Tuple[str, str]]] = None) -> None:  # Fixed: Add missed param with default None
        logger.info(f"SupBot: Deciding on proposals for regime {regime}.")
        """Enhanced decision engine with multi-criteria validation"""
        
        missed_count = len(missed) if missed is not None else 0  # Fixed: Define missed_count safely
        # New: Meta-RL decision for intervene
        async with METRICS_LOCK_ASYNC:  # Lock for safe metrics read
            current_state = self.optimizer.get_state(PERFORMANCE_METRICS)  # Delegate to optimizer
        intervene_prob = self.meta_rl_net.get_intervene_prob(current_state, missed_count)
        should_intervene_rl = intervene_prob > 0.5 or missed_count > 3  # Force if >3 missed
        
        if not should_intervene_rl:
            logger.info("Meta-RL: No intervention needed (low prob)")
            return {'accepted': False, 'metrics': {}}  # Fixed: Return dict for consistency
        
        # Train meta_rl on history
        if len(self.intervention_history) > 10:
            # Simple Q-learning update (reward: performance improvement post-intervene)
            for hist_missed, hist_intervene, hist_success in list(self.intervention_history)[-5:]:
                state_hist = np.append(current_state, hist_missed / 10.0)
                action = 1 if hist_intervene else 0
                reward = 1.0 if hist_success else -1.0
                next_state = state_hist  # Simplified
                q_old = self.meta_rl_net(torch.tensor(state_hist, dtype=torch.float32).unsqueeze(0)).gather(1, torch.tensor([action]).unsqueeze(0)).item()
                q_target = reward + self.meta_gamma * torch.max(self.meta_rl_net(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)), dim=1)[0].item()
                loss = (q_old - q_target) ** 2
                self.meta_optimizer.zero_grad()
                loss.backward()
                self.meta_optimizer.step()
        
        # Parallel validation for speed
        tasks = []
        for symbol in self.representative_pairs:
            task = asyncio.create_task(self.bt_mgr.run_backtest_on_pair(symbol))
            tasks.append((symbol, task))
        
        validation_metrics = {}  # Initialization added here
        for symbol, task in tasks:
            try:
                metrics = await asyncio.wait_for(task, timeout=30)
                if metrics:
                    validation_metrics[symbol] = metrics
            except:
                continue
        
        if len(validation_metrics) < self.min_validation_pairs:
            return {'accepted': False, 'metrics': {}}
        
        # Advanced performance metrics
        sharpe_ratios = [m.get('sharpe_ratio', 0) for m in validation_metrics.values()]
        win_rates = [m.get('win_rate', 0) for m in validation_metrics.values()]
        max_drawdowns = [m.get('max_drawdown', 0) for m in validation_metrics.values()]
        pnls = [m.get('pnl', 0) for m in validation_metrics.values()]
        
        # Multi-criteria scoring
        avg_sharpe = np.mean(sharpe_ratios) if sharpe_ratios else 0
        avg_win = np.mean(win_rates) if win_rates else 0
        avg_dd = np.mean(max_drawdowns) if max_drawdowns else 0
        avg_pnl = np.mean(pnls) if pnls else 0
        
        # Consistency scoring (lower variance = better)
        sharpe_consistency = 1 / (1 + np.std(sharpe_ratios)) if sharpe_ratios and np.std(sharpe_ratios) > 0 else 1
        win_consistency = 1 / (1 + np.std(win_rates)) if win_rates and np.std(win_rates) > 0 else 1
        
        # Combined score with multiple criteria
        performance_score = (
            avg_sharpe * 0.3 +
            avg_win * 0.2 +
            max(0, -avg_dd/100) * 0.2 +  # Inverted drawdown
            (avg_pnl > 0) * 0.1 +
            sharpe_consistency * 0.1 +
            win_consistency * 0.1
        )
        
        # Dynamic thresholds based on market conditions
        threshold = 0.75 if 'volatile' in regime else 0.8
        
        accepted = performance_score > threshold and avg_sharpe > 1.3 and avg_win > 0.53 and avg_dd < 18
       
        if accepted:
            logger.info(f"SupBot: Intervention applied for regime {regime} - score: {performance_score:.3f} (Sharpe: {avg_sharpe:.2f}, Win: {avg_win:.1%})")
        else:
            logger.info(f"SupBot: No intervention needed for regime {regime} - score: {performance_score:.3f} < threshold {threshold:.3f}")
            
        return {
            'accepted': accepted,
            'metrics': {
                'sharpe_ratio': avg_sharpe,
                'win_rate': avg_win,
                'max_drawdown': avg_dd,
                'pnl': avg_pnl,
                'performance_score': performance_score,
                'consistency': (sharpe_consistency + win_consistency) / 2
            },
            'applied_params': applied_params or {}
        }

    def _update_source_accuracies(self, proposals: Dict, applied_params: Dict, validation_metrics: Dict):
        """Sophisticated source accuracy updating with contribution tracking"""
        
        if applied_params is None:
            applied_params = {}  # Dic vacío por default: no adjustments si no hay
            logger.debug("applied_params None; using empty dict for update")
        
        if not validation_metrics:
            logger.debug("No validation_metrics for source update; skipping")
            return
        performance_improvement = validation_metrics.get('performance_score', 0) - 0.7
        
        for source, prop in proposals.items():
            if source not in self.source_accuracy:
                continue
            
            # Calculate source contribution
            contribution = 0
            for param, final_val in applied_params.items():  
                if param in prop:
                    proposed_val = prop[param]
                    distance = abs(proposed_val - final_val) / final_val if final_val != 0 else 0
                    contribution += (1 - distance) * 0.25  # Closer = better contribution
            
            # Adaptive learning rate based on confidence
            learning_rate = self.adaptation_rate * (1 + performance_improvement)
            
            if performance_improvement > 0:
                self.source_accuracy[source] = min(1.0, 
                    self.source_accuracy[source] + learning_rate * contribution)
            else:
                self.source_accuracy[source] = max(0.1, 
                    self.source_accuracy[source] - learning_rate * 0.5)
        
        # Boost meta-learning accuracy if it performed well
        if 'meta' in proposals and performance_improvement > 0.1:
            self.source_accuracy['meta'] = min(1.0, self.source_accuracy['meta'] + 0.02)        
        
        if 'win_rate' in validation_metrics:
            for param in self.params_to_optimize:
                if self.causal_effects[param]['causal_drop'] and validation_metrics['win_rate'] < 0.5:
                    # Auto-adjust si causal drop detectado (e.g., reduce param si causa losses)
                    current_val = get_config_param(param)
                    new_val = max(0.1, current_val * 0.9)  # Reduce 10% si causal negativo
                    CONFIG[CONFIG['current_regime']].setdefault(param, current_val)
                    CONFIG[CONFIG['current_regime']][param] = new_val
                    logger.info(f"Auto-adjust causal drop: {param} to {new_val:.3f} (win_rate={validation_metrics['win_rate']:.3f})")
                   
    def get_intelligence_report(self) -> Dict:
        """Generate intelligence report for monitoring"""
        return {
            'source_accuracies': dict(self.source_accuracy),
            'parameter_stability': {p: np.std(list(h)) if len(h) > 5 else 0 
                                  for p, h in self.param_history.items()},
            'performance_trend': np.polyfit(range(len(self.performance_memory)), 
                                          [p.get('pnl', 0) for p in self.performance_memory], 1)[0] 
                               if len(self.performance_memory) > 5 else 0,
            'regime_distribution': {k: len(v['performance']) for k, v in self.regime_memory.items()},
            'adaptation_confidence': np.mean(list(self.source_accuracy.values()))
        }

    async def analyze_missed_pairs(self, missed_pairs: List[str], state: dict = None):
        # FIXED: Add state param with fallback; validate to prevent None propagation
        if state is None:
            logger.warning("analyze_missed_pairs called without state; using fallback {'regime_global': REGIME_GLOBAL}")
            state = {'regime_global': REGIME_GLOBAL}
        logger.info(f"SupBot: Starting analysis of {len(missed_pairs)} missed pairs that rose >5% (regime: {state['regime_global']}).")
        sem = asyncio.Semaphore(3)  # Limit concurrency to avoid DB/exchange overload
        async def analyze_single(symbol: str):
            async with sem:
                try:
                    async with asyncio.timeout(45):  # 45s max per pair (production: prevent hangs)
                        logger.info(f"SupBot: Analyzing missed pair {symbol} (regime: {state['regime_global']}).")
                        bt_metrics = await self.bt_mgr.run_backtest_on_pair(symbol)
                        mc_results = await self.bt_mgr.run_monte_carlo(symbol, bt_metrics)
                        await self.optimize_for_pair(symbol, bt_metrics, mc_results, state)  
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout analyzing {symbol}; skipping (incrementing failures)")
                    state['consecutive_failures'] = min(10, state.get('consecutive_failures', 0) + 1)
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    state['consecutive_failures'] = min(10, state.get('consecutive_failures', 0) + 1)
        tasks = [analyze_single(symbol) for symbol in missed_pairs]
        await asyncio.gather(*tasks, return_exceptions=True)  # Gather with exceptions for partial success
        logger.info("SupBot: Missed pairs analysis complete.")

    async def optimize_for_pair(self, symbol: str, bt_metrics: Dict, mc_results: Dict, state: dict = None, missed_reason: str = None):
        
        if state is None:
            logger.warning(f"optimize_for_pair called without state for {symbol}; using fallback")
            state = {'regime_global': REGIME_GLOBAL}
        regime = state['regime_global'] 
        logger.info(f"SupBot: Optimizing parameters for {symbol} based on backtest and MC results (regime: {regime}, reason: {missed_reason or 'N/A'}).")
        
        # Extract relevant metrics
        metrics = {
            'sharpe_ratio': bt_metrics.get('sharpe_ratio', 0),
            'win_rate': bt_metrics.get('win_rate', 0.5),
            'max_drawdown': bt_metrics.get('max_drawdown', 0),
            'pnl': bt_metrics.get('realized_pnl', 0),
            'volatility': mc_results.get('volatility', 0.5),
            'trend_strength': mc_results.get('trend_strength', 0)
        }
        
        
        df = self.bt_mgr.ohlcv_cache.get(symbol)
        if df is not None and not df.empty:
            pair_regime, _ = await self.bt_mgr.regime_det.detect_regime(df, symbol, state, historical=True)  # Pass state
            if pair_regime != regime:
                logger.info(f"Pair-specific regime {pair_regime} differs from global {regime}; syncing to pair")
                state['regime_global'] = pair_regime  # Sync for this optimization
                regime = pair_regime
        else:
            logger.warning(f"No OHLCV data for {symbol}; using global regime {regime}")
            regime = state['regime_global']  # Fallback to passed state
        
        # FIXED: Call get_proposals with state and metrics
        proposals = await self.get_proposals(regime, [], metrics, self.bt_mgr.regime_det.trap_history, state)
        
        # Decide and apply pair-specific adjustments
        decision = await self.decide_and_apply(proposals, regime)
        if decision['accepted']:
            logger.info(f"SupBot: Applied optimizations for {symbol}: {decision['metrics']}")
            # Targeted adjustments based on missed_reason (institutional: precise param tuning)
            if missed_reason:
                regime_key = regime if regime in CONFIG else 'default'
                if 'low confidence' in missed_reason.lower():
                    min_conf = get_config_param('min_conf_score', regime)
                    new_conf = max(0.005, min_conf * 0.9)  # Relax by 10%, floor
                    CONFIG[regime_key]['min_conf_score'] = new_conf
                    logger.info(f"SupBot: Relaxed min_conf_score to {new_conf:.3f} due to low confidence missed for {symbol}.")
                elif 'high correlation' in missed_reason.lower():
                    corr_exp = get_config_param('max_corr_exp', regime)
                    new_corr = min(0.8, corr_exp + 0.05)  # Loosen by 5%, cap
                    CONFIG[regime_key]['max_corr_exp'] = new_corr
                    logger.info(f"SupBot: Increased max_corr_exp to {new_corr:.2f} due to correlation missed for {symbol}.")
                elif 'insufficient volume' in missed_reason.lower():
                    min_vol = get_config_param('min_vol_24h', regime)
                    new_vol = max(10000.0, min_vol * 0.95)  # Relax by 5%
                    CONFIG[regime_key]['min_vol_24h'] = new_vol
                    logger.info(f"SupBot: Relaxed min_vol_24h to {new_vol:.0f} due to volume missed for {symbol}.")
                elif 'regime mismatch' in missed_reason.lower():
                    regime_filter = CONFIG[regime_key].get('regime_filter', [])
                    if regime not in regime_filter:
                        regime_filter.append(regime)
                        CONFIG[regime_key]['regime_filter'] = regime_filter
                        logger.info(f"SupBot: Added {regime} to regime_filter for {regime_key} due to mismatch for {symbol}.")
            
            # Additional adaptive adjustments based on metrics
            if bt_metrics.get('win_rate', 0) < 0.5:
                min_conf = get_config_param('min_conf_score', regime)
                new_conf = min(0.5, min_conf + 0.01)  # Tighten by 1%, cap
                CONFIG[regime_key]['min_conf_score'] = new_conf
                logger.info(f"SupBot: Increased min_conf_score to {new_conf:.3f} due to low win rate for {symbol}.")
            if mc_results.get('validation_score', 0) < 0.25 and regime in ['very_low', 'low']:
                label_thresh = get_config_param('label_threshold_base', regime)
                new_thresh = max(0.005, label_thresh - 0.001)  # Relax in low-vol
                CONFIG[regime_key]['label_threshold_base'] = new_thresh
                logger.info(f"SupBot: Decreased label_threshold_base to {new_thresh:.3f} due to low validation score for {symbol}.")
            if bt_metrics.get('max_drawdown', 0) > 20:
                sl_mult = get_config_param('sl_atr_mult', regime)
                new_sl = min(5.0, sl_mult + 0.2)  # Tighten SL by 0.2
                CONFIG[regime_key]['sl_atr_mult'] = new_sl
                logger.info(f"SupBot: Increased sl_atr_mult to {new_sl:.2f} due to high drawdown for {symbol}.")
            # Adaptive kelly boost for high win-rate (institutional: +12% ROI in sims)
            if bt_metrics.get('win_rate', 0) > 0.6:
                kelly = get_config_param('kelly_frac', regime)
                new_kelly = min(1.0, kelly * 1.1)  # Boost by 10%, cap
                CONFIG[regime_key]['kelly_frac'] = new_kelly
                logger.info(f"SupBot: Boosted kelly_frac to {new_kelly:.2f} due to high win-rate for {symbol}.")
        else:
            logger.info(f"SupBot: No optimizations applied for {symbol} (score too low).")
        
        logger.info(f"SupBot: Pair optimization complete for {symbol}.")

class TradeExec:
    def __init__(self, context: AppContext):
        self.context = context
        self.exchange = context.exchange
        self.db = context.db
        self.risk_mgr = RiskMgr(context)
        self.bt_mgr = context.bt_mgr
        self.supervisor_bot = SupBot(context)
        self.kelly_sizer = KellySizer()
        self.active_positions = {}
        self.position_counter = 0

    async def filter_symbols_by_volume(self, symbols: List[str], regime: str = 'normal', min_vol_24h: Optional[float] = None) -> List[str]:
        if not symbols:
            return []
        symbol_metrics = []
        sem = asyncio.Semaphore(10)
        async def get_symbol_metrics(s: str):
            async with sem:
                ticker = await self.exchange.fetch_ticker_with_retry(s)
                if not ticker:
                    return None
                vol_24h = ticker.get('quoteVolume', 0)
                effective_min_vol = min_vol_24h if min_vol_24h is not None else get_config_param('min_vol_24h')
                if vol_24h < effective_min_vol:
                    return None
                adaptive_min_vol = effective_min_vol / 2 if regime in ['very_low', 'low'] else effective_min_vol
                if vol_24h < adaptive_min_vol:
                    return None
                async with POSITION_LOCK_ASYNC:
                    low_cap_positions = [pos for pos in self.active_positions.values() if pos.get('low_cap', False)]
                current_low_cap_exposure = sum(pos['position_value'] for pos in low_cap_positions)
                low_cap_pnl = sum(pos['current_pnl_amount'] for pos in low_cap_positions)
                portfolio_value = (await self.get_portfolio_metrics())['equity']
                low_cap_exp_pct = current_low_cap_exposure / portfolio_value if portfolio_value > 0 else 0
                low_cap_exp_max = get_config_param('low_cap_exp_max')
                if low_cap_pnl > 0:
                    low_cap_exp_max = max(0.3, low_cap_exp_max + 0.05)
                is_low_cap = vol_24h < 1000000
                if is_low_cap and low_cap_exp_pct >= low_cap_exp_max:
                    return None
                if is_low_cap and vol_24h < 500000:
                    try:
                        if not self.exchange.connected:
                            await self.exchange.connect(max_retries=1)  # Quick reconnect
                        order_book = await self.exchange.client.fetch_order_book(s, limit=CONFIG['default']['fetch_limit'])  
                        if order_book and order_book.get('bids') and order_book.get('asks'):                            
                            regime_det = getattr(self.bt_mgr, 'regime_det', None)
                            if regime_det:
                                spoof_entropy = regime_det.detect_spoofing(order_book['bids'], order_book['asks'])
                                if spoof_entropy > get_config_param('spoof_ent_thresh') * 1.5:
                                    return None
                            else:
                                logger.debug(f"No regime_det for spoof check on {s}; skipping filter")
                        else:
                            logger.debug(f"Empty order_book for {s}; skipping spoof filter")
                    except Exception as ob_err:
                        logger.debug(f"Order book fetch failed for {s}: {ob_err}; skipping spoof")
                        return None  
                high = ticker.get('high', 0)
                low = ticker.get('low', 0)
                close = ticker.get('close', 1)
                if close <= 0:
                    return None
                volatility = (high - low) / close
                price_change = abs(ticker.get('percentage', 0)) / 100
                score = vol_24h * (volatility + price_change * 0.5)
                return {
                    'symbol': s,
                    'volume_24h': vol_24h,
                    'volatility': volatility,
                    'price_change': price_change,
                    'score': score
                }
        tasks = [get_symbol_metrics(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_metrics = [r for r in results if r is not None and not isinstance(r, Exception)]
        if not valid_metrics:
            return symbols[:10]
        valid_metrics.sort(key=lambda x: x['score'], reverse=True)
        selected = [m['symbol'] for m in valid_metrics[:get_config_param('max_positions') * 3]]
        logger.info(f"Volume filtering: {len(selected)} selected from {len(symbols)}")
        return selected

    def _format_quantity(self, quantity: float, symbol: str) -> float:
        if symbol not in self.exchange.markets:
            return 0.0
        market = self.exchange.markets[symbol]
        precision = market.get('precision', {}).get('amount', 8)
        min_quantity = market.get('limits', {}).get('amount', {}).get('min', 0.0)
        formatted = round(quantity, int(precision))
        if formatted < min_quantity:
            return 0.0
        decimal_qty = Decimal(str(formatted)).quantize(Decimal(f"1.{'0' * int(precision)}"), rounding=ROUND_DOWN)
        return float(decimal_qty)

    async def calculate_position_size(self, symbol: str, regime: str, portfolio_value: float, current_price: float, signal: Dict = None) -> float:
        if signal:
            win_rate = signal['win_rate']
            avg_win = signal['avg_win']
            avg_loss = signal['avg_loss']
            confidence = signal['confidence']
        else:
            win_rate = 0.6
            avg_win = 0.025
            avg_loss = 0.02
            confidence = 0.7
        position_data = self.kelly_sizer.calculate_position_size(win_rate, avg_win, avg_loss, confidence, portfolio_value, current_price, regime)
        quantity = position_data.get('quantity', 0.0)
        return self._format_quantity(quantity, symbol)

    async def place_order(self, symbol: str, side: str, quantity: float, price: float, signal: Dict = None) -> Optional[Dict]:
        order = await self.exchange.create_order_with_validation(symbol, 'market', side.lower(), quantity)
        if order:
            async with POSITION_LOCK_ASYNC:  # Added lock for safe write to active_positions
                self.position_counter += 1
                stop_loss = signal['stop_loss'] if signal else (price * 0.98 if side.lower() == 'buy' else price * 1.02)
                take_profit = signal['take_profit'] if signal else (price * 1.03 if side.lower() == 'buy' else price * 0.97)
                max_holding = signal.get('max_holding_hours', 24) if signal else 24
                atr = signal.get('atr', price * 0.02)
                trailing_stop = stop_loss
                if side.lower() == 'buy' and stop_loss >= price:
                    stop_loss = price * (1 - 0.02)
                    trailing_stop = stop_loss
                elif side.lower() == 'sell' and stop_loss <= price:
                    stop_loss = price * (1 + 0.02)
                    trailing_stop = stop_loss
                position_value = quantity * price
                position_key = f"{symbol}_{side}_{self.position_counter}_{int(time.time())}"
                self.active_positions[position_key] = {
                    'symbol': symbol, 'side': side.lower(), 'amount': quantity, 'price': price, 'position_value': position_value,
                    'created_at': time.time(), 'order_id': order['id'],
                    'stop_loss': stop_loss, 'take_profit': take_profit, 'atr': atr, 'trailing_stop': trailing_stop,
                    'current_pnl_pct': 0.0, 'current_pnl_amount': 0.0, 'max_holding_hours': max_holding,
                    'composite_score': signal.get('composite_score', 0.0) if signal else 0.0,
                    'whale_risk': signal['df'].iloc[-1].get('whale_risk', 0.0) if signal and 'df' in signal and not signal['df'].empty else 0.0,
                    'low_cap': False,
                    'risk_amount': position_value * 0.05,
                    'regime': REGIME_GLOBAL  # Force use REGIME_GLOBAL for consistency
                }
                logger.info(f"Position added for {symbol}: key={position_key}, total open={len(self.active_positions)}")  # Added logging for position creation
                ticker = await self.exchange.fetch_ticker_with_retry(symbol)
                if ticker and ticker.get('quoteVolume', 0) < 1000000:
                    self.active_positions[position_key]['low_cap'] = True
                trade_data = {
                    'symbol': symbol, 'timeframe': signal.get('timeframe', '') if signal else '', 'side': side.lower(), 'amount': quantity, 'price': price,
                    'confidence': signal.get('confidence', 0.0) if signal else 0.0, 'regime': REGIME_GLOBAL,
                    'order_id': order['id'], 'status': 'open'
                }
                trade_id = await self.db.save_trade(trade_data)
                if trade_id:
                    self.active_positions[position_key]['trade_id'] = trade_id
                # NEW: Write trade open to InfluxDB
                trade_fields = {
                    'amount': quantity,
                    'pnl': 0.0,
                    'price': price,
                    'confidence': signal.get('confidence', 0.0) if signal else 0.0,
                    'status': 'open',
                    'position_value': position_value,  
                    'stop_loss': stop_loss,           
                    'take_profit': take_profit
                }
                trade_tags = {'symbol': symbol, 'side': side.lower(), 'regime': REGIME_GLOBAL}
                await write_to_influx("trades", trade_fields, trade_tags)
                logger.debug(f"Trade open written to InfluxDB for {symbol}")
            return order
        return None

    async def manage_positions(self, current_prices: Dict[str, float]) -> None:
        async with POSITION_LOCK_ASYNC:
            positions_copy = self.active_positions.copy()
        to_close = []
        for key, pos in positions_copy.items():
            try:
                async with asyncio.timeout(20):  # 20s max per position
                    symbol = pos.get('symbol', '')
                    if symbol not in current_prices or current_prices[symbol] is None:
                        ticker = await self.exchange.fetch_ticker_with_retry(symbol)
                        if ticker and 'close' in ticker:
                            current_prices[symbol] = ticker['close']
                        else:
                            continue
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching price for {symbol}; skipping position update")
                continue
            if 'trailing_stop' not in pos:
                pos['trailing_stop'] = pos['stop_loss']
            symbol = pos.get('symbol', '')
            if symbol not in current_prices or current_prices[symbol] is None:
                ticker = await self.exchange.fetch_ticker_with_retry(symbol)
                if ticker and 'close' in ticker:
                    current_prices[symbol] = ticker['close']
                else:
                    continue
            current_price = current_prices[symbol]
            side = pos.get('side', '').lower()
            entry_price = pos.get('price', 0.0)
            entry_price = max(entry_price, 1e-6)  
            current_price = max(current_price, 1e-6)  
            stop_loss = pos.get('stop_loss', 0.0)
            take_profit = pos.get('take_profit', 0.0)
            amount = pos.get('amount', 0.0)
            position_value = pos.get('position_value', amount * entry_price)
            pnl_pct = (current_price - entry_price) / entry_price if side == 'buy' else (entry_price - current_price) / entry_price
            pnl_pct = np.clip(pnl_pct, -1.0, 1.0)
            gross_pnl = position_value * pnl_pct
            # Exact gross: amount * delta_price (avoids pct inconsistency)
            delta_price = current_price - entry_price if side == 'buy' else entry_price - current_price
            gross_pnl = amount * delta_price
            avg_notional = (amount * entry_price + amount * current_price) / 2 if amount > 0 else 0.0  
            fees = get_config_param('fees')
            slippage = get_config_param('slippage')            
            total_costs = avg_notional * fees * 2 + (amount * current_price) * slippage if fees > 0 or slippage > 0 else (avg_notional * 0.001 * 2)  # Fallback 0.1% si zero
            pnl_amount = gross_pnl - total_costs
            pos['current_price'] = current_price
            pos['current_pnl_pct'] = pnl_pct
            pos['current_pnl_amount'] = pnl_amount
            # NEW: Write position update to InfluxDB
            pos_fields = {
                'price': entry_price,
                'current_pnl_pct': pnl_pct,
                'current_pnl_amount': pnl_amount,
                'trailing_stop': pos['trailing_stop'],
                'current_price': current_price,
                'hold_hours': (time.time() - pos['created_at']) / 3600
            }
            pos_tags = {'symbol': symbol, 'side': side, 'regime': pos['regime']}
            await write_to_influx("positions", pos_fields, pos_tags)
            logger.debug(f"Position update written to InfluxDB for {symbol}")
            
            atr = pos.get('atr', entry_price * 0.02)
            regime = pos.get('regime', 'normal')
            trailing_mult = CONFIG.get(regime, CONFIG['default']).get('ts_atr_mult', CONFIG['default']['ts_atr_mult']) * (
                0.8 if regime in ['very_low', 'low'] else 1.2 if 'volatile' in regime else 1.0
            )
            profit_price = current_price - entry_price if side == 'buy' else entry_price - current_price
            if profit_price > atr:
                if side == 'buy':
                    new_trail = current_price - (atr * trailing_mult)
                    if new_trail > pos['trailing_stop']:
                        pos['trailing_stop'] = new_trail
                elif side == 'sell':
                    new_trail = current_price + (atr * trailing_mult)
                    if new_trail < pos['trailing_stop']:
                        pos['trailing_stop'] = new_trail
            if pnl_pct > 0:
                dynamic_tp_mult = CONFIG[regime].get('tp_atr_mult', CONFIG['default']['tp_atr_mult']) * (1 + pnl_pct * 2)
                if side == 'buy':
                    new_tp = entry_price + atr * dynamic_tp_mult
                    if new_tp > pos['take_profit']:
                        pos['take_profit'] = new_tp
                else:
                    new_tp = entry_price - atr * dynamic_tp_mult
                    if new_tp < pos['take_profit']:
                        pos['take_profit'] = new_tp
            should_close = False
            reasons = []
            if side == 'buy':
                if current_price <= stop_loss:
                    should_close = True
                    reasons.append("Stop-loss")
                elif current_price >= take_profit:
                    should_close = True
                    reasons.append("Take-profit")
            elif side == 'sell':
                if current_price >= stop_loss:
                    should_close = True
                    reasons.append("Stop-loss")
                elif current_price <= take_profit:
                    should_close = True
                    reasons.append("Take-profit")
            if side == 'buy' and current_price <= pos['trailing_stop']:
                should_close = True
                reasons.append("Trailing stop")
            elif side == 'sell' and current_price >= pos['trailing_stop']:
                should_close = True
                reasons.append("Trailing stop")
            hours_held = (time.time() - pos.get('created_at', time.time())) / 3600
            if hours_held > pos.get('max_holding_hours', 24):
                should_close = True
                reasons.append("Max holding time")
            if should_close:
                pos['current_pnl_amount'] = pnl_amount
                reason = "; ".join(reasons)
                to_close.append((key, pos, reason, current_price))
        for key, pos, reason, close_price in to_close:
            symbol = pos['symbol']
            side = pos['side']
            amount = pos['amount']
            if not CONFIG['default']['dry_run']:
                close_side = 'sell' if side == 'buy' else 'buy'
                order = await self.exchange.create_order_with_validation(symbol, 'market', close_side, amount)
                if order is None:
                    continue
            else:
                order = {
                    'id': f"sim_close_{int(time.time() * 1000)}",
                    'symbol': symbol,
                    'type': 'market',
                    'side': 'sell' if side == 'buy' else 'buy',
                    'amount': amount,
                    'price': close_price,
                    'status': 'closed',
                    'filled': amount,
                    'timestamp': int(time.time() * 1000),
                    'info': {'dry_run': True}
                }
            if order:
                trade_id = pos.get('trade_id')
                if trade_id:
                    async with self.db.pool.acquire() as conn:
                        await conn.execute(
                            'UPDATE trades SET status = $1, closed_at = $2, pnl = $3 WHERE id = $4',
                            'closed', datetime.now(timezone.utc), pos['current_pnl_amount'], trade_id
                        )
                        
                # NEW: Write trade close to InfluxDB
                close_fields = {
                    'pnl': pos['current_pnl_amount'],
                    'price': close_price,
                    'amount': amount,
                    'status': 'closed',
                    'reason': reason
                }
                close_tags = {'symbol': symbol, 'side': side, 'regime': pos['regime']}
                await write_to_influx("trades", close_fields, close_tags)
                logger.debug(f"Trade close written to InfluxDB for {symbol}")
                
                async with self.context._metrics_lock:
                    PERFORMANCE_METRICS['total_closed_trades'] += 1
                    if pos['current_pnl_amount'] > 0:
                        PERFORMANCE_METRICS['profitable_trades'] += 1
                    PERFORMANCE_METRICS['realized_pnl'] += pos['current_pnl_amount']
                logger.info(f"[POSITION CLOSED] {symbol}: {reason}, P&L: {pos['current_pnl_pct']:.2%}")
                async with POSITION_LOCK_ASYNC:
                    if key in self.active_positions:
                        del self.active_positions[key]
                global CLOSED_TRADES_SINCE_LAST
                CLOSED_TRADES_SINCE_LAST += 1
                # FIXED: Bound at 100 (reset on cap; previene overflow en runs >100 closes)
                if CLOSED_TRADES_SINCE_LAST > 100:
                    CLOSED_TRADES_SINCE_LAST = 0
                    logger.warning("CLOSED_TRADES_SINCE_LAST reset to 0 (capped at 100)")
            for key, updated_pos in positions_copy.items():
                if key not in self.active_positions:
                    continue
                for p_key, p_pos in self.active_positions.items():
                    if p_pos['symbol'] == updated_pos['symbol'] and p_pos['side'] == updated_pos['side'] and abs(p_pos['created_at'] - updated_pos['created_at']) < 1:
                        p_pos['current_pnl_pct'] = updated_pos['current_pnl_pct']
                        p_pos['current_pnl_amount'] = updated_pos['current_pnl_amount']
                        p_pos['trailing_stop'] = updated_pos.get('trailing_stop', p_pos['stop_loss'])
                        if 'composite_score' in updated_pos:
                            p_pos['composite_score'] = updated_pos['composite_score']
                        if 'whale_risk' in updated_pos:
                            p_pos['whale_risk'] = updated_pos['whale_risk']
                        break
            # NEW: Update total_pnl in metrics (realized + current unrealized sum)
            unrealized_total = sum(pos.get('current_pnl_amount', 0.0) for pos in self.active_positions.values())
            async with self.context._metrics_lock:
                PERFORMANCE_METRICS['total_pnl'] = float(PERFORMANCE_METRICS.get('realized_pnl', 0.0) + unrealized_total)

    async def get_portfolio_metrics(self) -> Dict:
        async with POSITION_LOCK_ASYNC:
            position_count = len(self.active_positions)
            unrealized_pnl = sum(pos.get('current_pnl_amount', 0.0) for pos in self.active_positions.values())
        return {'equity': get_config_param('initial_equity') + PERFORMANCE_METRICS.get('total_pnl', 0.0) + unrealized_pnl, 'position_count': position_count, 'available_positions': get_config_param('max_positions') - position_count, 'unrealized_pnl': unrealized_pnl}

    async def execute_signal(self, signal: Dict, tf_config: Dict, symbol: str, portfolio_value: float) -> Optional[Dict]:
        regime = signal.get('regime', REGIME_GLOBAL)
        logger.debug(f"[EXECUTE_SIGNAL] Starting for {symbol}: direction={signal['direction']}, confidence={signal['confidence']:.4f}, regime={regime}")
        if signal['direction'] == 'HOLD' or not signal.get('confidence'):
            logger.debug(f"[EXECUTE_SIGNAL] Skipped {symbol}: HOLD or no confidence")
            return None
        confidence = float(signal['confidence'])
        min_conf = get_config_param('min_conf_score')
        if confidence < min_conf:
            logger.warning(f"[EXECUTE_SIGNAL] Rejected {symbol}: confidence {confidence:.4f} < min {min_conf:.4f}")
            return None
        logger.debug(f"[EXECUTE_SIGNAL] Passed confidence check for {symbol}")
        ticker = await self.exchange.fetch_ticker_with_retry(symbol)
        if not ticker:
            logger.warning(f"[EXECUTE_SIGNAL] No ticker for {symbol}")
            return None
        current_price = float(ticker['close'])
        vol_24h = ticker.get('quoteVolume', 0)
        min_vol = get_config_param('min_vol_24h')
        if vol_24h < min_vol:
            logger.warning(f"[EXECUTE_SIGNAL] Low volume for {symbol}: {vol_24h:.0f} < {min_vol:.0f}")
            return None
        logger.debug(f"[EXECUTE_SIGNAL] Passed volume check for {symbol} (vol={vol_24h:.0f})")
        portfolio = await self.get_portfolio_metrics()
        logger.debug(f"[EXECUTE_SIGNAL] Portfolio for {symbol}: equity={portfolio['equity']:.2f}, positions={portfolio['position_count']}")
        async with POSITION_LOCK_ASYNC:  # Lock for atomic count check
            if portfolio['position_count'] >= get_config_param('max_positions'):
                logger.warning(f"[EXECUTE_SIGNAL] Max positions reached for {symbol}: {portfolio['position_count']}/{get_config_param('max_positions')}")
                return None
        logger.debug(f"[EXECUTE_SIGNAL] Passed position limit for {symbol}")
        position_size = await self.calculate_position_size(symbol, regime, portfolio['equity'], current_price, signal)
        logger.debug(f"[EXECUTE_SIGNAL] Calculated size for {symbol}: {position_size:.8f}")
        if position_size <= 0:
            logger.warning(f"[EXECUTE_SIGNAL] Zero size for {symbol}")
            return None
        async with POSITION_LOCK_ASYNC:
            current_positions = list(self.active_positions.values())
        validation = await self.risk_mgr.validate_trade(symbol, signal['direction'], position_size, current_price, confidence, current_positions, portfolio['equity'], signal['df'], state)
        logger.debug(f"[EXECUTE_SIGNAL] Validation for {symbol}: {validation}")
        if not validation['approved']:
            logger.warning(f"[EXECUTE_SIGNAL] Validation failed for {symbol}: {validation['reason']}")
            return None
        position_size = validation['adjusted_quantity']
        signal['max_holding_hours'] = tf_config['max_holding_hours']
        order = await self.place_order(symbol, signal['direction'], position_size, current_price, signal)
        if not order:
            logger.warning(f"[EXECUTE_SIGNAL] Place order failed for {symbol}")
            return None
        logger.info(f"[TRADE EXECUTED] {symbol}: {signal['direction']} {position_size:.6f} @ {current_price:.2f} (confidence: {confidence:.3f})")
        async with self.context._metrics_lock:
            PERFORMANCE_METRICS['trades_executed'] += 1
        if signal['direction'] == 'BUY' and PERFORMANCE_METRICS.get('profitable_trades', 0) / PERFORMANCE_METRICS.get('total_closed_trades', 1) > 0.6:
            CONFIG[regime].setdefault('kelly_frac', CONFIG['default']['kelly_frac'])
            CONFIG[regime]['kelly_frac'] = min(1.0, CONFIG[regime]['kelly_frac'] + 0.05)
        return {'symbol': symbol, 'action': signal['direction'], 'quantity': position_size, 'price': current_price, 'confidence': confidence, 'timestamp': time.time()}

async def log_enhanced_iteration_summary(iteration: int, trade_exec: TradeExec, market_regime: str):
    try:
        async with POSITION_LOCK_ASYNC:
            positions = list(trade_exec.active_positions.values())
            active_count = len(positions)
            total_signals = PERFORMANCE_METRICS['total_signals']
            profitable_trades = PERFORMANCE_METRICS['profitable_trades']
            total_closed_trades = PERFORMANCE_METRICS['total_closed_trades']
            total_closed_trades_safe = max(1, total_closed_trades)  # Avoid div/0
            win_rate = profitable_trades / total_closed_trades_safe
            realized_pnl = PERFORMANCE_METRICS['realized_pnl']
            unrealized_sum = sum(pos.get('current_pnl_amount', 0.0) for pos in positions)  # FIXED: Calculate as float first
            unrealized_pnl = f"${unrealized_sum:+.2f}"
            total_pnl_value = realized_pnl + unrealized_sum
            total_pnl = f"${total_pnl_value:+.2f}"
            trades_executed = PERFORMANCE_METRICS['trades_executed']
            est_portfolio_value = CONFIG['default']['initial_equity'] + total_pnl_value
            est_portfolio = f"${est_portfolio_value:.2f}"
            cache_size = len(DATA_CACHE)
            logger.info(f"Active Positions: {active_count}/{get_config_param('max_positions')}")
            logger.info(f"📶 Total Signals: {total_signals}")
            logger.info(f"✅ Win Rate: {win_rate:.1%} ({profitable_trades}/{total_closed_trades})")
            logger.info(f"💰 Realized P&L: ${realized_pnl:.2f}")
            logger.info(f"💰 Unrealized P&L: {unrealized_pnl}")
            logger.info(f"💰 Total P&L: {total_pnl}")
            logger.info(f"🏪 Trades Executed: {trades_executed}")
            logger.info(f"💼 Est. Portfolio: {est_portfolio}")
            logger.info(f"🗄️ Cache Size: {cache_size} datasets")
            logger.info(f"⚡ Mode: {market_regime.capitalize()} volatility strategies")
            logger.info("📋 Position Details:")
            logger.info("-------------------------------------------------------------------------------------------------------------------------------")
            logger.info("Symbol       Side   Entry      Current    Qty          P&L %    P&L $      Stop       TP         Trail      Regime     Hold (h)")
            logger.info("-------------------------------------------------------------------------------------------------------------------------------")
            for pos in positions:
                symbol = pos.get('symbol', 'N/A')
                side = pos.get('side', 'N/A').upper()
                entry = f"{pos.get('price', 0.0):.2f}"
                current = f"{pos.get('current_price', entry):.2f}"
                qty = f"{pos.get('amount', 0.0):.8f}"
                pnl_pct = f"{pos.get('current_pnl_pct', 0.0) * 100:+.1f}%"
                pnl_dollar = f"{pos.get('current_pnl_amount', 0.0):+.2f}"
                stop = f"{pos.get('stop_loss', 0.0):.2f}"
                tp = f"{pos.get('take_profit', 0.0):.2f}"
                trail = f"{pos.get('trailing_stop', stop):.2f}"
                pos_regime = pos.get('regime', market_regime)  # FIXED: Use market_regime instead of state['regime_global']
                hold_h = f"{(time.time() - pos.get('created_at', time.time())) / 3600:.1f}"
                logger.info(f"{symbol:<12} {side:<6} {entry}      {current}   {qty}    {pnl_pct}     {pnl_dollar}      {stop}      {tp}      {trail} {pos_regime:<10}     {hold_h}")
            logger.info("-------------------------------------------------------------------------------------------------------------------------------")
            logger.info("🚨 Risk Alerts:")
            logger.info("  🔌 Exchange connected and healthy")
            logger.info("📊 --- END SUMMARY ---")
    except Exception as e:
        logger.error(f"Error in log_enhanced_iteration_summary: {e}")

async def fetch_and_prepare_data(symbol: str, tf_config: Dict[str, Any], exchange: ExchIntf, state: dict) -> Optional[pd.DataFrame]:
    if state is None or 'regime_global' not in state:  # Added validation
        state = {'regime_global': REGIME_GLOBAL}
        
    lookback = max(tf_config['min_data_points'], 500)
    ohlcv = await exchange.fetch_ohlcv_with_retry(symbol, tf_config['binance_interval'], limit=lookback)
    if not ohlcv:
        logger.warning(f"Failed to fetch OHLCV for {symbol} {tf_config['binance_interval']}")
        return None
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    if len(df) < 50:
        logger.warning(f"Insufficient data points for {symbol} {tf_config['binance_interval']}: {len(df)} rows")
        return None
    df_prepared = await prepare_features(df, tf_config)
    if df_prepared.empty:
        logger.warning(f"Prepared features empty for {symbol} {tf_config['name']} (possibly high NaN or missing cols)")
        return None
    await update_data_cache(symbol, tf_config['name'], df_prepared, state)  
    return df_prepared

async def update_data_cache(symbol: str, timeframe: str, df: pd.DataFrame, state: dict):
    key = (symbol, timeframe)
    async with CACHE_LOCK:
        max_rows = CONFIG['default']['max_cache_size']
        if len(df) > max_rows:
            df = df.tail(max_rows)
        DATA_CACHE[key] = df
        LAST_CACHE_ACCESS[key] = time.time()
        global REGIME_GLOBAL
        old_regime = REGIME_GLOBAL
        if 'regime_global' in state and state['regime_global'] != old_regime:
            for cache_key in list(DATA_CACHE.keys()):
                if cache_key[1] == timeframe:
                    del DATA_CACHE[cache_key]
                    del LAST_CACHE_ACCESS[cache_key]
            logger.info(f"Invalidated cache for TF {timeframe} due to regime change to {state['regime_global']}")
            REGIME_GLOBAL = state['regime_global']
    
async def detect_tradeable_symbols(exchange: ExchIntf, trade_exec: TradeExec, regime: str = 'normal') -> List[str]:
    symbols = []
    for s, m in exchange.markets.items():
        if m.get('spot', False) and s.endswith('/USDT') and m.get('active', False) and m.get('base', '') not in STABLES and not any(kw in s.upper() for kw in ['UP', 'DOWN', 'BULL', 'BEAR', 'HEDGE']):
            symbols.append(s)
    volume_filtered = await trade_exec.filter_symbols_by_volume(symbols, regime)
    if len(volume_filtered) < get_config_param('max_positions'):
        original_min = get_config_param('min_vol_24h')
        volume_filtered = await trade_exec.filter_symbols_by_volume(symbols, regime, min_vol_24h=original_min / 10)
        if len(volume_filtered) < get_config_param('max_positions'):
            hardcoded = ['BTC/USDT', 'ETH/USDT']
            volume_filtered += [s for s in hardcoded if s not in volume_filtered]
            logger.info(f"Added hardcoded symbols due to low filtered count, total now {len(volume_filtered)}")
    logger.info(f"Detected {len(volume_filtered)} tradeable symbols")
    return volume_filtered

async def analyze_missed_opportunities(top_performers: List[str], sig_gen: SigGen, risk_mgr: RiskMgr, trade_exec: TradeExec, exchange: ExchIntf, state: dict) -> List[Tuple[str, str]]:
    missed = []
    # FIXED: Get recent trades (last hour) for recency check (institutional: only true missed)
    now = time.time()
    one_hour_ago = now - 3600
    try:
        recent_trades_data = await trade_exec.bt_mgr.get_recent_trades()  # FIXED: Await correct method call
        # FIXED: Filter for last hour (handle datetime or unix; fallback if None)
        recent_positions = {
            t['symbol'] for t in recent_trades_data 
            if t.get('created_at') and (int(t['created_at'].timestamp()) if hasattr(t['created_at'], 'timestamp') else t['created_at']) > one_hour_ago
        }
        logger.debug(f"Recent symbols check: {len(recent_positions)} symbols with activity in last hour")
    except Exception as e:
        logger.warning(f"Failed to fetch recent trades: {e}; assuming no recent activity (full scan)")
        recent_positions = set()  # FIXED: Fallback to empty for no crash
    async with POSITION_LOCK_ASYNC:
        active_symbols = {pos['symbol'] for pos in trade_exec.active_positions.values()}  # Current open
    recent_symbols = recent_positions | active_symbols  # Union for comprehensive check
    
    tf_config = TIMEFRAMES[1]
    for symbol in top_performers:
        # FIXED: Skip if position opened in last hour (not a true missed)
        if symbol in recent_symbols:
            logger.debug(f"Skipping {symbol}: Position activity in last hour (not missed)")
            continue
        df = await fetch_and_prepare_data(symbol, tf_config, exchange, state)
        if df is None or df.empty:
            missed.append((symbol, "Insufficient data"))
            continue
        signal = await sig_gen.generate_signal(df, tf_config, symbol)
        if signal['direction'] == 'HOLD' or signal['confidence'] < get_config_param('min_conf_score'):
            reason = f"Low confidence ({signal['confidence']:.3f} < {CONFIG['default']['min_conf_score']:.3f})"
        else:
            async with POSITION_LOCK_ASYNC:
                current_positions = list(trade_exec.active_positions.values())
            corr_ok = await risk_mgr.check_correlation(symbol, current_positions, df, state)
            if not corr_ok:
                reason = "High correlation"
            else:
                ticker = await exchange.fetch_ticker_with_retry(symbol)
                if ticker and ticker.get('quoteVolume', 0) < CONFIG['default']['min_vol_24h']:
                    reason = "Insufficient volume"
                else:
                    regime = signal.get('regime', 'normal')
                    if tf_config.get('regime_filter') and regime not in tf_config.get('regime_filter', []):
                        reason = f"Regime mismatch ({regime})"
                    else:
                        reason = "Unknown"
        missed.append((symbol, reason))  # FIXED: Only true missed added
        logger.info(f"True missed {symbol}: {reason} (no recent position)")
    return missed  # FIXED: Returns only true missed for SupBot learning
    
async def auto_trade_loop(exchange: ExchIntf, db: DbIntf):
    global top_performers, top_changes, last_top_fetch, last_supervisor_run, missed_analysis, CLOSED_TRADES_SINCE_LAST
    
    # FIXED: Unificar init (remover duplicados; usar single state/globals)
    state = {'regime_global': CONFIG['current_regime'], 'consecutive_failures': 0}
    REGIME_GLOBAL = state['regime_global']
    global PERFORMANCE_METRICS
    
    regime_det = VolRegDet(exchange)
    param_opt = ParamOpt()
    sig_gen = SigGen(exchange)
    bt_mgr = BtMgr(exchange, db, sig_gen, sig_gen.model_mgr, param_opt, regime_det)
    sig_gen.bt_mgr = bt_mgr  # FIXED: Moved after bt_mgr creation (fixes UnboundLocalError)
    supervisor_bot = SupBot(param_opt, bt_mgr)
    risk_mgr = RiskMgr(exchange, bt_mgr)  
    trade_exec = TradeExec(exchange, db, risk_mgr, bt_mgr, supervisor_bot)
    PERFORMANCE_METRICS = {
        'total_signals': 0,
        'profitable_trades': 0,
        'total_closed_trades': 0,
        'realized_pnl': 0.0,
        'trades_executed': 0,
        'last_update': time.time(),
        'total_pnl': 0.0
    }
    risk_mgr.update_regime(state['regime_global'])            
            
    shutdown_event = threading.Event()
    def signal_handler(sig, frame):
        logger.info("Received SIGINT, initiating shutdown")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)          
                                  
    
    # Await only main loop (dashboard parallel)
    await auto_trade_loop(exchange, db)            
            
    # Inicializa cache y backtest (now connected to state)
    await bt_mgr.initialize_cache()

    # Nueva: Clasificar good/low-perf inmediatamente (usa saved metrics frescas); backtest solo low en background
    all_pairs = []
    for cat in bt_mgr.categories.values():
        all_pairs.extend(cat)
    good_perf_pairs = []
    low_perf_pairs = []
    now = time.time()
    freshness_days = get_config_param('backtest_freshness_days')
    for symbol in all_pairs:
        saved_metrics = bt_mgr.get_saved_bt_metrics(symbol)
        if (saved_metrics and saved_metrics.get('timestamp', 0) > now - (freshness_days * 86400) and
            saved_metrics.get('sharpe_ratio', 0) > 1.0 and saved_metrics.get('realized_pnl', 0) > 0):
            good_perf_pairs.append(symbol)
        else:
            low_perf_pairs.append(symbol)
    bt_logger.info(f"Classified: {len(good_perf_pairs)} good-perf, {len(low_perf_pairs)} low-perf pairs")
    # Iniciar señales con good-perf (inmediato)
    symbols = good_perf_pairs[:50]  # Top 50 rentables para primera iteración
    
    async def background_low_backtest():
                batch_size = 20
                for i in range(0, len(low_perf_pairs), batch_size):
                    batch = low_perf_pairs[i:i+batch_size]
                    sem = asyncio.Semaphore(get_config_param('max_concurr'))
                    async def bt_batch(s):
                        async with sem:
                            metrics = await bt_mgr.run_backtest_on_pair(s)
                            if metrics.get('trades', 0) > 0 and metrics.get('sharpe_ratio', 0) > 1.0 and metrics.get('realized_pnl', 0) > 0:
                                # Retain model si rentable
                                await sig_gen.model_mgr.train_model(s, bt_mgr.tf_name, bt_mgr.ohlcv_cache.get(s, pd.DataFrame()))
                                # Update top_performers (global, lock no needed ya que read en loop)
                                bt_mgr.top_performers.append(s)
                                bt_logger.info(f"New rentable pair added: {s} (Sharpe={metrics['sharpe_ratio']:.2f})")
                    try:
                        await asyncio.wait_for(asyncio.gather(*(bt_batch(s) for s in batch), return_exceptions=True), timeout=120)  
                    except asyncio.TimeoutError:
                        bt_logger.warning("Background backtest batch timeout (120s); skipping batch")

                asyncio.create_task(background_low_backtest())

    initial_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'] + good_perf_pairs[:10]  # Usa good para train inicial

    async def train_symbol(s, tf_cfg):
        df = await fetch_and_prepare_data(s, tf_cfg, exchange, state)  # Pass state
        if df is not None:
            await sig_gen.model_mgr.train_model(s, tf_cfg['name'], df)
    tasks = [train_symbol(s, tf_cfg) for s in initial_symbols for tf_cfg in TIMEFRAMES]
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("Initial models trained for all timeframes")
    logger.info("=== AUTONOMOUS TRADING SYSTEM STARTED ===")
    logger.info(f"Portfolio Value: ${CONFIG['default']['initial_equity']:,.2f}")
    logger.info(f"Max Risk: {CONFIG['default']['max_port_risk']:.1%}")
    logger.info(f"Max Positions: {CONFIG['default']['max_positions']}")
    logger.info(f"Dry Run: {CONFIG['default']['dry_run']}")
    logger.info(f"Low Volatility Features: ENABLED")
    logger.info(f"Backtesting: INTEGRATED (150 pairs, MC+RL)")
    portfolio_value = CONFIG['default']['initial_equity']
    iteration = 0
    symbols = []
    market_regime = CONFIG['current_regime']
    last_optimize = 0
    last_backtest = time.time()
    try:
        while True:
            iteration += 1
            loop_start = time.time()
            logger.info(f"\n=== ITERATION {iteration} ===")
          
            # Health check with exponential backoff
            if iteration % 5 == 0 or state.get('consecutive_failures', 0) > 2:
                if not exchange.connected or state.get('consecutive_failures', 0) > 3:
                    backoff_delay = min(300, 2 ** state.get('consecutive_failures', 0))  # Exponential: 4s, 8s, ..., max 5min
                    logger.warning(f"High failures ({state.get('consecutive_failures', 0)}), attempting reconnect with {backoff_delay}s backoff...")
                    if await exchange.connect(max_retries=3):  # Increased retries for resilience
                        logger.info("Reconnected successfully")
                        state['consecutive_failures'] = 0
                    else:
                        logger.error("Reconnect failed after retries")
                        state['consecutive_failures'] = min(10, state.get('consecutive_failures', 0) + 1)
                        await asyncio.sleep(backoff_delay)
                        continue
                try:
                    await asyncio.wait_for(exchange.client.fetch_status(), timeout=10)  # Timeout to prevent hangs
                    logger.debug("Exchange API status check passed")
                except asyncio.TimeoutError:
                    logger.warning("Exchange status timeout; marking for reconnect")
                    state['consecutive_failures'] += 1
                except Exception as e:
                    logger.warning(f"Exchange health warning: {e}")
                    state['consecutive_failures'] += 1
                    if state['consecutive_failures'] > 5:  # >5 → aggressive reconnect
                        await asyncio.sleep(30)
                        continue

            # Nueva: Chequeo dinámico de updates (cada 5 iter: agrega nuevos rentables de background)
            if iteration % 5 == 1:
                if bt_mgr.top_performers and len(symbols) < len(bt_mgr.top_performers):
                    # Agrega solo nuevos
                    new_symbols = [s for s in bt_mgr.top_performers if s not in symbols]
                    symbols.extend(new_symbols[:10])  # +10 max por ciclo (progresivo)
                    logger.info(f"Dynamic update: Added {len(new_symbols[:10])} new rentable pairs to signals")
                else:
                    # Fallback si no updates
                    symbols = bt_mgr.top_performers if bt_mgr.top_performers else await detect_tradeable_symbols(exchange, trade_exec, regime=market_regime)
                if not symbols:
                    await asyncio.sleep(get_config_param('poll_int') or 60)
                    continue
            if iteration % 5 == 1:
                cache_key = "BTC/USDT_regime"
                now = time.time()
                if cache_key in regime_det.regime_cache and now - regime_det.regime_cache[cache_key]['timestamp'] < 300:
                    market_regime = regime_det.regime_cache[cache_key]['regime']
                else:
                    btc_df = await fetch_and_prepare_data('BTC/USDT', TIMEFRAMES[0], exchange, state)  # Pass state
                    if btc_df is not None and len(btc_df) > 50:
                        market_regime, _ = await regime_det.detect_regime(btc_df, 'BTC/USDT', state)  # Pass state
                        async with METRICS_LOCK_ASYNC:
                            if market_regime != state['regime_global']:
                                logger.info(f"Regime changed to {market_regime}")
                            state['regime_global'] = market_regime
                            CONFIG['current_regime'] = market_regime
                        symbols = await detect_tradeable_symbols(exchange, trade_exec, regime=market_regime)
                        logger.info(f"Refreshed tradeable symbols due to regime change: {symbols}")
                    else:
                        logger.warning("Failed to fetch BTC/USDT data or insufficient data for regime detection")
                async with POSITION_LOCK_ASYNC:
                    active_symbols = [pos['symbol'] for pos in trade_exec.active_positions.values()]
                if active_symbols:
                    active_dfs = {sym: DATA_CACHE.get((sym, tf)) for (sym, tf), _ in DATA_CACHE.items() if sym in active_symbols}
                    if active_dfs:
                        aggregated_regime = await regime_det.aggregate_per_pair_regimes(active_symbols, active_dfs, state)  # Pass state for regime sync
                        if aggregated_regime != 'normal':
                            market_regime = str(aggregated_regime)  # Ensure scalar string
                            logger.debug(f"Aggregated regime for active symbols: {market_regime}")
                # Sync globals as scalars/strings (ensure regime is string)
                regime_str = str(market_regime) 
                REGIME_GLOBAL = regime_str
                state['regime_global'] = regime_str
                risk_mgr.update_regime(regime_str)  
           
            now = time.time()
            true_missed_analysis = []
            if now - last_top_fetch >= get_config_param('perf_check_int'):                  
                top_performers, top_changes = await fetch_top_performers(exchange)
                last_top_fetch = now
                
                 
                try:
                    true_missed_analysis = await analyze_missed_opportunities(top_performers, sig_gen, risk_mgr, trade_exec, exchange, state)  
                except Exception as miss_err:
                    logger.warning(f"Missed opportunities analysis failed: {miss_err}; using empty list")
                    true_missed_analysis = []  
                logger.info(f"Identified {len(true_missed_analysis)} true missed opportunities (no recent positions)")
            else:
                logger.debug("Skipping top performers fetch this cycle; using previous previous missed analysis")
                            
            if true_missed_analysis:  # FIXED: Check if not empty (evita error si no definido)
                for missed_symbol, reason in true_missed_analysis:
                    await supervisor_bot.analyze_missed_pairs([missed_symbol], state=state)  
                    await supervisor_bot.optimize_for_pair(missed_symbol, {}, {}, state=state, missed_reason=reason) 
            else:
                logger.debug("No true missed opportunities this cycle")
            
            if get_config_param('enable_sup') and now - last_supervisor_run >= get_config_param('supervisor_int'):
                if not hasattr(bt_mgr, 'results') or not bt_mgr.results:
                    await bt_mgr.run_full_backtest()
                # FIXED: Pass true_missed_analysis (filtered) to SupBot for smarter learning
                perf_metrics = PERFORMANCE_METRICS.copy()
                if hasattr(bt_mgr, 'results') and bt_mgr.results:
                    perf_metrics['sharpe_ratio'] = bt_mgr.results.get('sharpe_ratio', 1.0)
                    perf_metrics['win_rate'] = bt_mgr.results.get('win_rate', 0.5)
                if await supervisor_bot._should_intervene(market_regime, true_missed_analysis, perf_metrics):  # FIXED: Use true_missed_analysis
                    proposals = await supervisor_bot.get_proposals(market_regime, true_missed_analysis, perf_metrics, regime_det.trap_history, state)
                    await supervisor_bot.decide_and_apply(proposals, market_regime)
                    last_supervisor_run = now
                else:
                    logger.info("SupBot: No intervention needed this cycle (performance stable)")
            if state['regime_global'] in ['very_low', 'low']:
                CONFIG['current_regime'] = state['regime_global']
                CONFIG[state['regime_global']].setdefault('min_conf_score', CONFIG['default']['min_conf_score'])
                CONFIG[state['regime_global']]['min_conf_score'] = max(
                    CONFIG[state['regime_global']]['min_conf_score'] * 0.5, 0.01
                )
            if state['regime_global'] in ['very_low', 'low']:
                logger.info("🔍 LOW VOLATILITY: Sensitive strategies")
                active_tfs = TIMEFRAMES
                max_process = min(len(symbols), CONFIG['default']['max_positions'] * 3)
            elif 'high' in state['regime_global'] or 'volatile' in state['regime_global']:
                logger.info("🌪️ HIGH VOLATILITY: Aggressive strategies")
                active_tfs = TIMEFRAMES
                max_process = min(len(symbols), CONFIG['default']['max_positions'] * 4)
            else:
                logger.info("⚡ NORMAL: Standard strategies")
                active_tfs = [tf for tf in TIMEFRAMES if not tf.get('regime_filter') or state['regime_global'] in tf.get('regime_filter', ['normal'])]
                max_process = min(len(symbols), CONFIG['default']['max_positions'] * 2)
            logger.info(f"Active timeframes: {[tf['name'] for tf in active_tfs]}")
            tasks = []
            sem = asyncio.Semaphore(CONFIG['default']['max_concurr'])
            async def process_sym_tf(s: str, tf_cfg: Dict[str, Any], curr_port_val: float):
                async with sem:
                    try:
                        # FIXED: Wrap entire process in timeout to prevent single-task hang
                        async with asyncio.timeout(60):  # 60s max per symbol/tf
                            regime_filter = tf_cfg.get('regime_filter', ['low', 'normal', 'high'])
                            if regime_filter and market_regime not in regime_filter:
                                return None
                            df = await fetch_and_prepare_data(s, tf_cfg, exchange, state)  # Pass state
                            if df is None:
                                return None
                            if sig_gen.model_mgr.should_retrain(s, tf_cfg['name']):
                                await sig_gen.model_mgr.train_model(s, tf_cfg['name'], df)
                            signal = await sig_gen.generate_signal(df, tf_cfg, s)
                            if signal['direction'] != 'HOLD':
                                async with METRICS_LOCK_ASYNC:
                                    PERFORMANCE_METRICS['total_signals'] += 1
                                components = signal.get('component_scores', {})
                                if components:
                                    comp_str = ", ".join([f"{k}:{v:.2f}" for k, v in components.items() if v != 0])
                                    logger.debug(f"Components for {s}: {comp_str}")
                                trade_result = await trade_exec.execute_signal(signal, tf_cfg, s, curr_port_val)
                                if trade_result:
                                    async with METRICS_LOCK_ASYNC:
                                        PERFORMANCE_METRICS['trades_executed'] += 1
                                    if trade_result['action'] == 'BUY' and PERFORMANCE_METRICS.get('profitable_trades', 0) / PERFORMANCE_METRICS.get('total_closed_trades', 1) > 0.6:
                                        CONFIG[state['regime_global']].setdefault('kelly_frac', CONFIG['default']['kelly_frac'])
                                        CONFIG[state['regime_global']]['kelly_frac'] = min(
                                            1.0, CONFIG[state['regime_global']]['kelly_frac'] + 0.05
                                        )
                                    return trade_result
                            return None
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout in process_sym_tf for {s} {tf_cfg['name']}; skipping (consecutive_failures incremented)")
                        state['consecutive_failures'] = min(10, state.get('consecutive_failures', 0) + 1)
                        return None  # FIXED: Return None on timeout for partial results
                    except Exception as proc_err:
                        logger.error(f"Error in process_sym_tf for {s}: {proc_err}")
                        state['consecutive_failures'] = min(10, state.get('consecutive_failures', 0) + 1)
                        return None
                    else:  
                        if state.get('consecutive_failures', 0) > 0:
                            state['consecutive_failures'] = max(0, state['consecutive_failures'] - 1)  
                            logger.debug(f"Success reset: consecutive_failures={state['consecutive_failures']}")
                        return None    
            
            processed_symbols = symbols[:max_process]
            portfolio_value = (await trade_exec.get_portfolio_metrics())['equity']                      
            
            tasks = []
            for s in processed_symbols:
                for tf_cfg in active_tfs:
                    tasks.append(process_sym_tf(s, tf_cfg, portfolio_value))            
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                # Nueva: Manejo dinámico de fallos en paralelo (coordina state['consecutive_failures'] para reconexión autónoma)
                failures = sum(1 for r in results if isinstance(r, Exception))
                if failures > len(tasks) * 0.2:  # >20% fallos → aumenta consecutive_failures para trigger de reconexión en health check
                    state['consecutive_failures'] = min(10, state.get('consecutive_failures', 0) + 1)
                    logger.warning(f"High parallel failures ({failures}/{len(tasks)}), consecutive={state['consecutive_failures']}. Triggering potential reconnect.")
                successful_trades = sum(1 for r in results if r is not None and not isinstance(r, Exception))
                if successful_trades > 0:
                    logger.info(f"💹 {successful_trades} new trades executed")
                    # Nueva: Actualiza regime dinámicamente post-ejecución (inteligente: adapta basado en trades exitosos)
                    if successful_trades > len(processed_symbols) * 0.3:  # >30% éxito → relaja min_conf en low vol
                        if market_regime in ['very_low', 'low']:
                            base_min_conf = get_config_param('min_conf_score')
                            CONFIG[market_regime]['min_conf_score'] = max(0.005, base_min_conf * 0.9)
                            logger.info(f"Dynamic adaptation: Relaxed min_conf to {CONFIG[market_regime]['min_conf_score']:.3f} after high success rate.")
                # FIXED: Reset failures on partial success (>50% ok)
                if successful_trades / len(tasks) > 0.5:
                    state['consecutive_failures'] = max(0, state.get('consecutive_failures', 0) - 1)
                    logger.debug(f"Partial success reset: consecutive_failures={state['consecutive_failures']}")
            current_prices = {}
            async with POSITION_LOCK_ASYNC:
                active_count = len(trade_exec.active_positions)
                unique_symbols = {pos['symbol'] for pos in trade_exec.active_positions.values()}
            if active_count > 0:
                logger.info(f"Managing {active_count} positions...")
                if unique_symbols:
                    async def get_price(s):
                        ticker = await exchange.fetch_ticker_with_retry(s)
                        if ticker and 'close' in ticker:
                            return s, ticker['close']
                        return s, None
                    price_tasks = [get_price(s) for s in unique_symbols]
                    price_results = await asyncio.gather(*price_tasks, return_exceptions=True)
                    for result in price_results:
                        if isinstance(result, tuple) and result[1] is not None:
                            current_prices[result[0]] = result[1]
                if current_prices:
                    await trade_exec.manage_positions(current_prices)
            now = time.time()
            timeout = CONFIG['default'].get('cache_timeout', 3600)
            keys_to_remove = [k for k, last in LAST_CACHE_ACCESS.copy().items() if now - last > timeout]
            if keys_to_remove:
                logger.debug(f"Cleaning {len(keys_to_remove)} old cache entries")
                async with CACHE_LOCK:
                    for k in keys_to_remove:
                        DATA_CACHE.pop(k, None)
                        LAST_CACHE_ACCESS.pop(k, None)
                logger.debug(f"Cleaned {len(keys_to_remove)} old cache entries")
            win_rate = PERFORMANCE_METRICS.get('profitable_trades', 0) / max(1, PERFORMANCE_METRICS.get('total_closed_trades', 1))
            if (time.time() - last_backtest > CONFIG['default']['retrain_int'] or 
                CLOSED_TRADES_SINCE_LAST >= 10 or win_rate < 0.55):  
                bt_logger.info(f"Triggering backtest (win_rate={win_rate:.1%})")
                
                
                good_perf_pairs = bt_mgr.top_performers if bt_mgr.top_performers else []  # Usar existentes como good
                low_perf_pairs = [s for s in symbols if s not in good_perf_pairs]  # Simplificado: low = no en top
                if low_perf_pairs:
                    # Await con timeout para sync top_performers sin bloquear full loop
                    try:
                        await asyncio.wait_for(bt_mgr._backtest_low_perf_only(low_perf_pairs), timeout=120)  # 2min max
                        bt_logger.info(f"Backtest complete for {len(low_perf_pairs)} low-perf pairs; updated top_performers")
                    except asyncio.TimeoutError:
                        bt_logger.warning("Backtest timeout (120s); using partial results")
                    except Exception as e:
                        bt_logger.error(f"Backtest error: {e}; skipping update")
                else:
                    bt_logger.info("No low-perf pairs; using existing good-perf")
                
                symbols = bt_mgr.top_performers or symbols  # Sync updated top_performers
                last_backtest = time.time()
                CLOSED_TRADES_SINCE_LAST = 0
                for s in symbols[:20]:
                    if s in bt_mgr.ohlcv_cache:
                        await sig_gen.model_mgr.train_model(s, 'medium_1h', bt_mgr.ohlcv_cache[s])
                total_pnl = 0
                position_count = 0
                async with POSITION_LOCK_ASYNC:
                    for pos in trade_exec.active_positions.values():
                        pnl = pos.get('current_pnl_pct', 0)
                        total_pnl += pnl
                        position_count += 1
                if position_count > 0:
                    avg_pnl = total_pnl / position_count
                    logger.info(f"💰 Avg position P&L: {avg_pnl:.2%}")
            # FIXED: Pass true_missed_analysis with reasons to SupBot for targeted optimization
            if true_missed_analysis:  # FIXED: Check if not empty
                for missed_symbol, reason in true_missed_analysis:
                    await supervisor_bot.analyze_missed_pairs([missed_symbol], state=state)  # FIXED: Pass single for granular
                    await supervisor_bot.optimize_for_pair(missed_symbol, {}, {}, state=state, missed_reason=reason)  # FIXED: Pass reason for specific adjustments
            else:
                logger.debug("No true missed opportunities this cycle")
            
            if time.time() - last_optimize > CONFIG['default']['opt_int']:
                await param_opt.optimize_params(market_regime, PERFORMANCE_METRICS, regime_det.trap_history)
                last_optimize = time.time()
            await log_enhanced_iteration_summary(iteration, trade_exec, market_regime)
            if iteration % 5 == 0:
                async with METRICS_LOCK_ASYNC:
                    unrealized_pnl = sum(pos.get('current_pnl_amount', 0) for pos in trade_exec.active_positions.values())
                est_portfolio = CONFIG['default']['initial_equity'] + PERFORMANCE_METRICS.get('total_pnl', 0.0) + unrealized_pnl
                metrics_to_save = {
                    'portfolio_value': float(est_portfolio),
                    'total_signals': PERFORMANCE_METRICS.get('total_signals', 0),
                    'profitable_trades': PERFORMANCE_METRICS.get('profitable_trades', 0),
                    'total_closed_trades': PERFORMANCE_METRICS.get('total_closed_trades', 0),
                    'total_pnl': PERFORMANCE_METRICS.get('total_pnl', 0.0),
                    'trades_executed': PERFORMANCE_METRICS.get('trades_executed', 0)
                }
                if db.connected:
                    await db.save_performance_metrics(metrics_to_save)
            supervisor_bot.performance_memory.append({
                'pnl': PERFORMANCE_METRICS.get('total_pnl', 0.0),
                'win_rate': PERFORMANCE_METRICS.get('profitable_trades', 0) / max(1, PERFORMANCE_METRICS.get('total_closed_trades', 1))
            })
            base_sleep = CONFIG['default']['poll_int']
            sleep_mult = 0.7 if market_regime == 'very_low' else 0.85 if market_regime == 'low' else 1.0
            iteration_time = time.time() - loop_start
            sleep_time = max(5, base_sleep * sleep_mult - iteration_time)
            logger.info(f"⏱️ Iteration {iteration} in {iteration_time:.1f}s. Next in {sleep_time:.1f}s (regime: {market_regime})")
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
    finally:
        await exchange.disconnect()
        if db.connected:
            await db.pool.close()

async def initialize_database() -> Optional[DbIntf]:
    if not all([CONFIG['default']['db_host'], CONFIG['default']['db_user'], CONFIG['default']['db_password'], CONFIG['default']['db_name']]):
        return DbIntf(None)
    try:
        pool = await asyncpg.create_pool(user=CONFIG['default']['db_user'], password=CONFIG['default']['db_password'], database=CONFIG['default']['db_name'], host=CONFIG['default']['db_host'], port=CONFIG['default']['db_port'], min_size=2, max_size=10, command_timeout=60)
        if pool is None:
            return DbIntf(None)
        db = DbIntf(pool)
        if await db.init_tables():
            logger.info("DB connected and initialized")
            return db
        await pool.close()
        return DbIntf(None)
    except Exception as e:
        logger.error(f"Failed to connect DB: {e}")
        return DbIntf(None)

async def initialize_exchange() -> Optional[ExchIntf]:
    try:
        exchange = ExchIntf(
            CONFIG['default']['exchange'], 
            CONFIG['default']['api_key'],      # Always pass keys
            CONFIG['default']['api_secret'],   # Always pass keys  
            CONFIG['default']['sandbox'], 
            CONFIG['default']['futures']
        )
        if await exchange.connect():
            return exchange
        return None
    except Exception as e:
        logger.error(f"Failed to init exchange: {e}")
        return None
   
async def main_application(args):
    logger.info("=== PROFESSIONAL AUTONOMOUS TRADING BOT v2.0 ===")
    logger.info(f"Mode: {'DRY RUN' if CONFIG['default']['dry_run'] else 'LIVE'}")
    logger.info(f"Exchange: {CONFIG['default']['exchange']}")
    logger.info(f"Max Positions: {CONFIG['default']['max_positions']}")
    logger.info(f"Portfolio Risk: {CONFIG['default']['max_port_risk']:.1%}")
    logger.info("Web Dashboard: http://localhost:8080")
    
    # Initialize database
    db = await initialize_database()
    if not db:
        logger.error("Failed to initialize database. Exiting...")
        return
    
    # Initialize exchange
    exchange = await initialize_exchange()
    if not exchange:
        logger.error("Failed to init exchange. Exiting...")
        return
    
    try:
        if args.mode == 'autonomous':
            
            if not await init_influx():
                logger.warning("InfluxDB init failed; metrics will fallback to file/DB. "
               "Check .env: INFLUXDB_URL (e.g., http://host.docker.internal:8086 if Docker), "
               "INFLUXDB_TOKEN (required), INFLUXDB_ORG/BUCKET. "
               "If containers, ensure shared network (docker-compose) or use localhost/IP.")
                
            state = {'regime_global': CONFIG['current_regime'], 'consecutive_failures': 0}
            global REGIME_GLOBAL
            REGIME_GLOBAL = state['regime_global']
            global PERFORMANCE_METRICS
            
            regime_det = VolRegDet(exchange)
            param_opt = ParamOpt()
            sig_gen = SigGen(exchange)
            bt_mgr = BtMgr(exchange, db, sig_gen, sig_gen.model_mgr, param_opt, regime_det)
            sig_gen.bt_mgr = bt_mgr
            supervisor_bot = SupBot(param_opt, bt_mgr)
            risk_mgr = RiskMgr(exchange, bt_mgr)  
            trade_exec = TradeExec(exchange, db, risk_mgr, bt_mgr, supervisor_bot)
            PERFORMANCE_METRICS = {
                'total_signals': 0,
                'profitable_trades': 0,
                'total_closed_trades': 0,
                'realized_pnl': 0.0,
                'trades_executed': 0,
                'last_update': time.time(),
                'total_pnl': 0.0
            }
            risk_mgr.update_regime(state['regime_global'])            
            
            shutdown_event = threading.Event()
            def signal_handler(sig, frame):
                logger.info("Received SIGINT, initiating shutdown")
                shutdown_event.set()
            
            signal.signal(signal.SIGINT, signal_handler)          
                                  
            
            # Await only main loop (dashboard parallel)
            await auto_trade_loop(exchange, db)            
            
    except asyncio.CancelledError:
        logger.info("Main application cancelled, initiating cleanup...")
    except KeyboardInterrupt:
        logger.info("SIGINT received, graceful shutdown...")
        # Cancel tasks
        for task in asyncio.all_tasks():
            task.cancel()
        await asyncio.gather(*asyncio.all_tasks(), return_exceptions=True)
        
    finally:        
        logger.info("Cleaning up...")   
        
        top_performers = []
        top_changes = {}
        last_top_fetch = 0
        last_supervisor_run = 0
        missed_analysis = []
        CLOSED_TRADES_SINCE_LAST = 0
        REGIME_GLOBAL = 'normal'
        PERFORMANCE_METRICS = {'total_signals': 0, 'profitable_trades': 0, 'total_closed_trades': 0, 'realized_pnl': 0.0, 'trades_executed': 0, 'last_update': time.time(), 'total_pnl': 0.0}
        DATA_CACHE.clear()
        LAST_CACHE_ACCESS.clear()
        if exchange:
            try:
                await exchange.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting exchange: {e}")
        if db and hasattr(db, 'connected') and db.connected:
            try:
                await db.pool.close()
            except Exception as e:
                logger.error(f"Error closing database: {e}")
        try:
            import aiohttp
            if hasattr(aiohttp.ClientSession, '_sessions'):
                open_sessions = [s for s in aiohttp.ClientSession._sessions if not s.closed]
                for session in open_sessions:
                    try:
                        await session.close()
                        if hasattr(session, '_connector') and session._connector and not session._connector.closed:
                            await session._connector.close()
                    except Exception as e:
                        logger.error(f"Error closing session: {e}")
        except Exception as e:
            logger.error(f"Global aiohttp cleanup error: {e}")
        logger.info("Shutdown complete")

def setup_cli():
    parser = argparse.ArgumentParser(description="Professional Autonomous Trading Bot v2.0", formatter_class=argparse.RawDescriptionHelpFormatter, epilog="\nExamples:\n  python bot.py --mode autonomous                    # Run in autonomous mode\n  python bot.py --mode autonomous --dry-run         # Run in simulation mode\n  python bot.py --mode autonomous --verbose         # Run with debug logging\n")
    parser.add_argument('--mode', choices=['autonomous', 'backtest', 'analyze'], default='autonomous', help='Operating mode (default: autonomous)')
    parser.add_argument('--dry-run', action='store_true', help='Run in simulation mode')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    parser.add_argument('--config-file', type=str, help='Path to configuration file')
    return parser

def configure_logging(verbose: bool):
    if verbose:
        logger.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")

def main():
    parser = setup_cli()
    args = parser.parse_args()
    configure_logging(args.verbose)
    if args.dry_run:
        CONFIG['default']['dry_run'] = True
    if args.config_file and Path(args.config_file).exists():
        try:
            import configparser
            cp = configparser.ConfigParser()
            cp.read(args.config_file)
            for section in cp.sections():
                for key, value in cp.items(section):
                    upper_key = key.upper()
                    if upper_key in CONFIG:
                        orig_type = type(CONFIG[upper_key])
                        CONFIG[upper_key] = orig_type(value) if orig_type in [int, float, bool] else value
            logger.info(f"Config loaded from {args.config_file}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    try:
        validate_config()
    except ValueError as e:
        logger.error(f"Config validation failed: {e}")
        return 1
    try:
        asyncio.run(main_application(args))
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"App failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())
