#!/usr/bin/env python3
"""
Enterprise AI Trading Bot - Utility Functions

Common utilities for the enterprise trading bot.
"""

import os
import sys
import json
import asyncio
import logging
import hashlib
import hmac
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np


# =====================================================================================
# UTILITY FUNCTIONS
# =====================================================================================

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration"""
    log_level = getattr(logging, config.get('log_level', 'INFO').upper(), logging.INFO)
    enable_debug = config.get('enable_debug_logging', False)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('trading_bot.log')
        ]
    )
    
    return logging.getLogger(__name__)


def validate_environment() -> Dict[str, bool]:
    """Validate environment setup and dependencies"""
    checks = {}
    
    # Check Python version
    checks['python_version'] = sys.version_info >= (3, 8)
    
    # Check required environment variables
    required_vars = ['EXCHANGE_API_KEY', 'EXCHANGE_API_SECRET']
    checks['environment_vars'] = all(os.getenv(var) for var in required_vars)
    
    # Check required packages
    required_packages = ['ccxt', 'pandas', 'numpy']
    for package in required_packages:
        try:
            __import__(package)
            checks[f'package_{package}'] = True
        except ImportError:
            checks[f'package_{package}'] = False
    
    return checks


def calculate_position_size(capital: float, risk_per_trade: float, 
                          entry_price: float, stop_loss_price: float) -> float:
    """Calculate position size based on risk management"""
    try:
        risk_amount = capital * risk_per_trade
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk <= 0:
            return 0.0
        
        position_size = risk_amount / price_risk
        return max(0.0, position_size)
        
    except Exception as e:
        logging.error(f"Position size calculation failed: {e}")
        return 0.0


def calculate_pnl(entry_price: float, exit_price: float, size: float, side: str) -> Dict[str, float]:
    """Calculate P&L for a trade"""
    try:
        if side.upper() == 'BUY':
            pnl = (exit_price - entry_price) * size
            pnl_percentage = (exit_price - entry_price) / entry_price
        else:  # SELL
            pnl = (entry_price - exit_price) * size
            pnl_percentage = (entry_price - exit_price) / entry_price
        
        return {
            'pnl': pnl,
            'pnl_percentage': pnl_percentage,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'side': side.upper()
        }
        
    except Exception as e:
        logging.error(f"P&L calculation failed: {e}")
        return {'pnl': 0.0, 'pnl_percentage': 0.0}


def format_currency(amount: float, currency: str = 'USD') -> str:
    """Format currency amount"""
    try:
        if currency == 'USD':
            return f"${amount:,.2f}"
        elif currency == 'BTC':
            return f"₿{amount:.8f}"
        elif currency == 'ETH':
            return f"Ξ{amount:.6f}"
        else:
            return f"{amount:,.2f} {currency}"
    except Exception:
        return f"{amount:,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format percentage value"""
    try:
        return f"{value:.{decimals}f}%"
    except Exception:
        return "0.00%"


def generate_correlation_id() -> str:
    """Generate unique correlation ID"""
    import uuid
    return str(uuid.uuid4())


def safe_json_dumps(data: Any) -> str:
    """Safely serialize data to JSON"""
    try:
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        return json.dumps(data, default=json_serializer, indent=2)
    except Exception as e:
        logging.error(f"JSON serialization failed: {e}")
        return "{}"


def load_json_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Load JSON file safely"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON file {file_path}: {e}")
        return None


def save_json_file(data: Any, file_path: str) -> bool:
    """Save data to JSON file safely"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write(safe_json_dumps(data))
        return True
    except Exception as e:
        logging.error(f"Failed to save JSON file {file_path}: {e}")
        return False


def sanitize_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove or mask sensitive information from data"""
    sensitive_keys = ['password', 'secret', 'token', 'api_key', 'api_secret', 'private_key']
    
    def sanitize_recursive(obj):
        if isinstance(obj, dict):
            return {k: '[REDACTED]' if any(sensitive in k.lower() for sensitive in sensitive_keys) 
                   else sanitize_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_recursive(item) for item in obj]
        else:
            return obj
    
    return sanitize_recursive(data)


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio"""
    try:
        if not returns or len(returns) < 2:
            return 0.0
        
        # Convert to daily returns if needed
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio
        excess_return = mean_return - (risk_free_rate / 252)  # Daily risk-free rate
        sharpe = (excess_return / std_return) * np.sqrt(252)
        
        return float(sharpe)
        
    except Exception as e:
        logging.error(f"Sharpe ratio calculation failed: {e}")
        return 0.0


def calculate_max_drawdown(equity_curve: List[float]) -> Dict[str, float]:
    """Calculate maximum drawdown and related metrics"""
    try:
        if len(equity_curve) < 2:
            return {'max_drawdown': 0.0, 'max_drawdown_duration': 0, 'recovery_factor': 0.0}
        
        equity_series = pd.Series(equity_curve)
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        
        max_drawdown = float(drawdown.min())
        
        # Calculate drawdown duration
        in_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Recovery factor (if applicable)
        recovery_factor = 0.0
        if max_drawdown < 0:
            total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0]
            recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'recovery_factor': float(recovery_factor)
        }
        
    except Exception as e:
        logging.error(f"Max drawdown calculation failed: {e}")
        return {'max_drawdown': 0.0, 'max_drawdown_duration': 0, 'recovery_factor': 0.0}


def calculate_var(returns: List[float], confidence_level: float = 0.05) -> float:
    """Calculate Value at Risk (VaR)"""
    try:
        if not returns:
            return 0.0
        
        return float(np.percentile(returns, confidence_level * 100))
        
    except Exception as e:
        logging.error(f"VaR calculation failed: {e}")
        return 0.0


def encrypt_api_key(api_key: str, secret: str) -> str:
    """Encrypt API key using HMAC"""
    try:
        return hmac.new(secret.encode(), api_key.encode(), hashlib.sha256).hexdigest()
    except Exception as e:
        logging.error(f"API key encryption failed: {e}")
        return api_key


def create_directory_structure():
    """Create necessary directory structure"""
    directories = [
        'logs',
        'data',
        'models',
        'reports',
        'config',
        'cache',
        'backups'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def validate_symbol(symbol: str, exchange: str) -> bool:
    """Validate trading symbol format"""
    try:
        # Basic symbol format validation
        if '/' not in symbol:
            return False
        
        base, quote = symbol.split('/')
        
        # Check for valid characters
        valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        if not all(c in valid_chars for c in base + quote):
            return False
        
        return True
        
    except Exception:
        return False


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations"""
    try:
        # Remove invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Limit length
        if len(filename) > 200:
            filename = filename[:200]
        
        return filename
        
    except Exception:
        return "unnamed_file"


def get_file_hash(file_path: str) -> Optional[str]:
    """Calculate SHA256 hash of file"""
    try:
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logging.error(f"Failed to calculate file hash: {e}")
        return None


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries recursively"""
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            for k, v in d.items():
                if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                    result[k] = merge_dicts(result[k], v)
                else:
                    result[k] = v
    return result


def get_timeframe_in_seconds(timeframe: str) -> int:
    """Convert timeframe string to seconds"""
    timeframe_map = {
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1h': 3600,
        '4h': 14400,
        '1d': 86400
    }
    return timeframe_map.get(timeframe, 3600)


def format_timeframe(timeframe: str) -> str:
    """Format timeframe for display"""
    timeframe_formats = {
        '1m': '1 Minute',
        '5m': '5 Minutes',
        '15m': '15 Minutes',
        '30m': '30 Minutes',
        '1h': '1 Hour',
        '4h': '4 Hours',
        '1d': '1 Day'
    }
    return timeframe_formats.get(timeframe, timeframe)


def get_next_market_open_time(timezone_name: str = 'UTC') -> datetime:
    """Get next market open time (simplified)"""
    try:
        now = datetime.now(timezone.utc)
        
        # Simplified - assume 24/7 crypto markets
        return now + timedelta(minutes=1)
        
    except Exception:
        return datetime.now(timezone.utc)


def cleanup_old_files(directory: str, max_age_days: int = 30) -> int:
    """Clean up old files from directory"""
    try:
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        cleaned_count = 0
        
        for file_path in Path(directory).glob('*'):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime, timezone.utc)
                if file_time < cutoff_time:
                    file_path.unlink()
                    cleaned_count += 1
        
        return cleaned_count
        
    except Exception as e:
        logging.error(f"File cleanup failed: {e}")
        return 0


# =====================================================================================
# ASYNC UTILITIES
# =====================================================================================

async def wait_with_timeout(coro, timeout: float = 30.0):
    """Wait for coroutine with timeout"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logging.warning(f"Operation timed out after {timeout} seconds")
        return None


async def retry_async(func, max_attempts: int = 3, delay: float = 1.0, *args, **kwargs):
    """Retry async function with exponential backoff"""
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                await asyncio.sleep(delay * (2 ** attempt))
            else:
                break
    
    raise last_exception


# =====================================================================================
# DATA UTILITIES
# =====================================================================================

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate basic technical indicators"""
    try:
        result = df.copy()
        
        # Moving averages
        result['sma_20'] = df['close'].rolling(20).mean()
        result['sma_50'] = df['close'].rolling(50).mean()
        result['ema_12'] = df['close'].ewm(span=12).mean()
        result['ema_26'] = df['close'].ewm(span=26).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        result['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        result['macd'] = result['ema_12'] - result['ema_26']
        result['macd_signal'] = result['macd'].ewm(span=9).mean()
        result['macd_histogram'] = result['macd'] - result['macd_signal']
        
        # Bollinger Bands
        bb_std = df['close'].rolling(20).std()
        result['bb_upper'] = result['sma_20'] + (bb_std * 2)
        result['bb_lower'] = result['sma_20'] - (bb_std * 2)
        result['bb_position'] = (df['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
        
        return result
        
    except Exception as e:
        logging.error(f"Technical indicators calculation failed: {e}")
        return df


# =====================================================================================
# MAIN EXECUTION
# =====================================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enterprise Trading Bot Utilities')
    parser.add_argument('--validate-env', action='store_true', help='Validate environment setup')
    parser.add_argument('--create-dirs', action='store_true', help='Create directory structure')
    parser.add_argument('--cleanup-files', type=int, help='Clean up files older than N days')
    parser.add_argument('--calc-sharpe', type=str, help='Calculate Sharpe ratio from returns file')
    parser.add_argument('--calc-drawdown', type=str, help='Calculate max drawdown from equity file')
    
    args = parser.parse_args()
    
    if args.validate_env:
        checks = validate_environment()
        print("Environment validation:")
        for check, result in checks.items():
            status = "✓" if result else "✗"
            print(f"  {status} {check}")
    
    if args.create_dirs:
        create_directory_structure()
        print("Directory structure created")
    
    if args.cleanup_files:
        cleaned = cleanup_old_files('.', args.cleanup_files)
        print(f"Cleaned up {cleaned} old files")
    
    if args.calc_sharpe:
        returns = load_json_file(args.calc_sharpe) or []
        sharpe = calculate_sharpe_ratio(returns)
        print(f"Sharpe Ratio: {sharpe:.4f}")
    
    if args.calc_drawdown:
        equity = load_json_file(args.calc_drawdown) or []
        drawdown_stats = calculate_max_drawdown(equity)
        print(f"Max Drawdown: {drawdown_stats['max_drawdown']:.2%}")
        print(f"Max DD Duration: {drawdown_stats['max_drawdown_duration']} periods")
        print(f"Recovery Factor: {drawdown_stats['recovery_factor']:.2f}")