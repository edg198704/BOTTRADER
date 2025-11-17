import pandas as pd
import numpy as np
from bot_core.logger import get_logger

logger = get_logger(__name__)

def create_dataframe(ohlcv_data: list) -> pd.DataFrame | None:
    """Create DataFrame from OHLCV data with complete validation."""
    try:
        if not ohlcv_data or len(ohlcv_data) == 0:
            logger.warning("OHLCV data is empty")
            return None
        
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if set(required_cols) - set(df.columns):
            logger.error("Missing required columns in OHLCV data")
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=['timestamp', 'close', 'open', 'high', 'low'], inplace=True)
        df['volume'].fillna(0, inplace=True)

        if len(df) < 20:
            logger.warning("Insufficient data after cleaning", final_rows=len(df))
            return None

        df.set_index('timestamp', inplace=True)
        return df
        
    except Exception as e:
        logger.error("Failed to create DataFrame", error=str(e), exc_info=True)
        return None

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate a comprehensive set of technical indicators."""
    if df is None or len(df) < 50:
        logger.debug("DataFrame has insufficient data for all indicators.", data_length=len(df) if df is not None else 0)
        return df

    df_out = df.copy()

    # RSI
    delta = df_out['close'].diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=13, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    df_out['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df_out['close'].ewm(span=12, adjust=False).mean()
    ema26 = df_out['close'].ewm(span=26, adjust=False).mean()
    df_out['macd'] = ema12 - ema26
    df_out['macd_signal'] = df_out['macd'].ewm(span=9, adjust=False).mean()
    df_out['macd_hist'] = df_out['macd'] - df_out['macd_signal']

    # Bollinger Bands
    sma_20 = df_out['close'].rolling(20).mean()
    std_20 = df_out['close'].rolling(20).std()
    df_out['bb_upper'] = sma_20 + (std_20 * 2)
    df_out['bb_lower'] = sma_20 - (std_20 * 2)
    df_out['bb_middle'] = sma_20

    # ATR (for Risk Manager)
    high_low = df_out['high'] - df_out['low']
    high_close = abs(df_out['high'] - df_out['close'].shift())
    low_close = abs(df_out['low'] - df_out['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_out['atr'] = tr.ewm(alpha=1/14, adjust=False).mean()

    # ADX
    plus_dm = df_out['high'].diff()
    minus_dm = df_out['low'].diff().mul(-1)
    plus_dm[plus_dm < 0] = 0
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < 0] = 0
    minus_dm[minus_dm < plus_dm] = 0
    tr14 = tr.rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / tr14)
    minus_di = 100 * (minus_dm.rolling(14).sum() / tr14)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_out['adx'] = dx.rolling(14).mean()

    # SMAs for crossover strategy
    df_out['sma_fast'] = df_out['close'].rolling(window=10).mean()
    df_out['sma_slow'] = df_out['close'].rolling(window=20).mean()

    df_out.dropna(inplace=True)
    logger.debug("Technical indicators calculated", row_count=len(df_out))
    return df_out
