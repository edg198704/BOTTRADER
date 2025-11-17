import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def create_dataframe(ohlcv_data: list) -> pd.DataFrame | None:
    """Create DataFrame from OHLCV data with validation."""
    try:
        if not ohlcv_data or len(ohlcv_data) < 20:
            logger.warning("Insufficient OHLCV data to create DataFrame.")
            return None

        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        if len(df) < 20:
            logger.warning(f"Insufficient data rows ({len(df)}) after cleaning.")
            return None

        return df
    except Exception as e:
        logger.error(f"Error creating DataFrame: {e}", exc_info=True)
        return None

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all necessary technical indicators."""
    if df is None or len(df) < 50:
        logger.warning(f"DataFrame has insufficient data ({len(df) if df is not None else 0}) for all indicators.")
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

    # Bollinger Bands
    sma_20 = df_out['close'].rolling(20).mean()
    std_20 = df_out['close'].rolling(20).std()
    df_out['bb_upper'] = sma_20 + (std_20 * 2)
    df_out['bb_lower'] = sma_20 - (std_20 * 2)

    # ATR (for Risk Manager)
    high_low = df_out['high'] - df_out['low']
    high_close = abs(df_out['high'] - df_out['close'].shift())
    low_close = abs(df_out['low'] - df_out['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_out['atr'] = tr.ewm(alpha=1/14, adjust=False).mean()

    df_out.dropna(inplace=True)
    logger.debug(f"Technical indicators calculated for DataFrame with {len(df_out)} rows.")
    return df_out
