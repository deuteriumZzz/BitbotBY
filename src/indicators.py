import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет технические индикаторы к DataFrame
    """
    try:
        df = df.copy()

        # RSI
        df["rsi"] = calculate_rsi(df["close"])

        # MACD
        df["macd"], df["macd_signal"] = calculate_macd(df["close"])
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Bollinger Bands
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = calculate_bollinger_bands(
            df["close"]
        )
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # Moving Averages
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()
        df["ema_12"] = df["close"].ewm(span=12).mean()
        df["ema_26"] = df["close"].ewm(span=26).mean()

        # Volatility
        df["atr"] = calculate_atr(df, period=14)
        df["volatility"] = df["close"].rolling(window=20).std()

        # Momentum
        df["momentum"] = df["close"].pct_change(period=10)

        # Volume indicators
        df["volume_sma"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]

        # Fill NaN values
        df = df.fillna(method="bfill").fillna(method="ffill")

        return df

    except Exception as e:
        logger.error(f"Error adding indicators: {e}")
        return df


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series]:
    """Calculate MACD indicator"""
    fast_ema = series.ewm(span=fast).mean()
    slow_ema = series.ewm(span=slow).mean()
    macd = fast_ema - slow_ema
    macd_signal = macd.ewm(span=signal).mean()
    return macd, macd_signal


def calculate_bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands"""
    middle_band = series.rolling(window=period).mean()
    std_dev = series.rolling(window=period).std()
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    return upper_band, middle_band, lower_band


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize data for ML models"""
    df_normalized = df.copy()
    for column in df.columns:
        if df[column].std() != 0:
            df_normalized[column] = (df[column] - df[column].mean()) / df[column].std()
    return df_normalized.fillna(0)
