import asyncio
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import talib


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for given DataFrame"""
    if df.empty:
        return df

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values

    # RSI
    df["rsi"] = talib.RSI(close, timeperiod=14)

    # MACD
    macd, macd_signal, macd_hist = talib.MACD(close)
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    # Bollinger Bands
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = talib.BBANDS(close, timeperiod=20)

    # Moving Averages
    df["sma_20"] = talib.SMA(close, timeperiod=20)
    df["sma_50"] = talib.SMA(close, timeperiod=50)
    df["ema_12"] = talib.EMA(close, timeperiod=12)
    df["ema_26"] = talib.EMA(close, timeperiod=26)

    # Stochastic
    df["stoch_k"], df["stoch_d"] = talib.STOCH(high, low, close)

    # Volume indicators
    df["volume_sma"] = talib.SMA(volume, timeperiod=20)
    df["obv"] = talib.OBV(close, volume)

    # Volatility
    df["atr"] = talib.ATR(high, low, close, timeperiod=14)
    df["natr"] = talib.NATR(high, low, close, timeperiod=14)

    # Trend indicators
    df["adx"] = talib.ADX(high, low, close, timeperiod=14)
    df["cci"] = talib.CCI(high, low, close, timeperiod=14)

    return df


def generate_signals(df: pd.DataFrame) -> Dict[str, bool]:
    """Generate trading signals based on technical indicators"""
    if df.empty:
        return {}

    latest = df.iloc[-1]

    signals = {
        "rsi_overbought": latest["rsi"] > 70 if pd.notna(latest["rsi"]) else False,
        "rsi_oversold": latest["rsi"] < 30 if pd.notna(latest["rsi"]) else False,
        "macd_bullish": latest["macd"] > latest["macd_signal"]
        if pd.notna(latest["macd"])
        else False,
        "macd_bearish": latest["macd"] < latest["macd_signal"]
        if pd.notna(latest["macd"])
        else False,
        "price_above_sma20": latest["close"] > latest["sma_20"]
        if pd.notna(latest["sma_20"])
        else False,
        "price_below_sma20": latest["close"] < latest["sma_20"]
        if pd.notna(latest["sma_20"])
        else False,
        "golden_cross": latest["sma_20"] > latest["sma_50"]
        if pd.notna(latest["sma_50"])
        else False,
        "death_cross": latest["sma_20"] < latest["sma_50"]
        if pd.notna(latest["sma_50"])
        else False,
        "stoch_overbought": latest["stoch_k"] > 80
        if pd.notna(latest["stoch_k"])
        else False,
        "stoch_oversold": latest["stoch_k"] < 20
        if pd.notna(latest["stoch_k"])
        else False,
    }

    return signals


def calculate_position_size(
    account_balance: float, risk_per_trade: float, entry_price: float, stop_loss: float
) -> float:
    """Calculate position size based on risk management"""
    risk_amount = account_balance * risk_per_trade
    risk_per_share = abs(entry_price - stop_loss)

    if risk_per_share == 0:
        return 0

    position_size = risk_amount / risk_per_share
    return position_size


def format_price(price: float) -> str:
    """Format price with appropriate precision"""
    if price >= 1000:
        return f"{price:.2f}"
    elif price >= 1:
        return f"{price:.4f}"
    else:
        return f"{price:.8f}"


async def async_retry(func, max_retries: int = 3, delay: float = 1.0, **kwargs):
    """Retry async function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return await func(**kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(delay * (2**attempt))
    return None


def calculate_volatility(df: pd.DataFrame, period: int = 20) -> float:
    """Calculate price volatility"""
    if len(df) < period:
        return 0

    returns = df["close"].pct_change().dropna()
    if len(returns) < period:
        return 0

    return returns.tail(period).std()


def normalize_data(data: np.ndarray) -> np.ndarray:
    """Normalize data to [0, 1] range"""
    if len(data) == 0:
        return data

    min_val = np.min(data)
    max_val = np.max(data)

    if max_val == min_val:
        return np.zeros_like(data)

    return (data - min_val) / (max_val - min_val)


def create_lagged_features(
    df: pd.DataFrame, columns: List[str], lags: List[int]
) -> pd.DataFrame:
    """Create lagged features for time series data"""
    df_copy = df.copy()

    for col in columns:
        for lag in lags:
            df_copy[f"{col}_lag_{lag}"] = df_copy[col].shift(lag)

    return df_copy


def calculate_correlation_matrix(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation matrix for multiple symbols"""
    returns = prices_df.pct_change().dropna()
    return returns.corr()


def safe_divide(a: float, b: float) -> float:
    """Safe division with zero check"""
    if b == 0:
        return 0
    return a / b


def format_timestamp(timestamp: Any) -> str:
    """Format timestamp to ISO string"""
    if isinstance(timestamp, str):
        return timestamp
    elif isinstance(timestamp, datetime):
        return timestamp.isoformat()
    elif isinstance(timestamp, (int, float)):
        return datetime.fromtimestamp(timestamp).isoformat()
    else:
        return datetime.now().isoformat()


def validate_market_data(data: Dict) -> bool:
    """Validate market data structure"""
    required_fields = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    return all(field in data for field in required_fields)


def calculate_performance_metrics(trades: List[Dict]) -> Dict[str, float]:
    """Calculate trading performance metrics"""
    if not trades:
        return {}

    winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
    losing_trades = [t for t in trades if t.get("pnl", 0) < 0]

    total_pnl = sum(t.get("pnl", 0) for t in trades)
    avg_win = (
        np.mean([t.get("pnl", 0) for t in winning_trades]) if winning_trades else 0
    )
    avg_loss = np.mean([t.get("pnl", 0) for t in losing_trades]) if losing_trades else 0

    return {
        "total_trades": len(trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": len(winning_trades) / len(trades) if trades else 0,
        "total_pnl": total_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else float("inf"),
    }
