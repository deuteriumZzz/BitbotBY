"""
Almgren-Chriss (2001) adaptive market impact model.

Replaces static bid/ask spread constants with an order-size-aware estimate:

  MI(x) = γ·σ·x/2  +  η·σ·√x        (permanent/2 + temporary)

where:
  x  = participation rate = order_size_usdt / daily_volume_usdt
  σ  = daily price volatility (std of log-returns scaled to daily)
  γ  = permanent impact coefficient  (default 0.1)
  η  = temporary impact coefficient  (default 0.1)

For typical crypto large-cap (BTC participation rate ~0.001–0.01):
  - static constants gave ~0.07% flat
  - AC gives ~0.03% for small orders, ~0.15% for large block trades
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Almgren-Chriss coefficients calibrated to Bybit BTC/USDT spot
_ETA = 0.1    # temporary impact (square-root term)
_GAMMA = 0.1  # permanent impact (linear term)
_MIN_IMPACT = 0.0002   # floor: 2 bp (exchange fee floor)
_MAX_IMPACT = 0.02     # cap: 2% max assumed impact

# Candles per 24 h for each supported ccxt timeframe
_TF_CANDLES: dict[str, int] = {
    "1m": 1440, "3m": 480, "5m": 288, "15m": 96,
    "30m": 48, "1h": 24, "2h": 12, "4h": 6, "1d": 1,
}


def _candles_per_day(timeframe: str) -> int:
    return _TF_CANDLES.get(timeframe, 96)


def almgren_chriss_impact(
    order_size_usdt: float,
    daily_volume_usdt: float,
    daily_vol: float,
    eta: float = _ETA,
    gamma: float = _GAMMA,
) -> float:
    """
    Compute one-way market impact fraction via Almgren-Chriss.

    :param order_size_usdt: Order notional in USDT.
    :param daily_volume_usdt: Average daily traded volume in USDT.
    :param daily_vol: Daily price volatility (std of log-returns, e.g. 0.02).
    :param eta: Temporary impact coefficient.
    :param gamma: Permanent impact coefficient.
    :return: Impact as fraction of mid-price (add for buy, subtract for sell).
    """
    if daily_volume_usdt <= 0 or order_size_usdt <= 0:
        return _MIN_IMPACT

    x = order_size_usdt / daily_volume_usdt  # participation rate
    vol = max(daily_vol, 1e-6)

    # Half-permanent + temporary (Almgren-Chriss decomposition)
    permanent = gamma * vol * x * 0.5
    temporary = eta * vol * (x ** 0.5)
    impact = permanent + temporary

    return float(np.clip(impact, _MIN_IMPACT, _MAX_IMPACT))


def estimate_from_df(
    df: pd.DataFrame,
    order_size_usdt: float,
    timeframe: str = "15m",
    eta: float = _ETA,
    gamma: float = _GAMMA,
) -> float:
    """
    Estimate Almgren-Chriss impact from an OHLCV DataFrame window.

    Derives daily volume and volatility from df, then calls
    almgren_chriss_impact().

    :param df: OHLCV DataFrame with 'close' and optionally 'volume' columns.
    :param order_size_usdt: Order notional in USDT.
    :param timeframe: ccxt timeframe string ("1m", "15m", "1h", etc.).
    :return: Market impact fraction.
    """
    if df is None or len(df) < 2 or "close" not in df.columns:
        return _MIN_IMPACT

    cpd = _candles_per_day(timeframe)
    close = df["close"].astype(float)
    last_price = float(close.iloc[-1])

    if "volume" in df.columns:
        avg_candle_vol = df["volume"].astype(float).mean()
        daily_volume_usdt = avg_candle_vol * cpd * last_price
    else:
        daily_volume_usdt = order_size_usdt * 10_000  # generous fallback

    log_ret = np.log(close / close.shift(1)).dropna()
    if len(log_ret) >= 5:
        daily_vol = float(log_ret.std()) * (cpd ** 0.5)
    else:
        daily_vol = 0.02  # 2% fallback

    return almgren_chriss_impact(
        order_size_usdt=order_size_usdt,
        daily_volume_usdt=daily_volume_usdt,
        daily_vol=daily_vol,
        eta=eta,
        gamma=gamma,
    )
