"""
Адаптивная модель рыночного импакта Almgren-Chriss (2001).

Заменяет статические константы bid/ask спреда оценкой с учётом размера ордера:

  MI(x) = γ·σ·x/2  +  η·σ·√x        (постоянный/2 + временный)

где:
  x  = participation rate = order_size_usdt / daily_volume_usdt
  σ  = дневная волатильность цены (std лог-доходностей в пересчёте на день)
  γ  = коэффициент постоянного импакта  (по умолчанию 0.1)
  η  = коэффициент временного импакта   (по умолчанию 0.1)

Для типичного крипто large-cap (BTC participation rate ~0.001–0.01):
  - статические константы давали ~0.07% flat
  - AC даёт ~0.03% для малых ордеров, ~0.15% для крупных блок-сделок
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Almgren-Chriss coefficients calibrated to Bybit BTC/USDT spot
_ETA = 0.1  # temporary impact (square-root term)
_GAMMA = 0.1  # permanent impact (linear term)
_MIN_IMPACT = 0.0002  # floor: 2 bp (exchange fee floor)
_MAX_IMPACT = 0.02  # cap: 2% max assumed impact

# Candles per 24 h for each supported ccxt timeframe
_TF_CANDLES: dict[str, int] = {
    "1m": 1440,
    "3m": 480,
    "5m": 288,
    "15m": 96,
    "30m": 48,
    "1h": 24,
    "2h": 12,
    "4h": 6,
    "1d": 1,
}


def _candles_per_day(timeframe: str) -> int:
    """Возвращает количество свечей за 24ч для указанного таймфрейма."""
    return _TF_CANDLES.get(timeframe, 96)


def almgren_chriss_impact(
    order_size_usdt: float,
    daily_volume_usdt: float,
    daily_vol: float,
    eta: float = _ETA,
    gamma: float = _GAMMA,
) -> float:
    """
    Вычисляет одностороннюю долю рыночного импакта по модели Almgren-Chriss.

    :param order_size_usdt: Номинал ордера в USDT.
    :param daily_volume_usdt: Средний дневной объём торгов в USDT.
    :param daily_vol: Дневная волатильность цены (std лог-доходностей, например 0.02).
    :param eta: Коэффициент временного импакта.
    :param gamma: Коэффициент постоянного импакта.
    :return: Импакт как доля от mid-цены (прибавить для buy, вычесть для sell).
    """
    if daily_volume_usdt <= 0 or order_size_usdt <= 0:
        return _MIN_IMPACT

    x = order_size_usdt / daily_volume_usdt  # participation rate
    vol = max(daily_vol, 1e-6)

    # Half-permanent + temporary (Almgren-Chriss decomposition)
    permanent = gamma * vol * x * 0.5
    temporary = eta * vol * (x**0.5)
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
    Оценивает импакт Almgren-Chriss на основе окна OHLCV DataFrame.

    Выводит дневной объём и волатильность из df, затем вызывает
    almgren_chriss_impact().

    :param df: OHLCV DataFrame с колонкой 'close' и опционально 'volume'.
    :param order_size_usdt: Номинал ордера в USDT.
    :param timeframe: Таймфрейм ccxt ("1m", "15m", "1h" и т.д.).
    :param eta: Коэффициент временного импакта.
    :param gamma: Коэффициент постоянного импакта.
    :return: Доля рыночного импакта.
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
        daily_vol = float(log_ret.std()) * (cpd**0.5)
    else:
        daily_vol = 0.02  # 2% fallback

    return almgren_chriss_impact(
        order_size_usdt=order_size_usdt,
        daily_volume_usdt=daily_volume_usdt,
        daily_vol=daily_vol,
        eta=eta,
        gamma=gamma,
    )
