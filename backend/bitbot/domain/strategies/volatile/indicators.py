"""Экспериментальные индикаторные стратегии (volatile): порты из Argent-Systems.

Каждая — функция веса `*_signals(df, allow_short)` из Argent без изменения логики и
её объявление через `indicator_strategy` рядом. Преимущество не доказано: по
умолчанию выключены. Совпадение с оригиналом проверяет
tests/strategies/test_strategies_ported.py.
"""

import numpy as np
import pandas as pd

from bitbot.domain.strategies.ported_base import (
    indicator_strategy,
)


# ── community_waddah_attar_explosion ──
def _waddah_attar_signals(
    df: pd.DataFrame,
    macd_fast: int = 20,
    macd_slow: int = 40,
    sensitivity: float = 150.0,
    bb_period: int = 20,
    bb_std: float = 2.0,
    dz_period: int = 100,
    dz_mult: float = 3.7,
    allow_short: bool = False,
    **_: object,
) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]

    macd = (
        close.ewm(span=macd_fast, adjust=False).mean()
        - close.ewm(span=macd_slow, adjust=False).mean()
    )
    trend = (macd - macd.shift(1)) * sensitivity

    bb_mid = close.rolling(bb_period).mean()
    bb_stdev = close.rolling(bb_period).std()
    explosion_line = 2 * bb_std * bb_stdev
    _ = bb_mid  # midline unused directly, kept for readability of the BB construction

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    dead_zone = tr.rolling(dz_period).mean() * dz_mult

    bullish_strong = (trend > 0) & (trend > explosion_line) & (trend > dead_zone)
    bearish_strong = (
        (trend < 0) & (trend.abs() > explosion_line) & (trend.abs() > dead_zone)
    )
    weak = ~bullish_strong & ~bearish_strong

    weight = pd.Series(np.nan, index=df.index)
    weight[weak] = 0.0
    weight[bearish_strong] = -1.0 if allow_short else 0.0
    weight[bullish_strong] = 1.0
    return weight.ffill().fillna(0.0)


WaddahAttarStrategy = indicator_strategy(
    "waddah_attar",
    _waddah_attar_signals,
    description="Waddah Attar Explosion: вход на сильном импульсе выше «мёртвой зоны».",
    min_bars=140,
    market_type="volatile",
    markets="both",
    risk="high",
)
