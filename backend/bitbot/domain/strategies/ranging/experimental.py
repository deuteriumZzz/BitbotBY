"""Экспериментальные стратегии (ranging): порты из Argent-Systems, классами.

Преимущество не доказано ни у одной: по умолчанию выключены (`experimental = True`
у `PortedStrategy`). Как вес позиции Argent превращается в сигнал входа —
`ported_base.py`. Индикаторные порты этого семейства — в indicators.py.
"""

import numpy as np
import pandas as pd

from bitbot.domain.strategies.ported_base import (
    PortedStrategy,
    _hold_state,
    _rsi,
    _true_range,
)

# ── 5. Возврат к VWAP ────────────────────────────────────────────────────────


class VWAPReversionStrategy(PortedStrategy):
    """Отклонение от скользящего VWAP + RSI. По koopatroopa787/hermes-trading (MIT).

    Из Argent: reversion_vwap.
    """

    name = "vwap_reversion"
    description = (
        "Цена ушла от VWAP больше чем на 1% и RSI в крайней зоне: ставка на возврат "
        "к «справедливой цене»."
    )
    recommended_timeframes = ["15m", "1h"]
    risk_level = "medium"
    market_type = "ranging"
    min_bars = 40

    def __init__(
        self,
        period: int = 20,
        dev: float = 0.01,
        rsi_period: int = 14,
        oversold: float = 35,
        overbought: float = 65,
    ) -> None:
        super().__init__()
        self.period, self.dev, self.rsi_n = period, dev, rsi_period
        self.os, self.ob = oversold, overbought

    def weights(self, df: pd.DataFrame) -> pd.Series:
        c, v = df["close"], df["volume"]
        typical = (df["high"] + df["low"] + c) / 3
        vwap = (typical * v).rolling(self.period).sum() / v.rolling(
            self.period
        ).sum().replace(0, np.nan)
        dev = (c - vwap) / vwap.replace(0, np.nan)
        rsi = _rsi(c, self.rsi_n)
        return _hold_state(
            (dev < -self.dev) & (rsi < self.os),
            (dev > self.dev) & (rsi > self.ob),
            flat=(c - vwap).abs() / vwap.replace(0, np.nan) < self.dev * 0.25,
        )


class StochRSIStrategy(PortedStrategy):
    """Стохастик по RSI: вход из зон перепроданности/перекупленности. Из Argent:
    reversion_stoch_rsi.
    """

    name = "stoch_rsi"
    description = (
        "StochRSI: покупка при перепроданности (<20), продажа при "
        "перекупленности (>80)."
    )
    recommended_timeframes = ["15m", "1h"]
    risk_level = "medium"
    market_type = "ranging"
    min_bars = 60

    def __init__(
        self,
        rsi_period: int = 14,
        stoch_period: int = 14,
        oversold: float = 20,
        overbought: float = 80,
    ) -> None:
        super().__init__()
        self.rsi_n, self.st_n = rsi_period, stoch_period
        self.os, self.ob = oversold, overbought

    def weights(self, df: pd.DataFrame) -> pd.Series:
        rsi = _rsi(df["close"], self.rsi_n)
        lo, hi = rsi.rolling(self.st_n).min(), rsi.rolling(self.st_n).max()
        st = 100 * (rsi - lo) / (hi - lo).replace(0, np.nan)
        return _hold_state(st < self.os, st > self.ob)


class WilliamsRStrategy(PortedStrategy):
    """Williams %R: возврат из крайних зон. Из Argent: reversion_williams_r."""

    name = "williams_r"
    description = (
        "Williams %R: покупка при глубокой перепроданности (<−80), продажа при "
        "перекупленности (>−20)."
    )
    recommended_timeframes = ["15m", "1h"]
    risk_level = "medium"
    market_type = "ranging"
    min_bars = 30

    def __init__(
        self, period: int = 14, oversold: float = -80, overbought: float = -20
    ) -> None:
        super().__init__()
        self.period, self.os, self.ob = period, oversold, overbought

    def weights(self, df: pd.DataFrame) -> pd.Series:
        hh = df["high"].rolling(self.period).max()
        ll = df["low"].rolling(self.period).min()
        pr = -100 * (hh - df["close"]) / (hh - ll).replace(0, np.nan)
        return _hold_state(pr < self.os, pr > self.ob)


class KeltnerFadeStrategy(PortedStrategy):
    """Против касания границы канала Кельтнера, цель — средняя. Из Argent:
    reversion_keltner_fade.
    """

    name = "keltner_fade"
    description = (
        "Канал Кельтнера: цена вышла за границу (EMA ± 2 ATR) — ставка на "
        "возврат к средней."
    )
    recommended_timeframes = ["15m", "1h"]
    risk_level = "medium"
    market_type = "ranging"
    min_bars = 40

    def __init__(self, period: int = 20, atr_period: int = 10, mult: float = 2.0):
        super().__init__()
        self.period, self.atr_n, self.mult = period, atr_period, mult

    def weights(self, df: pd.DataFrame) -> pd.Series:
        c = df["close"]
        mid = c.ewm(span=self.period, adjust=False).mean()
        atr = _true_range(df).ewm(alpha=1 / self.atr_n, adjust=False).mean()
        return _hold_state(
            c < mid - self.mult * atr,
            c > mid + self.mult * atr,
            flat=(c - mid).abs() < atr * 0.1,
        )
