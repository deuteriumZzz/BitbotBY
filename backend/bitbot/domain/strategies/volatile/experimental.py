"""Экспериментальные стратегии (volatile): порты из Argent-Systems, классами.

Преимущество не доказано ни у одной: по умолчанию выключены (`experimental = True`
у `PortedStrategy`). Как вес позиции Argent превращается в сигнал входа —
`ported_base.py`. Индикаторные порты этого семейства — в indicators.py.
"""

import numpy as np
import pandas as pd

from bitbot.domain.strategies.ported_base import (
    PortedStrategy,
    _rsi,
    _true_range,
)

# ── 2. Dual Thrust ───────────────────────────────────────────────────────────


class DualThrustStrategy(PortedStrategy):
    """Пробой диапазона Dual Thrust (Michael Chalek), по jesse-ai (MIT).

    Диапазон считается по предыдущим N барам; вход при закрытии за open ± k·range.
    Из Argent: niche_dual_thrust.
    """

    name = "dual_thrust"
    description = (
        "Dual Thrust: вход при закрытии за границей open ± k·диапазон, где диапазон "
        "считается по последним 20 барам."
    )
    recommended_timeframes = ["15m", "1h"]
    risk_level = "medium"
    market_type = "volatile"
    min_bars = 25

    def __init__(self, period: int = 20, k1: float = 0.5, k2: float = 0.5) -> None:
        super().__init__()
        self.period, self.k1, self.k2 = period, k1, k2

    def weights(self, df: pd.DataFrame) -> pd.Series:
        h, lo, c, o = df["high"], df["low"], df["close"], df["open"]
        hh = h.rolling(self.period).max().shift(1)
        hc = c.rolling(self.period).max().shift(1)
        ll = lo.rolling(self.period).min().shift(1)
        lc = c.rolling(self.period).min().shift(1)
        rng = pd.concat([hh - lc, hc - ll], axis=1).max(axis=1)
        up, down = c > o + self.k1 * rng, c < o - self.k2 * rng
        w = pd.Series(np.nan, index=df.index)
        w[up] = 1.0
        w[down] = -1.0
        return w.ffill().fillna(0.0)


# ── 3. TTM Squeeze ───────────────────────────────────────────────────────────


def _linreg_endpoint(y: pd.Series, length: int) -> pd.Series:
    """Конец скользящей линейной регрессии; то же, что rolling.apply, но быстро."""
    x = np.arange(length, dtype=float)
    xm = x.mean()
    sxx = ((x - xm) ** 2).sum()
    k = 1.0 / length + (x - xm) * (length - 1 - xm) / sxx
    out = np.full(len(y), np.nan)
    if len(y) >= length:
        # окно j: y[j-L+1..j]; свёртка с развёрнутыми весами
        out[length - 1 :] = np.convolve(y.to_numpy(), k[::-1], mode="valid")
    return pd.Series(out, index=y.index)


class TTMSqueezeStrategy(PortedStrategy):
    """TTM Squeeze (John Carter): Боллинджер внутри Кельтнера, затем выход.

    Вход в момент выхода из сжатия в сторону импульса. Из Argent:
    community_ttm_squeeze_carter.
    """

    name = "ttm_squeeze"
    description = (
        "TTM Squeeze: после сжатия волатильности (Боллинджер внутри Кельтнера) "
        "вход по направлению импульса."
    )
    recommended_timeframes = ["15m", "1h"]
    risk_level = "medium"
    market_type = "volatile"
    min_bars = 60

    def __init__(
        self, period: int = 20, bb_std: float = 2.0, kc_atr_mult: float = 1.5
    ) -> None:
        super().__init__()
        self.period, self.bb_std, self.kc_mult = period, bb_std, kc_atr_mult

    def weights(self, df: pd.DataFrame) -> pd.Series:
        p = self.period
        h, lo, c = df["high"], df["low"], df["close"]
        mid = c.rolling(p).mean()
        sd = c.rolling(p).std()
        atr = _true_range(df).rolling(p).mean()
        on = ((mid - self.bb_std * sd) > (mid - self.kc_mult * atr)) & (
            (mid + self.bb_std * sd) < (mid + self.kc_mult * atr)
        )
        fired = (~on) & on.shift(1, fill_value=False)
        don_mid = (h.rolling(p).max() + lo.rolling(p).min()) / 2
        momentum = _linreg_endpoint(c - (don_mid + mid) / 2, p)
        flip = (momentum * momentum.shift(1)) < 0
        w = pd.Series(np.nan, index=df.index)
        w[flip] = 0.0
        w[fired & (momentum < 0)] = -1.0
        w[fired & (momentum > 0)] = 1.0
        return w.ffill().fillna(0.0)


# ── 6. Отскок после капитуляции ──────────────────────────────────────────────


class CrashBounceStrategy(PortedStrategy):
    """Покупка после сильного обвала, но не свободного падения (только лонг).

    Идея из NostalgiaForInfinity (GPL-3.0); код написан заново по описанию.
    Из Argent: nfi_crash_bounce_capitulation. Высокий риск: «ловля ножа».
    """

    name = "crash_bounce"
    description = (
        "Отскок после капитуляции: обвал 15–30%, перепроданность, всплеск объёма и "
        "первая зелёная свеча-возврат. Только покупка."
    )
    recommended_timeframes = ["15m", "1h"]
    risk_level = "high"
    market_type = "volatile"
    min_bars = 160

    def weights(self, df: pd.DataFrame) -> pd.Series:
        o, h, c, v = df["open"], df["high"], df["close"], df["volume"]
        roc = c.pct_change(96) * 100
        hard_crash = (roc < -15.0) & (roc > -30.0)
        oversold = _rsi(c, 3) < 15.0
        fresh = c.pct_change(6) * 100 < 0
        spike = v / v.rolling(96).mean().replace(0, np.nan) > 1.3
        near_low = c < c.rolling(48).min() * 1.08
        reclaim = (c > o) & (c > h.shift(1))
        entry = hard_crash & oversold & fresh & spike & near_low & reclaim
        w = pd.Series(np.nan, index=df.index)
        w[c > c.rolling(48).mean()] = 0.0
        w[entry] = 1.0
        return w.ffill().fillna(0.0)
