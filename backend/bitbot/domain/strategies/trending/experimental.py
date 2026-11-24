"""Экспериментальные стратегии (trending): порты из Argent-Systems, классами.

Преимущество не доказано ни у одной: по умолчанию выключены (`experimental = True`
у `PortedStrategy`). Как вес позиции Argent превращается в сигнал входа —
`ported_base.py`. Индикаторные порты этого семейства — в indicators.py.
"""

import numpy as np
import pandas as pd

from bitbot.domain.strategies.ported_base import (
    PortedStrategy,
    _hold_state,
    _true_range,
)

# ── 1. SuperTrend ────────────────────────────────────────────────────────────


class SuperTrendStrategy(PortedStrategy):
    """Полоса по ATR вокруг (high+low)/2, которая переворачивается при пробое.

    Вход на перевороте: вверх — покупка, вниз — продажа. Из Argent: trend_supertrend.
    """

    name = "supertrend"
    description = (
        "Разворот полосы ATR (SuperTrend): вход, когда цена закрылась за полосой "
        "и тренд сменил направление."
    )
    recommended_timeframes = ["15m", "1h"]
    risk_level = "medium"
    market_type = "trending"
    min_bars = 30

    def __init__(self, period: int = 10, multiplier: float = 3.0) -> None:
        super().__init__()
        self.period, self.multiplier = period, multiplier

    def weights(self, df: pd.DataFrame) -> pd.Series:
        atr = _true_range(df).ewm(alpha=1 / self.period, adjust=False).mean()
        hl2 = (df["high"] + df["low"]) / 2
        upper = (hl2 + self.multiplier * atr).to_numpy()
        lower = (hl2 - self.multiplier * atr).to_numpy()
        close = df["close"].to_numpy()
        n = len(df)
        fu, fl, d = np.zeros(n), np.zeros(n), np.ones(n)
        fu[0], fl[0] = upper[0], lower[0]
        for i in range(1, n):
            fu[i] = (
                upper[i]
                if upper[i] < fu[i - 1] or close[i - 1] > fu[i - 1]
                else fu[i - 1]
            )
            fl[i] = (
                lower[i]
                if lower[i] > fl[i - 1] or close[i - 1] < fl[i - 1]
                else fl[i - 1]
            )
            if d[i - 1] == 1:
                d[i] = -1 if close[i] < fl[i] else 1
            else:
                d[i] = 1 if close[i] > fu[i] else -1
        return pd.Series(d, index=df.index)


# ── 4. Рождение тренда по ADX ───────────────────────────────────────────────


class ADXTrendBirthStrategy(PortedStrategy):
    """ADX пересекает порог при перевесе одной из сторон DI и цене по ту же сторону EMA.

    Идея из NostalgiaForInfinity (GPL-3.0); код здесь написан заново по описанию.
    Из Argent: nfi_adx_trend_birth.
    """

    name = "adx_trend_birth"
    description = (
        "ADX пересёк порог 20, одна сторона DI сильнее, цена по ту же сторону EMA200: "
        "ловит начало тренда."
    )
    recommended_timeframes = ["15m", "1h"]
    risk_level = "medium"
    market_type = "trending"
    min_bars = 220

    def __init__(
        self, adx_period: int = 14, adx_threshold: float = 20.0, ema_period: int = 200
    ) -> None:
        super().__init__()
        self.n, self.thr, self.ema_n = adx_period, adx_threshold, ema_period

    def weights(self, df: pd.DataFrame) -> pd.Series:
        h, lo, c = df["high"], df["low"], df["close"]
        up, dn = h.diff(), -lo.diff()
        plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
        minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
        a = 1 / self.n
        atr = _true_range(df).ewm(alpha=a, adjust=False).mean().replace(0, np.nan)
        pdi = (100 * plus_dm.ewm(alpha=a, adjust=False).mean() / atr).fillna(0.0)
        mdi = (100 * minus_dm.ewm(alpha=a, adjust=False).mean() / atr).fillna(0.0)
        dx = (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan) * 100
        adx = dx.ewm(alpha=a, adjust=False).mean().fillna(0.0)
        ema = c.ewm(span=self.ema_n, adjust=False).mean()
        cross = (adx > self.thr) & (adx.shift(1) <= self.thr)
        bulls, uptrend = pdi > mdi, c > ema
        return _hold_state(
            cross & bulls & uptrend,
            cross & ~bulls & ~uptrend,
            flat=(~bulls) | (~uptrend),
        )


# ── 7. Шорт в медвежьем тренде ───────────────────────────────────────────────


class BearHedgeStrategy(PortedStrategy):
    """Шорт, когда цена ниже длинной средней и волатильность выше обычной.

    Собственная идея автора Argent, не проверена. Из Argent: hedge_trend_vol_bear.
    Даёт только сигналы на продажу, поэтому только для фьючерсов.
    """

    name = "bear_hedge"
    description = (
        "Медвежий режим: цена ниже SMA200 и волатильность заметно выше нормы. "
        "Только шорт."
    )
    recommended_timeframes = ["15m", "1h"]
    risk_level = "high"
    market_type = "trending"
    markets = "futures"
    min_bars = 320

    def weights(self, df: pd.DataFrame) -> pd.Series:
        c = df["close"]
        down = c < c.rolling(200).mean()
        vol = c.pct_change().rolling(14).std()
        z = (vol - vol.rolling(90).mean()) / vol.rolling(90).std().replace(0, np.nan)
        return -((down & (z >= 1.0)).astype(float).fillna(0.0))


# ── Вторая партия: индикаторные варианты из Argent ───────────────────────────
# Те же правила, что у первой: вес как в Argent (с шортом, allow_short=True),
# сигнал только в момент входа. Для спота шорт-сигналы фильтрует сам бот.


class IchimokuStrategy(PortedStrategy):
    """Цена выше облака Ишимоку — лонг, ниже — шорт, внутри облака — вне рынка.

    Из Argent: trend_ichimoku (классика Хосоды, без лицензии).
    """

    name = "ichimoku"
    description = "Ишимоку: вход, когда цена выходит за облако (Кумо) вверх или вниз."
    recommended_timeframes = ["15m", "1h"]
    risk_level = "medium"
    market_type = "trending"
    min_bars = 90

    def weights(self, df: pd.DataFrame) -> pd.Series:
        h, lo, c = df["high"], df["low"], df["close"]

        def mid(n: int) -> pd.Series:
            return (h.rolling(n).max() + lo.rolling(n).min()) / 2

        span_a = ((mid(9) + mid(26)) / 2).shift(26)
        span_b = mid(52).shift(26)
        top = pd.concat([span_a, span_b], axis=1).max(axis=1)
        bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)
        w = pd.Series(0.0, index=df.index)
        w[c > top] = 1.0
        w[c < bottom] = -1.0
        return w


class ParabolicSARStrategy(PortedStrategy):
    """Parabolic SAR Уайлдера: вход на развороте точки SAR. Из Argent:
    trend_parabolic_sar.
    """

    name = "parabolic_sar"
    description = (
        "Parabolic SAR: вход, когда цена пересекла точку SAR и тренд развернулся."
    )
    recommended_timeframes = ["15m", "1h"]
    risk_level = "medium"
    market_type = "trending"
    min_bars = 30

    def __init__(
        self, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2
    ) -> None:
        super().__init__()
        self.af0, self.step, self.af_max = af_start, af_step, af_max

    def weights(self, df: pd.DataFrame) -> pd.Series:
        high, low = df["high"].to_numpy(), df["low"].to_numpy()
        n = len(df)
        trend, sar = np.ones(n), np.zeros(n)
        ep, af = high[0], self.af0
        sar[0] = low[0]
        for i in range(1, n):
            lo2 = low[i - 2] if i >= 2 else low[i - 1]
            hi2 = high[i - 2] if i >= 2 else high[i - 1]
            cand = sar[i - 1] + af * (ep - sar[i - 1])
            if trend[i - 1] == 1:
                cand = min(cand, low[i - 1], lo2)
                if low[i] < cand:
                    trend[i], sar[i], ep, af = -1, ep, low[i], self.af0
                else:
                    trend[i], sar[i] = 1, cand
                    if high[i] > ep:
                        ep, af = high[i], min(af + self.step, self.af_max)
            else:
                cand = max(cand, high[i - 1], hi2)
                if high[i] > cand:
                    trend[i], sar[i], ep, af = 1, ep, high[i], self.af0
                else:
                    trend[i], sar[i] = -1, cand
                    if low[i] < ep:
                        ep, af = low[i], min(af + self.step, self.af_max)
        return pd.Series(trend, index=df.index)


class VortexStrategy(PortedStrategy):
    """Vortex: VI+ выше VI− — лонг, иначе шорт. Из Argent: trend_vortex."""

    name = "vortex"
    description = (
        "Vortex: вход, когда восходящее вихревое движение обгоняет нисходящее "
        "(или наоборот)."
    )
    recommended_timeframes = ["15m", "1h"]
    risk_level = "medium"
    market_type = "trending"
    min_bars = 30

    def __init__(self, period: int = 14) -> None:
        super().__init__()
        self.period = period

    def weights(self, df: pd.DataFrame) -> pd.Series:
        h, lo = df["high"], df["low"]
        tr_sum = _true_range(df).rolling(self.period).sum().replace(0, np.nan)
        vi_p = (h - lo.shift(1)).abs().rolling(self.period).sum() / tr_sum
        vi_m = (lo - h.shift(1)).abs().rolling(self.period).sum() / tr_sum
        return pd.Series(np.where(vi_p > vi_m, 1.0, -1.0), index=df.index).where(
            vi_p.notna() & vi_m.notna(), 0.0
        )


class OBVTrendStrategy(PortedStrategy):
    """OBV выше/ниже своей средней: объём подтверждает движение. Из Argent:
    volume_obv_trend.
    """

    name = "obv_trend"
    description = (
        "OBV: вход, когда накопленный объём пересекает свою среднюю (поток "
        "объёма подтверждает тренд)."
    )
    recommended_timeframes = ["15m", "1h"]
    risk_level = "medium"
    market_type = "trending"
    min_bars = 40

    def __init__(self, ma_period: int = 20) -> None:
        super().__init__()
        self.ma_n = ma_period

    def weights(self, df: pd.DataFrame) -> pd.Series:
        obv = (np.sign(df["close"].diff()).fillna(0.0) * df["volume"]).cumsum()
        ma = obv.rolling(self.ma_n).mean()
        return pd.Series(np.where(obv > ma, 1.0, -1.0), index=df.index).where(
            ma.notna(), 0.0
        )


class MFITrendStrategy(PortedStrategy):
    """MFI («RSI с объёмом») с гистерезисом 45/55. Из Argent:
    volume_mfi_trend_filter.
    """

    name = "mfi_trend"
    description = (
        "MFI: вход, когда денежный поток устойчиво выше 55 (покупатели) или "
        "ниже 45 (продавцы)."
    )
    recommended_timeframes = ["15m", "1h"]
    risk_level = "medium"
    market_type = "trending"
    min_bars = 30

    def __init__(
        self, period: int = 14, bull_level: float = 55.0, bear_level: float = 45.0
    ) -> None:
        super().__init__()
        self.period, self.bull, self.bear = period, bull_level, bear_level

    def weights(self, df: pd.DataFrame) -> pd.Series:
        typical = (df["high"] + df["low"] + df["close"]) / 3
        flow = typical * df["volume"]
        rising = typical > typical.shift(1)
        pos = flow.where(rising, 0.0).rolling(self.period).sum()
        neg = flow.where(~rising, 0.0).rolling(self.period).sum()
        mfi = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
        return _hold_state(mfi > self.bull, mfi < self.bear)
