"""Стратегии, перенесённые из Argent-Systems (github.com/deuteriumZzz/Argent-Systems).

В Argent каждая стратегия — функция signals(df), которая возвращает целевой вес
позиции на каждый бар (+1 лонг, 0 вне рынка, −1 шорт). Нашему боту нужен сигнал на
последнем баре: «открыть сейчас или нет». Поэтому здесь:

* вес считается так же, как в Argent (все операции причинные: вес бара зависит
  только от этого и прошлых баров);
* сигнал buy/sell выдаётся только в момент ВХОДА — когда вес на последнем баре
  изменился и не равен нулю. Пока позиция «держится», стратегия молчит;
  выходом занимаются стоп, тейк и защиты бота, а не вес;
* уверенность у всех портов одинаковая: у них нет честной самооценки, а в
  турнире локального отбора побеждает самая уверенная — завышать её нельзя.

Ни у одной из этих стратегий нет доказанного преимущества (по собственной проверке
Argent ни одна ценовая стратегия не значима по Deflated Sharpe). Это материал для
бэктеста и разнообразия, а не обещание прибыли.
"""

from typing import Any, Dict, Optional, Tuple, Type

import numpy as np
import pandas as pd

from .strategies_base import BaseStrategy

# Сколько последних свечей считаем вживую: хватает для самых длинных окон
# (200 + 14 + 90) с запасом на «прогрев» рекурсивных индикаторов.
_LIVE_TAIL = 1000
_CONFIDENCE = 0.70


class PortedStrategy(BaseStrategy):
    """База: вес позиции по Argent → сигнал входа на последнем баре."""

    confidence = _CONFIDENCE
    min_bars = 60
    experimental = True

    def __init__(self) -> None:
        # Вес, посчитанный заранее на всей истории (только бэктест, см. prepare()).
        self._cache: Optional[Tuple[pd.DataFrame, pd.Series]] = None

    def weights(self, df: pd.DataFrame) -> pd.Series:
        """Целевой вес позиции на каждый бар: +1, 0 или −1."""
        raise NotImplementedError

    def prepare(self, df: pd.DataFrame) -> None:
        """Бэктест: считает вес на всей истории один раз, а не на каждом шаге.

        Допустимо, потому что вес причинный: вес на префиксе истории равен
        префиксу веса на всей истории (это проверяется тестом).
        """
        self._cache = (df, self.weights(df))

    def _weights_for(self, data: pd.DataFrame) -> pd.Series:
        if self._cache is not None:
            full, w = self._cache
            n = len(data)
            if (
                n <= len(full)
                and data.index[0] == full.index[0]
                and data["close"].iloc[-1] == full["close"].iloc[n - 1]
                and data["volume"].iloc[-1] == full["volume"].iloc[n - 1]
            ):
                return w.iloc[:n]
        return self.weights(data.tail(_LIVE_TAIL))

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        if len(data) < self.min_bars:
            return {"action": "hold", "confidence": 0.50}
        w = self._weights_for(data)
        now, before = float(w.iloc[-1]), float(w.iloc[-2])
        if now != before and now != 0.0:
            return {
                "action": "buy" if now > 0 else "sell",
                "confidence": self.confidence,
                "price": float(data["close"].iloc[-1]),
            }
        return {"action": "hold", "confidence": 0.45}


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()],
        axis=1,
    ).max(axis=1)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def _hold_state(
    events_long: pd.Series, events_short: pd.Series, flat: Optional[pd.Series] = None
) -> pd.Series:
    """Вес, который «залипает» между событиями (как ffill в Argent)."""
    w = pd.Series(np.nan, index=events_long.index)
    if flat is not None:
        w[flat] = 0.0
    w[events_short] = -1.0
    w[events_long] = 1.0
    return w.ffill().fillna(0.0)


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


PORTED_STRATEGIES: Dict[str, Type[BaseStrategy]] = {
    cls.name: cls
    for cls in (
        SuperTrendStrategy,
        DualThrustStrategy,
        TTMSqueezeStrategy,
        ADXTrendBirthStrategy,
        VWAPReversionStrategy,
        CrashBounceStrategy,
        BearHedgeStrategy,
        IchimokuStrategy,
        ParabolicSARStrategy,
        VortexStrategy,
        StochRSIStrategy,
        WilliamsRStrategy,
        KeltnerFadeStrategy,
        OBVTrendStrategy,
        MFITrendStrategy,
    )
}
