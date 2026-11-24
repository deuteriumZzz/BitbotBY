"""Экспериментальные стратегии для бокового рынка (флэт): порты из Argent-Systems.

Преимущество не доказано ни у одной: по умолчанию выключены (`experimental = True`
у `PortedStrategy`), это материал для бэктеста и разнообразия. Как вес позиции
Argent превращается в сигнал входа — `ported_base.py`. Индикаторные функции
(`*_signals`) взяты из Argent без изменения логики; совпадение с оригиналом
проверяет tests/strategies/test_strategies_ported.py.
"""

import numpy as np
import pandas as pd

from bitbot.domain.strategies.ported_base import (
    PortedStrategy,
    _hold_state,
    _rsi,
    _true_range,
    indicator_strategy,
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


# ── nfi_cci_aroon_reversal ──
def _cci_aroon_cci(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> pd.Series:
    typical = (high + low + close) / 3
    sma = typical.rolling(period).mean()
    mad = typical.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (typical - sma) / (0.015 * mad.replace(0, np.nan))


def _cci_aroon_aroon_up(high: pd.Series, period: int) -> pd.Series:
    # bars since the highest high in the window, scaled to 0-100
    bars_since_high = high.rolling(period + 1).apply(
        lambda x: period - np.argmax(x[::-1]), raw=True
    )
    return 100 * (period - bars_since_high) / period


def _cci_aroon_aroon_down(low: pd.Series, period: int) -> pd.Series:
    # mirror of _cci_aroon_aroon_up: bars since the lowest low in the window
    bars_since_low = low.rolling(period + 1).apply(
        lambda x: period - np.argmin(x[::-1]), raw=True
    )
    return 100 * (period - bars_since_low) / period


def _cci_aroon_signals(
    df: pd.DataFrame,
    cci_period: int = 20,
    cci_ceiling: float = 50.0,
    cci_floor: float = 200.0,
    cci_exit: float = 100.0,
    aroon_period: int = 14,
    aroon_min: float = 50.0,
    trend_ema_period: int = 100,
    trend_slope_lookback: int = 10,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]

    cci = _cci_aroon_cci(high, low, close, cci_period)
    cci_recovering = (cci > cci.shift(1)) & (cci < -cci_ceiling) & (cci > -cci_floor)

    aroon_up = _cci_aroon_aroon_up(high, aroon_period)
    fresh_upswing = aroon_up > aroon_min

    trend_ema = close.ewm(span=trend_ema_period, adjust=False).mean()
    not_falling_knife = trend_ema > trend_ema.shift(trend_slope_lookback)

    long_entry = cci_recovering & fresh_upswing & not_falling_knife
    long_exit = cci > cci_exit

    weight = pd.Series(np.nan, index=df.index)
    # exit set first so a same-bar entry (below) can override the flatten
    weight[long_exit] = 0.0
    if allow_short:
        aroon_down = _cci_aroon_aroon_down(low, aroon_period)
        cci_topping = (cci < cci.shift(1)) & (cci > cci_ceiling) & (cci < cci_floor)
        short_entry = (
            cci_topping
            & (aroon_down > aroon_min)
            & (trend_ema < trend_ema.shift(trend_slope_lookback))
        )
        weight[short_entry] = -1.0
    weight[long_entry] = 1.0
    return weight.ffill().fillna(0.0)


CciAroonStrategy = indicator_strategy(
    "cci_aroon",
    _cci_aroon_signals,
    description="CCI + Aroon: отскок из перепроданности при свежем восходящем движении и растущем тренде.",
    min_bars=130,
    market_type="ranging",
    markets="both",
    risk="medium",
)


# ── niche_fisher_transform ──
def _fisher_transform_fisher(median_price: pd.Series, period: int) -> pd.Series:
    lo = median_price.rolling(period).min()
    hi = median_price.rolling(period).max()
    norm = (2 * (median_price - lo) / (hi - lo).replace(0, np.nan) - 1).clip(
        -0.999, 0.999
    )
    return (0.5 * np.log((1 + norm) / (1 - norm))).fillna(0.0)


def _fisher_transform_signals(
    df: pd.DataFrame,
    fisher_period: int = 12,
    smooth_period: int = 8,
    baseline_period: int = 21,
    **_,
) -> pd.Series:
    median_price = (df["high"] + df["low"]) / 2
    fisher = _fisher_transform_fisher(median_price, fisher_period)
    fisher_smooth = fisher.ewm(span=smooth_period, adjust=False).mean()

    baseline = df["close"].ewm(span=baseline_period, adjust=False).mean()
    baseline_up = baseline.diff() > 0

    entries = (
        (fisher_smooth > fisher_smooth.shift(1)) & (fisher_smooth < 0) & baseline_up
    )
    exits = (fisher_smooth.shift(1) > 0) & (fisher_smooth < 0)

    weight = pd.Series(float("nan"), index=df.index)
    weight[entries] = 1.0
    weight[exits] = 0.0
    return weight.ffill().fillna(0.0)


FisherTransformStrategy = indicator_strategy(
    "fisher_transform",
    _fisher_transform_signals,
    description="Fisher Transform: покупка на развороте вверх из отрицательной зоны при растущем тренде. Только лонг.",
    min_bars=60,
    market_type="ranging",
    markets="both",
    risk="medium",
)


# ── niche_kdj ──
def _kdj_signals(
    df: pd.DataFrame, period: int = 9, allow_short: bool = False, **_
) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]

    llv = low.rolling(period).min()
    hhv = high.rolling(period).max()
    rsv = ((close - llv) / (hhv - llv).replace(0, float("nan")) * 100).fillna(50.0)

    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    j = 3 * k - 2 * d

    bullish = (j > k) & (j > d)
    short_weight = -1.0 if allow_short else 0.0
    return bullish.map({True: 1.0, False: short_weight})


KdjStrategy = indicator_strategy(
    "kdj",
    _kdj_signals,
    description="KDJ: вход, когда линия J уходит за K и D (ускорение стохастика).",
    min_bars=40,
    market_type="ranging",
    markets="both",
    risk="medium",
)


# ── pattern_rsi_divergence ──
def _rsi_divergence_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _rsi_divergence_signals(
    df: pd.DataFrame,
    rsi_period: int = 14,
    pivot_window: int = 3,
    lookback: int = 30,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]
    rsi = _rsi_divergence_rsi(close, rsi_period)

    window = pivot_window * 2 + 1
    is_low = (
        (close == close.rolling(window, center=True).min()).fillna(False).to_numpy()
    )
    is_high = (
        (close == close.rolling(window, center=True).max()).fillna(False).to_numpy()
    )

    close_np = close.to_numpy()
    rsi_np = rsi.to_numpy()
    n = len(df)

    weight = np.zeros(n)
    position = 0.0
    short_weight = -1.0 if allow_short else 0.0
    last_low_idx, last_high_idx = -1, -1

    # `confirm_idx` lags the current bar `t` by `pivot_window` bars - that's
    # the earliest point a centered rolling min/max at that index is fully
    # known, so this stays causal (no look-ahead) despite using a centered
    # window to find the pivots themselves.
    for t in range(window - 1, n):
        confirm_idx = t - pivot_window

        if is_low[confirm_idx]:
            if (
                last_low_idx != -1
                and confirm_idx - last_low_idx <= lookback
                and close_np[confirm_idx] < close_np[last_low_idx]
                and rsi_np[confirm_idx] > rsi_np[last_low_idx]
            ):
                position = 1.0
            last_low_idx = confirm_idx

        if is_high[confirm_idx]:
            if (
                last_high_idx != -1
                and confirm_idx - last_high_idx <= lookback
                and close_np[confirm_idx] > close_np[last_high_idx]
                and rsi_np[confirm_idx] < rsi_np[last_high_idx]
            ):
                position = short_weight
            last_high_idx = confirm_idx

        weight[t] = position

    return pd.Series(weight, index=df.index)


RsiDivergenceStrategy = indicator_strategy(
    "rsi_divergence",
    _rsi_divergence_signals,
    description="Дивергенция RSI: цена обновила экстремум, а RSI — нет; вход после подтверждения пивота.",
    min_bars=80,
    market_type="ranging",
    markets="both",
    risk="medium",
)


# ── pattern_shooting_star ──
def _shooting_star_signals(
    df: pd.DataFrame,
    lower_wick_max: float = 0.2,
    body_size_mult: float = 0.5,
    body_lookback: int = 20,
    stop_pct: float = 0.05,
    holding_period: int = 7,
    allow_short: bool = True,
    **_,
) -> pd.Series:
    open_, high, low, close = df["open"], df["high"], df["low"], df["close"]

    body = (open_ - close).abs()
    avg_body = body.rolling(body_lookback).mean()

    red = open_ >= close
    small_lower_wick = (close - low) < lower_wick_max * (close - open_).abs()
    small_body = body < body_size_mult * avg_body
    long_upper_wick = (high - open_) >= 2 * (open_ - close)
    uptrend = (close >= close.shift(1)) & (close.shift(1) >= close.shift(2))

    trigger = (
        (red & small_lower_wick & small_body & long_upper_wick & uptrend)
        .fillna(False)
        .to_numpy()
    )
    close_np = close.to_numpy()
    n = len(df)

    weight = np.zeros(n)
    short_weight = -1.0 if allow_short else 0.0
    in_trade = False
    entry_price = 0.0
    bars_held = 0

    for i in range(n):
        if in_trade:
            bars_held += 1
            if (
                abs(close_np[i] / entry_price - 1) > stop_pct
                or bars_held >= holding_period
            ):
                in_trade = False
        elif trigger[i]:
            in_trade = True
            entry_price = close_np[i]
            bars_held = 0
        weight[i] = short_weight if in_trade else 0.0

    return pd.Series(weight, index=df.index)


ShootingStarStrategy = indicator_strategy(
    "shooting_star",
    _shooting_star_signals,
    description="Падающая звезда: свеча с длинной верхней тенью после роста — шорт на 7 свечей. Только продажа.",
    min_bars=40,
    market_type="ranging",
    markets="futures",
    risk="high",
)


# ── reversion_cci ──
def _cci_reversion_signals(
    df: pd.DataFrame,
    period: int = 20,
    threshold: float = 100,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    sma = typical.rolling(period).mean()
    mad = typical.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (typical - sma) / (0.015 * mad.replace(0, np.nan))

    long_entries = cci < -threshold
    short_entries = cci > threshold

    weight = pd.Series(np.nan, index=df.index)
    weight[long_entries] = 1.0
    weight[short_entries] = -1.0 if allow_short else 0.0
    return weight.ffill().fillna(0.0)


CciReversionStrategy = indicator_strategy(
    "cci_reversion",
    _cci_reversion_signals,
    description="CCI: покупка ниже −100, продажа выше +100 (возврат к средней).",
    min_bars=40,
    market_type="ranging",
    markets="both",
    risk="medium",
)


# ── reversion_ifr2_trend_filtered ──
def _ifr2_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _ifr2_signals(
    df: pd.DataFrame,
    rsi_period: int = 2,
    oversold: float = 10,
    trend_ema_period: int = 50,
    slope_lookback: int = 5,
    exit_lookback: int = 2,
    **_,
) -> pd.Series:
    close = df["close"]
    rsi = _ifr2_rsi(close, rsi_period)
    trend_ema = close.ewm(span=trend_ema_period, adjust=False).mean()
    uptrend = trend_ema > trend_ema.shift(slope_lookback)

    entries = (rsi < oversold) & uptrend
    exits = close > close.shift(1).rolling(exit_lookback).max()

    weight = pd.Series(np.nan, index=df.index)
    weight[entries] = 1.0
    weight[exits] = 0.0
    return weight.ffill().fillna(0.0)


Ifr2Strategy = indicator_strategy(
    "ifr2",
    _ifr2_signals,
    description="IFR2: RSI(2) ниже 10 при растущем тренде. Только лонг.",
    min_bars=60,
    market_type="ranging",
    markets="both",
    risk="medium",
)


# ── reversion_stochastic ──
def _stochastic_signals(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    oversold: float = 20,
    overbought: float = 80,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    span = (highest_high - lowest_low).replace(0, np.nan)
    percent_k = 100 * (close - lowest_low) / span
    percent_d = percent_k.rolling(d_period).mean()

    long_entries = (percent_k < oversold) & (percent_d < oversold)
    short_entries = (percent_k > overbought) & (percent_d > overbought)

    weight = pd.Series(np.nan, index=df.index)
    weight[long_entries] = 1.0
    weight[short_entries] = -1.0 if allow_short else 0.0
    return weight.ffill().fillna(0.0)


StochasticStrategy = indicator_strategy(
    "stochastic",
    _stochastic_signals,
    description="Стохастик: вход из зон перепроданности (<20) и перекупленности (>80).",
    min_bars=40,
    market_type="ranging",
    markets="both",
    risk="medium",
)


# ── volume_profile_poc_reversion ──
def _poc_reversion_rolling_poc(
    typical: pd.Series, volume: pd.Series, window: int, num_bins: int
) -> pd.Series:
    typ = typical.to_numpy(dtype=float)
    vol = volume.to_numpy(dtype=float)
    n = len(typ)
    poc = np.full(n, np.nan)

    for i in range(window - 1, n):
        seg_typ = typ[i - window + 1 : i + 1]
        seg_vol = vol[i - window + 1 : i + 1]
        lo, hi = seg_typ.min(), seg_typ.max()
        if hi <= lo:
            poc[i] = seg_typ[-1]
            continue
        edges = np.linspace(lo, hi, num_bins + 1)
        idx = np.clip(np.digitize(seg_typ, edges) - 1, 0, num_bins - 1)
        vol_by_bin = np.bincount(idx, weights=seg_vol, minlength=num_bins)
        top = vol_by_bin.argmax()
        poc[i] = (edges[top] + edges[top + 1]) / 2

    return pd.Series(poc, index=typical.index)


def _poc_reversion_signals(
    df: pd.DataFrame,
    profile_window: int = 48,
    num_bins: int = 12,
    dev_pct: float = 0.01,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]
    typical = (df["high"] + df["low"] + close) / 3

    poc = _poc_reversion_rolling_poc(typical, df["volume"], profile_window, num_bins)
    deviation = (close - poc) / poc.replace(0, np.nan)

    weight = pd.Series(np.nan, index=df.index)
    weight[deviation.abs() <= dev_pct * 0.25] = 0.0
    weight[deviation < -dev_pct] = 1.0
    weight[deviation > dev_pct] = -1.0 if allow_short else 0.0
    return weight.ffill().fillna(0.0)


PocReversionStrategy = indicator_strategy(
    "poc_reversion",
    _poc_reversion_signals,
    description="Профиль объёма: отклонение от точки максимального объёма (POC) на 1% — ставка на возврат.",
    min_bars=80,
    market_type="ranging",
    markets="both",
    risk="medium",
)


# ── academic_bollinger_band_filtered_reversion ──
def _bollinger_reentry_signals(
    df: pd.DataFrame,
    period: int = 20,
    num_std: float = 2.0,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std

    reentry_long = (close.shift(1) < lower.shift(1)) & (close >= lower)
    reentry_short = (close.shift(1) > upper.shift(1)) & (close <= upper)
    crossed_up = (close.shift(1) < mid.shift(1)) & (close >= mid)
    crossed_down = (close.shift(1) > mid.shift(1)) & (close <= mid)
    touch_mid = crossed_up | crossed_down

    weight = pd.Series(np.nan, index=df.index)
    weight[touch_mid] = 0.0
    weight[reentry_long] = 1.0
    weight[reentry_short] = -1.0 if allow_short else 0.0
    return weight.ffill().fillna(0.0)


BollingerReentryStrategy = indicator_strategy(
    "bollinger_reentry",
    _bollinger_reentry_signals,
    description="Возврат в полосы Боллинджера: вход, когда цена вернулась внутрь полосы после выхода за неё.",
    min_bars=40,
    market_type="ranging",
    markets="both",
    risk="medium",
)


# ── academic_spike_reversal_proxy ──
def _spike_reversal_signals(
    df: pd.DataFrame,
    volume_z_threshold: float = 2.5,
    return_z_threshold: float = 1.5,
    vol_lookback: int = 48,
    holding_period: int = 6,
    **_,
) -> pd.Series:
    ret = df["close"].pct_change()

    vol_mean = df["volume"].rolling(vol_lookback).mean()
    vol_std = df["volume"].rolling(vol_lookback).std()
    volume_z = (df["volume"] - vol_mean) / vol_std

    ret_mean = ret.rolling(vol_lookback).mean()
    ret_std = ret.rolling(vol_lookback).std()
    return_z = (ret - ret_mean) / ret_std

    event = (volume_z > volume_z_threshold) & (return_z.abs() > return_z_threshold)
    direction = -np.sign(ret)

    event_positions = np.flatnonzero(event.fillna(False).to_numpy())
    w = np.zeros(len(df))
    d = direction.fillna(0.0).to_numpy()
    n = len(w)
    # ponytail: later overlapping events simply overwrite earlier ones' tail;
    # a priority queue of active events would resolve overlaps more precisely
    # if that ever matters for a real backtest.
    for i in event_positions:
        end = min(i + holding_period, n)
        w[i:end] = d[i]

    return pd.Series(w, index=df.index)


SpikeReversalStrategy = indicator_strategy(
    "spike_reversal",
    _spike_reversal_signals,
    description="Разворот после всплеска объёма и цены: ставка против движения на 6 свечей.",
    min_bars=80,
    market_type="ranging",
    markets="both",
    risk="high",
)
