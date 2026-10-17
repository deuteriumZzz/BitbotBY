# flake8: noqa
# mypy: ignore-errors
"""Вторая партия портов из Argent-Systems: индикаторные стратегии (34 штуки).

Функции расчёта веса взяты из Argent без изменений логики (это ваш же репозиторий,
у всех источник — классика или открытая формула): так вес гарантированно совпадает с
оригиналом. Здесь они обёрнуты в тот же адаптер `PortedStrategy`, что и первая партия:
сигнал buy/sell выдаётся только в момент входа, выходом занимаются стоп, тейк и защиты
бота, уверенность у всех 0.70.

Шорт-сигналы включены (`allow_short=True`); на споте бот их не исполняет. Исключение —
`shooting_star`: она торгует только вниз, поэтому помечена «только фьючерсы». Не взяты
`volume_inventory_skew_adaptive` (дробный вес позиции, а не вход/выход) и
`pattern_london_breakout` (зависит от часов сессии в индексе свечей).

Преимущества не доказано ни у одной: это материал для бэктеста и разнообразия.
Файл исключён из flake8 и mypy осознанно: код взят как есть, чтобы не терять совпадение
с оригиналом; проверка совпадения — tests/test_strategies_ported.py.
"""

from typing import Any, Dict, Type

import numpy as np
import pandas as pd

from .strategies_base import BaseStrategy
from .strategies_ported import PortedStrategy


# ── community_chaikin_oscillator ──
def _chaikin_osc_signals(
    df: pd.DataFrame,
    fast_period: int = 3,
    slow_period: int = 10,
    entry_threshold: float = 0.0,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]

    rng = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / rng
    adl = (mfm * volume).fillna(0.0).cumsum()

    oscillator = (
        adl.ewm(span=fast_period, adjust=False).mean()
        - adl.ewm(span=slow_period, adjust=False).mean()
    )

    weight = pd.Series(np.nan, index=df.index)
    weight[oscillator > entry_threshold] = 1.0
    weight[oscillator < -entry_threshold] = -1.0 if allow_short else 0.0
    return weight.ffill().fillna(0.0)


# ── community_coppock_curve ──
def _coppock_wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def _coppock_signals(
    df: pd.DataFrame,
    roc_long: int = 14,
    roc_short: int = 11,
    wma_period: int = 10,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]

    roc_l = (close / close.shift(roc_long) - 1) * 100
    roc_s = (close / close.shift(roc_short) - 1) * 100
    coppock = _coppock_wma(roc_l + roc_s, wma_period)

    cross_up = (coppock > 0) & (coppock.shift(1) <= 0)
    cross_down = (coppock < 0) & (coppock.shift(1) >= 0)

    weight = pd.Series(np.nan, index=df.index)
    weight[cross_down] = -1.0 if allow_short else 0.0
    weight[cross_up] = 1.0
    return weight.ffill().fillna(0.0)


# ── community_coral_trend ──
def _coral_trend_signals(
    df: pd.DataFrame,
    period: int = 21,
    cd: float = 0.4,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]

    di = (period - 1) / 2 + 1
    alpha = 2 / (di + 1)

    i1 = close.ewm(alpha=alpha, adjust=False).mean()
    i2 = i1.ewm(alpha=alpha, adjust=False).mean()
    i3 = i2.ewm(alpha=alpha, adjust=False).mean()
    i4 = i3.ewm(alpha=alpha, adjust=False).mean()
    i5 = i4.ewm(alpha=alpha, adjust=False).mean()
    i6 = i5.ewm(alpha=alpha, adjust=False).mean()

    c3 = 3 * (cd**2 + cd**3)
    c4 = -3 * (2 * cd**2 + cd + cd**3)
    c5 = 3 * cd + 1 + cd**3 + 3 * cd**2

    coral = -(cd**3) * i6 + c3 * i5 + c4 * i4 + c5 * i3

    bullish = coral > coral.shift(1)
    short_weight = -1.0 if allow_short else 0.0
    return bullish.map({True: 1.0, False: short_weight})


# ── community_kst_oscillator ──
def _kst_roc(close: pd.Series, period: int) -> pd.Series:
    return (close / close.shift(period) - 1) * 100


def _kst_signals(
    df: pd.DataFrame,
    roc1: int = 10,
    sma1: int = 10,
    roc2: int = 15,
    sma2: int = 10,
    roc3: int = 20,
    sma3: int = 10,
    roc4: int = 30,
    sma4: int = 15,
    signal_period: int = 9,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]

    rcma1 = _kst_roc(close, roc1).rolling(sma1).mean()
    rcma2 = _kst_roc(close, roc2).rolling(sma2).mean()
    rcma3 = _kst_roc(close, roc3).rolling(sma3).mean()
    rcma4 = _kst_roc(close, roc4).rolling(sma4).mean()

    kst = rcma1 + 2 * rcma2 + 3 * rcma3 + 4 * rcma4
    signal_line = kst.rolling(signal_period).mean()

    bullish = kst > signal_line
    short_weight = -1.0 if allow_short else 0.0
    return bullish.map({True: 1.0, False: short_weight})


# ── community_rvi_oscillator ──
def _rvi_symmetric_filter(series: pd.Series) -> pd.Series:
    return (series + 2 * series.shift(1) + 2 * series.shift(2) + series.shift(3)) / 6


def _rvi_signals(
    df: pd.DataFrame,
    period: int = 10,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    open_, high, low, close = df["open"], df["high"], df["low"], df["close"]

    num = _rvi_symmetric_filter(close - open_).rolling(period).sum()
    denom = _rvi_symmetric_filter(high - low).rolling(period).sum()
    rvi = num / denom.replace(0, np.nan)
    signal_line = _rvi_symmetric_filter(rvi)

    bullish = rvi > signal_line
    short_weight = -1.0 if allow_short else 0.0
    return bullish.map({True: 1.0, False: short_weight})


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
    **_,
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


# ── community_zlsma_trend ──
def _zlsma_rolling_linreg_endpoint(series: pd.Series, length: int) -> pd.Series:
    x = np.arange(length, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _endpoint(y: np.ndarray) -> float:
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / x_var
        intercept = y_mean - slope * x_mean
        return slope * (length - 1) + intercept

    return series.rolling(length).apply(_endpoint, raw=True)


def _zlsma_signals(
    df: pd.DataFrame,
    length: int = 32,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]

    lsma = _zlsma_rolling_linreg_endpoint(close, length)
    lsma2 = _zlsma_rolling_linreg_endpoint(lsma, length)
    zlsma = lsma + (lsma - lsma2)

    zlsma_rising = zlsma > zlsma.shift(1)
    above = (close > zlsma) & zlsma_rising
    below = (close < zlsma) & ~zlsma_rising

    weight = pd.Series(np.nan, index=df.index)
    weight[above] = 1.0
    weight[below] = -1.0 if allow_short else 0.0
    return weight.ffill().fillna(0.0)


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


# ── nfi_engulfing_trend_continuation ──
def _engulfing_trend_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _engulfing_trend_signals(
    df: pd.DataFrame,
    trend_ema_period: int = 200,
    rsi_period: int = 14,
    rsi_min: float = 50.0,
    fast_rsi_period: int = 3,
    fast_rsi_min: float = 55.0,
    roc_period: int = 24,
    roc_ceiling: float = 8.0,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    open_, close = df["open"], df["close"]
    prev_open, prev_close = open_.shift(1), close.shift(1)

    bullish_engulf = (
        (close > open_)
        & (prev_close < prev_open)
        & (close >= prev_open)
        & (open_ <= prev_close)
    )
    bearish_engulf = (
        (close < open_)
        & (prev_close > prev_open)
        & (close <= prev_open)
        & (open_ >= prev_close)
    )

    ema = close.ewm(span=trend_ema_period, adjust=False).mean()
    uptrend = close > ema
    downtrend = close < ema

    rsi = _engulfing_trend_rsi(close, rsi_period)
    fast_rsi = _engulfing_trend_rsi(close, fast_rsi_period)
    roc = close.pct_change(roc_period) * 100

    long_entry = (
        bullish_engulf
        & uptrend
        & (rsi > rsi_min)
        & (fast_rsi > fast_rsi_min)
        & (roc < roc_ceiling)
    )

    weight = pd.Series(np.nan, index=df.index)
    # trend flips first so a same-bar entry (below) can override the flatten
    weight[~uptrend] = 0.0
    if allow_short:
        weight[~downtrend] = 0.0
        short_entry = (
            bearish_engulf
            & downtrend
            & (rsi < (100 - rsi_min))
            & (fast_rsi < (100 - fast_rsi_min))
            & (roc > -roc_ceiling)
        )
        weight[short_entry] = -1.0
    weight[long_entry] = 1.0
    return weight.ffill().fillna(0.0)


# ── niche_chaikin_money_flow ──
def _chaikin_money_flow_signals(
    df: pd.DataFrame,
    period: int = 20,
    entry_threshold: float = 0.05,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]

    rng = (high - low).replace(0, float("nan"))
    mfm = ((close - low) - (high - close)) / rng
    mfv = (mfm * volume).fillna(0.0)

    cmf = mfv.rolling(period).sum() / volume.rolling(period).sum().replace(
        0, float("nan")
    )

    weight = pd.Series(float("nan"), index=df.index)
    weight[cmf > entry_threshold] = 1.0
    weight[cmf < -entry_threshold] = -1.0 if allow_short else 0.0
    return weight.ffill().fillna(0.0)


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


# ── niche_klinger_volume ──
def _klinger_signals(
    df: pd.DataFrame,
    fast_period: int = 34,
    slow_period: int = 55,
    signal_period: int = 13,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]

    hlc = (high + low + close) / 3
    trend = np.sign(hlc.diff()).replace(0, np.nan).ffill().fillna(1.0)

    dm = high - low
    trend_flip = trend != trend.shift(1)
    cm = np.zeros(len(df))
    dm_arr = dm.to_numpy()
    flip_arr = trend_flip.to_numpy()
    for i in range(len(df)):
        if i == 0 or flip_arr[i]:
            cm[i] = dm_arr[i]
        else:
            cm[i] = cm[i - 1] + dm_arr[i]
    cm_series = pd.Series(cm, index=df.index).replace(0, np.nan)

    vf = volume * (2 * (dm / cm_series - 1)).abs().fillna(0.0) * trend * 100
    kvo = (
        vf.ewm(span=fast_period, adjust=False).mean()
        - vf.ewm(span=slow_period, adjust=False).mean()
    )
    signal = kvo.ewm(span=signal_period, adjust=False).mean()

    bullish = kvo > signal
    short_weight = -1.0 if allow_short else 0.0
    return bullish.map({True: 1.0, False: short_weight})


# ── pattern_awesome_oscillator ──
def _awesome_saucer_signals(
    df: pd.DataFrame,
    fast_period: int = 5,
    slow_period: int = 34,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    midpoint = (df["high"] + df["low"]) / 2
    ao = midpoint.rolling(fast_period).mean() - midpoint.rolling(slow_period).mean()

    open_, close = df["open"], df["close"]
    red = open_ > close
    green = open_ < close

    bullish_saucer = (
        red
        & green.shift(1)
        & green.shift(2)
        & (ao.shift(1) > ao.shift(2))
        & (ao.shift(1) < 0)
        & (ao < 0)
    )
    bearish_saucer = (
        green
        & red.shift(1)
        & red.shift(2)
        & (ao.shift(1) < ao.shift(2))
        & (ao.shift(1) > 0)
        & (ao > 0)
    )

    short_weight = -1.0 if allow_short else 0.0
    weight = pd.Series(
        np.where(ao > 0, 1.0, np.where(ao < 0, short_weight, np.nan)), index=df.index
    )
    weight[bullish_saucer] = 1.0
    weight[bearish_saucer] = short_weight
    return weight.ffill().fillna(0.0)


# ── pattern_heikin_ashi ──
def _heikin_ashi_heikin_ashi(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)

    ha_close = (o + h + l + c) / 4
    ha_open = np.empty(len(df))
    ha_open[0] = o[0]
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2

    ha_high = np.maximum(np.maximum(ha_open, ha_close), h)
    ha_low = np.minimum(np.minimum(ha_open, ha_close), l)
    return ha_open, ha_high, ha_low, ha_close


def _heikin_ashi_signals(df: pd.DataFrame, allow_short: bool = False, **_) -> pd.Series:
    ha_open, ha_high, ha_low, ha_close = _heikin_ashi_heikin_ashi(df)

    bullish_no_lower_wick = (ha_close > ha_open) & np.isclose(ha_open, ha_low)
    bearish_no_upper_wick = (ha_close < ha_open) & np.isclose(ha_open, ha_high)

    weight = np.zeros(len(df))
    weight[bullish_no_lower_wick] = 1.0
    weight[bearish_no_upper_wick] = -1.0 if allow_short else 0.0
    return pd.Series(weight, index=df.index)


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


# ── trend_adx_dmi ──
def _adx_dmi_dmi_adx(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)

    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)


def _adx_dmi_signals(
    df: pd.DataFrame,
    period: int = 14,
    adx_threshold: float = 20.0,
    fast: int = 10,
    slow: int = 30,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]
    adx = _adx_dmi_dmi_adx(df, period)
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()

    trending = adx > adx_threshold
    bullish = fast_ma > slow_ma

    short_weight = -1.0 if allow_short else 0.0
    weight = pd.Series(0.0, index=df.index)
    weight[trending & bullish] = 1.0
    weight[trending & ~bullish] = short_weight
    return weight


# ── trend_chandelier_exit ──
def _chandelier_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _chandelier_signals(
    df: pd.DataFrame,
    period: int = 22,
    mult: float = 3.0,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    atr = _chandelier_atr(df, period)
    long_stop_basic = (high.rolling(period).max() - mult * atr).to_numpy()
    short_stop_basic = (low.rolling(period).min() + mult * atr).to_numpy()
    close_v = close.to_numpy()

    n = len(df)
    long_stop = np.zeros(n)
    short_stop = np.zeros(n)
    direction = np.ones(n)

    long_stop[0] = long_stop_basic[0]
    short_stop[0] = short_stop_basic[0]

    for i in range(1, n):
        long_stop[i] = (
            max(long_stop_basic[i], long_stop[i - 1])
            if close_v[i - 1] > long_stop[i - 1]
            else long_stop_basic[i]
        )
        short_stop[i] = (
            min(short_stop_basic[i], short_stop[i - 1])
            if close_v[i - 1] < short_stop[i - 1]
            else short_stop_basic[i]
        )

        if direction[i - 1] == -1 and close_v[i] > short_stop[i - 1]:
            direction[i] = 1
        elif direction[i - 1] == 1 and close_v[i] < long_stop[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

    short_weight = -1.0 if allow_short else 0.0
    return pd.Series(direction, index=df.index).map({1: 1.0, -1: short_weight})


# ── trend_halftrend ──
def _halftrend_signals(
    df: pd.DataFrame, amplitude: int = 4, allow_short: bool = False, **_
) -> pd.Series:
    ma_high = df["high"].rolling(amplitude).mean().to_numpy()
    ma_low = df["low"].rolling(amplitude).mean().to_numpy()
    close_v = df["close"].to_numpy()
    n = len(df)

    trend = np.ones(n)
    ref = np.zeros(n)
    ref[0] = ma_low[0] if not np.isnan(ma_low[0]) else close_v[0]

    for i in range(1, n):
        if trend[i - 1] == 1:
            candidate = (
                min(ma_low[i], ref[i - 1]) if not np.isnan(ma_low[i]) else ref[i - 1]
            )
            if close_v[i] < candidate:
                trend[i] = -1
                ref[i] = ma_high[i] if not np.isnan(ma_high[i]) else candidate
            else:
                trend[i] = 1
                ref[i] = candidate
        else:
            candidate = (
                max(ma_high[i], ref[i - 1]) if not np.isnan(ma_high[i]) else ref[i - 1]
            )
            if close_v[i] > candidate:
                trend[i] = 1
                ref[i] = ma_low[i] if not np.isnan(ma_low[i]) else candidate
            else:
                trend[i] = -1
                ref[i] = candidate

    short_weight = -1.0 if allow_short else 0.0
    return pd.Series(trend, index=df.index).map({1: 1.0, -1: short_weight})


# ── trend_ssl_channel ──
def _ssl_channel_signals(
    df: pd.DataFrame, period: int = 10, allow_short: bool = False, **_
) -> pd.Series:
    close = df["close"]
    sma_high = df["high"].rolling(period).mean()
    sma_low = df["low"].rolling(period).mean()

    direction = pd.Series(np.nan, index=df.index)
    direction[close > sma_high] = 1.0
    direction[close < sma_low] = -1.0
    direction = direction.ffill().fillna(0.0)

    if not allow_short:
        direction = direction.clip(lower=0.0)
    return direction


# ── volume_ad_line_trend ──
def _ad_line_trend_signals(
    df: pd.DataFrame,
    fast_period: int = 10,
    slow_period: int = 30,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]

    rng = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / rng
    mfv = (mfm * volume).fillna(0.0)
    adl = mfv.cumsum()

    fast_ema = adl.ewm(span=fast_period, adjust=False).mean()
    slow_ema = adl.ewm(span=slow_period, adjust=False).mean()

    short_weight = -1.0 if allow_short else 0.0
    return (fast_ema > slow_ema).map({True: 1.0, False: short_weight})


# ── volume_ease_of_movement ──
def _ease_of_movement_signals(
    df: pd.DataFrame,
    sma_period: int = 14,
    volume_scale: float = 1_000_000.0,
    entry_threshold: float = 0.0,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    high, low, volume = df["high"], df["low"], df["volume"]

    midpoint = (high + low) / 2
    distance_moved = midpoint.diff()
    box_ratio = (volume / volume_scale) / (high - low).replace(0, np.nan)
    raw_eom = (distance_moved / box_ratio).fillna(0.0)
    eom = raw_eom.rolling(sma_period).mean()

    weight = pd.Series(np.nan, index=df.index)
    weight[eom > entry_threshold] = 1.0
    weight[eom < -entry_threshold] = -1.0 if allow_short else 0.0
    return weight.ffill().fillna(0.0)


# ── volume_elder_force_index ──
def _elder_force_signals(
    df: pd.DataFrame,
    fast_period: int = 2,
    slow_period: int = 13,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close, volume = df["close"], df["volume"]
    force = close.diff() * volume

    fi_fast = force.ewm(span=fast_period, adjust=False).mean()
    fi_slow = force.ewm(span=slow_period, adjust=False).mean()

    trend_up = fi_slow > 0
    trend_down = fi_slow < 0

    long_entry = trend_up & (fi_fast > 0) & (fi_fast.shift(1) <= 0)
    short_entry = trend_down & (fi_fast < 0) & (fi_fast.shift(1) >= 0)
    long_exit = ~trend_up

    weight = pd.Series(np.nan, index=df.index)
    # exit set first so a same-bar entry (below) can override the flatten
    weight[long_exit] = 0.0
    if allow_short:
        weight[short_entry] = -1.0
    weight[long_entry] = 1.0
    return weight.ffill().fillna(0.0)


# ── volume_vwap_trend_filter ──
def _vwap_trend_signals(
    df: pd.DataFrame,
    vwap_period: int = 20,
    slope_lookback: int = 5,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close, volume = df["close"], df["volume"]
    typical = (df["high"] + df["low"] + close) / 3
    pv = (typical * volume).rolling(vwap_period).sum()
    vol = volume.rolling(vwap_period).sum()
    vwap = pv / vol.replace(0, np.nan)

    vwap_rising = vwap > vwap.shift(slope_lookback)
    above = (close > vwap) & vwap_rising
    below = (close < vwap) & ~vwap_rising

    weight = pd.Series(np.nan, index=df.index)
    weight[above] = 1.0
    weight[below] = -1.0 if allow_short else 0.0
    return weight.ffill().fillna(0.0)


# ── volume_weighted_macd ──
def _vw_macd_vwma(close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    return (close * volume).rolling(period).sum() / volume.rolling(
        period
    ).sum().replace(0, np.nan)


def _vw_macd_signals(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close, volume = df["close"], df["volume"]

    vw_macd = _vw_macd_vwma(close, volume, fast_period) - _vw_macd_vwma(
        close, volume, slow_period
    )
    signal_line = vw_macd.ewm(span=signal_period, adjust=False).mean()

    bullish = vw_macd > signal_line
    short_weight = -1.0 if allow_short else 0.0
    return bullish.map({True: 1.0, False: short_weight})


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


# ── academic_dual_momentum_absolute ──
def _dual_momentum_signals(
    df: pd.DataFrame,
    lookback: int = 252,
    risk_free_rate: float = 0.0,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]
    trailing_return = close.pct_change(lookback)
    excess = trailing_return - risk_free_rate

    if allow_short:
        weight = np.sign(excess)
    else:
        weight = (excess > 0).astype(float)

    return pd.Series(weight, index=df.index).fillna(0.0).clip(-1.0, 1.0)


# ── academic_gtaa_ma_timing ──
def _gtaa_ma_signals(
    df: pd.DataFrame,
    ma_lookback: int = 210,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]
    sma = close.rolling(ma_lookback, min_periods=ma_lookback).mean()

    if allow_short:
        weight = np.sign(close - sma)
    else:
        weight = (close > sma).astype(float)

    return pd.Series(weight, index=df.index).fillna(0.0).clip(-1.0, 1.0)


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


IND_SPECS = {
    "chaikin_osc": (
        "Осциллятор Чайкина: вход, когда поток денег (A/D) ускоряется вверх или вниз.",
        60,
        "trending",
        "both",
        "medium",
    ),
    "coppock": (
        "Кривая Коппока: вход при пересечении нуля (разворот импульса).",
        80,
        "trending",
        "both",
        "medium",
    ),
    "coral_trend": (
        "Coral Trend: вход при развороте сглаженной линии тренда.",
        100,
        "trending",
        "both",
        "medium",
    ),
    "kst": (
        "KST (Know Sure Thing): вход, когда сумма импульсов пересекает свою сигнальную линию.",
        100,
        "trending",
        "both",
        "medium",
    ),
    "rvi": (
        "RVI: вход, когда индекс относительной энергии пересекает сигнальную линию.",
        40,
        "trending",
        "both",
        "medium",
    ),
    "waddah_attar": (
        "Waddah Attar Explosion: вход на сильном импульсе выше «мёртвой зоны».",
        140,
        "volatile",
        "both",
        "high",
    ),
    "zlsma": (
        "ZLSMA: вход, когда цена по сторону растущей линии с нулевым запаздыванием.",
        100,
        "trending",
        "both",
        "medium",
    ),
    "cci_aroon": (
        "CCI + Aroon: отскок из перепроданности при свежем восходящем движении и растущем тренде.",
        130,
        "ranging",
        "both",
        "medium",
    ),
    "engulfing_trend": (
        "Поглощение по тренду: бычье/медвежье поглощение, подтверждённое EMA200 и RSI.",
        220,
        "trending",
        "both",
        "medium",
    ),
    "chaikin_money_flow": (
        "Chaikin Money Flow: вход, когда денежный поток устойчиво выше или ниже порога.",
        40,
        "trending",
        "both",
        "medium",
    ),
    "fisher_transform": (
        "Fisher Transform: покупка на развороте вверх из отрицательной зоны при растущем тренде. Только лонг.",
        60,
        "ranging",
        "both",
        "medium",
    ),
    "kdj": (
        "KDJ: вход, когда линия J уходит за K и D (ускорение стохастика).",
        40,
        "ranging",
        "both",
        "medium",
    ),
    "klinger": (
        "Klinger: вход, когда объёмный осциллятор пересекает сигнальную линию.",
        100,
        "trending",
        "both",
        "medium",
    ),
    "awesome_saucer": (
        "Awesome Oscillator: знак осциллятора и паттерн «блюдце» против него.",
        60,
        "trending",
        "both",
        "medium",
    ),
    "heikin_ashi": (
        "Heikin Ashi: свеча без нижней (или верхней) тени — признак сильного движения.",
        30,
        "trending",
        "both",
        "medium",
    ),
    "rsi_divergence": (
        "Дивергенция RSI: цена обновила экстремум, а RSI — нет; вход после подтверждения пивота.",
        80,
        "ranging",
        "both",
        "medium",
    ),
    "shooting_star": (
        "Падающая звезда: свеча с длинной верхней тенью после роста — шорт на 7 свечей. Только продажа.",
        40,
        "ranging",
        "futures",
        "high",
    ),
    "cci_reversion": (
        "CCI: покупка ниже −100, продажа выше +100 (возврат к средней).",
        40,
        "ranging",
        "both",
        "medium",
    ),
    "ifr2": (
        "IFR2: RSI(2) ниже 10 при растущем тренде. Только лонг.",
        60,
        "ranging",
        "both",
        "medium",
    ),
    "stochastic": (
        "Стохастик: вход из зон перепроданности (<20) и перекупленности (>80).",
        40,
        "ranging",
        "both",
        "medium",
    ),
    "adx_dmi": (
        "ADX/DMI: тренд при ADX выше 20, направление по средним 10/30.",
        60,
        "trending",
        "both",
        "medium",
    ),
    "chandelier": (
        "Chandelier Exit: вход при смене стороны трейлинг-стопа от максимума/минимума.",
        60,
        "trending",
        "both",
        "medium",
    ),
    "halftrend": (
        "HalfTrend: вход при смене направления линии HalfTrend.",
        30,
        "trending",
        "both",
        "medium",
    ),
    "ssl_channel": (
        "SSL Channel: вход, когда цена закрылась за средней максимумов (вверх) или минимумов (вниз).",
        30,
        "trending",
        "both",
        "medium",
    ),
    "ad_line_trend": (
        "Линия накопления/распределения: быстрая EMA пересекает медленную.",
        60,
        "trending",
        "both",
        "medium",
    ),
    "ease_of_movement": (
        "Ease of Movement: вход, когда лёгкость движения цены меняет знак.",
        40,
        "trending",
        "both",
        "medium",
    ),
    "elder_force": (
        "Индекс силы Элдера: вход при пересечении нуля быстрой силой по знаку медленной.",
        40,
        "trending",
        "both",
        "medium",
    ),
    "vwap_trend": (
        "VWAP-тренд: цена по сторону растущего или падающего VWAP.",
        40,
        "trending",
        "both",
        "medium",
    ),
    "vw_macd": (
        "MACD по VWMA: вход, когда объёмно-взвешенный MACD пересекает сигнальную линию.",
        60,
        "trending",
        "both",
        "medium",
    ),
    "poc_reversion": (
        "Профиль объёма: отклонение от точки максимального объёма (POC) на 1% — ставка на возврат.",
        80,
        "ranging",
        "both",
        "medium",
    ),
    "bollinger_reentry": (
        "Возврат в полосы Боллинджера: вход, когда цена вернулась внутрь полосы после выхода за неё.",
        40,
        "ranging",
        "both",
        "medium",
    ),
    "dual_momentum": (
        "Двойной моментум: лонг, пока доходность за период выше нуля, иначе шорт. Окно 252 свечи.",
        270,
        "trending",
        "both",
        "medium",
    ),
    "gtaa_ma": (
        "Тайминг по скользящей: лонг выше SMA(210), шорт ниже.",
        230,
        "trending",
        "both",
        "medium",
    ),
    "spike_reversal": (
        "Разворот после всплеска объёма и цены: ставка против движения на 6 свечей.",
        80,
        "ranging",
        "both",
        "high",
    ),
}

IND_FUNCS = {
    "chaikin_osc": _chaikin_osc_signals,
    "coppock": _coppock_signals,
    "coral_trend": _coral_trend_signals,
    "kst": _kst_signals,
    "rvi": _rvi_signals,
    "waddah_attar": _waddah_attar_signals,
    "zlsma": _zlsma_signals,
    "cci_aroon": _cci_aroon_signals,
    "engulfing_trend": _engulfing_trend_signals,
    "chaikin_money_flow": _chaikin_money_flow_signals,
    "fisher_transform": _fisher_transform_signals,
    "kdj": _kdj_signals,
    "klinger": _klinger_signals,
    "awesome_saucer": _awesome_saucer_signals,
    "heikin_ashi": _heikin_ashi_signals,
    "rsi_divergence": _rsi_divergence_signals,
    "shooting_star": _shooting_star_signals,
    "cci_reversion": _cci_reversion_signals,
    "ifr2": _ifr2_signals,
    "stochastic": _stochastic_signals,
    "adx_dmi": _adx_dmi_signals,
    "chandelier": _chandelier_signals,
    "halftrend": _halftrend_signals,
    "ssl_channel": _ssl_channel_signals,
    "ad_line_trend": _ad_line_trend_signals,
    "ease_of_movement": _ease_of_movement_signals,
    "elder_force": _elder_force_signals,
    "vwap_trend": _vwap_trend_signals,
    "vw_macd": _vw_macd_signals,
    "poc_reversion": _poc_reversion_signals,
    "bollinger_reentry": _bollinger_reentry_signals,
    "dual_momentum": _dual_momentum_signals,
    "gtaa_ma": _gtaa_ma_signals,
    "spike_reversal": _spike_reversal_signals,
}


def _make(short: str) -> Type[BaseStrategy]:
    desc, min_bars, market_type, markets, risk = IND_SPECS[short]
    fn = IND_FUNCS[short]

    def weights(self: Any, df: pd.DataFrame) -> pd.Series:
        return fn(df, allow_short=True).astype(float).fillna(0.0)

    return type(
        "".join(p.title() for p in short.split("_")) + "Strategy",
        (PortedStrategy,),
        {
            "name": short,
            "description": desc,
            "recommended_timeframes": ["15m", "1h"],
            "risk_level": risk,
            "market_type": market_type,
            "markets": markets,
            "min_bars": min_bars,
            "weights": weights,
        },
    )


IND_STRATEGIES: Dict[str, Type[BaseStrategy]] = {k: _make(k) for k in IND_SPECS}
