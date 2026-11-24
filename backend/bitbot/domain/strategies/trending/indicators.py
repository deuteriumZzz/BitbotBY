"""Экспериментальные индикаторные стратегии (trending): порты из Argent-Systems.

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


ChaikinOscStrategy = indicator_strategy(
    "chaikin_osc",
    _chaikin_osc_signals,
    description="Осциллятор Чайкина: вход, когда поток денег (A/D) ускоряется вверх или вниз.",
    min_bars=60,
    market_type="trending",
    markets="both",
    risk="medium",
)


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


CoppockStrategy = indicator_strategy(
    "coppock",
    _coppock_signals,
    description="Кривая Коппока: вход при пересечении нуля (разворот импульса).",
    min_bars=80,
    market_type="trending",
    markets="both",
    risk="medium",
)


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


CoralTrendStrategy = indicator_strategy(
    "coral_trend",
    _coral_trend_signals,
    description="Coral Trend: вход при развороте сглаженной линии тренда.",
    min_bars=100,
    market_type="trending",
    markets="both",
    risk="medium",
)


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


KstStrategy = indicator_strategy(
    "kst",
    _kst_signals,
    description="KST (Know Sure Thing): вход, когда сумма импульсов пересекает свою сигнальную линию.",
    min_bars=100,
    market_type="trending",
    markets="both",
    risk="medium",
)


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


RviStrategy = indicator_strategy(
    "rvi",
    _rvi_signals,
    description="RVI: вход, когда индекс относительной энергии пересекает сигнальную линию.",
    min_bars=40,
    market_type="trending",
    markets="both",
    risk="medium",
)


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


ZlsmaStrategy = indicator_strategy(
    "zlsma",
    _zlsma_signals,
    description="ZLSMA: вход, когда цена по сторону растущей линии с нулевым запаздыванием.",
    min_bars=100,
    market_type="trending",
    markets="both",
    risk="medium",
)


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


EngulfingTrendStrategy = indicator_strategy(
    "engulfing_trend",
    _engulfing_trend_signals,
    description="Поглощение по тренду: бычье/медвежье поглощение, подтверждённое EMA200 и RSI.",
    min_bars=220,
    market_type="trending",
    markets="both",
    risk="medium",
)


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


ChaikinMoneyFlowStrategy = indicator_strategy(
    "chaikin_money_flow",
    _chaikin_money_flow_signals,
    description="Chaikin Money Flow: вход, когда денежный поток устойчиво выше или ниже порога.",
    min_bars=40,
    market_type="trending",
    markets="both",
    risk="medium",
)


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


KlingerStrategy = indicator_strategy(
    "klinger",
    _klinger_signals,
    description="Klinger: вход, когда объёмный осциллятор пересекает сигнальную линию.",
    min_bars=100,
    market_type="trending",
    markets="both",
    risk="medium",
)


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


AwesomeSaucerStrategy = indicator_strategy(
    "awesome_saucer",
    _awesome_saucer_signals,
    description="Awesome Oscillator: знак осциллятора и паттерн «блюдце» против него.",
    min_bars=60,
    market_type="trending",
    markets="both",
    risk="medium",
)


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


HeikinAshiStrategy = indicator_strategy(
    "heikin_ashi",
    _heikin_ashi_signals,
    description="Heikin Ashi: свеча без нижней (или верхней) тени — признак сильного движения.",
    min_bars=30,
    market_type="trending",
    markets="both",
    risk="medium",
)


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


AdxDmiStrategy = indicator_strategy(
    "adx_dmi",
    _adx_dmi_signals,
    description="ADX/DMI: тренд при ADX выше 20, направление по средним 10/30.",
    min_bars=60,
    market_type="trending",
    markets="both",
    risk="medium",
)


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


ChandelierStrategy = indicator_strategy(
    "chandelier",
    _chandelier_signals,
    description="Chandelier Exit: вход при смене стороны трейлинг-стопа от максимума/минимума.",
    min_bars=60,
    market_type="trending",
    markets="both",
    risk="medium",
)


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


HalftrendStrategy = indicator_strategy(
    "halftrend",
    _halftrend_signals,
    description="HalfTrend: вход при смене направления линии HalfTrend.",
    min_bars=30,
    market_type="trending",
    markets="both",
    risk="medium",
)


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


SslChannelStrategy = indicator_strategy(
    "ssl_channel",
    _ssl_channel_signals,
    description="SSL Channel: вход, когда цена закрылась за средней максимумов (вверх) или минимумов (вниз).",
    min_bars=30,
    market_type="trending",
    markets="both",
    risk="medium",
)


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


AdLineTrendStrategy = indicator_strategy(
    "ad_line_trend",
    _ad_line_trend_signals,
    description="Линия накопления/распределения: быстрая EMA пересекает медленную.",
    min_bars=60,
    market_type="trending",
    markets="both",
    risk="medium",
)


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


EaseOfMovementStrategy = indicator_strategy(
    "ease_of_movement",
    _ease_of_movement_signals,
    description="Ease of Movement: вход, когда лёгкость движения цены меняет знак.",
    min_bars=40,
    market_type="trending",
    markets="both",
    risk="medium",
)


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


ElderForceStrategy = indicator_strategy(
    "elder_force",
    _elder_force_signals,
    description="Индекс силы Элдера: вход при пересечении нуля быстрой силой по знаку медленной.",
    min_bars=40,
    market_type="trending",
    markets="both",
    risk="medium",
)


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


VwapTrendStrategy = indicator_strategy(
    "vwap_trend",
    _vwap_trend_signals,
    description="VWAP-тренд: цена по сторону растущего или падающего VWAP.",
    min_bars=40,
    market_type="trending",
    markets="both",
    risk="medium",
)


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


VwMacdStrategy = indicator_strategy(
    "vw_macd",
    _vw_macd_signals,
    description="MACD по VWMA: вход, когда объёмно-взвешенный MACD пересекает сигнальную линию.",
    min_bars=60,
    market_type="trending",
    markets="both",
    risk="medium",
)


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


DualMomentumStrategy = indicator_strategy(
    "dual_momentum",
    _dual_momentum_signals,
    description="Двойной моментум: лонг, пока доходность за период выше нуля, иначе шорт. Окно 252 свечи.",
    min_bars=270,
    market_type="trending",
    markets="both",
    risk="medium",
)


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


GtaaMaStrategy = indicator_strategy(
    "gtaa_ma",
    _gtaa_ma_signals,
    description="Тайминг по скользящей: лонг выше SMA(210), шорт ниже.",
    min_bars=230,
    market_type="trending",
    markets="both",
    risk="medium",
)
