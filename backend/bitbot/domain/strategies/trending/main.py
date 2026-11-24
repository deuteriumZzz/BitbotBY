"""Основные стратегии для трендового рынка: включены по умолчанию.
"""

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from bitbot.domain.strategies.base import BaseStrategy


class EMACrossoverStrategy(BaseStrategy):
    """
    Стратегия на пересечении EMA 12 и EMA 26.

    Покупка при бычьем кроссовере, продажа при медвежьем.
    Подходит для трендовых рынков.
    """

    name = "ema_crossover"
    description = (
        "Торговля на пересечении EMA 12 и EMA 26. "
        "Покупка при бычьем кроссовере, продажа при медвежьем. "
        "Подходит для трендовых рынков."
    )
    recommended_timeframes = ["1h", "4h", "1d"]
    risk_level = "medium"
    market_type = "trending"

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Генерирует сигнал на основе пересечения EMA 12/26.

        :param data: DataFrame с индикаторами ema_short, ema_long.
        :return: Сигнал с action и confidence.
        """
        close = self._last(data, "close")
        ema_s = self._last(data, "ema_short", close)
        ema_l = self._last(data, "ema_long", close)
        ema_s_p = self._prev(data, "ema_short", ema_s)
        ema_l_p = self._prev(data, "ema_long", ema_l)
        ema_s_p2 = self._prev2(data, "ema_short", ema_s)
        ema_l_p2 = self._prev2(data, "ema_long", ema_l)

        # Confirmed bullish: current AND prev bar ema_s > ema_l
        # AND the bar before that had ema_s <= ema_l (actual crossover)
        bullish_cross = ema_s > ema_l and ema_s_p > ema_l_p and ema_s_p2 <= ema_l_p2
        bearish_cross = ema_s < ema_l and ema_s_p < ema_l_p and ema_s_p2 >= ema_l_p2

        if bullish_cross:
            return {"action": "buy", "confidence": 0.82, "price": close}
        if bearish_cross:
            return {"action": "sell", "confidence": 0.78, "price": close}
        if ema_s > ema_l:
            return {"action": "buy", "confidence": 0.60, "price": close}
        if ema_s < ema_l:
            return {"action": "sell", "confidence": 0.58, "price": close}
        return {"action": "hold", "confidence": 0.50}


class MACDCrossoverStrategy(BaseStrategy):
    """
    Стратегия на пересечении MACD и сигнальной линии.

    Покупка при бычьем кроссовере MACD/Signal выше нуля,
    продажа при медвежьем.
    """

    name = "macd_crossover"
    description = (
        "Покупка при бычьем кроссовере MACD/Signal выше нуля, "
        "продажа при медвежьем. "
        "Сила сигнала зависит от положения относительно нулевой линии."
    )
    recommended_timeframes = ["1h", "4h", "1d"]
    risk_level = "medium"
    market_type = "trending"

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Генерирует сигнал на основе пересечения MACD/Signal.

        :param data: DataFrame с индикаторами macd, macd_signal.
        :return: Сигнал с action и confidence.
        """
        close = self._last(data, "close")
        macd = self._last(data, "macd")
        sig = self._last(data, "macd_signal")
        macd_p = self._prev(data, "macd", macd)
        sig_p = self._prev(data, "macd_signal", sig)

        bullish_cross = macd > sig and macd_p <= sig_p
        bearish_cross = macd < sig and macd_p >= sig_p

        if bullish_cross:
            conf = 0.88 if macd > 0 else 0.68
            return {"action": "buy", "confidence": conf, "price": close}
        if bearish_cross:
            conf = 0.88 if macd < 0 else 0.68
            return {"action": "sell", "confidence": conf, "price": close}
        if macd > sig and macd > 0:
            return {"action": "buy", "confidence": 0.58, "price": close}
        if macd < sig and macd < 0:
            return {"action": "sell", "confidence": 0.58, "price": close}
        return {"action": "hold", "confidence": 0.50}


class SwingTradingStrategy(BaseStrategy):
    """
    Стратегия свинг-трейдинга на поворотных точках тренда.

    Требует подтверждения от EMA, MACD, RSI и средней линии Боллинджера.
    Лучший таймфрейм: 4h–1d.
    """

    name = "swing_trading"
    description = (
        "Открытие позиций на поворотных точках тренда. "
        "Требует подтверждения от EMA, MACD, RSI и средней линии Боллинджера. "
        "Лучший таймфрейм: 4h–1d."
    )
    recommended_timeframes = ["4h", "1d"]
    risk_level = "medium"
    market_type = "trending"

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Генерирует свинг-сигнал с множественным подтверждением.

        :param data: DataFrame с индикаторами ema_short, ema_long, macd, rsi, bb_middle.
        :return: Сигнал с action и confidence.
        """
        close = self._last(data, "close")
        rsi = self._last(data, "rsi", 50)
        ema_s = self._last(data, "ema_short", close)
        ema_l = self._last(data, "ema_long", close)
        macd = self._last(data, "macd")
        macd_sig = self._last(data, "macd_signal")
        bb_m = self._last(data, "bb_middle", close)

        trend_up = ema_s > ema_l
        trend_down = ema_s < ema_l
        macd_up = macd > macd_sig
        macd_down = macd < macd_sig

        if trend_up and macd_up and 40 < rsi < 65 and close > bb_m:
            return {"action": "buy", "confidence": 0.85, "price": close}
        if trend_down and macd_down and 35 < rsi < 60 and close < bb_m:
            return {"action": "sell", "confidence": 0.85, "price": close}
        if trend_up and macd_up and rsi < 65:
            return {"action": "buy", "confidence": 0.65, "price": close}
        if trend_down and macd_down and rsi > 35:
            return {"action": "sell", "confidence": 0.65, "price": close}
        return {"action": "hold", "confidence": 0.50}


class TrendFollowingStrategy(BaseStrategy):
    """
    Консервативная стратегия следования тренду через SMA/EMA.

    Открытие позиции только при подтверждении тренда:
    SMA20 > SMA50, EMA12 > EMA26, цена выше SMA20.
    """

    name = "trend_following"
    description = (
        "Открытие позиции только при подтверждении тренда: "
        "SMA20 > SMA50, EMA12 > EMA26, цена выше SMA20. "
        "Консервативная стратегия с низким риском."
    )
    recommended_timeframes = ["4h", "1d"]
    risk_level = "low"
    market_type = "trending"

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Генерирует сигнал следования тренду.

        :param data: DataFrame с индикаторами ema_short, ema_long, sma_20, sma_50, rsi.
        :return: Сигнал с action и confidence.
        """
        close = self._last(data, "close")
        ema_s = self._last(data, "ema_short", close)
        ema_l = self._last(data, "ema_long", close)
        sma20 = self._last(data, "sma_20", close)
        sma50 = self._last(data, "sma_50", close)
        rsi = self._last(data, "rsi", 50)

        strong_up = sma20 > sma50 and ema_s > ema_l and close > sma20
        strong_down = sma20 < sma50 and ema_s < ema_l and close < sma20

        if strong_up and 45 < rsi < 70:
            return {"action": "buy", "confidence": 0.80, "price": close}
        if strong_up:
            return {"action": "buy", "confidence": 0.62, "price": close}
        if strong_down and 30 < rsi < 55:
            return {"action": "sell", "confidence": 0.80, "price": close}
        if strong_down:
            return {"action": "sell", "confidence": 0.62, "price": close}
        return {"action": "hold", "confidence": 0.50}


# 16. FIBONACCI RETRACEMENT (ICT tier-1)


class FibonacciRetracementStrategy(BaseStrategy):
    """
    Вход на откате к уровням Фибоначчи от последнего значимого свинга.

    Определяет свинг-хай/лоу за последние 50 свечей, направление ноги — по
    хронологии (какой экстремум сформировался раньше), затем ищет вход в
    "золотой зоне" отката 50-61.8% с подтверждением разворота по RSI и
    направлению последней свечи.
    """

    name = "fibonacci_retracement"
    description = (
        "Вход на откате в зону 50-61.8% Фибоначчи от последнего значимого "
        "свинга, с подтверждением разворота по RSI. Требует тренда."
    )
    recommended_timeframes = ["1h", "4h", "1d"]
    risk_level = "medium"
    market_type = "trending"

    _LOOKBACK = 50

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        if len(data) < self._LOOKBACK:
            return {"action": "hold", "confidence": 0.50}

        window = data.iloc[-self._LOOKBACK :]
        close = self._last(data, "close")
        prev_close = self._prev(data, "close", close)
        rsi = self._last(data, "rsi", 50)

        high_pos = int(window["high"].values.argmax())
        low_pos = int(window["low"].values.argmin())
        swing_high = float(window["high"].iloc[high_pos])
        swing_low = float(window["low"].iloc[low_pos])
        diff = swing_high - swing_low
        if diff <= 0:
            return {"action": "hold", "confidence": 0.50}

        uptrend_leg = low_pos < high_pos

        if uptrend_leg:
            level_50 = swing_high - diff * 0.5
            level_618 = swing_high - diff * 0.618
        else:
            level_50 = swing_low + diff * 0.5
            level_618 = swing_low + diff * 0.618
        zone_lo, zone_hi = sorted((level_50, level_618))

        if not (zone_lo <= close <= zone_hi):
            return {"action": "hold", "confidence": 0.40}

        if uptrend_leg and rsi < 55 and close >= prev_close:
            depth = (swing_high - close) / diff
            conf = round(0.60 + (depth - 0.5) / 0.118 * 0.22, 2)
            return {"action": "buy", "confidence": conf, "price": close}

        if not uptrend_leg and rsi > 45 and close <= prev_close:
            depth = (close - swing_low) / diff
            conf = round(0.60 + (depth - 0.5) / 0.118 * 0.22, 2)
            return {"action": "sell", "confidence": conf, "price": close}

        return {"action": "hold", "confidence": 0.45}


# 18. FAIR VALUE GAP (ICT tier-1)


class FairValueGapStrategy(BaseStrategy):
    """
    Вход на возврате цены в незаполненный Fair Value Gap (ICT).

    FVG — трёхсвечный ценовой дисбаланс: агрессивное движение (средняя
    свеча) оставляет "дыру" без торговли между хаем свечи 1 и лоу свечи 3
    (бычий FVG) либо между лоу свечи 1 и хаем свечи 3 (медвежий FVG).
    Такие зоны статистически стремятся быть "закрытыми" — цена
    возвращается в них хотя бы частично. Стратегия ищет самый свежий
    ещё не пробитый насквозь гэп за lookback свечей и торгует отскок при
    возврате цены в его границы, в направлении породившего гэп импульса
    (не против него).
    """

    name = "fair_value_gap"
    description = (
        "Вход на возврате цены в последний незаполненный Fair Value Gap "
        "(ICT) — по направлению породившего гэп импульса."
    )
    recommended_timeframes = ["15m", "1h", "4h"]
    risk_level = "medium"
    market_type = "trending"

    _LOOKBACK = 40

    def _find_last_fvg(
        self, window: pd.DataFrame
    ) -> Optional[Tuple[str, float, float, int]]:
        """Ищет самый свежий непробитый насквозь FVG в окне.

        :return: (направление_импульса, low_зоны, high_зоны, возраст_в_свечах) или None.
        """
        n = len(window)
        for i in range(n - 3, -1, -1):
            c1 = window.iloc[i]
            c3 = window.iloc[i + 2]
            if c1["high"] < c3["low"]:
                zone_lo, zone_hi, direction = c1["high"], c3["low"], "bullish"
            elif c1["low"] > c3["high"]:
                zone_lo, zone_hi, direction = c3["high"], c1["low"], "bearish"
            else:
                continue

            filled = False
            for j in range(i + 3, n):
                c = window.iloc[j]
                if direction == "bullish" and c["low"] <= zone_lo:
                    filled = True
                    break
                if direction == "bearish" and c["high"] >= zone_hi:
                    filled = True
                    break
            if filled:
                continue

            age = n - 1 - (i + 2)
            return direction, float(zone_lo), float(zone_hi), age
        return None

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        if len(data) < self._LOOKBACK:
            return {"action": "hold", "confidence": 0.50}

        window = data.iloc[-self._LOOKBACK :]
        close = self._last(data, "close")
        rsi = self._last(data, "rsi", 50)

        fvg = self._find_last_fvg(window)
        if fvg is None:
            return {"action": "hold", "confidence": 0.40}
        direction, zone_lo, zone_hi, age = fvg

        if not (zone_lo <= close <= zone_hi):
            return {"action": "hold", "confidence": 0.40}

        freshness = max(0.0, 1.0 - age / self._LOOKBACK)

        if direction == "bullish" and rsi < 60:
            conf = round(min(0.58 + freshness * 0.22, 0.85), 2)
            return {"action": "buy", "confidence": conf, "price": close}

        if direction == "bearish" and rsi > 40:
            conf = round(min(0.58 + freshness * 0.22, 0.85), 2)
            return {"action": "sell", "confidence": conf, "price": close}

        return {"action": "hold", "confidence": 0.45}
