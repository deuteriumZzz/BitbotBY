"""Основные стратегии для бокового рынка (флэт): включены по умолчанию.
"""

from typing import Any, Dict, List

import pandas as pd

from bitbot.domain.strategies.base import BaseStrategy


class RSIMomentumStrategy(BaseStrategy):
    """
    Стратегия RSI Momentum на перепроданности/перекупленности.

    Покупка при RSI < 30 (перепроданность),
    продажа при RSI > 70 (перекупленность).
    Оптимальна при боковом движении цены.
    """

    name = "rsi_momentum"
    description = (
        "Покупка при RSI < 30 (перепроданность), "
        "продажа при RSI > 70 (перекупленность). "
        "Оптимальна при боковом движении цены."
    )
    recommended_timeframes = ["15m", "1h", "4h"]
    risk_level = "medium"
    market_type = "ranging"

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Генерирует сигнал на основе значения RSI.

        :param data: DataFrame с индикатором rsi.
        :return: Сигнал с action и confidence.
        """
        close = self._last(data, "close")
        rsi = self._last(data, "rsi", 50)

        if rsi < 25:
            return {"action": "buy", "confidence": 0.90, "price": close}
        if rsi < 30:
            return {"action": "buy", "confidence": 0.78, "price": close}
        # RSI 30-70 is neutral territory — no directional edge, skip.
        if rsi > 75:
            return {"action": "sell", "confidence": 0.90, "price": close}
        if rsi > 70:
            return {"action": "sell", "confidence": 0.78, "price": close}
        return {"action": "hold", "confidence": 0.50}


class MeanReversionStrategy(BaseStrategy):
    """
    Стратегия возврата к среднему по Bollinger Bands.

    Требует экстремального отклонения (BB + RSI + momentum).
    Работает на боковых рынках с чёткими уровнями.
    """

    name = "mean_reversion"
    description = (
        "Торговля на возврате цены к средней линии Боллинджера. "
        "Требует экстремального отклонения (BB + RSI + momentum). "
        "Работает на боковых рынках с чёткими уровнями."
    )
    recommended_timeframes = ["1h", "4h"]
    risk_level = "medium"
    market_type = "ranging"

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Генерирует сигнал возврата к среднему.

        :param data: DataFrame с индикаторами bb_upper, bb_lower,
            bb_middle, rsi, momentum.
        :return: Сигнал с action и confidence.
        """
        close = self._last(data, "close")
        rsi = self._last(data, "rsi", 50)
        bb_u = self._last(data, "bb_upper", close * 1.02)
        bb_l = self._last(data, "bb_lower", close * 0.98)
        bb_m = self._last(data, "bb_middle", close)
        mom = self._last(data, "momentum", 0)

        extreme_low = close < bb_l and rsi < 30
        extreme_high = close > bb_u and rsi > 70

        # Mean-reversion is strongest when momentum is already turning around:
        # extreme_low + mom > 0 means price bouncing up from oversold → high confidence.
        # extreme_low + mom < 0 means still falling → enter with lower confidence.
        if extreme_low and mom > 0:
            return {"action": "buy", "confidence": 0.88, "price": close}
        if extreme_low:
            return {"action": "buy", "confidence": 0.72, "price": close}
        if close < bb_l:
            return {"action": "buy", "confidence": 0.62, "price": close}
        if extreme_high and mom < 0:
            return {"action": "sell", "confidence": 0.88, "price": close}
        if extreme_high:
            return {"action": "sell", "confidence": 0.72, "price": close}
        if close > bb_u:
            return {"action": "sell", "confidence": 0.62, "price": close}
        if close < bb_m and rsi < 45:
            return {"action": "buy", "confidence": 0.55, "price": close}
        if close > bb_m and rsi > 55:
            return {"action": "sell", "confidence": 0.55, "price": close}
        return {"action": "hold", "confidence": 0.50}


# 15. SUPPORT / RESISTANCE (ICT tier-1)


class SupportResistanceStrategy(BaseStrategy):
    """
    Отскок от кластеризованных уровней поддержки/сопротивления.

    В отличие от `breakout` (пробой 20-свечного хай/лоу), строит реальные
    уровни через кластеризацию локальных свинг-хай/лоу за последние 60
    свечей — уровень тем сильнее, чем больше раз цена его касалась. Сигнал
    даётся на отскоке (не пробое) от сильного уровня с подтверждением RSI.
    """

    name = "support_resistance"
    description = (
        "Отскок от уровня поддержки/сопротивления, построенного кластеризацией "
        "свинг-хай/лоу за 60 свечей. Сила уровня = число касаний."
    )
    recommended_timeframes = ["1h", "4h"]
    risk_level = "medium"
    market_type = "ranging"

    _LOOKBACK = 60
    _SWING_WINDOW = 3
    _CLUSTER_TOL = 0.005
    _TOUCH_TOL = 0.003

    def _swing_levels(
        self, window: pd.DataFrame, col: str, is_high: bool
    ) -> List[float]:
        w = self._SWING_WINDOW
        n = len(window)
        levels = []
        for i in range(w, n - w):
            local = window[col].iloc[i - w : i + w + 1]
            val = window[col].iloc[i]
            if is_high and val == local.max():
                levels.append(float(val))
            elif not is_high and val == local.min():
                levels.append(float(val))
        return levels

    @staticmethod
    def _cluster(levels: List[float], price: float, tol_pct: float) -> List[Any]:
        """Группирует близкие уровни, возвращает [(средний_уровень, касания), ...]."""
        if not levels:
            return []
        levels = sorted(levels)
        tol = price * tol_pct
        clusters: List[List[float]] = [[levels[0]]]
        for lv in levels[1:]:
            if lv - clusters[-1][-1] <= tol:
                clusters[-1].append(lv)
            else:
                clusters.append([lv])
        return [(sum(c) / len(c), len(c)) for c in clusters]

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        if len(data) < self._LOOKBACK:
            return {"action": "hold", "confidence": 0.50}

        window = data.iloc[-self._LOOKBACK :]
        close = self._last(data, "close")
        rsi = self._last(data, "rsi", 50)

        highs = self._cluster(
            self._swing_levels(window, "high", True), close, self._CLUSTER_TOL
        )
        lows = self._cluster(
            self._swing_levels(window, "low", False), close, self._CLUSTER_TOL
        )
        supports = [(lv, touches) for lv, touches in lows if lv < close]
        resistances = [(lv, touches) for lv, touches in highs if lv > close]
        touch_tol = close * self._TOUCH_TOL

        if supports:
            level, touches = max(supports, key=lambda x: x[1])
            if close - level <= touch_tol and rsi < 45:
                conf = round(min(0.55 + touches * 0.08, 0.88), 2)
                return {"action": "buy", "confidence": conf, "price": close}

        if resistances:
            level, touches = max(resistances, key=lambda x: x[1])
            if level - close <= touch_tol and rsi > 55:
                conf = round(min(0.55 + touches * 0.08, 0.88), 2)
                return {"action": "sell", "confidence": conf, "price": close}

        return {"action": "hold", "confidence": 0.40}


# 20. LIQUIDITY SWEEP (ICT tier-1)


class LiquiditySweepStrategy(BaseStrategy):
    """
    Разворот на снятии ликвидности за предыдущим свинг-хай/лоу (ICT stop hunt).

    За предыдущим свинг-экстремумом скапливаются стоп-лоссы (ликвидность).
    "Снятие ликвидности" — цена коротким фитилём пробивает этот уровень,
    исполняя чужие стопы, но закрывается обратно внутри диапазона —
    типичный признак ложного пробоя перед разворотом, а не начала нового
    тренда. В отличие от `breakout` (торгует ПРОДОЛЖЕНИЕ пробоя), эта
    стратегия торгует его ОТКАЗ: сигнал только когда текущая свеча
    пробила уровень фитилём, но закрылась обратно за него.
    """

    name = "liquidity_sweep"
    description = (
        "Разворот на ложном пробое (снятии ликвидности) предыдущего "
        "свинг-хай/лоу — фитиль пробивает уровень, закрытие возвращается "
        "внутрь диапазона."
    )
    recommended_timeframes = ["15m", "1h", "4h"]
    risk_level = "high"
    market_type = "ranging"

    _LOOKBACK = 30

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        if len(data) < self._LOOKBACK + 1:
            return {"action": "hold", "confidence": 0.50}

        prior = data.iloc[-(self._LOOKBACK + 1) : -1]
        cur = data.iloc[-1]
        rsi = self._last(data, "rsi", 50)

        swing_high = float(prior["high"].max())
        swing_low = float(prior["low"].min())
        cur_high, cur_low, cur_close = (
            float(cur["high"]),
            float(cur["low"]),
            float(cur["close"]),
        )

        if cur_high > swing_high and cur_close < swing_high and rsi > 50:
            depth = (cur_high - swing_high) / swing_high
            conf = round(min(0.60 + min(depth / 0.01, 1.0) * 0.20, 0.85), 2)
            return {"action": "sell", "confidence": conf, "price": cur_close}

        if cur_low < swing_low and cur_close > swing_low and rsi < 50:
            depth = (swing_low - cur_low) / swing_low
            conf = round(min(0.60 + min(depth / 0.01, 1.0) * 0.20, 0.85), 2)
            return {"action": "buy", "confidence": conf, "price": cur_close}

        return {"action": "hold", "confidence": 0.40}


# 21. AMD — ACCUMULATION / MANIPULATION / DISTRIBUTION (ICT tier-1)


class AMDStrategy(BaseStrategy):
    """
    Accumulation-Manipulation-Distribution (ICT "Power of Three").

    Классическая модель ICT для дневного/сессионного диапазона: узкий
    флэт-диапазон накопления (Accumulation) → ложный вынос цены за его
    границу в одну сторону, снимающий ликвидность (Manipulation / "Judas
    swing") → настоящее направленное движение в противоположную сторону
    (Distribution). В OHLCV-приближении: диапазон накопления ищется по
    тесноте свечей за accum-окно (до недавних _MANIP_WINDOW свечей),
    манипуляция — по выносу хая/лоу за его границы в этом недавнем окне,
    дистрибуция — по закрытию цены за ПРОТИВОПОЛОЖНОЙ границей диапазона
    там же. Требование закрытия за противоположной границей (не просто
    в середину) — сознательно строгий фильтр: сигнал только на полностью
    состоявшейся дистрибуции, а не на первом отскоке.
    """

    name = "amd"
    description = (
        "ICT Power of Three: узкий диапазон накопления → ложный вынос за "
        "его границу (манипуляция) → закрытие за противоположной границей "
        "(дистрибуция) — вход по направлению дистрибуции."
    )
    recommended_timeframes = ["1h", "4h", "1d"]
    risk_level = "high"
    market_type = "ranging"

    _ACCUM_LOOKBACK = 20
    _MANIP_WINDOW = 6
    _MAX_RANGE_PCT = 0.05

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        total_needed = self._ACCUM_LOOKBACK + self._MANIP_WINDOW
        if len(data) < total_needed:
            return {"action": "hold", "confidence": 0.50}

        accum = data.iloc[-total_needed : -self._MANIP_WINDOW]
        recent = data.iloc[-self._MANIP_WINDOW :]
        cur_close = self._last(data, "close")
        rsi = self._last(data, "rsi", 50)

        range_high = float(accum["high"].max())
        range_low = float(accum["low"].min())
        mid = (range_high + range_low) / 2
        if mid <= 0:
            return {"action": "hold", "confidence": 0.40}
        width_pct = (range_high - range_low) / mid
        if width_pct > self._MAX_RANGE_PCT:
            return {"action": "hold", "confidence": 0.40}

        manip_up = float(recent["high"].max()) > range_high
        manip_down = float(recent["low"].min()) < range_low
        span = max(range_high - range_low, 1e-9)

        # Манипуляция вверх (Judas swing вверх) → настоящая дистрибуция вниз
        if manip_up and cur_close < range_low and rsi < 50:
            depth = (range_low - cur_close) / span
            conf = round(min(0.62 + min(depth, 1.0) * 0.20, 0.88), 2)
            return {"action": "sell", "confidence": conf, "price": cur_close}

        # Манипуляция вниз (Judas swing вниз) → настоящая дистрибуция вверх
        if manip_down and cur_close > range_high and rsi > 50:
            depth = (cur_close - range_high) / span
            conf = round(min(0.62 + min(depth, 1.0) * 0.20, 0.88), 2)
            return {"action": "buy", "confidence": conf, "price": cur_close}

        return {"action": "hold", "confidence": 0.45}
