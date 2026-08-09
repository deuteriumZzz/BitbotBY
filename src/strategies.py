"""
Торговые стратегии и оркестратор стратегий.

Содержит базовый класс BaseStrategy, конкретные реализации (EMA Crossover,
RSI Momentum, MACD, Bollinger Bands, Scalping, Swing, Breakout, Mean Reversion,
Trend Following), реестр STRATEGY_REGISTRY и класс TradingStrategy-оркестратор.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

import pandas as pd

from config import Config

from .redis_client import RedisClient

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Базовый абстрактный класс для всех торговых стратегий.

    Определяет интерфейс generate_signal() и вспомогательные методы
    для безопасного доступа к значениям DataFrame.
    """

    name: str = ""
    description: str = ""
    recommended_timeframes: List[str] = []
    risk_level: str = "medium"  # low / medium / high
    market_type: str = "any"  # trending / ranging / volatile / any

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Генерирует торговый сигнал на основе рыночных данных.

        :param data: DataFrame с OHLCV и техническими индикаторами.
        :return: Словарь с ключами action, confidence и опционально price.
        """

    def get_info(self) -> Dict[str, Any]:
        """
        Возвращает метаданные стратегии.

        :return: Словарь с name, description, recommended_timeframes,
            risk_level, market_type.
        """
        return {
            "name": self.name,
            "description": self.description,
            "recommended_timeframes": self.recommended_timeframes,
            "risk_level": self.risk_level,
            "market_type": self.market_type,
        }

    def _last(self, data: pd.DataFrame, col: str, default: float = 0.0) -> float:
        """Возвращает последнее значение колонки или default."""
        if col in data.columns and len(data) > 0:
            val = data[col].iloc[-1]
            return float(val) if pd.notna(val) else default
        return default

    def _prev(self, data: pd.DataFrame, col: str, default: float = 0.0) -> float:
        """Возвращает предпоследнее значение колонки или default."""
        if col in data.columns and len(data) > 1:
            val = data[col].iloc[-2]
            return float(val) if pd.notna(val) else default
        return default

    def _prev2(self, data: pd.DataFrame, col: str, default: float = 0.0) -> float:
        """Возвращает третье с конца значение колонки или default."""
        if col in data.columns and len(data) > 2:
            val = data[col].iloc[-3]
            return float(val) if pd.notna(val) else default
        return default


# ─────────────────────────────────────────────────────────────────────────────
# 1. EMA CROSSOVER
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# 2. RSI MOMENTUM
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# 3. MACD CROSSOVER
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# 4. BOLLINGER BANDS
# ─────────────────────────────────────────────────────────────────────────────


class BollingerBandsStrategy(BaseStrategy):
    """
    Стратегия Bollinger Bands с подтверждением RSI.

    Покупка при касании нижней BB + RSI < 40,
    продажа при касании верхней BB + RSI > 60.
    """

    name = "bollinger_bands"
    description = (
        "Покупка при касании нижней BB + RSI < 40, "
        "продажа при касании верхней BB + RSI > 60. "
        "Работает в волатильных рынках с чёткими уровнями."
    )
    recommended_timeframes = ["15m", "1h", "4h"]
    risk_level = "medium"
    market_type = "volatile"

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Генерирует сигнал на основе положения цены относительно полос Боллинджера.

        :param data: DataFrame с индикаторами bb_upper, bb_lower, rsi.
        :return: Сигнал с action и confidence.
        """
        close = self._last(data, "close")
        bb_u = self._last(data, "bb_upper", close * 1.02)
        bb_l = self._last(data, "bb_lower", close * 0.98)
        rsi = self._last(data, "rsi", 50)

        band_width = bb_u - bb_l
        if band_width == 0:
            return {"action": "hold", "confidence": 0.50}
        pos = (close - bb_l) / band_width  # 0 = нижняя, 1 = верхняя

        if close <= bb_l and rsi < 30:
            return {"action": "buy", "confidence": 0.90, "price": close}
        if close <= bb_l and rsi < 40:
            return {"action": "buy", "confidence": 0.75, "price": close}
        if pos < 0.2 and rsi < 45:
            return {"action": "buy", "confidence": 0.60, "price": close}
        if close >= bb_u and rsi > 70:
            return {"action": "sell", "confidence": 0.90, "price": close}
        if close >= bb_u and rsi > 60:
            return {"action": "sell", "confidence": 0.75, "price": close}
        if pos > 0.8 and rsi > 55:
            return {"action": "sell", "confidence": 0.60, "price": close}
        return {"action": "hold", "confidence": 0.50}


# ─────────────────────────────────────────────────────────────────────────────
# 5. SCALPING
# ─────────────────────────────────────────────────────────────────────────────


class ScalpingStrategy(BaseStrategy):
    """
    Стратегия скальпинга с использованием RSI, EMA и объёма.

    Работает на 1m–15m, требует высокого volume_ratio (>1.5).
    Высокий риск, высокая частота сделок.
    """

    name = "scalping"
    description = (
        "Микросделки с использованием RSI, EMA и объёма. "
        "Работает на 1m–15m, требует высокого volume_ratio (>1.5). "
        "Высокий риск, высокая частота сделок."
    )
    recommended_timeframes = ["1m", "5m", "15m"]
    risk_level = "high"
    market_type = "volatile"

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Генерирует скальпинговый сигнал на основе RSI, EMA и объёма.

        :param data: DataFrame с индикаторами rsi, ema_short, volume_ratio, macd.
        :return: Сигнал с action и confidence.
        """
        close = self._last(data, "close")
        rsi = self._last(data, "rsi", 50)
        ema_s = self._last(data, "ema_short", close)
        vol_ratio = self._last(data, "volume_ratio", 1.0)
        macd = self._last(data, "macd")
        macd_sig = self._last(data, "macd_signal")

        high_vol = vol_ratio > 1.5

        # Пороги 32/68 почти никогда не совпадают с close>ema_s / macd>macd_sig:
        # пока RSI ещё глубоко перепродан, цена и MACD обычно ещё не развернулись.
        # 40/60 ловит подтверждённый разворот, а не гипотетическое "ещё дно".
        if rsi < 40 and close > ema_s and macd > macd_sig:
            conf = 0.75 if high_vol else 0.58
            return {"action": "buy", "confidence": conf, "price": close}
        if rsi > 60 and close < ema_s and macd < macd_sig:
            conf = 0.75 if high_vol else 0.58
            return {"action": "sell", "confidence": conf, "price": close}
        return {"action": "hold", "confidence": 0.40}


# ─────────────────────────────────────────────────────────────────────────────
# 6. SWING TRADING
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# 7. BREAKOUT
# ─────────────────────────────────────────────────────────────────────────────


class BreakoutStrategy(BaseStrategy):
    """
    Стратегия прорыва уровней поддержки/сопротивления.

    Покупка при пробое 20-свечного максимума с высоким объёмом,
    продажа при пробое минимума.
    """

    name = "breakout"
    description = (
        "Покупка при пробое 20-свечного максимума с высоким объёмом, "
        "продажа при пробое минимума. "
        "Работает на трендовых движениях с волатильностью."
    )
    recommended_timeframes = ["1h", "4h"]
    risk_level = "high"
    market_type = "volatile"

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Генерирует сигнал прорыва исторических уровней.

        :param data: DataFrame с OHLCV и volume_ratio.
        :return: Сигнал с action и confidence.
        """
        if len(data) < 21:
            return {"action": "hold", "confidence": 0.50}

        close = self._last(data, "close")
        vol_ratio = self._last(data, "volume_ratio", 1.0)

        prev = data.iloc[-21:-1]
        resistance = float(prev["high"].max())
        support = float(prev["low"].min())

        high_vol = vol_ratio > 1.8

        if close > resistance:
            conf = 0.85 if high_vol else 0.65
            return {"action": "buy", "confidence": conf, "price": close}
        if close < support:
            conf = 0.85 if high_vol else 0.65
            return {"action": "sell", "confidence": conf, "price": close}
        return {"action": "hold", "confidence": 0.40}


# ─────────────────────────────────────────────────────────────────────────────
# 8. MEAN REVERSION
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# 9. TREND FOLLOWING
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# 10. VOLUME SPIKE
# ─────────────────────────────────────────────────────────────────────────────


class VolumeSpikeStrategy(BaseStrategy):
    """
    Стратегия на аномальном всплеске объёма.

    Входит в сделку когда объём текущей свечи в N раз превышает
    среднее за 20 свечей (N = VOLUME_SPIKE_THRESHOLD, по умолчанию 2.5).
    Направление определяется по EMA и RSI.
    Эффективна на 5m–15m, особенно на альткоинах.
    """

    name = "volume_spike"
    description = (
        "Вход при аномальном росте объёма (x2.5+ от среднего за 20 свечей). "
        "Направление по EMA + RSI. Лучший таймфрейм: 5m–15m, альткоины."
    )
    recommended_timeframes = ["5m", "15m"]
    risk_level = "high"
    market_type = "volatile"

    # Переопределяется подклассами для профиль-специфичных порогов (0 = из Config)
    _threshold: float = 0.0

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        if len(data) < 21:
            return {"action": "hold", "confidence": 0.50}

        close = self._last(data, "close")
        ema_s = self._last(data, "ema_short", close)
        rsi = self._last(data, "rsi", 50)

        threshold = self._threshold or Config.VOLUME_SPIKE_THRESHOLD
        avg_vol = float(data["volume"].iloc[-21:-1].mean())
        cur_vol = float(data["volume"].iloc[-1])

        if avg_vol <= 0:
            return {"action": "hold", "confidence": 0.50}

        ratio = cur_vol / avg_vol
        if ratio < threshold:
            return {"action": "hold", "confidence": 0.40}

        # Уверенность растёт пропорционально силе спайка (cap 0.90)
        spike_bonus = min((ratio - threshold) * 0.05, 0.15)

        if close > ema_s and rsi < 70:
            conf = round(min(0.75 + spike_bonus, 0.90), 2)
            return {"action": "buy", "confidence": conf, "price": close}
        if close < ema_s and rsi > 30:
            conf = round(min(0.75 + spike_bonus, 0.90), 2)
            return {"action": "sell", "confidence": conf, "price": close}
        return {"action": "hold", "confidence": 0.45}


# ─────────────────────────────────────────────────────────────────────────────
# 11. VOLUME SPIKE BLUECHIP
# ─────────────────────────────────────────────────────────────────────────────


class VolumeSpikeBlueChipStrategy(VolumeSpikeStrategy):
    """
    Volume spike для блючип-профиля.

    Порог 2.0x — BTC/ETH торгуются миллиарды в сутки, даже 2x превышение
    среднего уже означает нетипичную активность (новость, институционал).
    Оптимальный таймфрейм: 15m–1h.
    """

    name = "volume_spike_bluechip"
    description = (
        "Вход при 2x+ всплеске объёма на блючипах. "
        "Направление по EMA + RSI. Лучший таймфрейм: 15m–1h."
    )
    recommended_timeframes = ["15m", "1h"]
    risk_level = "medium"
    market_type = "volatile"

    _threshold: float = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# 12. VOLUME SPIKE ALTCOIN
# ─────────────────────────────────────────────────────────────────────────────


class VolumeSpikeAltcoinStrategy(VolumeSpikeStrategy):
    """
    Volume spike для альткоин-профиля.

    Порог 3.0x — у альткоинов больше случайного шума в объёме,
    нужен более строгий фильтр чтобы не входить на ложных спайках.
    Оптимальный таймфрейм: 5m–15m.
    """

    name = "volume_spike_altcoin"
    description = (
        "Вход при 3x+ всплеске объёма на альткоинах. "
        "Направление по EMA + RSI. Лучший таймфрейм: 5m–15m."
    )
    recommended_timeframes = ["5m", "15m"]
    risk_level = "high"
    market_type = "volatile"

    _threshold: float = 3.0


# ─────────────────────────────────────────────────────────────────────────────
# 13. VOLUME SPIKE MEME
# ─────────────────────────────────────────────────────────────────────────────


class VolumeSpikeMemStrategy(VolumeSpikeStrategy):
    """
    Агрессивный volume spike для мемкоинов.

    Порог 5x вместо 2.5x — мемы часто имеют фоновый «шум» объёма,
    настоящий памп виден только при кратном превышении.
    Оптимальный таймфрейм: 1m–3m.
    """

    name = "volume_spike_meme"
    description = (
        "Вход при 5x+ всплеске объёма. Порог вдвое агрессивнее стандартного. "
        "Направление по EMA + RSI. Лучший таймфрейм: 1m–3m, мемкоины."
    )
    recommended_timeframes = ["1m", "3m"]
    risk_level = "high"
    market_type = "volatile"

    _threshold: float = 5.0


# ─────────────────────────────────────────────────────────────────────────────
# 12. MOMENTUM BURST
# ─────────────────────────────────────────────────────────────────────────────


class MomentumBurstStrategy(BaseStrategy):
    """
    Стратегия резкого ценового импульса за 3 свечи.

    Входим в продолжение памп-движения (+4%+ за 3 свечи) при подтверждении
    объёмом. Предназначена для 1m–3m, мемкоины и высоковолатильные альткоины.
    """

    name = "momentum_burst"
    description = (
        "Вход при резком ценовом импульсе (+4%+ за 3 свечи) с объёмным подтверждением. "
        "Предназначена для 1m–3m таймфреймов, мемкоины."
    )
    recommended_timeframes = ["1m", "3m"]
    risk_level = "high"
    market_type = "volatile"

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        if len(data) < 4:
            return {"action": "hold", "confidence": 0.50}

        close = self._last(data, "close")
        close_3 = float(data["close"].iloc[-4])
        if close_3 <= 0:
            return {"action": "hold", "confidence": 0.50}

        move_3 = (close - close_3) / close_3

        avg_vol = (
            float(data["volume"].iloc[-21:-1].mean())
            if len(data) >= 21
            else float(data["volume"].mean())
        )
        cur_vol = self._last(data, "volume")
        vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 0.0

        rsi = self._last(data, "rsi", 50)

        # Сильный импульс вверх с объёмным подтверждением — входим в продолжение
        if move_3 >= 0.04 and vol_ratio >= 2.0 and rsi < 80:
            bonus = min((move_3 - 0.04) * 5, 0.15)
            conf = round(min(0.72 + bonus, 0.90), 2)
            return {"action": "buy", "confidence": conf, "price": close}
        if move_3 >= 0.02 and vol_ratio >= 1.5 and rsi < 75:
            return {"action": "buy", "confidence": 0.62, "price": close}

        # Резкое падение с объёмом → шорт
        if move_3 <= -0.04 and vol_ratio >= 2.0 and rsi > 20:
            bonus = min((abs(move_3) - 0.04) * 5, 0.15)
            conf = round(min(0.72 + bonus, 0.90), 2)
            return {"action": "sell", "confidence": conf, "price": close}

        return {"action": "hold", "confidence": 0.40}


# ─────────────────────────────────────────────────────────────────────────────
# 13. PUMP EXIT
# ─────────────────────────────────────────────────────────────────────────────


class PumpExitStrategy(BaseStrategy):
    """
    Стратегия выхода из памп-позиции при угасании объёма.

    Когда объём падает ниже 40% от пикового значения за последние 5 свечей,
    а цена уже выросла на 5%+ — вероятен дамп. Сигнализирует sell.
    Работает как exit-стратегия в паре с volume_spike_meme и momentum_burst.
    """

    name = "pump_exit"
    description = (
        "Sell-сигнал при угасании памп-объёма: объём < 40% от пика последних 5 свечей "
        "и цена выросла на 5%+. Используется совместно с volume_spike_meme."
    )
    recommended_timeframes = ["1m", "3m", "5m"]
    risk_level = "high"
    market_type = "volatile"

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        if len(data) < 5:
            return {"action": "hold", "confidence": 0.50}

        close = self._last(data, "close")
        vols = list(data["volume"].iloc[-5:])
        peak_vol = max(vols[:-1])
        cur_vol = vols[-1]

        if peak_vol <= 0:
            return {"action": "hold", "confidence": 0.50}

        vol_decline = cur_vol / peak_vol

        avg_vol = (
            float(data["volume"].iloc[-21:-1].mean())
            if len(data) >= 21
            else float(data["volume"].mean())
        )

        recent_low = float(data["close"].iloc[-10:].min()) if len(data) >= 10 else close
        pump_move = (close - recent_low) / recent_low if recent_low > 0 else 0.0

        rsi = self._last(data, "rsi", 50)

        # Объём упал до <40% от пика + цена выросла на 5%+ → памп выдыхается
        if vol_decline < 0.4 and pump_move >= 0.05 and rsi > 60:
            return {"action": "sell", "confidence": 0.85, "price": close}

        # Объём упал до <60% от пика + умеренный рост + ниже среднего → предупреждение
        if vol_decline < 0.6 and pump_move >= 0.03 and cur_vol < avg_vol:
            return {"action": "sell", "confidence": 0.68, "price": close}

        return {"action": "hold", "confidence": 0.40}


# ─────────────────────────────────────────────────────────────────────────────
# 14. SENTIMENT
# ─────────────────────────────────────────────────────────────────────────────


class SentimentStrategy(BaseStrategy):
    """
    Стратегия сентимента для мемкоинов.

    Читает колонку social_sentiment, инжектированную TradingBot из SentimentPoller
    (взвешенная оценка Twitter + Telegram, -1.0..1.0).
    Hold если колонка отсутствует или оценка близка к нулю.
    """

    name = "sentiment"
    description = (
        "Сигнал buy/sell из оценки сентимента Twitter + Telegram. "
        "Предназначена для профиля мемкоинов. Требует запущенный SentimentPoller."
    )
    recommended_timeframes = ["1m", "3m", "5m"]
    risk_level = "high"
    market_type = "volatile"

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        close = self._last(data, "close")
        score = self._last(data, "social_sentiment", 0.0)

        buy_threshold = Config.SENTIMENT_BUY_THRESHOLD
        sell_threshold = Config.SENTIMENT_SELL_THRESHOLD

        if score >= buy_threshold:
            conf = round(min(0.60 + (score - buy_threshold) * 0.5, 0.88), 2)
            return {"action": "buy", "confidence": conf, "price": close}
        if score <= sell_threshold:
            conf = round(min(0.60 + (abs(score) - abs(sell_threshold)) * 0.5, 0.85), 2)
            return {"action": "sell", "confidence": conf, "price": close}
        return {"action": "hold", "confidence": 0.40}


# ─────────────────────────────────────────────────────────────────────────────
# 15. SUPPORT / RESISTANCE (ICT tier-1)
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# 16. FIBONACCI RETRACEMENT (ICT tier-1)
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# 17. CVD DIVERGENCE (ICT tier-1, приближённый — не тик-данные)
# ─────────────────────────────────────────────────────────────────────────────


class CVDDivergenceStrategy(BaseStrategy):
    """
    Дивергенция кумулятивной дельты объёма (приближённый CVD).

    Настоящий order-flow CVD требует тик-данных с классификацией
    покупок/продаж на уровне сделок — в OHLCV-свечах их нет. Здесь дельта
    свечи приближается направлением тела (+volume если close>open, -volume
    если close<open) — грубее честного footprint-CVD, поэтому явно
    помечено как приближение, а не тик-точный расчёт. Ищет классическую
    дивергенцию: цена обновляет экстремум, а приближённый CVD — нет
    (накопление/распределение против текущего движения цены).
    """

    name = "cvd_divergence"
    description = (
        "Дивергенция цены и приближённой кумулятивной дельты объёма "
        "(по направлению тела свечи, не тик-данные). Разворотный сигнал."
    )
    recommended_timeframes = ["15m", "1h"]
    risk_level = "high"
    market_type = "any"

    _LOOKBACK = 30

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        if len(data) < self._LOOKBACK:
            return {"action": "hold", "confidence": 0.50}

        window = data.iloc[-self._LOOKBACK :]
        close = self._last(data, "close")
        rsi = self._last(data, "rsi", 50)

        direction = (window["close"] - window["open"]).apply(
            lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
        )
        cvd = (direction * window["volume"]).cumsum()

        half = self._LOOKBACK // 2
        price_first, price_second = (
            window["close"].iloc[:half],
            window["close"].iloc[half:],
        )
        cvd_first, cvd_second = cvd.iloc[:half], cvd.iloc[half:]
        avg_vol = float(window["volume"].mean()) * half + 1e-9

        price_new_low = price_second.min() < price_first.min()
        cvd_higher_low = cvd_second.min() > cvd_first.min()
        price_new_high = price_second.max() > price_first.max()
        cvd_lower_high = cvd_second.max() < cvd_first.max()

        if price_new_low and cvd_higher_low and rsi < 50:
            spread = abs(float(cvd_second.min() - cvd_first.min()))
            conf = round(min(0.60 + min(spread / avg_vol, 1.0) * 0.2, 0.82), 2)
            return {"action": "buy", "confidence": conf, "price": close}

        if price_new_high and cvd_lower_high and rsi > 50:
            spread = abs(float(cvd_first.max() - cvd_second.max()))
            conf = round(min(0.60 + min(spread / avg_vol, 1.0) * 0.2, 0.82), 2)
            return {"action": "sell", "confidence": conf, "price": close}

        return {"action": "hold", "confidence": 0.40}


# ─────────────────────────────────────────────────────────────────────────────
# 18. FAIR VALUE GAP (ICT tier-1)
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# 19. INVERSE FAIR VALUE GAP (ICT tier-1)
# ─────────────────────────────────────────────────────────────────────────────


class InverseFairValueGapStrategy(BaseStrategy):
    """
    Вход на ретесте инвертированного Fair Value Gap (IFVG, ICT).

    Когда FVG пробивается насквозь (цена закрывается по другую сторону
    гэпа), он теряет исходную полярность: пробитый вниз бычий FVG
    становится сопротивлением, пробитый вверх медвежий — поддержкой.
    Стратегия ищет самый свежий насквозь пробитый гэп за lookback свечей
    и торгует отскок при ретесте его границ с уже развёрнутой стороны —
    по направлению пробоя, а не исходного гэпа. Более рискованная и
    контр-трендовая по духу, чем FairValueGapStrategy выше, поэтому
    отдельная стратегия, а не режим той же.
    """

    name = "inverse_fair_value_gap"
    description = (
        "Вход на ретесте насквозь пробитого (инвертированного) Fair "
        "Value Gap (ICT) — по направлению пробоя, полярность зоны развёрнута."
    )
    recommended_timeframes = ["15m", "1h", "4h"]
    risk_level = "high"
    market_type = "volatile"

    _LOOKBACK = 40

    def _find_last_inverted_fvg(
        self, window: pd.DataFrame
    ) -> Optional[Tuple[str, float, float, int]]:
        """Ищет самый свежий насквозь пробитый (инвертированный) FVG.

        :return: (новое_направление_после_инверсии, low_зоны, high_зоны,
            возраст_пробоя_в_свечах) или None.
        """
        n = len(window)
        for i in range(n - 3, -1, -1):
            c1 = window.iloc[i]
            c3 = window.iloc[i + 2]
            if c1["high"] < c3["low"]:
                zone_lo, zone_hi, origin = c1["high"], c3["low"], "bullish"
            elif c1["low"] > c3["high"]:
                zone_lo, zone_hi, origin = c3["high"], c1["low"], "bearish"
            else:
                continue

            breach_idx = None
            for j in range(i + 3, n):
                c = window.iloc[j]
                if origin == "bullish" and c["close"] < zone_lo:
                    breach_idx = j
                    break
                if origin == "bearish" and c["close"] > zone_hi:
                    breach_idx = j
                    break
            if breach_idx is None:
                continue

            age = n - 1 - breach_idx
            new_direction = "bearish" if origin == "bullish" else "bullish"
            return new_direction, float(zone_lo), float(zone_hi), age
        return None

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        if len(data) < self._LOOKBACK:
            return {"action": "hold", "confidence": 0.50}

        window = data.iloc[-self._LOOKBACK :]
        close = self._last(data, "close")
        rsi = self._last(data, "rsi", 50)

        ifvg = self._find_last_inverted_fvg(window)
        if ifvg is None:
            return {"action": "hold", "confidence": 0.40}
        direction, zone_lo, zone_hi, age = ifvg

        if not (zone_lo <= close <= zone_hi):
            return {"action": "hold", "confidence": 0.40}

        freshness = max(0.0, 1.0 - age / self._LOOKBACK)

        if direction == "bearish" and rsi > 45:
            conf = round(min(0.55 + freshness * 0.20, 0.80), 2)
            return {"action": "sell", "confidence": conf, "price": close}

        if direction == "bullish" and rsi < 55:
            conf = round(min(0.55 + freshness * 0.20, 0.80), 2)
            return {"action": "buy", "confidence": conf, "price": close}

        return {"action": "hold", "confidence": 0.45}


# ─────────────────────────────────────────────────────────────────────────────
# 20. LIQUIDITY SWEEP (ICT tier-1)
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# 21. AMD — ACCUMULATION / MANIPULATION / DISTRIBUTION (ICT tier-1)
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {
    "ema_crossover": EMACrossoverStrategy,
    "rsi_momentum": RSIMomentumStrategy,
    "macd_crossover": MACDCrossoverStrategy,
    "bollinger_bands": BollingerBandsStrategy,
    "scalping": ScalpingStrategy,
    "swing_trading": SwingTradingStrategy,
    "breakout": BreakoutStrategy,
    "mean_reversion": MeanReversionStrategy,
    "trend_following": TrendFollowingStrategy,
    "volume_spike": VolumeSpikeStrategy,
    # profile-specific volume spike variants
    "volume_spike_bluechip": VolumeSpikeBlueChipStrategy,
    "volume_spike_altcoin": VolumeSpikeAltcoinStrategy,
    "volume_spike_meme": VolumeSpikeMemStrategy,
    # meme-profile strategies
    "momentum_burst": MomentumBurstStrategy,
    "pump_exit": PumpExitStrategy,
    "sentiment": SentimentStrategy,
    # ICT tier-1
    "support_resistance": SupportResistanceStrategy,
    "fibonacci_retracement": FibonacciRetracementStrategy,
    "cvd_divergence": CVDDivergenceStrategy,
    "fair_value_gap": FairValueGapStrategy,
    "inverse_fair_value_gap": InverseFairValueGapStrategy,
    "liquidity_sweep": LiquiditySweepStrategy,
    "amd": AMDStrategy,
}


def get_all_strategies() -> List[Dict[str, Any]]:
    """Возвращает описания всех доступных стратегий."""
    return [cls().get_info() for cls in STRATEGY_REGISTRY.values()]


def create_strategy(name: str) -> BaseStrategy:
    """
    Создаёт экземпляр стратегии по имени.

    :param name: Название стратегии из STRATEGY_REGISTRY.
    :return: Экземпляр стратегии.
    :raises ValueError: Если стратегия с таким именем не найдена.
    """
    if name not in STRATEGY_REGISTRY:
        available = list(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Неизвестная стратегия '{name}'. Доступные: {available}")
    return STRATEGY_REGISTRY[name]()


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────


class TradingStrategy:
    """
    Оркестратор стратегий.

    Управляет выбором активной стратегии, генерацией сигналов и сохранением
    состояния в Redis. Поддерживает переключение стратегии на лету через
    switch_strategy() — используется при AI-рекомендации или выборе пользователя.
    """

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.redis = RedisClient()
        self.strategy: BaseStrategy = create_strategy(strategy_name)
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Инициализирует оркестратор и логирует информацию об активной стратегии."""
        self.logger.info(
            "Strategy initialized: %s | risk=%s | timeframes=%s",
            self.strategy_name,
            self.strategy.risk_level,
            self.strategy.recommended_timeframes,
        )

    def switch_strategy(self, new_name: str):
        """
        Переключает активную стратегию без перезапуска бота.

        :param new_name: Название новой стратегии.
        """
        self.strategy = create_strategy(new_name)
        self.strategy_name = new_name
        self.logger.info("Switched strategy → %s", new_name)

    async def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Генерирует торговый сигнал и сохраняет состояние в Redis.

        :param data: DataFrame с OHLCV и индикаторами.
        :return: Сигнал с ключами action, confidence и опционально price.
        """
        signal = self.strategy.generate_signal(data)

        state = {
            "last_signal": signal,
            "timestamp": pd.Timestamp.now().isoformat(),
            "strategy": self.strategy_name,
            "market_conditions": {
                "price": float(data["close"].iloc[-1]),
                "volume": (
                    float(data["volume"].iloc[-1]) if "volume" in data.columns else 0
                ),
                "volatility": float(data["close"].std()),
            },
        }
        self.redis.save_trading_state(self.strategy_name, state)
        self.redis.publish_signal(
            {
                "strategy": self.strategy_name,
                "signal": signal,
                "timestamp": state["timestamp"],
            }
        )
        return signal

    @staticmethod
    def list_strategies() -> List[Dict[str, Any]]:
        """Возвращает список всех стратегий с их описаниями."""
        return get_all_strategies()

    def get_current_strategy_info(self) -> Dict[str, Any]:
        """Возвращает информацию об активной стратегии."""
        return self.strategy.get_info()
