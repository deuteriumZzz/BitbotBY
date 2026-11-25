"""Основные стратегии для волатильного рынка: включены по умолчанию.
"""

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from bitbot.config.settings import Config
from bitbot.domain.strategies.base import BaseStrategy


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

        high_vol = vol_ratio > self.param("high_volume", 1.5)

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

        high_vol = vol_ratio > self.param("high_volume", 1.8)

        if close > resistance:
            conf = 0.85 if high_vol else 0.65
            return {"action": "buy", "confidence": conf, "price": close}
        if close < support:
            conf = 0.85 if high_vol else 0.65
            return {"action": "sell", "confidence": conf, "price": close}
        return {"action": "hold", "confidence": 0.40}


class VolumeSpikeStrategy(BaseStrategy):
    """
    Стратегия на аномальном всплеске объёма.
    Входит в сделку когда объём текущей свечи в N раз превышает
    среднее за 20 свечей. N зависит от профиля монет (SAC_PROFILE):
    Bluechip 2.0, Altcoin 3.0, Meme 5.0; без профиля — VOLUME_SPIKE_THRESHOLD
    (по умолчанию 2.5). Направление определяется по EMA и RSI.
    Эффективна на 5m–15m, особенно на альткоинах.

    Раньше это были четыре стратегии (по одной на профиль) с одинаковой
    логикой и разным порогом — теперь одна, а профиль задаёт только порог.
    """

    name = "volume_spike"
    description = (
        "Вход при аномальном росте объёма: порог зависит от профиля монет "
        "(Bluechip 2x, Altcoin 3x, Meme 5x), иначе x2.5. "
        "Направление по EMA + RSI. Лучший таймфрейм: 5m–15m."
    )
    recommended_timeframes = ["5m", "15m"]
    risk_level = "high"
    market_type = "volatile"

    @staticmethod
    def threshold() -> float:
        """Порог всплеска для текущего профиля монет."""
        return VolumeSpikeStrategy().param("spike", Config.VOLUME_SPIKE_THRESHOLD)

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        if len(data) < 21:
            return {"action": "hold", "confidence": 0.50}

        close = self._last(data, "close")
        ema_s = self._last(data, "ema_short", close)
        rsi = self._last(data, "rsi", 50)

        threshold = self.threshold()
        avg_vol = float(data["volume"].iloc[-21:-1].mean())
        cur_vol = float(data["volume"].iloc[-1])

        if avg_vol <= 0:
            return {"action": "hold", "confidence": 0.50}

        ratio = cur_vol / avg_vol
        if ratio < threshold:
            return {"action": "hold", "confidence": 0.40}

        # Уверенность считается от АБСОЛЮТНОГО ratio, а не от (ratio -
        # threshold): иначе вариант с более низким порогом систематически
        # получал бы равную или бОльшую confidence при любом одинаковом
        # ratio и всегда побеждал бы в турнире recommend_strategy_local —
        # смысл более строгих профильных порогов (altcoin/meme) обнулялся
        # бы. Теперь одинаковый по силе реальный спайк даёт одинаковую
        # confidence независимо от того, какой профильный вариант его поймал.
        conf = round(min(0.65 + (ratio - 1.0) * 0.05, 0.90), 2)

        if close > ema_s and rsi < 70:
            return {"action": "buy", "confidence": conf, "price": close}
        if close < ema_s and rsi > 30:
            return {"action": "sell", "confidence": conf, "price": close}
        return {"action": "hold", "confidence": 0.45}


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
        strong, strong_vol = self.param("strong_move", 0.04), self.param(
            "strong_volume", 2.0
        )
        weak, weak_vol = self.param("weak_move", 0.02), self.param("weak_volume", 1.5)

        # Сильный импульс вверх с объёмным подтверждением — входим в продолжение
        if move_3 >= strong and vol_ratio >= strong_vol and rsi < 80:
            bonus = min((move_3 - strong) * 5, 0.15)
            conf = round(min(0.72 + bonus, 0.90), 2)
            return {"action": "buy", "confidence": conf, "price": close}
        if move_3 >= weak and vol_ratio >= weak_vol and rsi < 75:
            return {"action": "buy", "confidence": 0.62, "price": close}

        # Резкое падение с объёмом → шорт
        if move_3 <= -strong and vol_ratio >= strong_vol and rsi > 20:
            bonus = min((abs(move_3) - strong) * 5, 0.15)
            conf = round(min(0.72 + bonus, 0.90), 2)
            return {"action": "sell", "confidence": conf, "price": close}

        return {"action": "hold", "confidence": 0.40}


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
    markets = "futures"  # сигнал только sell: на споте это выход, а не вход

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
        if (
            vol_decline < 0.4
            and pump_move >= self.param("strong_pump", 0.05)
            and rsi > 60
        ):
            return {"action": "sell", "confidence": 0.85, "price": close}

        # Объём упал до <60% от пика + умеренный рост + ниже среднего → предупреждение
        if (
            vol_decline < 0.6
            and pump_move >= self.param("weak_pump", 0.03)
            and cur_vol < avg_vol
        ):
            return {"action": "sell", "confidence": 0.68, "price": close}

        return {"action": "hold", "confidence": 0.40}


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


# 19. INVERSE FAIR VALUE GAP (ICT tier-1)


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
