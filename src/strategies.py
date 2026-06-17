import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type

import pandas as pd

from .redis_client import RedisClient

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Базовый класс для всех торговых стратегий."""

    name: str = ""
    description: str = ""
    recommended_timeframes: List[str] = []
    risk_level: str = "medium"   # low / medium / high
    market_type: str = "any"     # trending / ranging / volatile / any

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        pass

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "recommended_timeframes": self.recommended_timeframes,
            "risk_level": self.risk_level,
            "market_type": self.market_type,
        }

    def _last(self, data: pd.DataFrame, col: str, default: float = 0.0) -> float:
        if col in data.columns and len(data) > 0:
            val = data[col].iloc[-1]
            return float(val) if pd.notna(val) else default
        return default

    def _prev(self, data: pd.DataFrame, col: str, default: float = 0.0) -> float:
        if col in data.columns and len(data) > 1:
            val = data[col].iloc[-2]
            return float(val) if pd.notna(val) else default
        return default


# ─────────────────────────────────────────────────────────────────────────────
# 1. EMA CROSSOVER
# ─────────────────────────────────────────────────────────────────────────────

class EMACrossoverStrategy(BaseStrategy):
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
        close = self._last(data, "close")
        ema_s = self._last(data, "ema_short", close)
        ema_l = self._last(data, "ema_long", close)
        ema_s_p = self._prev(data, "ema_short", ema_s)
        ema_l_p = self._prev(data, "ema_long", ema_l)

        bullish_cross = ema_s > ema_l and ema_s_p <= ema_l_p
        bearish_cross = ema_s < ema_l and ema_s_p >= ema_l_p

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
    name = "rsi_momentum"
    description = (
        "Покупка при RSI < 30 (перепроданность), продажа при RSI > 70 (перекупленность). "
        "Оптимальна при боковом движении цены."
    )
    recommended_timeframes = ["15m", "1h", "4h"]
    risk_level = "medium"
    market_type = "ranging"

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        close = self._last(data, "close")
        rsi = self._last(data, "rsi", 50)

        if rsi < 25:
            return {"action": "buy", "confidence": 0.90, "price": close}
        if rsi < 30:
            return {"action": "buy", "confidence": 0.78, "price": close}
        if rsi < 40:
            return {"action": "buy", "confidence": 0.57, "price": close}
        if rsi > 75:
            return {"action": "sell", "confidence": 0.90, "price": close}
        if rsi > 70:
            return {"action": "sell", "confidence": 0.78, "price": close}
        if rsi > 60:
            return {"action": "sell", "confidence": 0.57, "price": close}
        return {"action": "hold", "confidence": 0.50}


# ─────────────────────────────────────────────────────────────────────────────
# 3. MACD CROSSOVER
# ─────────────────────────────────────────────────────────────────────────────

class MACDCrossoverStrategy(BaseStrategy):
    name = "macd_crossover"
    description = (
        "Покупка при бычьем кроссовере MACD/Signal выше нуля, "
        "продажа при медвежьем. Сила сигнала зависит от положения относительно нулевой линии."
    )
    recommended_timeframes = ["1h", "4h", "1d"]
    risk_level = "medium"
    market_type = "trending"

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
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
        close = self._last(data, "close")
        bb_u = self._last(data, "bb_upper", close * 1.02)
        bb_l = self._last(data, "bb_lower", close * 0.98)
        bb_m = self._last(data, "bb_middle", close)
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
        close = self._last(data, "close")
        rsi = self._last(data, "rsi", 50)
        ema_s = self._last(data, "ema_short", close)
        vol_ratio = self._last(data, "volume_ratio", 1.0)
        macd = self._last(data, "macd")
        macd_sig = self._last(data, "macd_signal")

        high_vol = vol_ratio > 1.5

        if rsi < 32 and close > ema_s and macd > macd_sig:
            conf = 0.75 if high_vol else 0.58
            return {"action": "buy", "confidence": conf, "price": close}
        if rsi > 68 and close < ema_s and macd < macd_sig:
            conf = 0.75 if high_vol else 0.58
            return {"action": "sell", "confidence": conf, "price": close}
        return {"action": "hold", "confidence": 0.40}


# ─────────────────────────────────────────────────────────────────────────────
# 6. SWING TRADING
# ─────────────────────────────────────────────────────────────────────────────

class SwingTradingStrategy(BaseStrategy):
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
        close = self._last(data, "close")
        rsi = self._last(data, "rsi", 50)
        bb_u = self._last(data, "bb_upper", close * 1.02)
        bb_l = self._last(data, "bb_lower", close * 0.98)
        bb_m = self._last(data, "bb_middle", close)
        mom = self._last(data, "momentum", 0)

        extreme_low = close < bb_l and rsi < 30
        extreme_high = close > bb_u and rsi > 70

        if extreme_low and mom < 0:
            return {"action": "buy", "confidence": 0.88, "price": close}
        if extreme_low:
            return {"action": "buy", "confidence": 0.72, "price": close}
        if close < bb_l:
            return {"action": "buy", "confidence": 0.62, "price": close}
        if extreme_high and mom > 0:
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
    name = "trend_following"
    description = (
        "Открытие позиции только при подтверждении тренда: SMA20 > SMA50, EMA12 > EMA26, цена выше SMA20. "
        "Консервативная стратегия с низким риском."
    )
    recommended_timeframes = ["4h", "1d"]
    risk_level = "low"
    market_type = "trending"

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
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
# REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {
    "ema_crossover":   EMACrossoverStrategy,
    "rsi_momentum":    RSIMomentumStrategy,
    "macd_crossover":  MACDCrossoverStrategy,
    "bollinger_bands": BollingerBandsStrategy,
    "scalping":        ScalpingStrategy,
    "swing_trading":   SwingTradingStrategy,
    "breakout":        BreakoutStrategy,
    "mean_reversion":  MeanReversionStrategy,
    "trend_following": TrendFollowingStrategy,
}


def get_all_strategies() -> List[Dict[str, Any]]:
    """Возвращает описания всех доступных стратегий."""
    return [cls().get_info() for cls in STRATEGY_REGISTRY.values()]


def create_strategy(name: str) -> BaseStrategy:
    """Создаёт экземпляр стратегии по имени."""
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
        self.logger.info(
            f"Strategy initialized: {self.strategy_name} | "
            f"risk={self.strategy.risk_level} | "
            f"timeframes={self.strategy.recommended_timeframes}"
        )

    def switch_strategy(self, new_name: str):
        """Переключает активную стратегию без перезапуска бота."""
        self.strategy = create_strategy(new_name)
        self.strategy_name = new_name
        self.logger.info(f"Switched strategy → {new_name}")

    async def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Генерирует торговый сигнал и сохраняет состояние в Redis."""
        signal = self.strategy.generate_signal(data)

        state = {
            "last_signal": signal,
            "timestamp": pd.Timestamp.now().isoformat(),
            "strategy": self.strategy_name,
            "market_conditions": {
                "price": float(data["close"].iloc[-1]),
                "volume": float(data["volume"].iloc[-1]) if "volume" in data.columns else 0,
                "volatility": float(data["close"].std()),
            },
        }
        self.redis.save_trading_state(self.strategy_name, state)
        self.redis.publish_signal({
            "strategy": self.strategy_name,
            "signal": signal,
            "timestamp": state["timestamp"],
        })
        return signal

    @staticmethod
    def list_strategies() -> List[Dict[str, Any]]:
        """Возвращает список всех стратегий с их описаниями."""
        return get_all_strategies()

    def get_current_strategy_info(self) -> Dict[str, Any]:
        """Возвращает информацию об активной стратегии."""
        return self.strategy.get_info()
