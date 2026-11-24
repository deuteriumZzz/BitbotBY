"""Стратегии: реестр, создание по имени и оркестратор TradingStrategy.

Стратегии разложены по семействам — по рынку, на котором работают (`market_type`):
trending/, ranging/, volatile/, any/. В каждом main.py — основные (включены по
умолчанию), в experimental.py и indicators.py — порты из Argent без доказанного
преимущества (классами и индикаторные; выключены по умолчанию, см. experimental_names()). База — base.py, адаптер
портов — ported_base.py.

Новая стратегия: класс в файле своего семейства + имя в _ORDER. Порядок реестра —
порядок в турнире локального отбора (при равной уверенности побеждает первая),
поэтому он задан явно, а не порядком импорта файлов. Имена не меняются: на них
завязаны настройки и история сделок.
"""

import logging
from typing import Any, Dict, List, Tuple, Type

import pandas as pd

from bitbot.domain.strategies.any import main as any_main
from bitbot.domain.strategies.base import BaseStrategy
from bitbot.domain.strategies.ranging import experimental as ranging_experimental
from bitbot.domain.strategies.ranging import indicators as ranging_indicators
from bitbot.domain.strategies.ranging import main as ranging_main
from bitbot.domain.strategies.trending import experimental as trending_experimental
from bitbot.domain.strategies.trending import indicators as trending_indicators
from bitbot.domain.strategies.trending import main as trending_main
from bitbot.domain.strategies.volatile import experimental as volatile_experimental
from bitbot.domain.strategies.volatile import indicators as volatile_indicators
from bitbot.domain.strategies.volatile import main as volatile_main
from bitbot.storage.redis_client import RedisClient

logger = logging.getLogger(__name__)


_FAMILIES = (
    trending_main,
    ranging_main,
    volatile_main,
    any_main,
    trending_experimental,
    ranging_experimental,
    volatile_experimental,
    trending_indicators,
    ranging_indicators,
    volatile_indicators,
)


# Порядок реестра = порядок в турнире (см. докстринг модуля).
_ORDER: Tuple[str, ...] = (
    "ema_crossover",
    "rsi_momentum",
    "macd_crossover",
    "bollinger_bands",
    "scalping",
    "swing_trading",
    "breakout",
    "mean_reversion",
    "trend_following",
    "volume_spike",
    "momentum_burst",
    "pump_exit",
    "sentiment",
    "support_resistance",
    "fibonacci_retracement",
    "cvd_divergence",
    "fair_value_gap",
    "inverse_fair_value_gap",
    "liquidity_sweep",
    "amd",
    "supertrend",
    "dual_thrust",
    "ttm_squeeze",
    "adx_trend_birth",
    "vwap_reversion",
    "crash_bounce",
    "bear_hedge",
    "ichimoku",
    "parabolic_sar",
    "vortex",
    "stoch_rsi",
    "williams_r",
    "keltner_fade",
    "obv_trend",
    "mfi_trend",
    "chaikin_osc",
    "coppock",
    "coral_trend",
    "kst",
    "rvi",
    "waddah_attar",
    "zlsma",
    "cci_aroon",
    "engulfing_trend",
    "chaikin_money_flow",
    "fisher_transform",
    "kdj",
    "klinger",
    "awesome_saucer",
    "heikin_ashi",
    "rsi_divergence",
    "shooting_star",
    "cci_reversion",
    "ifr2",
    "stochastic",
    "adx_dmi",
    "chandelier",
    "halftrend",
    "ssl_channel",
    "ad_line_trend",
    "ease_of_movement",
    "elder_force",
    "vwap_trend",
    "vw_macd",
    "poc_reversion",
    "bollinger_reentry",
    "dual_momentum",
    "gtaa_ma",
    "spike_reversal",
)


def _collect() -> Dict[str, Type[BaseStrategy]]:
    found: Dict[str, Type[BaseStrategy]] = {}
    for module in _FAMILIES:
        for obj in vars(module).values():
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseStrategy)
                and obj.__dict__.get("name")
            ):
                if found.setdefault(obj.name, obj) is not obj:
                    raise RuntimeError(f"две стратегии с именем {obj.name}")
    stray = set(found) ^ set(_ORDER)
    if stray:
        raise RuntimeError(f"стратегия не в _ORDER или без класса: {sorted(stray)}")
    return {name: found[name] for name in _ORDER}


STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = _collect()


# Профильные варианты volume_spike слиты в одну стратегию; старые имена остаются
# в истории сделок и в Redis, поэтому create_strategy() их по-прежнему понимает.
LEGACY_ALIASES: Dict[str, str] = {
    "volume_spike_bluechip": "volume_spike",
    "volume_spike_altcoin": "volume_spike",
    "volume_spike_meme": "volume_spike",
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
    name = LEGACY_ALIASES.get(name, name)
    if name not in STRATEGY_REGISTRY:
        available = list(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Неизвестная стратегия '{name}'. Доступные: {available}")
    return STRATEGY_REGISTRY[name]()


def experimental_names() -> List[str]:
    """Стратегии без доказанного преимущества: выключены по умолчанию."""
    return [n for n, c in STRATEGY_REGISTRY.items() if c.experimental]


# ORCHESTRATOR


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
