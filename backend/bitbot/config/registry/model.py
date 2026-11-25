"""Модель настройки (Setting) и помощники описания: группы, «сразу» через RuntimeConfig, варианты.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


HotSet = Callable[[Any, Any], Any]  # (RuntimeConfig, value) -> bool | None


HotGet = Callable[[Any], Any]  # (RuntimeConfig) -> value


@dataclass(frozen=True)
class Setting:
    key: str
    group: str
    label: str
    type: str  # int | float | pct (доля 0..1) | bool | choice | str
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    unit: str = ""
    options: Tuple[Tuple[str, str], ...] = ()
    hint: str = ""
    attr: Optional[str] = None  # атрибут Config, если имя отличается от key
    readonly: bool = False
    hot_set: Optional[HotSet] = field(default=None, compare=False, repr=False)
    hot_get: Optional[HotGet] = field(default=None, compare=False, repr=False)
    check: Optional[Callable[[Any], Optional[str]]] = field(
        default=None, compare=False, repr=False
    )

    @property
    def hot(self) -> bool:
        return self.hot_set is not None

    @property
    def config_attr(self) -> str:
        return self.attr or self.key

    def public(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("hot_set", "hot_get", "check", "attr"):
            d.pop(k, None)
        d["options"] = [{"value": v, "label": lbl} for v, lbl in self.options]
        d["hot"] = self.hot
        d["basic"] = self.key in BASIC
        return d


# Что видно в «Все параметры» сразу; остальное — по «Показать расширенные».
BASIC = frozenset(
    {
        "MARKET_TYPE",
        "TIMEFRAME",
        "SCAN_TOP_N",
        "TRADING_HOURS",
        "AUTO_EXECUTE",
        "LOCAL_FALLBACK",
        "PAPER_GUARDS_OFF",
        "TELEGRAM_CONFIRM_TIMEOUT",
        "CONFIRM_ON_TIMEOUT",
        "INITIAL_BALANCE",
        "RISK_PER_TRADE",
        "MAX_POSITIONS",
        "STOP_LOSS_PERCENT",
        "DAILY_LOSS_LIMIT",
        "MAX_DRAWDOWN_PERCENT",
        "DRAWDOWN_SCALE_ENABLED",
        "MACRO_BLACKOUT_ENABLED",
        "LEVERAGE_MODE",
        "LEVERAGE_MAX",
        "TRAILING_STOP_ATR_MULT",
        "PARTIAL_TP_ENABLED",
        "MIN_SIGNAL_CONFIDENCE",
        "MIN_SIGNAL_CONFIDENCE_PAPER",
        "MODE",
        "AI_PROVIDER",
        "CHRONOS_ENABLED",
        "SEASON_SWITCH_MODE",
        "LIVE_ALPHA_KILL_ENABLED",
    }
)


def _rc(setter: str, getter: str) -> Dict[str, Any]:
    return {
        "hot_set": lambda rc, v: getattr(rc, setter)(v),
        "hot_get": lambda rc: getattr(rc, getter)(),
    }


def _redis_key(key: str, getter: str) -> Dict[str, Any]:
    """Значение, которое бот читает через RuntimeConfig.getter с фолбэком на Config."""
    return {
        "hot_set": lambda rc, v: rc._set(key, str(v)) or True,
        "hot_get": lambda rc: getattr(rc, getter)(),
    }


# Настройки на RuntimeConfig.hot — их ключи стирает и «Сброс настроек».
HOT_KEYS: List[str] = []


def _hot(key: str) -> Dict[str, Any]:
    """Меняется сразу, без своего геттера: бот читает RuntimeConfig.hot(key)."""
    HOT_KEYS.append(key)
    return {
        "hot_set": lambda rc, v: rc.set_hot(key, v),
        "hot_get": lambda rc: rc.hot(key),
    }


def _opts(*pairs: str) -> Tuple[Tuple[str, str], ...]:
    return tuple((p, p) for p in pairs)


def _hours_ok(v: str) -> bool:
    try:
        a, b = (int(x) for x in v.split("-"))
    except ValueError:
        return False
    return 0 <= a <= 23 and 0 <= b <= 23 and a != b


_G_TRADE = "Торговля"


_G_RISK = "Риск на сделку"


_G_GUARD = "Аварийная защита"


_G_PORT = "Защита портфеля"


_G_LEV = "Плечо"


_G_EXIT = "Выход из сделки"


_G_FILTER = "Фильтры входа"


_G_AI = "AI и решения"


_G_KILL = "Kill-switch стратегий"


_G_NEWS = "Новости и соцсети"


_G_SEASON = "Сезон рынка"


_G_TRUST = "Сделки по доверенным каналам"


_G_SAC = "Обучение SAC и бэктест"


_G_MON = "Мониторинг и логи"


_G_SERVER = "Сервер"


_STRATEGIES = (
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
)
