"""
Реестр настроек бота — единый источник для UI «Все параметры» и /api/config.

Каждая настройка описана один раз: подпись по-русски, тип, границы и способ
применения. «Сразу» (hot) — через RuntimeConfig или ключ Redis, который бот
читает каждый цикл. Остальные хранятся в Redis-хэше `bot:config_overrides` и
применяются при старте бота (`apply_overrides`), до `Config.validate()`.

Приоритет значения: изменение из UI > переменная окружения > значение по умолчанию.
Ключи API здесь не хранятся — для них отдельное зашифрованное хранилище.

DQN_WEIGHT, AI_WEIGHT, DQN_SOLO_CONFIDENCE, AI_STRATEGY_SELECTION из .env.example
намеренно не включены: код их не читает (веса Hybrid зашиты по режиму рынка
в signal_combiner.py), и ручка в UI ничего бы не меняла.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from src import ai_providers
from src.runtime_config import CONFIRM_TIMEOUT_MAX, CONFIRM_TIMEOUT_MIN

logger = logging.getLogger(__name__)

OVERRIDES_KEY = "bot:config_overrides"

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
_G_LEV = "Плечо"
_G_EXIT = "Выход из сделки"
_G_FILTER = "Фильтры входа"
_G_AI = "AI и решения"
_G_KILL = "Kill-switch стратегий"
_G_NEWS = "Новости и соцсети"
_G_SEASON = "Сезон рынка"
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

SETTINGS: Tuple[Setting, ...] = (
    # ── Торговля ──
    Setting(
        "MARKET_TYPE",
        _G_TRADE,
        "Рынок",
        "choice",
        "linear",
        options=(("linear", "Фьючерсы"), ("spot", "Спот")),
        hint="Спот — только покупки. Фьючерсы — лонг и шорт с плечом.",
        **_rc("set_market_type", "get_market_type"),
    ),
    Setting(
        "TESTNET",
        _G_TRADE,
        "Тестовая сеть Bybit (testnet)",
        "bool",
        False,
        hint="Ордера идут на testnet.bybit.com — нужны отдельные ключи testnet. "
        "Для проверки Live без реальных денег.",
    ),
    Setting(
        "TIMEFRAME",
        _G_TRADE,
        "Таймфрейм свечей",
        "choice",
        "15m",
        options=_opts("1m", "3m", "5m", "15m", "1h", "4h", "1d"),
        hint="Профиль рынка при переключении ставит своё значение.",
        **_redis_key("bot:timeframe", "get_timeframe"),
    ),
    Setting(
        "TRADING_INTERVAL",
        _G_TRADE,
        "Как часто бот проверяет рынок",
        "int",
        30,
        min=10,
        max=3600,
        unit="с",
        hint="Пауза между циклами. Меньше — быстрее реакция, но больше AI-запросов.",
    ),
    Setting(
        "SCAN_TOP_N",
        _G_TRADE,
        "Сколько монет сканировать",
        "int",
        20,
        min=1,
        max=200,
        hint="Топ монет по объёму за 24 часа.",
        **_rc("set_scan_top_n", "get_scan_top_n"),
    ),
    Setting(
        "TRADING_SYMBOL",
        _G_TRADE,
        "Монета по умолчанию",
        "str",
        "BTC/USDT",
        attr="SYMBOL",
        hint="Используется, когда сканер не нашёл кандидатов.",
    ),
    Setting(
        "TRADING_HOURS",
        _G_TRADE,
        "Торговые часы (UTC)",
        "str",
        "",
        hint="Например 8-22. Пусто — круглосуточно.",
        check=lambda v: (
            None if v == "" or _hours_ok(v) else "Формат: 8-22, часы от 0 до 23."
        ),
        **_rc("set_trading_hours", "get_trading_hours"),
    ),
    Setting(
        "AUTO_EXECUTE",
        _G_TRADE,
        "Открывать сделки автоматически (без подтверждения)",
        "bool",
        False,
        hint="В Paper бот не спрашивает всегда. В Live выключено — каждая сделка "
        "приходит на подтверждение.",
        **_rc("set_auto_execute", "get_auto_execute"),
    ),
    Setting(
        "TELEGRAM_CONFIRM_TIMEOUT",
        _G_TRADE,
        "Сколько ждать подтверждения",
        "int",
        60,
        min=CONFIRM_TIMEOUT_MIN,
        max=CONFIRM_TIMEOUT_MAX,
        unit="с",
        hint="Время на решение по сделке. Не больше 2 минут: цена уходит, "
        "и сигнал устаревает.",
        **_rc("set_confirm_timeout", "get_confirm_timeout"),
    ),
    Setting(
        "CONFIRM_ON_TIMEOUT",
        _G_TRADE,
        "Если не ответил вовремя",
        "choice",
        "execute",
        options=(("execute", "Открыть сделку"), ("skip", "Пропустить")),
        **_rc("set_confirm_on_timeout", "get_confirm_on_timeout"),
    ),
    Setting(
        "INITIAL_BALANCE",
        _G_TRADE,
        "Стартовый баланс Paper",
        "float",
        10000.0,
        min=100,
        max=10_000_000,
        step=100,
        unit="$",
    ),
    Setting(
        "COMMISSION_RATE",
        _G_TRADE,
        "Комиссия в расчётах",
        "pct",
        0.001,
        min=0,
        max=0.01,
        step=0.0001,
        hint="Для бэктеста и paper-торговли.",
    ),
    # ── Риск на сделку ──
    Setting(
        "RISK_PER_TRADE",
        _G_RISK,
        "Риск на сделку",
        "pct",
        0.02,
        min=0.001,
        max=0.1,
        step=0.001,
        hint="Сколько баланса бот готов потерять, если сработает стоп.",
        **_rc("set_risk_per_trade", "get_risk_per_trade"),
    ),
    Setting(
        "MAX_POSITIONS",
        _G_RISK,
        "Сделок одновременно",
        "int",
        3,
        min=1,
        max=20,
        hint="В Paper можно до 15 — статистика копится быстрее.",
        **_rc("set_max_positions", "get_max_positions"),
    ),
    Setting(
        "MAX_POSITION_SIZE",
        _G_RISK,
        "Максимальный размер позиции",
        "pct",
        0.1,
        min=0.01,
        max=1,
        step=0.01,
        hint="Доля баланса в одной позиции.",
    ),
    Setting(
        "STOP_LOSS_PERCENT",
        _G_RISK,
        "Стоп-лосс по умолчанию",
        "pct",
        0.05,
        min=0.005,
        max=0.3,
        step=0.005,
        hint="Профиль рынка может задать свой.",
    ),
    Setting(
        "MAX_PORTFOLIO_EXPOSURE",
        _G_RISK,
        "Общая нагрузка портфеля",
        "float",
        1.5,
        min=0,
        max=10,
        step=0.1,
        unit="× баланса",
        hint="0 — без ограничения.",
    ),
    Setting(
        "DAILY_LOSS_LIMIT",
        _G_RISK,
        "Дневной лимит убытка",
        "pct",
        0.05,
        min=0.005,
        max=0.5,
        step=0.005,
        hint="При превышении бот закрывает всё и ждёт новых суток UTC.",
    ),
    Setting(
        "CIRCUIT_BREAKER_LOSSES",
        _G_RISK,
        "Стоп после убытков подряд",
        "int",
        3,
        min=0,
        max=20,
        unit="сделок",
        hint="0 — выключено.",
    ),
    Setting(
        "SYMBOL_COOLDOWN_HOURS",
        _G_RISK,
        "Пауза по монете после сделки",
        "float",
        3.0,
        min=0,
        max=72,
        step=0.5,
        unit="ч",
    ),
    # ── Аварийная защита ──
    Setting(
        "MAX_DRAWDOWN_PERCENT",
        _G_GUARD,
        "Предельная просадка",
        "pct",
        0.2,
        min=0.01,
        max=0.5,
        step=0.01,
        hint="От пика баланса. Бот закрывает всё и встаёт на паузу.",
    ),
    Setting(
        "DRAWDOWN_HALT_HOURS",
        _G_GUARD,
        "Пауза после остановки",
        "float",
        4.0,
        min=0,
        max=72,
        unit="ч",
    ),
    Setting(
        "DRAWDOWN_CONFIRM_CYCLES",
        _G_GUARD,
        "Подтверждений до остановки",
        "int",
        3,
        min=1,
        max=10,
        unit="циклов",
        hint="Защита от ложного срабатывания на резком проколе цены.",
        **_rc("set_drawdown_confirm_cycles", "get_drawdown_confirm_cycles"),
    ),
    Setting(
        "ATR_SPIKE_MULT",
        _G_GUARD,
        "Всплеск волатильности",
        "float",
        3.0,
        min=1.5,
        max=10,
        step=0.5,
        unit="× ATR",
        hint="Если волатильность выросла во столько раз — закрыть всё и пауза 1 ч.",
    ),
    Setting(
        "ATR_SPIKE_LOOKBACK",
        _G_GUARD,
        "Окно средней волатильности",
        "int",
        20,
        min=5,
        max=200,
        unit="свечей",
    ),
    Setting(
        "DRAWDOWN_SCALE_ENABLED",
        _G_GUARD,
        "Уменьшать сделки при просадке",
        "bool",
        True,
        **_rc("set_drawdown_scale_enabled", "get_drawdown_scale_enabled"),
    ),
    Setting(
        "DRAWDOWN_SCALE_THRESHOLD",
        _G_GUARD,
        "С какой просадки уменьшать",
        "pct",
        0.1,
        min=0.01,
        max=0.5,
        step=0.01,
    ),
    Setting(
        "DRAWDOWN_SCALE_FACTOR",
        _G_GUARD,
        "Размер сделки при просадке",
        "pct",
        0.5,
        min=0.1,
        max=1,
        step=0.05,
        hint="0.5 — сделки вдвое меньше обычных.",
    ),
    Setting(
        "MACRO_BLACKOUT_ENABLED",
        _G_GUARD,
        "Пауза на макро-новостях",
        "bool",
        True,
        hint="Не входить за 30 мин до и после FOMC, CPI, NFP. Нужен ключ Finnhub.",
    ),
    # ── Плечо ──
    Setting(
        "LEVERAGE_MODE",
        _G_LEV,
        "Режим плеча",
        "choice",
        "volatility",
        options=(
            ("fixed", "Фикс"),
            ("volatility", "По волатильности"),
            ("full", "Макс"),
        ),
        **_rc("set_leverage_mode", "get_leverage_mode"),
    ),
    Setting(
        "LEVERAGE", _G_LEV, "Плечо в режиме «Фикс»", "int", 3, min=1, max=20, unit="×"
    ),
    Setting(
        "LEVERAGE_MIN", _G_LEV, "Минимальное плечо", "int", 1, min=1, max=20, unit="×"
    ),
    Setting(
        "LEVERAGE_MAX", _G_LEV, "Максимальное плечо", "int", 5, min=1, max=20, unit="×"
    ),
    Setting(
        "LEVERAGE_TARGET_RISK",
        _G_LEV,
        "Риск на одно движение ATR",
        "pct",
        0.01,
        min=0.001,
        max=0.1,
        step=0.001,
        hint="По нему подбирается плечо в режимах «По волатильности» и «Макс».",
        **_rc("set_leverage_target_risk", "get_leverage_target_risk"),
    ),
    Setting(
        "LEVERAGE_DD_THRESHOLD",
        _G_LEV,
        "Просадка, при которой плечо снижается",
        "pct",
        0.15,
        min=0.05,
        max=0.5,
        step=0.01,
        hint="В режиме «Макс» плечо падает до 30% от расчётного. "
        "Отдельно от предельной просадки.",
        **_rc("set_max_drawdown_percent", "get_max_drawdown_percent"),
    ),
    # ── Выход из сделки ──
    Setting(
        "TRAILING_STOP_ATR_MULT",
        _G_EXIT,
        "Отступ трейлинг-стопа",
        "float",
        1.0,
        min=0.2,
        max=5,
        step=0.1,
        unit="× ATR",
    ),
    Setting("PARTIAL_TP_ENABLED", _G_EXIT, "Частичная фиксация прибыли", "bool", True),
    Setting(
        "PARTIAL_TP_TRIGGER",
        _G_EXIT,
        "Когда фиксировать",
        "pct",
        0.6,
        min=0.1,
        max=0.95,
        step=0.05,
        hint="Какую часть пути до тейк-профита должна пройти цена.",
    ),
    Setting(
        "PARTIAL_TP_FRACTION",
        _G_EXIT,
        "Какую часть закрывать",
        "pct",
        0.5,
        min=0.1,
        max=0.9,
        step=0.05,
    ),
    # ── Фильтры входа ──
    Setting(
        "MIN_SIGNAL_CONFIDENCE",
        _G_FILTER,
        "Мин. уверенность сигнала · Live",
        "float",
        0.65,
        min=0.4,
        max=0.95,
        step=0.01,
        hot_set=lambda rc, v: rc.set_signal_confidence(v, paper=False),
        hot_get=lambda rc: rc.get_signal_confidence(paper=False),
    ),
    Setting(
        "MIN_SIGNAL_CONFIDENCE_PAPER",
        _G_FILTER,
        "Мин. уверенность сигнала · Paper",
        "float",
        0.6,
        min=0.4,
        max=0.95,
        step=0.01,
        hint="Ниже, чем в Live, чтобы быстрее копить сделки для обучения.",
        hot_set=lambda rc, v: rc.set_signal_confidence(v, paper=True),
        hot_get=lambda rc: rc.get_signal_confidence(paper=True),
    ),
    Setting(
        "MAX_CORRELATION",
        _G_FILTER,
        "Макс. корреляция с открытыми · Live",
        "float",
        0.7,
        min=0,
        max=1,
        step=0.05,
        hint="0 — фильтр выключен.",
    ),
    Setting(
        "MAX_CORRELATION_PAPER",
        _G_FILTER,
        "Макс. корреляция с открытыми · Paper",
        "float",
        0.85,
        min=0,
        max=1,
        step=0.05,
    ),
    Setting(
        "CORRELATION_WINDOW",
        _G_FILTER,
        "Окно корреляции",
        "int",
        50,
        min=10,
        max=500,
        unit="свечей",
    ),
    Setting(
        "MIN_VOLUME_USDT",
        _G_FILTER,
        "Мин. объём за 24 ч",
        "float",
        1_000_000.0,
        min=0,
        max=10_000_000_000,
        step=100_000,
        unit="$",
        **_redis_key("bot:min_volume_usdt", "get_min_volume_usdt"),
    ),
    Setting(
        "MAX_VOLUME_USDT",
        _G_FILTER,
        "Макс. объём за 24 ч",
        "float",
        0.0,
        min=0,
        max=100_000_000_000,
        step=1_000_000,
        unit="$",
        hint="0 — без ограничения. Для Altcoin отсекает BTC и ETH.",
        **_redis_key("bot:max_volume_usdt", "get_max_volume_usdt"),
    ),
    Setting(
        "MAX_SPREAD_PCT",
        _G_FILTER,
        "Макс. спред",
        "float",
        0.3,
        min=0.01,
        max=5,
        step=0.01,
        unit="%",
    ),
    Setting(
        "VOLUME_SPIKE_THRESHOLD",
        _G_FILTER,
        "Порог всплеска объёма",
        "float",
        2.5,
        min=1.5,
        max=6,
        step=0.1,
        unit="× среднего",
    ),
    # ── AI и решения ──
    Setting(
        "MODE",
        _G_AI,
        "Кто принимает решение",
        "choice",
        "ai",
        options=(
            ("local", "Локальные стратегии (без AI)"),
            ("ai", "AI"),
            ("dqn", "SAC"),
            ("hybrid", "Hybrid"),
        ),
        **_rc("set_mode", "get_mode"),
    ),
    Setting(
        "AI_PROVIDER",
        _G_AI,
        "AI-провайдер",
        "choice",
        "auto",
        options=(
            ("auto", "Авто — по цепочке"),
            *((p.id, p.name) for p in ai_providers.PROVIDERS),
        ),
        hint="Выбранный провайдер идёт первым; остальная цепочка — в разделе AI.",
        **_rc("set_ai_provider", "get_ai_provider"),
    ),
    Setting(
        "CHRONOS_ENABLED",
        _G_AI,
        "Прогноз Chronos в режиме Hybrid",
        "bool",
        False,
        hint="Модель Amazon Chronos чуть повышает или понижает уверенность сигнала.",
        hot_set=lambda rc, v: rc.set_chronos_enabled(v),
        hot_get=lambda rc: rc.get_chronos_enabled(),
    ),
    Setting(
        "AI_DAILY_BUDGET",
        _G_AI,
        "AI-запросов в сутки",
        "int",
        200,
        min=0,
        max=10000,
        hint="Сверх лимита — простой анализ настроения без AI.",
    ),
    Setting("AI_MAX_SYMBOLS", _G_AI, "Монет в AI за цикл", "int", 30, min=1, max=100),
    Setting(
        "ACTIVE_STRATEGY",
        _G_AI,
        "Стратегия по умолчанию",
        "choice",
        "ema_crossover",
        options=_opts(*_STRATEGIES),
        attr="DEFAULT_STRATEGY",
    ),
    # ── Kill-switch ──
    Setting(
        "LIVE_ALPHA_KILL_ENABLED",
        _G_KILL,
        "Отключать убыточные стратегии",
        "bool",
        True,
        hint="В live стратегия выключается, если её результат хуже порога.",
    ),
    Setting(
        "LIVE_ALPHA_KILL_MIN_TRADES",
        _G_KILL,
        "Оценивать после",
        "int",
        20,
        min=5,
        max=500,
        unit="сделок",
    ),
    Setting(
        "LIVE_ALPHA_KILL_EV_THRESHOLD",
        _G_KILL,
        "Порог среднего результата сделки",
        "float",
        -0.5,
        min=-10,
        max=0,
        step=0.1,
    ),
    # ── Новости и соцсети ──
    Setting(
        "NEWS_UPDATE_INTERVAL",
        _G_NEWS,
        "Обновлять новости каждые",
        "int",
        900,
        min=60,
        max=86400,
        unit="с",
    ),
    Setting(
        "TWITTER_MAX_RESULTS",
        _G_NEWS,
        "Твитов за запрос",
        "int",
        10,
        min=10,
        max=100,
        hint="Меньше — экономнее тариф Basic.",
    ),
    Setting(
        "TELEGRAM_SENTIMENT_CHANNELS",
        _G_NEWS,
        "Telegram-каналы для мониторинга",
        "str",
        "CryptoMoonAlerts,pumpsignalscrypto,memecoinsnews",
        hint="Через запятую, без @.",
    ),
    Setting(
        "SENTIMENT_POLL_INTERVAL",
        _G_NEWS,
        "Опрашивать соцсети каждые",
        "int",
        300,
        min=60,
        max=3600,
        unit="с",
    ),
    Setting(
        "SENTIMENT_CACHE_TTL",
        _G_NEWS,
        "Хранить оценку настроения",
        "int",
        600,
        min=60,
        max=7200,
        unit="с",
    ),
    Setting(
        "SENTIMENT_BUY_THRESHOLD",
        _G_NEWS,
        "Порог покупки по настроению",
        "float",
        0.4,
        min=0,
        max=1,
        step=0.05,
    ),
    Setting(
        "SENTIMENT_SELL_THRESHOLD",
        _G_NEWS,
        "Порог продажи по настроению",
        "float",
        -0.4,
        min=-1,
        max=0,
        step=0.05,
    ),
    # ── Сезон рынка ──
    Setting(
        "SEASON_SWITCH_MODE",
        _G_SEASON,
        "При смене сезона",
        "choice",
        "alert",
        options=(("alert", "Спросить в Telegram"), ("auto", "Переключить самому")),
        **_rc("set_season_switch_mode", "get_season_switch_mode"),
    ),
    Setting(
        "SEASON_CHECK_INTERVAL_H",
        _G_SEASON,
        "Проверять сезон каждые",
        "float",
        4.0,
        min=1,
        max=48,
        unit="ч",
    ),
    # ── Обучение SAC ──
    Setting(
        "ONLINE_LEARNING_MODE",
        _G_SAC,
        "Автообучение",
        "choice",
        "periodic",
        options=(
            ("disabled", "Выключено"),
            ("periodic", "Периодически"),
            ("online", "После каждой сделки"),
            ("hybrid", "Периодически + веса стратегий"),
        ),
        hint="«После каждой сделки» — рискованно, только для экспериментов.",
    ),
    Setting(
        "ONLINE_LEARNING_TRIGGER",
        _G_SAC,
        "Переобучать каждые",
        "int",
        50,
        min=10,
        max=1000,
        unit="сделок",
    ),
    Setting(
        "ONLINE_LEARNING_GRADIENT_STEPS",
        _G_SAC,
        "Шагов за онлайн-обновление",
        "int",
        50,
        min=1,
        max=50,
    ),
    Setting(
        "TRAIN_TOP_N",
        _G_SAC,
        "Монет для обучения",
        "int",
        20,
        min=1,
        max=100,
        **_rc("set_train_top_n", "get_train_top_n"),
    ),
    Setting(
        "TRAIN_MIN_CANDLES",
        _G_SAC,
        "Минимум свечей на монету",
        "int",
        2880,
        min=500,
        max=50000,
        step=100,
    ),
    Setting(
        "TOTAL_TIMESTEPS",
        _G_SAC,
        "Шагов обучения",
        "int",
        50000,
        min=1000,
        max=5_000_000,
        step=1000,
    ),
    Setting(
        "EXPERIENCES_MAX",
        _G_SAC,
        "Хранить записей опыта",
        "int",
        10000,
        min=1000,
        max=1_000_000,
        step=1000,
    ),
    Setting(
        "BT_HOLDOUT_RATIO",
        _G_SAC,
        "Отложенная выборка в бэктесте",
        "pct",
        0.2,
        min=0.05,
        max=0.5,
        step=0.05,
        hint="Последняя часть истории, на которой проверяется стратегия.",
    ),
    Setting("SAC_MODEL_PATH", _G_SAC, "Файл модели SAC", "str", "models/sac_model.zip"),
    # ── Мониторинг ──
    Setting(
        "SILENT_DEATH_HOURS",
        _G_MON,
        "Тревога «бот молчит» через",
        "float",
        6.0,
        min=0.5,
        max=48,
        step=0.5,
        unit="ч",
    ),
    Setting(
        "REGIME_CACHE_TTL",
        _G_MON,
        "Кэш режима рынка",
        "int",
        300,
        min=30,
        max=3600,
        unit="с",
    ),
    Setting(
        "LOG_LEVEL",
        _G_MON,
        "Подробность логов",
        "choice",
        "INFO",
        options=_opts("DEBUG", "INFO", "WARNING", "ERROR"),
    ),
    Setting("JSON_LOGS", _G_MON, "Логи в формате JSON", "bool", False),
    # ── Сервер: только в .env ──
    Setting(
        "REDIS_URL",
        _G_SERVER,
        "Адрес Redis",
        "str",
        "redis://localhost:6379/0",
        readonly=True,
        hint="Меняется только в .env: настройки сами хранятся в Redis.",
    ),
    Setting(
        "HEALTH_PORT",
        _G_SERVER,
        "Порт health-сервера",
        "int",
        8080,
        readonly=True,
        hint="Меняется только в .env и docker-compose.",
    ),
    Setting(
        "WEBAPP_URL",
        _G_SERVER,
        "Адрес приложения для Telegram",
        "str",
        "",
        hint="Кнопка «Открыть терминал» в боте: https://<домен>/app/. "
        "Только https — так требует Telegram.",
        check=lambda v: (
            None
            if not v or str(v).startswith("https://")
            else "Только https-адрес (требование Telegram)."
        ),
    ),
    Setting(
        "DASHBOARD_USER",
        _G_SERVER,
        "Логин дашборда",
        "str",
        "admin",
        readonly=True,
        hint="Меняется в .env, затем make htpasswd.",
    ),
)

BY_KEY: Dict[str, Setting] = {s.key: s for s in SETTINGS}


# ── Значения ──────────────────────────────────────────────────────────────────


def coerce(s: Setting, raw: Any) -> Any:
    """Приводит значение к типу настройки. ValueError — с текстом для пользователя."""
    if s.type == "bool":
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str) and raw.lower() in ("true", "false", "1", "0"):
            return raw.lower() in ("true", "1")
        raise ValueError("Нужно true или false.")
    if s.type in ("int", "float", "pct"):
        if isinstance(raw, bool):
            raise ValueError("Нужно число.")
        try:
            num = float(raw)
        except (TypeError, ValueError):
            raise ValueError("Нужно число.") from None
        if s.type == "int":
            if num != int(num):
                raise ValueError("Нужно целое число.")
            num = int(num)
        if s.min is not None and num < s.min:
            raise ValueError(f"Не меньше {_fmt(s, s.min)}.")
        if s.max is not None and num > s.max:
            raise ValueError(f"Не больше {_fmt(s, s.max)}.")
        value: Any = num
    elif s.type == "choice":
        value = str(raw)
        if value not in {v for v, _ in s.options}:
            raise ValueError("Нет такого варианта.")
    else:
        value = str(raw).strip()
    if s.check:
        msg = s.check(value)
        if msg:
            raise ValueError(msg)
    return value


def _fmt(s: Setting, v: float) -> str:
    if s.type == "pct":
        return f"{v * 100:g}%"
    return f"{v:g}{(' ' + s.unit) if s.unit else ''}"


def effective_config(config_cls: Any, extra: Optional[Dict[str, Any]] = None) -> Any:
    """Экземпляр Config с действующими значениями класса (+ extra).

    Config — dataclass: значения по умолчанию его полей вычисляются при импорте,
    и `Config()` не видит того, что подставлено позже в атрибуты класса
    (переопределения из UI, ключи из хранилища, Paper/Live из Telegram).
    """
    cand = config_cls()
    if dataclasses.is_dataclass(cand):
        for f in dataclasses.fields(cand):
            setattr(cand, f.name, getattr(config_cls, f.name, getattr(cand, f.name)))
    for k, v in (extra or {}).items():
        setattr(cand, k, v)
    return cand


def load_overrides(r: Any) -> Dict[str, Any]:
    """Переопределения из UI (Redis-хэш). Битые записи пропускаются."""
    if r is None:
        return {}
    try:
        items = list((r.hgetall(OVERRIDES_KEY) or {}).items())
    except Exception as e:
        logger.debug("load_overrides failed: %s", e)
        return {}
    out: Dict[str, Any] = {}
    for k, v in items:
        k = k.decode() if isinstance(k, bytes) else str(k)
        v = v.decode() if isinstance(v, bytes) else v
        s = BY_KEY.get(k)
        if s is None or s.hot or s.readonly:
            continue
        try:
            out[k] = coerce(s, json.loads(v))
        except (ValueError, TypeError):
            logger.warning("Ignoring invalid config override %s=%r", k, v)
    return out


def _set_config(config_cls: Any, s: Setting, value: Any) -> None:
    # Часть настроек модули читают через os.getenv, часть — через Config.
    os.environ[s.key] = str(value).lower() if isinstance(value, bool) else str(value)
    if hasattr(config_cls, s.config_attr):
        setattr(config_cls, s.config_attr, value)


def apply_overrides(config_cls: Any, r: Any) -> List[str]:
    """Применяет переопределения при старте бота, до Config.validate().

    Если с ними Config.validate() падает, а без них — нет, переопределения
    откатываются: неудачная настройка из UI не должна мешать боту стартовать.
    """
    overrides = load_overrides(r)
    if not overrides:
        return []
    saved = {
        k: (getattr(config_cls, BY_KEY[k].config_attr, None), os.environ.get(k))
        for k in overrides
    }
    for k, v in overrides.items():
        _set_config(config_cls, BY_KEY[k], v)
    try:
        effective_config(config_cls).validate()
    except ValueError as e:
        for k, (attr_val, env_val) in saved.items():
            if hasattr(config_cls, BY_KEY[k].config_attr):
                setattr(config_cls, BY_KEY[k].config_attr, attr_val)
            if env_val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = env_val
        try:
            effective_config(config_cls).validate()
        except ValueError:
            # Конфиг не проходит и без переопределений — дело не в них.
            for k, v in overrides.items():
                _set_config(config_cls, BY_KEY[k], v)
            return list(overrides)
        logger.error("UI config overrides rejected at startup (%s) — using .env", e)
        return []
    logger.info("Applied UI config overrides: %s", ", ".join(sorted(overrides)))
    return list(overrides)


def current_values(config_cls: Any, rc: Any, r: Any) -> Dict[str, Dict[str, Any]]:
    """{key: {value, source}}; source — live (hot) | ui | env | default."""
    overrides = load_overrides(r)
    out: Dict[str, Dict[str, Any]] = {}
    for s in SETTINGS:
        if s.hot and rc is not None:
            try:
                value = s.hot_get(rc)  # type: ignore[misc]
            except Exception:
                value = None
            if value is not None:
                # Действующее значение у бота (Redis или фолбэк на Config).
                out[s.key] = {"value": value, "source": "live"}
                continue
        if s.key in overrides:
            out[s.key] = {"value": overrides[s.key], "source": "ui"}
            continue
        if s.key in os.environ:
            try:
                out[s.key] = {"value": coerce(s, os.environ[s.key]), "source": "env"}
                continue
            except ValueError:
                pass
        value = getattr(config_cls, s.config_attr, s.default)
        out[s.key] = {"value": value, "source": "default"}
    return out


def validate_patch(
    config_cls: Any, patch: Dict[str, Any], current: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """(значения, ошибки по ключам). None в значении — вернуть к .env / умолчанию."""
    values: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    for k, raw in patch.items():
        s = BY_KEY.get(k)
        if s is None:
            errors[k] = "Нет такой настройки."
        elif s.readonly:
            errors[k] = "Меняется только в .env на сервере."
        elif raw is None:
            if s.hot:
                errors[k] = "Укажите значение."
            else:
                values[k] = None
        else:
            try:
                values[k] = coerce(s, raw)
            except ValueError as e:
                errors[k] = str(e)
    if errors:
        return values, errors

    merged = {k: v["value"] for k, v in current.items()}
    merged.update({k: v for k, v in values.items() if v is not None})
    if merged.get("LEVERAGE_MIN", 1) > merged.get("LEVERAGE_MAX", 20):
        errors["LEVERAGE_MIN"] = "Минимальное плечо больше максимального."
    if errors:
        return values, errors

    # Та же проверка, что бот делает при старте: Config.validate() на кандидате.
    # Наличие ключей (Bybit для Live, хотя бы один AI-ключ) — забота хранилища
    # ключей: у дашборда их в env может не быть, и тогда любой PATCH падал бы.
    from src.secrets_store import SERVICES  # noqa: PLC0415

    attrs: Dict[str, Any] = {
        f.env: "configured-elsewhere"
        for svc in SERVICES
        for f in svc.fields
        if f.secret and hasattr(config_cls, f.env)
    }
    attrs.update(
        {
            BY_KEY[k].config_attr: v
            for k, v in merged.items()
            if k in BY_KEY and hasattr(config_cls, BY_KEY[k].config_attr)
        }
    )
    try:
        effective_config(config_cls, attrs).validate()
    except ValueError as e:
        errors["_config"] = str(e)
    return values, errors


def apply_patch(rc: Any, r: Any, values: Dict[str, Any]) -> Dict[str, List[str]]:
    """Записывает уже проверенные значения. hot — сразу, остальные — в хэш."""
    applied_hot: List[str] = []
    pending: List[str] = []
    failed: List[str] = []
    for k, v in values.items():
        s = BY_KEY[k]
        if s.hot:
            ok = s.hot_set(rc, v)  # type: ignore[misc]
            (failed if ok is False else applied_hot).append(k)
        elif v is None:
            r.hdel(OVERRIDES_KEY, k)
            pending.append(k)
        else:
            r.hset(OVERRIDES_KEY, k, json.dumps(v))
            pending.append(k)
    return {"applied_hot": applied_hot, "pending_restart": pending, "failed": failed}
