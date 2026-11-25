"""Runtime-конфигурация поверх Redis.

Позволяет менять настройки бота без перезапуска контейнера.
Все значения хранятся в Redis и переопределяют .env при наличии.

Пакет разложен по группам цепочкой классов (каждый видит методы нижних):
base → trading → risk → models → market → RuntimeConfig. Ключи Redis — keys.py,
пресеты и справочники — presets.py, текущий выбор (профиль, рынок, окружение
дочерних процессов) — active.py.
"""

from __future__ import annotations

import logging
from typing import Iterable

from bitbot.config.runtime.active import (  # noqa: F401 — публичный API пакета
    active_market_type,
    active_profile,
    child_env,
    experiences_path,
    set_active_market_type,
    set_active_profile,
)
from bitbot.config.runtime.keys import (
    _KEY_AI_PROVIDER,
    _KEY_AUTO_EXECUTE,
    _KEY_DISABLED_STRATS,
    _KEY_DRAWDOWN_CONFIRM,
    _KEY_DRAWDOWN_SCALE,
    _KEY_EXCLUDED,
    _KEY_EXPERIMENTAL_SEEN,
    _KEY_FORCED,
    _KEY_LEVERAGE_DD,
    _KEY_LEVERAGE_MODE,
    _KEY_LEVERAGE_TARGET_RISK,
    _KEY_LOCAL_FALLBACK,
    _KEY_MARKET_PROFILE,
    _KEY_MARKET_TYPE,
    _KEY_MAX_POSITIONS,
    _KEY_MAX_VOLUME,
    _KEY_MIN_VOLUME,
    _KEY_MODE,
    _KEY_PAPER_GUARDS_OFF,
    _KEY_PAUSED,
    _KEY_RISK_PER_TRADE,
    _KEY_SCAN_TOP_N,
    _KEY_TIMEFRAME,
    _KEY_TRADING_HOURS,
    _KEY_TRAIN_TOP_N,
    HOT_PREFIX,
)
from bitbot.config.runtime.market import MarketSettings
from bitbot.config.runtime.presets import (  # noqa: F401 — публичный API пакета
    _MARKET_PROFILES,
    BLUECHIP_BASES,
    CONFIRM_TIMEOUT_MAX,
    CONFIRM_TIMEOUT_MIN,
    RISK_PRESETS,
    confirm_policy,
)

logger = logging.getLogger(__name__)


class RuntimeConfig(MarketSettings):
    """
    Обёртка над Redis для хранения изменяемых во время работы настроек.

    Fallback: если ключ в Redis отсутствует, возвращает значение из Config (.env).
    """

    # ── Startup ───────────────────────────────────────────────────────────────
    _ALL_KEYS = (
        _KEY_MODE,
        _KEY_SCAN_TOP_N,
        _KEY_PAUSED,
        _KEY_AUTO_EXECUTE,
        _KEY_LOCAL_FALLBACK,
        _KEY_PAPER_GUARDS_OFF,
        _KEY_FORCED,
        _KEY_EXCLUDED,
        _KEY_DISABLED_STRATS,
        _KEY_TRADING_HOURS,
        _KEY_MAX_POSITIONS,
        _KEY_RISK_PER_TRADE,
        _KEY_DRAWDOWN_SCALE,
        _KEY_TRAIN_TOP_N,
        _KEY_AI_PROVIDER,
        _KEY_MARKET_TYPE,
        _KEY_LEVERAGE_MODE,
        _KEY_LEVERAGE_TARGET_RISK,
        _KEY_LEVERAGE_DD,
        _KEY_DRAWDOWN_CONFIRM,
        _KEY_MARKET_PROFILE,
        _KEY_TIMEFRAME,
        _KEY_MIN_VOLUME,
        _KEY_MAX_VOLUME,
    )

    def reset_to_defaults(self, experimental: Iterable[str]) -> None:
        """Сбрасывает все Runtime-настройки к значениям по умолчанию.

        Кнопка «Сброс настроек» в приложении. `experimental` — имена
        экспериментальных стратегий: после сброса они снова выключены.
        """
        from bitbot.config.registry import HOT_KEYS  # noqa: PLC0415

        hot = (f"{HOT_PREFIX}{k}" for k in HOT_KEYS)
        try:
            self._r.redis_client.delete(*self._ALL_KEYS, _KEY_EXPERIMENTAL_SEEN, *hot)
            logger.info("RuntimeConfig: сброс к .env-дефолтам выполнен")
        except Exception as e:
            logger.warning("RuntimeConfig reset failed: %s", e)
            return
        self.ensure_experimental_disabled(experimental)

    # ── Summary ───────────────────────────────────────────────────────────────
    def summary(self) -> dict:
        return {
            "mode": self.get_mode(),
            "scan_top_n": self.get_scan_top_n(),
            "paused": self.is_paused(),
            "auto_execute": self.get_auto_execute(),
            "forced_symbols": sorted(self.get_forced_symbols()),
            "excluded_symbols": sorted(self.get_excluded_symbols()),
        }
