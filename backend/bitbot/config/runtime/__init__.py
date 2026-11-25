"""Runtime-конфигурация поверх Redis.

Позволяет менять настройки бота без перезапуска контейнера.
Все значения хранятся в Redis и переопределяют умолчания Config.

Пакет разложен по группам цепочкой классов (каждый видит методы нижних):
base → trading → risk → models → market → RuntimeConfig. Ключи Redis — keys.py,
пресеты и справочники — presets.py, текущий выбор (профиль, рынок, окружение
дочерних процессов) — active.py.
"""

from __future__ import annotations

import logging
from typing import Iterable

from bitbot.config import keys
from bitbot.config.runtime.active import (  # noqa: F401 — публичный API пакета
    active_market_type,
    active_profile,
    child_env,
    experiences_path,
    set_active_market_type,
    set_active_profile,
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

    Fallback: если ключ в Redis отсутствует, возвращает значение из Config.
    """

    # ── Startup ───────────────────────────────────────────────────────────────
    _ALL_KEYS = (
        keys.MODE,
        keys.SCAN_TOP_N,
        keys.PAUSED,
        keys.AUTO_EXECUTE,
        keys.LOCAL_FALLBACK,
        keys.PAPER_GUARDS_OFF,
        keys.FORCED_SYMBOLS,
        keys.EXCLUDED_SYMBOLS,
        keys.DISABLED_STRATEGIES,
        keys.TRADING_HOURS,
        keys.MAX_POSITIONS,
        keys.RISK_PER_TRADE,
        keys.DRAWDOWN_SCALE_ENABLED,
        keys.TRAIN_TOP_N,
        keys.AI_PROVIDER,
        keys.MARKET_TYPE,
        keys.LEVERAGE_MODE,
        keys.LEVERAGE_TARGET_RISK,
        keys.LEVERAGE_DD_THRESHOLD,
        keys.DRAWDOWN_CONFIRM_CYCLES,
        keys.MARKET_PROFILE,
        keys.TIMEFRAME,
        keys.MIN_VOLUME_USDT,
        keys.MAX_VOLUME_USDT,
    )

    def reset_to_defaults(self, experimental: Iterable[str]) -> None:
        """Сбрасывает все Runtime-настройки к значениям по умолчанию.

        Кнопка «Сброс настроек» в приложении. `experimental` — имена
        экспериментальных стратегий: после сброса они снова выключены.
        """
        from bitbot.config.registry import HOT_KEYS  # noqa: PLC0415

        hot = (f"{keys.HOT_PREFIX}{k}" for k in HOT_KEYS)
        try:
            self._r.redis_client.delete(
                *self._ALL_KEYS, keys.EXPERIMENTAL_STRATEGIES_SEEN, *hot
            )
            logger.info("RuntimeConfig: сброс к умолчаниям выполнен")
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
