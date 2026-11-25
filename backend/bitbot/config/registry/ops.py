"""Настройки эксплуатации: мониторинг, логи, сервер.
"""

from __future__ import annotations

import logging
from typing import Tuple

from bitbot.config.registry.model import (
    _G_MON,
    _G_SERVER,
    Setting,
    _hours_ok,
    _opts,
    _rc,
)

logger = logging.getLogger(__name__)


SETTINGS: Tuple[Setting, ...] = (
    Setting(
        "QUIET_HOURS",
        _G_MON,
        "Тихие часы уведомлений (UTC)",
        "str",
        "",
        hint="В эти часы приходят только критичные: бот не отвечает, просадка, "
        "дневной лимит, всплеск волатильности. Например 15-0. Пусто — без тишины.",
        check=lambda v: (
            None if v == "" or _hours_ok(v) else "Формат: 15-0, часы от 0 до 23."
        ),
        **_rc("set_quiet_hours", "get_quiet_hours"),
    ),
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
