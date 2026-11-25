"""Вся таблица настроек в порядке приложения и поиск по ключу.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

from bitbot.config.registry import feeds, models, ops, risk, trading
from bitbot.config.registry.model import (
    _G_AI,
    _G_EXIT,
    _G_FILTER,
    _G_GUARD,
    _G_KILL,
    _G_LEV,
    _G_MON,
    _G_NEWS,
    _G_PORT,
    _G_RISK,
    _G_SAC,
    _G_SEASON,
    _G_SERVER,
    _G_TRADE,
    _G_TRUST,
    Setting,
)

logger = logging.getLogger(__name__)


# Порядок групп — как в приложении («Все параметры»): по первому появлению группы;
# внутри группы — порядок описания в файле темы.
_GROUP_ORDER = (
    _G_TRADE,
    _G_RISK,
    _G_GUARD,
    _G_LEV,
    _G_EXIT,
    _G_FILTER,
    _G_AI,
    _G_PORT,
    _G_KILL,
    _G_NEWS,
    _G_TRUST,
    _G_SEASON,
    _G_SAC,
    _G_MON,
    _G_SERVER,
)

SETTINGS: Tuple[Setting, ...] = tuple(
    sorted(
        (
            *trading.SETTINGS,
            *risk.SETTINGS,
            *models.SETTINGS,
            *feeds.SETTINGS,
            *ops.SETTINGS,
        ),
        key=lambda s: _GROUP_ORDER.index(s.group),
    )
)
BY_KEY: Dict[str, Setting] = {s.key: s for s in SETTINGS}
