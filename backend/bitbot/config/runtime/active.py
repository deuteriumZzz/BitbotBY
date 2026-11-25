"""Что выбрано сейчас — для кода без ссылки на RuntimeConfig и для дочерних процессов.
"""

from __future__ import annotations

import logging
from typing import Dict

from bitbot.config.settings import Config

logger = logging.getLogger(__name__)


# ── Что выбрано сейчас: для кода без ссылки на RuntimeConfig ─────────────────
# Профиль монет и тип рынка ставит RuntimeConfig при смене и бот при старте
# (из Redis). Раньше они копировались в os.environ, и после рестарта копия
# терялась; дочерние процессы получают их явно — `child_env()`.
_active_profile: str = ""


_active_market_type: str = ""


_EXPERIENCES_BY_PROFILE = {
    "altcoin": "data/experiences_altcoin.jsonl",
    "meme": "data/experiences_meme.jsonl",
}


def set_active_profile(name: str) -> None:
    global _active_profile
    _active_profile = name or ""


def active_profile() -> str:
    """Профиль монет (bluechip | altcoin | meme); от него порог volume_spike."""
    return _active_profile


def set_active_market_type(market_type: str) -> None:
    global _active_market_type
    _active_market_type = market_type or ""


def active_market_type() -> str:
    """spot | linear; до первой установки — значение из Config."""
    return _active_market_type or getattr(Config, "MARKET_TYPE", "linear")


def experiences_path(profile: str) -> str:
    """Файл опыта сделок для обучения SAC своего профиля (bluechip — общий)."""
    return _EXPERIENCES_BY_PROFILE.get(profile or "", "data/experiences.jsonl")


def child_env() -> Dict[str, str]:
    """Что дочерний процесс (обучение, тюнинг, отчёты) должен знать о режиме бота.

    Режим и профиль меняются в приложении на ходу; дочерний процесс читает их из
    окружения, поэтому их передают явно, а не через копию в os.environ бота.
    """
    profile = active_profile()
    return {
        "PAPER_TRADING": "true" if Config.PAPER_TRADING else "false",
        "MARKET_TYPE": active_market_type(),
        "SAC_PROFILE": profile,
        "EXPERIENCES_PATH": experiences_path(profile),
    }
