"""Основа RuntimeConfig: чтение и запись правок в Redis, «горячие» значения.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Set

from bitbot.config.runtime.keys import HOT_PREFIX
from bitbot.config.settings import Config

logger = logging.getLogger(__name__)


class RuntimeBase:
    def __init__(self, redis_client: Any) -> None:
        self._r = redis_client  # RedisClient instance

    def _get(self, key: str) -> str | None:
        try:
            raw = self._r.redis_client.get(key)
            if raw is None:
                return None
            if isinstance(raw, bytes):
                return raw.decode()
            if isinstance(raw, str):
                return raw
            return None  # неожиданный тип (напр. MagicMock в тестах)
        except Exception:
            return None

    def _set(self, key: str, value: str) -> None:
        try:
            self._r.redis_client.set(key, value)
        except Exception as e:
            logger.warning("RuntimeConfig set failed: %s", e)

    def hot(self, key: str) -> Any:
        """Настройка реестра, которая меняется сразу (`bot:hot:KEY`, JSON).

        Нет в Redis — прежнее значение из UI (до того как настройка стала
        «сразу»), иначе Config.
        """
        raw: Any = self._get(f"{HOT_PREFIX}{key}")
        if raw is None:
            try:
                raw = self._r.redis_client.hget("bot:config_overrides", key)
            except Exception:
                raw = None
            if isinstance(raw, bytes):
                raw = raw.decode()
        if isinstance(raw, str):
            try:
                value = json.loads(raw)
            except ValueError:
                value = None
            if value is not None:
                return value
        return getattr(Config, key, None)

    def set_hot(self, key: str, value: Any) -> bool:
        self._set(f"{HOT_PREFIX}{key}", json.dumps(value))
        return True

    # ── Symbol overrides ──────────────────────────────────────────────────────
    def _smembers(self, key: str) -> Set[str]:
        try:
            raw = self._r.redis_client.smembers(key)
            return {m.decode() if isinstance(m, bytes) else m for m in raw}
        except Exception:
            return set()
