"""Одноразовая ссылка входа в приложение из Telegram (/app).

Бот выпускает токен и отправляет владельцу ссылку
`http://localhost:<порт>/app/#local=<токен>`.
Панель принимает его тем же локальным входом, что и ссылку лаунчера
(src/dashboard/auth.check_local_login): только на localhost, один раз, 5 минут.
В Redis лежит только SHA-256 токена.
"""

from __future__ import annotations

import hashlib
import secrets
from typing import Any

TTL = 300
_PREFIX = "bot:login_link:"


def _key(token: str) -> str:
    return _PREFIX + hashlib.sha256(token.encode()).hexdigest()


def issue(r: Any) -> str:
    token = secrets.token_urlsafe(32)
    r.set(_key(token), "1", ex=TTL)
    return token


def consume(r: Any, token: str) -> bool:
    """True один раз для живого токена: удаление в Redis атомарно."""
    return bool(token) and bool(r.delete(_key(str(token))))
