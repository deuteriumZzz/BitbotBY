"""
Вход в приложение без паролей.

- Telegram MiniApp: подпись initData (HMAC-SHA256 от токена бота, алгоритм
  Telegram), данные не старше суток, войти может только владелец бота
  (TELEGRAM_CHAT_ID).
- Desktop: бот присылает владельцу в Telegram 6-значный код на 5 минут.

Обе ветки выдают сессию: случайный токен, в Redis хранится только его SHA-256.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Tuple

from src.settings import login_link, secrets_store

SESSION_TTL = 12 * 3600
INIT_DATA_MAX_AGE = 24 * 3600
CODE_TTL = 300
CODE_MAX_ATTEMPTS = 5
CODE_RESEND_SEC = 30

_CODE_KEY = "bot:desktop_code"
_CODE_RESEND_KEY = "bot:desktop_code_sent"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def owner_credentials() -> Tuple[Optional[str], Optional[str]]:
    """(токен бота, chat_id владельца) — из хранилища ключей или .env."""
    return (
        secrets_store.get_value("TELEGRAM_BOT_TOKEN"),
        secrets_store.get_value("TELEGRAM_CHAT_ID"),
    )


def telegram_auth_configured() -> bool:
    token, owner = owner_credentials()
    return bool(token and owner)


def verify_init_data(
    init_data: str,
    bot_token: str,
    owner_id: str,
    now: Optional[float] = None,
    max_age: int = INIT_DATA_MAX_AGE,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """(пользователь, "") при успехе, иначе (None, код ошибки)."""
    if not init_data or not bot_token:
        return None, "bad_init_data"
    pairs = dict(urllib.parse.parse_qsl(init_data, keep_blank_values=True))
    received = pairs.pop("hash", "")
    if not received:
        return None, "bad_init_data"
    check_string = "\n".join(f"{k}={v}" for k, v in sorted(pairs.items()))
    secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    expected = hmac.new(secret, check_string.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, received):
        return None, "bad_signature"
    try:
        auth_date = int(pairs.get("auth_date", "0"))
        user = json.loads(pairs.get("user", "{}"))
    except ValueError:
        return None, "bad_init_data"
    if auth_date <= 0 or (now or time.time()) - auth_date > max_age:
        return None, "expired"
    if str(user.get("id")) != str(owner_id):
        return None, "not_owner"
    return user, ""


# ── Сессии ────────────────────────────────────────────────────────────────────


def _session_key(token: str) -> str:
    return "bot:session:" + _sha(token)


def create_session(r: Any, subject: str, source: str) -> str:
    token = secrets.token_urlsafe(32)
    r.set(
        _session_key(token),
        json.dumps({"sub": subject, "src": source, "iat": int(time.time())}),
        ex=SESSION_TTL,
    )
    return token


def check_session(r: Any, token: str) -> Optional[Dict[str, Any]]:
    if r is None or not token:
        return None
    try:
        raw = r.get(_session_key(token))
    except Exception:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode()
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def drop_session(r: Any, token: str) -> None:
    if r is not None and token:
        r.delete(_session_key(token))


def bearer_token(headers: Any) -> str:
    value = headers.get("Authorization", "")
    return value[7:].strip() if value.startswith("Bearer ") else ""


# ── Код для Desktop ───────────────────────────────────────────────────────────


def issue_desktop_code(r: Any) -> Optional[str]:
    """Новый код или None, если прошлый запрошен меньше CODE_RESEND_SEC назад."""
    if not r.set(_CODE_RESEND_KEY, "1", nx=True, ex=CODE_RESEND_SEC):
        return None
    code = f"{secrets.randbelow(10**6):06d}"
    r.set(_CODE_KEY, json.dumps({"h": _sha(code), "tries": 0}), ex=CODE_TTL)
    return code


def check_desktop_code(r: Any, code: str) -> str:
    """ok | expired | locked | wrong."""
    raw = r.get(_CODE_KEY)
    if not raw:
        return "expired"
    state = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    if state["tries"] >= CODE_MAX_ATTEMPTS:
        return "locked"
    if hmac.compare_digest(_sha(str(code).strip()), state["h"]):
        r.delete(_CODE_KEY)
        return "ok"
    state["tries"] += 1
    r.set(_CODE_KEY, json.dumps(state), keepttl=True)
    return "locked" if state["tries"] >= CODE_MAX_ATTEMPTS else "wrong"


# ── Локальный вход лаунчера ───────────────────────────────────────────────────
#
# Только для запуска на своём компьютере: лаунчер включает BITBOT_LOCAL_MODE=1,
# кладёт в окружение случайный токен запуска и открывает приложение со ссылкой
# `#local=<токен>`. На сервере переменных нет, и этот вход выключен.
# Защита: токен ≥ 32 символов, сравнение за постоянное время, использовать можно
# один раз, 5 неверных попыток — блокировка на 10 минут, и запрос должен прийти
# на адрес localhost (защита от DNS-rebinding), а порт лаунчер открывает только
# на 127.0.0.1.

LOCAL_TOKEN_MIN = 32
LOCAL_MAX_FAILS = 5
LOCAL_LOCK_SEC = 600
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]"}
_LOCAL_FAIL_KEY = "bot:local_login_fails"


def local_login_token() -> str:
    """Токен запуска, если локальный вход включён; иначе пустая строка."""
    if os.getenv("BITBOT_LOCAL_MODE") != "1":
        return ""
    token = os.getenv("BITBOT_LOCAL_LOGIN_TOKEN", "")
    return token if len(token) >= LOCAL_TOKEN_MIN else ""


def _host_only(host_header: str) -> str:
    host = host_header.strip().lower()
    if host.startswith("["):  # [::1]:8080
        return host.split("]", 1)[0] + "]"
    return host.split(":", 1)[0]


def check_local_login(r: Any, token: str, host_header: str) -> str:
    """ok | disabled | host | locked | wrong | used."""
    expected = local_login_token()
    if not expected:
        return "disabled"
    if _host_only(host_header) not in _LOCAL_HOSTS:
        return "host"
    fails = r.get(_LOCAL_FAIL_KEY)
    if int(fails or 0) >= LOCAL_MAX_FAILS:
        return "locked"
    if login_link.consume(r, token):  # ссылка из Telegram (/app)
        return "ok"
    if not hmac.compare_digest(str(token).encode(), expected.encode()):
        if r.incr(_LOCAL_FAIL_KEY) == 1:
            r.expire(_LOCAL_FAIL_KEY, LOCAL_LOCK_SEC)
        return "wrong"
    # Один раз на токен запуска: повторное открытие той же ссылки не пускает.
    if not r.set(f"bot:local_login_used:{_sha(expected)[:16]}", "1", nx=True, ex=86400):
        return "used"
    return "ok"


def _send_owner_sync(text: str) -> bool:
    token, chat_id = owner_credentials()
    if not token or not chat_id:
        return False
    body = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    try:
        with urllib.request.urlopen(  # noqa: S310 — фиксированный хост Telegram
            f"https://api.telegram.org/bot{token}/sendMessage", data=body, timeout=10
        ) as resp:
            return resp.status == 200
    except Exception:
        return False


async def send_owner(text: str) -> bool:
    return await asyncio.to_thread(_send_owner_sync, text)
