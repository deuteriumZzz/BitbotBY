"""
Вход в Telegram для чтения каналов (Telethon) — из приложения, без консоли.

«Войти» → Telegram присылает код → код (и облачный пароль, если включена
двухэтапная проверка) вводится в приложении → готовая StringSession уходит
в шифрованное хранилище. Бот потом только подключается с ней и код не просит.

Незавершённый вход живёт в памяти процесса дашборда не дольше _TTL секунд.
Код, пароль и сессию не пишем ни в логи, ни в ответы API.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_TTL = 300.0
_pending: Dict[str, Any] = {}


class LoginError(Exception):
    def __init__(self, code: str, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def _rpc(e: Exception) -> LoginError:
    # Имя ошибки Telegram, без текста запроса: в нём бывает номер телефона.
    return LoginError("telegram_error", f"Telegram отказал: {type(e).__name__}.")


async def _drop() -> None:
    client = _pending.get("client")
    _pending.clear()
    if client is not None:
        try:
            await client.disconnect()
        except Exception as e:
            logger.debug("telethon login: disconnect failed: %s", e)


async def _current() -> Dict[str, Any]:
    if not _pending or time.monotonic() - _pending["at"] > _TTL:
        await _drop()
        raise LoginError(
            "no_login", "Вход не начат или устарел — нажмите «Войти» ещё раз.", 409
        )
    return _pending


async def send_code(api_id: str, api_hash: str, phone: str) -> str:
    """Просит Telegram прислать код. Возвращает, куда он придёт: app | sms."""
    from telethon import TelegramClient, errors  # noqa: PLC0415
    from telethon.sessions import StringSession  # noqa: PLC0415
    from telethon.tl.types.auth import SentCodeTypeApp  # noqa: PLC0415

    await _drop()
    client = TelegramClient(StringSession(), int(api_id), api_hash)
    try:
        await client.connect()
        sent = await client.send_code_request(phone)
    except errors.FloodWaitError as e:
        await client.disconnect()
        raise LoginError(
            "flood", f"Telegram просит подождать {e.seconds} с перед новым кодом.", 429
        ) from None
    except errors.ApiIdInvalidError:
        await client.disconnect()
        raise LoginError(
            "bad_api", "API ID или API Hash не подходят — проверьте их."
        ) from None
    except (errors.PhoneNumberInvalidError, errors.PhoneNumberBannedError):
        await client.disconnect()
        raise LoginError(
            "bad_phone", "Telegram не принимает этот номер телефона."
        ) from None
    except OSError:
        await client.disconnect()
        raise LoginError(
            "unreachable", "Telegram не ответил. Повторите.", 502
        ) from None
    except errors.RPCError as e:
        await client.disconnect()
        raise _rpc(e) from None
    _pending.update(
        client=client, phone=phone, hash=sent.phone_code_hash, at=time.monotonic()
    )
    return "app" if isinstance(sent.type, SentCodeTypeApp) else "sms"


async def _finish(client: Any) -> Tuple[str, str]:
    me = await client.get_me()
    session = client.session.save()
    name = f"@{me.username}" if getattr(me, "username", None) else ""
    await _drop()  # отключение не выходит из аккаунта: сессия остаётся рабочей
    return session, name


async def submit_code(code: str) -> Optional[Tuple[str, str]]:
    """(сессия, имя) — вход выполнен; None — нужен облачный пароль."""
    from telethon import errors  # noqa: PLC0415

    p = await _current()
    try:
        await p["client"].sign_in(p["phone"], code, phone_code_hash=p["hash"])
    except errors.SessionPasswordNeededError:
        return None
    except errors.PhoneCodeInvalidError:
        raise LoginError("bad_code", "Код не подошёл — проверьте цифры.") from None
    except errors.PhoneCodeExpiredError:
        await _drop()
        raise LoginError(
            "code_expired", "Код устарел — нажмите «Войти» ещё раз.", 410
        ) from None
    except errors.RPCError as e:
        raise _rpc(e) from None
    return await _finish(p["client"])


async def submit_password(password: str) -> Tuple[str, str]:
    from telethon import errors  # noqa: PLC0415

    p = await _current()
    try:
        await p["client"].sign_in(password=password)
    except errors.PasswordHashInvalidError:
        raise LoginError("bad_password", "Облачный пароль не подошёл.") from None
    except errors.RPCError as e:
        raise _rpc(e) from None
    return await _finish(p["client"])
