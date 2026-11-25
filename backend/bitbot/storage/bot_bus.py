"""
Шина между процессом бота и дашбордом через Redis.

События: бот публикует `{type, ts, data}` в канал `bot:events`, дашборд
раздаёт их клиентам по WebSocket.

Команды: дашборд кладёт `{request_id, type, args, source}` в список
`bot:commands`, бот забирает их `CommandConsumer`-ом, выполняет и пишет
результат в `bot:cmdres:{request_id}`. Повтор с тем же `request_id`
не исполняется дважды.

Модуль работает с «сырым» redis.Redis — и с decode_responses=True
(дашборд), и с bytes (RedisClient бота).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Awaitable, Callable, Dict, Optional

from bitbot.config import keys

logger = logging.getLogger(__name__)

RESULT_TTL = 600  # сек: сколько хранится результат и метка идемпотентности

Handler = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


def _text(raw: Any) -> str:
    return raw.decode() if isinstance(raw, bytes) else raw


def emit(r: Any, type_: str, data: Optional[Dict[str, Any]] = None) -> None:
    """Публикует событие. Никогда не бросает: события не должны ломать торговлю."""
    if r is None:
        return
    try:
        r.publish(
            keys.EVENTS,
            json.dumps({"type": type_, "ts": time.time(), "data": data or {}}),
        )
    except Exception as e:
        logger.debug("emit %s failed: %s", type_, e)


def enqueue(
    r: Any,
    type_: str,
    args: Optional[Dict[str, Any]],
    request_id: str,
    source: str,
) -> bool:
    """Ставит команду в очередь. False — команда с таким request_id уже была."""
    if not r.set(f"{keys.CMD_PREFIX}{request_id}", "1", nx=True, ex=RESULT_TTL):
        return False
    r.lpush(
        keys.COMMANDS,
        json.dumps(
            {
                "request_id": request_id,
                "type": type_,
                "args": args or {},
                "source": source,
            }
        ),
    )
    return True


def get_result(r: Any, request_id: str) -> Optional[Dict[str, Any]]:
    raw = r.get(f"{keys.CMDRES_PREFIX}{request_id}")
    return json.loads(_text(raw)) if raw else None


def error(code: str, message: str) -> Dict[str, Any]:
    """Ответ об ошибке: message — по-русски, для показа пользователю."""
    return {"ok": False, "error": {"code": code, "message": message}}


class CommandConsumer:
    """Забирает команды из очереди и исполняет их в процессе бота."""

    def __init__(self, r: Any, handlers: Dict[str, Handler]) -> None:
        self._r = r
        self._handlers = handlers

    async def handle(self, raw: Any) -> Optional[Dict[str, Any]]:
        try:
            cmd = json.loads(_text(raw))
            request_id = str(cmd["request_id"])
            type_ = str(cmd["type"])
        except (ValueError, KeyError, TypeError) as e:
            logger.warning("Bad command payload dropped: %s", e)
            return None

        handler = self._handlers.get(type_)
        if handler is None:
            result = error("unknown_command", f"Бот не знает команду «{type_}».")
        else:
            try:
                result = await handler(cmd.get("args") or {})
                result = result if "ok" in result else {"ok": True, **result}
            except Exception:
                logger.exception("Command %s (%s) failed", type_, request_id)
                result = error(
                    "failed", "Команда не выполнилась. Подробности — в логах бота."
                )

        logger.info(
            "Command %s from %s → %s",
            type_,
            cmd.get("source", "?"),
            "ok" if result.get("ok") else result["error"]["code"],
        )
        try:
            self._r.set(
                f"{keys.CMDRES_PREFIX}{request_id}", json.dumps(result), ex=RESULT_TTL
            )
        except Exception as e:
            logger.warning("Saving command result failed: %s", e)
        emit(
            self._r,
            "command.done",
            {"request_id": request_id, "type": type_, "result": result},
        )
        return result

    async def run(self, poll_timeout: int = 2) -> None:
        """Бесконечный цикл; останавливается отменой задачи."""
        warned = False
        while True:
            try:
                item = await asyncio.to_thread(
                    self._r.brpop, keys.COMMANDS, timeout=poll_timeout
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Без Redis бот работает дальше; предупреждаем один раз.
                (logger.debug if warned else logger.warning)(
                    "Command queue unavailable: %s", e
                )
                warned = True
                await asyncio.sleep(5)
                continue
            warned = False
            if item:
                await self.handle(item[1])
