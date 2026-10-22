"""Сторож «бот упал»: живёт в панели, отдельно от бота, и пишет владельцу в Telegram.

Бот сам не может сообщить, что он упал или завис: его проверка цикла работает
изнутри цикла. Панель видит пульс бота (data/healthcheck.txt) и время начала
последнего цикла (bot:last_cycle — его пишет след решения, и в паузе тоже).
Если выключен Docker или компьютер, сторожа нет вместе с ботом — сообщить некому.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

from src.dashboard import auth, data

logger = logging.getLogger(__name__)

CHECK_EVERY = 60
# Запас сверх интервала торговли: полный скан с AI бывает долгим.
STALL_GRACE = 900
_ALERT_KEY = "bot:watchdog_alert"


def problem(alive: bool, cycle_age: Optional[float], interval: int) -> Optional[str]:
    """Что не так с ботом, или None. cycle_age — секунд с начала последнего цикла."""
    if not alive:
        return "процесс бота не отвечает больше 2 минут"
    if cycle_age is not None and cycle_age > interval + STALL_GRACE:
        return f"торговый цикл не начинался {int(cycle_age // 60)} мин"
    return None


def _cycle_age(r: Any, now: float) -> Optional[float]:
    try:
        at = json.loads(r.get("bot:last_cycle") or "{}").get("at")
    except (ValueError, AttributeError):
        return None
    return now - float(at) if at else None


async def check_once(r: Any, started: float, now: Optional[float] = None) -> None:
    """Одна проверка: тревога при первой беде, «снова работает» — когда прошло."""
    from src.dashboard.routers.control import _runtime_config  # noqa: PLC0415

    now = time.time() if now is None else now
    cfg = _runtime_config(r)
    if cfg is None:
        return
    interval = cfg.get_trading_interval()
    # Сразу после запуска панели бот ещё поднимается — время цикла не судим.
    age = _cycle_age(r, now) if now - started > interval + STALL_GRACE else None
    issue = problem(data.check_healthcheck(), age, interval)
    alerted = r.get(_ALERT_KEY)
    if issue and not alerted:
        r.set(_ALERT_KEY, issue)
        if cfg.get_alert_enabled("bot_stalled"):
            await auth.send_owner(
                f"🛑 Бот не работает: {issue}.\n"
                "Перезапустите ярлыком BitbotBY или командой "
                "python3 launcher/launch.py start"
            )
    elif not issue and alerted:
        r.delete(_ALERT_KEY)
        if cfg.get_alert_enabled("bot_stalled"):
            await auth.send_owner("✅ Бот снова работает.")


async def run() -> None:
    started = time.time()
    while True:
        await asyncio.sleep(CHECK_EVERY)
        r = data.get_redis()
        if r is None:
            continue
        try:
            await check_once(r, started)
        except Exception as e:  # сторож не должен ронять панель
            logger.warning("watchdog check failed: %s", e)
