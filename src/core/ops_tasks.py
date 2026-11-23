"""Операционные фоновые задачи бота: WS-shadow, бэкапы БД, аварийное управление.

Всё здесь — вспомогательное и либо выключено флагом (WS-shadow), либо только защитное
(бэкапы, аварийные уровни по явной команде владельца). Торговые решения не затрагиваются.

Аварийные уровни (см. `src/ops/emergency.py`), действия бота:
  1 block_entries     — пауза новых входов (`RuntimeConfig.set_paused`);
  2 cancel_orders     — НИЧЕГО не отменяет: входы рыночные, отложенных входных ордеров нет,
                        а биржевые SL/TP — защита позиций, снимать её нельзя (ADR-002);
  3 reduce_positions  — закрыть убыточные, у прибыльных SL на безубыток (как hard drawdown);
  4 full_stop         — закрыть ВСЕ позиции и оставить паузу.
Повышение уровней требует подтверждения в команде; снятие аварии (`reset`) паузу не
снимает — торговлю возобновляет владелец командой /resume.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

from config import Config
from src.ops.emergency import LEVELS, EmergencyControl

logger = logging.getLogger(__name__)

EMERGENCY_PATH = os.path.join("data", "emergency.json")
BACKUP_DIR = os.path.join("data", "backups")
STALE_ALERT_S = 90.0
ALERT_COOLDOWN_S = 1800.0

_LEVEL_TEXT = {
    1: "запрет новых входов (пауза)",
    2: "отмена отложенных ордеров (входные ордера рыночные, биржевые SL/TP не снимаются — ничего не сделает)",
    3: "закрыть убыточные позиции, у прибыльных SL на безубыток",
    4: "закрыть ВСЕ позиции и оставить паузу",
}


# ── WS Shadow ─────────────────────────────────────────────────────────────────
async def ws_shadow_loop(
    bot: Any,
    *,
    connect: Optional[Callable[[], Awaitable[Any]]] = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    interval: float = 15.0,
    stop: Optional[asyncio.Event] = None,
) -> None:
    """Публичный WS-поток параллельно REST: метрики в Prometheus, тревога при тишине. Решений не принимает."""
    from src.infra.health_server import publish_ws_metrics  # noqa: PLC0415
    from src.market.ws.client import MarketStream, default_connect, run_stream, topics  # noqa: PLC0415

    symbols = [s.strip() for s in Config.WS_SHADOW_SYMBOLS.split(",") if s.strip()]
    stream = MarketStream(symbols)
    bot.ws_stream = stream
    stop = stop or asyncio.Event()
    runner = asyncio.create_task(run_stream(connect or default_connect(), stream, topics(symbols), stop))
    last_alert = -ALERT_COOLDOWN_S
    try:
        while not stop.is_set() and not runner.done():
            await sleep(interval)
            publish_ws_metrics(stream.health.snapshot())
            silent = stream.health.silent_streams(STALE_ALERT_S)
            if silent and time.monotonic() - last_alert >= ALERT_COOLDOWN_S:
                last_alert = time.monotonic()
                with contextlib.suppress(Exception):
                    await bot.telegram.notify("⚠️ WS-shadow: потоки замолчали: " + ", ".join(silent[:6]))
    finally:
        stop.set()
        runner.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await runner


# ── Бэкапы БД ─────────────────────────────────────────────────────────────────
def trades_db_path() -> str:
    from src.core.trade_history import _db_path  # noqa: PLC0415

    return _db_path()


async def backup_once(bot: Any, path: Optional[str] = None) -> Dict[str, Any]:
    """Одна проверенная копия БД сделок; ошибка — алерт в Telegram. Пустой файл пропускается."""
    from src.infra import db_health  # noqa: PLC0415

    path = path or trades_db_path()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {"ok": True, "skipped": True, "source": path}
    res = await asyncio.to_thread(db_health.backup, path, BACKUP_DIR, Config.DB_BACKUP_KEEP)
    with contextlib.suppress(Exception):
        bot._ledger_write("db_backup", ok=res["ok"], backup=res.get("backup"), error=res.get("error"))
    if not res["ok"]:
        with contextlib.suppress(Exception):
            await bot.telegram.notify(f"⚠️ Резервная копия БД не создана: {res['error']}")
    return res


async def db_backup_loop(
    bot: Any,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    first_delay: float = 60.0,
) -> None:
    await sleep(first_delay)
    while True:
        try:
            await backup_once(bot)
        except Exception as exc:  # noqa: BLE001 — задача бэкапа не должна падать насовсем
            logger.warning("DB backup failed: %s", exc)
        await sleep(max(1.0, Config.DB_BACKUP_INTERVAL_H) * 3600)


# ── Аварийное управление ──────────────────────────────────────────────────────
def build_emergency(bot: Any, path: str = EMERGENCY_PATH) -> EmergencyControl:
    async def block_entries() -> None:
        bot._runtime_config.set_paused(True)

    async def cancel_orders() -> None:
        logger.info("Emergency L2: отложенных входных ордеров нет, биржевые SL/TP сохранены")

    async def reduce_positions() -> None:
        async with bot._monitored_lock:
            symbols = [s for s, p in bot._monitored.items() if p]
        prices: Dict[str, float] = {}
        for sym in symbols:
            price = await bot.api.get_current_price(sym)
            if price:
                prices[sym] = float(price)
        await bot._position_monitor.close_losers_tighten_winners(
            bot._monitored, bot._monitored_lock, prices, reason="emergency"
        )

    async def full_stop() -> None:
        await bot._close_all_open_positions("emergency_full_stop")
        bot._save_monitored_state()
        bot._runtime_config.set_paused(True)

    return EmergencyControl(
        path,
        {
            "block_entries": block_entries,
            "cancel_orders": cancel_orders,
            "reduce_positions": reduce_positions,
            "full_stop": full_stop,
        },
    )


def emergency_ctl(bot: Any) -> EmergencyControl:
    ctl = getattr(bot, "_emergency_ctl", None)
    if ctl is None:
        ctl = bot._emergency_ctl = build_emergency(bot)
    return ctl


async def apply_emergency_on_start(bot: Any) -> None:
    """Рестарт не должен молча возобновить торговлю при активной аварии."""
    ctl = emergency_ctl(bot)
    if ctl.level >= 1:
        bot._runtime_config.set_paused(True)
        with contextlib.suppress(Exception):
            await bot.telegram.notify(
                f"⛔ Активен аварийный уровень {ctl.level}: торговля остаётся на паузе. "
                "Снять: /emergency reset, затем /resume."
            )


async def emergency_command(bot: Any, args: List[str], who: str) -> str:
    """Разбор `/emergency [уровень [confirm] | reset]` → текст ответа. Уровни — с подтверждением."""
    ctl = emergency_ctl(bot)
    state = ctl.state()
    usage = "Использование: /emergency | /emergency 1-4 [confirm] | /emergency reset"
    if not args:
        if ctl.level == 0:
            return "Аварийный уровень: 0 (норма).\n" + usage
        return (
            f"Аварийный уровень: {ctl.level} ({LEVELS[ctl.level]}) от {state.get('by')} "
            f"({state.get('ts')}): {state.get('reason')}\n" + usage
        )
    cmd = args[0].lower()
    if cmd == "reset":
        ctl.reset(who, "снято владельцем")
        return "Аварийный уровень снят. Торговля остаётся на паузе — возобновите командой /resume."
    if not cmd.isdigit() or int(cmd) not in LEVELS:
        return usage
    level = int(cmd)
    if level <= ctl.level:
        return f"Уровень {ctl.level} уже активен — понижать можно только командой reset."
    if len(args) < 2 or args[1].lower() != "confirm":
        steps = "\n".join(f"  {lv}. {_LEVEL_TEXT[lv]}" for lv in range(ctl.level + 1, level + 1))
        return f"Будет выполнено:\n{steps}\nПодтвердите: /emergency {level} confirm"
    res = await ctl.aescalate(level, who, "команда владельца")
    lines = [f"Аварийный уровень {res['level']}: выполнено {', '.join(res['executed']) or 'ничего'}."]
    if res["errors"]:
        lines.append("Ошибки: " + "; ".join(f"{k}: {v}" for k, v in res["errors"].items()))
    return "\n".join(lines)
