"""Операционные фоновые задачи бота: WS-shadow, бэкапы БД, аварийное управление.

Всё здесь — вспомогательное и либо выключено флагом (WS-shadow), либо только защитное
(бэкапы, аварийные уровни по явной команде владельца). Торговые решения не затрагиваются.

Аварийные уровни (см. `backend/bitbot/ops/emergency.py`), действия бота:
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
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

from bitbot.config.settings import Config
from bitbot.ops.emergency import DEFAULT_PATH, EmergencyControl, plan

logger = logging.getLogger(__name__)

EMERGENCY_PATH = DEFAULT_PATH
BACKUP_DIR = os.path.join("data", "backups")
STALE_ALERT_S = 90.0
ALERT_COOLDOWN_S = 1800.0


# ── WS Shadow ─────────────────────────────────────────────────────────────────
CANDLE_CHECK_S = 900.0


async def candle_check(
    bot: Any, stream: Any, timeframe: str = "15m", limit: int = 60
) -> Dict[str, Dict[str, Any]]:
    """Сверка подтверждённых свечей WS с REST по каждому символу; результат — в метрики Prometheus."""
    from bitbot.market.ws.candles import compare_with_rest  # noqa: PLC0415
    from bitbot.market.ws.client import to_bybit  # noqa: PLC0415
    from bitbot.workers.health_server import publish_candle_check  # noqa: PLC0415

    out: Dict[str, Dict[str, Any]] = {}
    for sym in [s.strip() for s in Config.WS_SHADOW_SYMBOLS.split(",") if s.strip()]:
        ws = stream.candles[to_bybit(sym)].ohlcv()[-30:]
        if len(ws) < 3:
            out[sym] = {"skipped": "мало подтверждённых свечей WS"}
            continue
        rest = await bot.api.exchange.fetch_ohlcv(sym, timeframe, limit=limit)
        out[sym] = compare_with_rest(ws, rest)
    publish_candle_check(out)
    return out


async def ws_shadow_loop(
    bot: Any,
    *,
    connect: Optional[Callable[[], Awaitable[Any]]] = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    interval: float = 15.0,
    stop: Optional[asyncio.Event] = None,
) -> None:
    """Публичный WS-поток параллельно REST: метрики в Prometheus, тревога при тишине. Решений не принимает."""
    from bitbot.market.ws.client import (  # noqa: PLC0415
        MarketStream,
        default_connect,
        run_stream,
        topics,
    )
    from bitbot.workers.health_server import publish_ws_metrics  # noqa: PLC0415

    symbols = [s.strip() for s in Config.WS_SHADOW_SYMBOLS.split(",") if s.strip()]
    stream = MarketStream(symbols)
    bot.ws_stream = stream
    stop = stop or asyncio.Event()
    runner = asyncio.create_task(
        run_stream(connect or default_connect(), stream, topics(symbols), stop)
    )
    last_alert = -ALERT_COOLDOWN_S
    last_check = time.monotonic()
    last_check_alert = -ALERT_COOLDOWN_S * 2
    try:
        while not stop.is_set() and not runner.done():
            await sleep(interval)
            publish_ws_metrics(stream.health.snapshot())
            # сверка свечей WS и REST раз в CANDLE_CHECK_S: расхождение — тревога (откат на REST и так штатный)
            if time.monotonic() - last_check >= CANDLE_CHECK_S:
                last_check = time.monotonic()
                with contextlib.suppress(Exception):
                    res = await candle_check(bot, stream)
                    bad = sorted(s for s, r in res.items() if r.get("ok") is False)
                    if (
                        bad
                        and time.monotonic() - last_check_alert >= ALERT_COOLDOWN_S * 2
                    ):
                        last_check_alert = time.monotonic()
                        await bot.telegram.notify(
                            "⚠️ WS≠REST: свечи расходятся: " + ", ".join(bad)
                        )
            silent = stream.health.silent_streams(STALE_ALERT_S)
            if silent and time.monotonic() - last_alert >= ALERT_COOLDOWN_S:
                last_alert = time.monotonic()
                with contextlib.suppress(Exception):
                    await bot.telegram.notify(
                        "⚠️ WS-shadow: потоки замолчали: " + ", ".join(silent[:6])
                    )
    finally:
        stop.set()
        runner.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await runner


# ── Бэкапы БД ─────────────────────────────────────────────────────────────────
def trades_db_path() -> str:
    from bitbot.storage.trade_history import _db_path  # noqa: PLC0415

    return _db_path()


async def backup_once(bot: Any, path: Optional[str] = None) -> Dict[str, Any]:
    """Одна проверенная копия БД сделок; ошибка — алерт в Telegram. Пустой файл пропускается."""
    from bitbot.storage import db_health  # noqa: PLC0415

    path = path or trades_db_path()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {"ok": True, "skipped": True, "source": path}
    res = await asyncio.to_thread(
        db_health.backup, path, BACKUP_DIR, Config.DB_BACKUP_KEEP
    )
    bot._last_backup = {  # для карточки «Защита и наблюдение» в приложении
        "ok": res["ok"],
        "at": datetime.now(timezone.utc).isoformat(),
        "path": res.get("backup"),
        "error": res.get("error"),
    }
    with contextlib.suppress(Exception):
        bot._ledger_write(
            "db_backup", ok=res["ok"], backup=res.get("backup"), error=res.get("error")
        )
    if not res["ok"]:
        with contextlib.suppress(Exception):
            await bot.telegram.notify(
                f"⚠️ Резервная копия БД не создана: {res['error']}"
            )
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
        except (
            Exception
        ) as exc:  # noqa: BLE001 — задача бэкапа не должна падать насовсем
            logger.warning("DB backup failed: %s", exc)
        await sleep(max(1.0, Config.DB_BACKUP_INTERVAL_H) * 3600)


# ── Состояние защиты и наблюдения (для приложения и журнала) ───────────────────
OPS_FLAGS = (
    "WS_SHADOW_ENABLED",
    "AI_SANITIZE_PROMPT",
    "AI_STRICT_VALIDATION",
    "AI_RAG_CONTEXT",
    "RISK_GATE_SHADOW",
    "DB_BACKUP_ENABLED",
    "FUNDING_IN_PNL",
    "SAC_V2_ENABLED",
    "EVENT_STREAM_ENABLED",
    "DATA_QUALITY_GATE",
)


def ops_status(bot: Any) -> Dict[str, Any]:
    """Что сейчас включено и как себя чувствует наблюдение: флаги, WS, копия БД, аварийный уровень.

    Публикуется в Redis `bot:ops_status` каждый цикл; приложение показывает это карточкой
    «Защита и наблюдение». Только чтение состояния, ничего не меняет.
    """
    from bitbot.ai.analyzer import PROMPT_VERSION  # noqa: PLC0415

    ws: Optional[Dict[str, Any]] = None
    stream = getattr(bot, "ws_stream", None)
    if stream is not None:
        snap = stream.health.snapshot()
        lags = [
            s["lag_ms"]["p99"]
            for s in snap["streams"].values()
            if s["lag_ms"]["p99"] is not None
        ]
        ws = {
            "streams": len(snap["streams"]),
            "silent": stream.health.silent_streams(STALE_ALERT_S),
            "reconnects": snap["reconnects"],
            "lag_p99_ms": max(lags) if lags else None,
            "gaps": sum(s["gaps"] for s in snap["streams"].values()),
        }
    try:
        level = int(emergency_ctl(bot).level)
    except Exception:  # noqa: BLE001 — статус не должен падать из-за файла аварии
        level = 0
    return {
        "mode": "paper" if Config.PAPER_TRADING else "live",
        "prompt_version": PROMPT_VERSION,
        "flags": {name: getattr(Config, name, False) is True for name in OPS_FLAGS},
        "forecaster": Config.FORECASTER,
        "ws": ws,
        "backup": getattr(bot, "_last_backup", None),
        "emergency": level,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def run_config_record(bot: Any) -> Dict[str, Any]:
    """Запись `run_config` в журнал при старте: какие флаги были включены с этого момента.

    По ней статистику «до» и «после» включения флагов можно разделить (журнал append-only).
    """
    st = ops_status(bot)
    rec = {k: st[k] for k in ("mode", "prompt_version", "flags", "forecaster")}
    with contextlib.suppress(Exception):
        bot._ledger_write("run_config", **rec)
    return rec


# ── RAG по Ledger ─────────────────────────────────────────────────────────────
RAG_REFRESH_S = 600.0


async def rag_refresh_loop(
    bot: Any,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    path: Optional[str] = None,
) -> None:
    """Раз в 10 минут пересобирает индекс по журналу в потоке (чтение до 50 МБ не должно блокировать цикл)."""
    from bitbot.ai.ledger_rag import LedgerRAG  # noqa: PLC0415

    while True:
        try:
            ledger_path = path or getattr(getattr(bot, "ledger", None), "path", None)
            bot.rag = await asyncio.to_thread(
                LedgerRAG.from_ledger, ledger_path or "data/decision_ledger.jsonl"
            )
        except (
            Exception
        ) as exc:  # noqa: BLE001 — прежний индекс остаётся, цикл не падает
            logger.warning("RAG refresh failed: %s", exc)
        await sleep(RAG_REFRESH_S)


def rag_context(bot: Any, symbols: List[str]) -> str:
    """Синхронный провайдер для `AIAnalyzer._rag_block`: только чтение готового индекса (быстро)."""
    from bitbot.ai.ledger_rag import safe_context  # noqa: PLC0415

    rag = getattr(bot, "rag", None)
    if rag is None:
        return ""
    parts = []
    for sym in symbols:
        r = safe_context(rag, sym, 2, symbol=sym, only_with_outcome=True)
        if r["context"]:
            parts.append(r["context"])
    return "\n".join(parts)


# ── Аварийное управление ──────────────────────────────────────────────────────
def build_emergency(bot: Any, path: str = EMERGENCY_PATH) -> EmergencyControl:
    async def block_entries() -> None:
        bot._runtime_config.set_paused(True)

    async def cancel_orders() -> None:
        logger.info(
            "Emergency L2: отложенных входных ордеров нет, биржевые SL/TP сохранены"
        )

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
                "Снять: приложение → Цепочка → Аварийное управление → «Сбросить», "
                "затем «Продолжить».",
                screen="bots",
            )


async def emergency_apply(
    bot: Any, action: str, confirm: bool, who: str
) -> Dict[str, Any]:
    """Аварийный уровень из приложения: "1"…"4" или "reset".

    Без `confirm` ничего не выполняется — возвращается список шагов для
    подтверждения. Уровень только растёт; сброс снимает аварию, но торговля
    остаётся на паузе (возобновить — отдельной командой).
    """
    ctl = emergency_ctl(bot)
    step = plan(ctl.level, action)
    if not step["ok"]:
        return step
    if not confirm:  # и для сброса: снятие аварии — тоже осознанное действие
        return {"level": ctl.level, "preview": True, **step}
    if step.get("reset"):
        ctl.reset(who, "снято владельцем")
        return {"level": 0, "reset": True}
    res = await ctl.aescalate(step["level"], who, "команда владельца")
    return {
        "level": res["level"],
        "executed": res["executed"],
        "errors": res["errors"],
    }
