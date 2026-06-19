"""
HTTP-сервер для мониторинга: GET /health (JSON) + GET /metrics (Prometheus).

Запускается как фоновая asyncio-задача через start_health_server().
Не требует новых зависимостей: fastapi и prometheus-client уже в requirements.txt.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest

if TYPE_CHECKING:
    from src.trading_bot import TradingBot

logger = logging.getLogger(__name__)

# ── Prometheus gauges/counters ────────────────────────────────────────────────
_g_open_positions = Gauge("bot_open_positions", "Открытых позиций")
_g_paper_balance = Gauge("bot_paper_balance_usdt", "Paper-баланс USDT")
_g_win_rate = Gauge("bot_win_rate", "Win rate по закрытым сделкам")
_g_total_pnl = Gauge("bot_total_pnl_usdt", "Суммарный реализованный PnL")
_g_consec_losses = Gauge("bot_consecutive_losses", "Убытков подряд (circuit breaker)")
_g_ai_calls = Gauge("bot_ai_calls_today", "AI-вызовов за сутки")
cycles_counter = Counter("bot_cycles_total", "Всего итераций торгового цикла")

_bot: "TradingBot | None" = None
app = FastAPI(docs_url=None, redoc_url=None, title="BitbotBY health")


async def _snapshot() -> Dict[str, Any]:
    """Собирает метрики бота и обновляет Prometheus gauges."""
    if _bot is None:
        return {}

    async with _bot._monitored_lock:
        positions = dict(_bot._monitored)

    _g_open_positions.set(len(positions))
    _g_paper_balance.set(_bot._paper_balance)
    _g_consec_losses.set(_bot._consecutive_losses)
    _g_ai_calls.set(_bot.news._ai_calls_today)

    stats: Dict[str, Any] = {}
    try:
        stats = await _bot.trade_history.get_summary()
        _g_win_rate.set(stats.get("win_rate", 0.0))
        _g_total_pnl.set(stats.get("total_pnl", 0.0))
    except Exception:
        pass

    return {
        "is_running": _bot.is_running,
        "open_positions": len(positions),
        "symbols": list(positions.keys()),
        "consecutive_losses": _bot._consecutive_losses,
        "paper_balance_usdt": round(_bot._paper_balance, 2),
        "current_regime": _bot._current_regime,
        "ai_scorer": getattr(_bot.news, "_ai_scorer", "unknown"),
        "ai_calls_today": _bot.news._ai_calls_today,
        "closed_trades": stats.get("closed_trades", 0),
        "win_rate": round(stats.get("win_rate", 0.0), 4),
        "total_pnl_usdt": round(stats.get("total_pnl", 0.0), 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health")
async def health() -> JSONResponse:
    data = await _snapshot()
    status = "ok" if (data.get("is_running") is True) else "stopped"
    return JSONResponse({"status": status, **data})


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    await _snapshot()
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def start_health_server(bot: "TradingBot", port: int = 8080) -> None:
    """
    Запускает HTTP сервер мониторинга как asyncio-задачу.

    Эндпоинты:
      GET /health  — JSON статус бота
      GET /metrics — Prometheus-формат для scraping

    :param bot: Экземпляр TradingBot (ссылка сохраняется в модуле).
    :param port: Порт (по умолчанию 8080).
    """
    global _bot
    _bot = bot

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        loop="none",
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    asyncio.create_task(server.serve())
    logger.info("Health server started on :%d  (/health  /metrics)", port)
