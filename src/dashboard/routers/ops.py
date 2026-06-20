"""Operations routes: /metrics, /health, /webhook/alerts."""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import urllib.parse
import urllib.request

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from config import Config
from src.dashboard.data import (
    check_healthcheck,
    g_balance,
    g_pnl,
    g_running,
    g_trades,
    g_win_rate,
    get_stats,
    redis_get,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/metrics")
async def metrics() -> Response:
    portfolio = redis_get("portfolio_state") or {}
    stats = get_stats()
    g_running.set(1 if check_healthcheck() else 0)
    g_balance.set(portfolio.get("balance", Config.INITIAL_BALANCE))
    g_pnl.set(stats.get("total_pnl", 0))
    g_win_rate.set(stats.get("win_rate", 0))
    g_trades.set(stats.get("total_trades", 0))
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


def _tg_escape(s: str) -> str:
    """Strip Telegram Markdown special chars from untrusted content."""
    return s.replace("*", "").replace("_", "").replace("`", "").replace("[", "")


def _send_telegram_sync(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return
    try:
        data = urllib.parse.urlencode(
            {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        ).encode()
        urllib.request.urlopen(
            f"https://api.telegram.org/bot{token}/sendMessage", data, timeout=5
        )
    except Exception as e:
        logger.warning("Telegram alert send failed: %s", e)


@router.post("/webhook/alerts")
async def alert_webhook(request: Request) -> JSONResponse:
    """Receive Alertmanager webhook and forward alerts to Telegram."""
    expected = os.getenv("ALERTMANAGER_WEBHOOK_SECRET", "")
    if expected:
        provided = request.headers.get("X-Webhook-Secret", "")
        if not hmac.compare_digest(provided.encode(), expected.encode()):
            return JSONResponse({"error": "Forbidden"}, status_code=403)

    payload = await request.json()
    alerts = payload.get("alerts", [])
    if not alerts:
        return JSONResponse({"status": "no_alerts"})

    lines = []
    for alert in alerts:
        status = alert.get("status", "firing")
        name = _tg_escape(str(alert.get("labels", {}).get("alertname", "unknown")))
        severity = _tg_escape(str(alert.get("labels", {}).get("severity", "warning")))
        summary = _tg_escape(str(alert.get("annotations", {}).get("summary", "")))
        description = _tg_escape(
            str(alert.get("annotations", {}).get("description", ""))
        )
        icon = "\U0001f534" if status == "firing" else "✅"
        label = "CRITICAL" if severity == "critical" else "WARNING"
        msg = f"{icon} [{label}] {name}"
        if summary:
            msg += f"\n{summary}"
        if description:
            msg += f"\n{description}"
        lines.append(msg)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _send_telegram_sync, "\n\n".join(lines))
    return JSONResponse({"status": "ok", "forwarded": len(lines)})
