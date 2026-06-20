"""Data-layer helpers for the dashboard: Redis, SQLite, healthcheck, Prometheus."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from prometheus_client import Gauge

logger = logging.getLogger(__name__)

_DB_PATH = os.path.join("data", "trades.db")

# ── Prometheus gauges ─────────────────────────────────────────────────────────

g_running = Gauge("bitbot_running", "1 if bot is alive (healthcheck < 120s)")
g_balance = Gauge("bitbot_balance_usdt", "Free USDT balance")
g_pnl = Gauge("bitbot_total_pnl_usdt", "Cumulative PnL in USDT")
g_win_rate = Gauge("bitbot_win_rate_percent", "Win rate of closed trades")
g_trades = Gauge("bitbot_total_trades", "Total trades in database")

# ── Redis ─────────────────────────────────────────────────────────────────────


def get_redis():
    """Return a connected redis.Redis client, or None if unavailable."""
    try:
        import redis as redis_lib

        r = redis_lib.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD") or None,
            decode_responses=True,
        )
        r.ping()
        return r
    except Exception:
        return None


def redis_get(key: str) -> Optional[dict]:
    """Fetch and JSON-decode a Redis key; return None on miss or error."""
    r = get_redis()
    if not r:
        return None
    try:
        raw = r.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None


# ── SQLite ────────────────────────────────────────────────────────────────────


def get_trades(limit: int = 50) -> List[dict]:
    """Return up to *limit* most-recent trades from the SQLite database."""
    if not os.path.exists(_DB_PATH):
        return []
    try:
        conn = sqlite3.connect(_DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT symbol, strategy, action,
                   entry_price, exit_price,
                   quantity, pnl, pnl_pct,
                   confidence, commission,
                   entry_time, exit_time, status
            FROM trades
            ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error("DB read error: %s", e)
        return []


def get_stats() -> dict:
    """Return aggregate win-rate / PnL stats from SQLite."""
    if not os.path.exists(_DB_PATH):
        return {}
    try:
        conn = sqlite3.connect(_DB_PATH)
        row = conn.execute(
            """
            SELECT
              COUNT(*) as total,
              SUM(CASE WHEN status='closed' THEN 1 ELSE 0 END) as closed,
              SUM(CASE WHEN status='closed' AND pnl>0 THEN 1 ELSE 0 END) as wins,
              SUM(CASE WHEN status='closed' THEN pnl ELSE 0 END) as total_pnl,
              SUM(commission) as total_comm
            FROM trades
            """
        ).fetchone()
        conn.close()
        total, closed, wins, pnl, comm = row
        return {
            "total_trades": total or 0,
            "closed_trades": closed or 0,
            "win_rate": round(wins / closed * 100, 1) if closed else 0,
            "total_pnl": round(pnl or 0, 2),
            "total_commissions": round(comm or 0, 2),
        }
    except Exception as e:
        logger.error("Stats query error: %s", e)
        return {}


# ── Healthcheck ───────────────────────────────────────────────────────────────


def check_healthcheck() -> bool:
    """Return True if data/healthcheck.txt was updated within the last 120 s."""
    hc = Path("data/healthcheck.txt")
    if not hc.exists():
        return False
    try:
        content = hc.read_text().strip()
        if len(content) > 50:
            return False
        ts = datetime.fromisoformat(content)
        return (datetime.now() - ts).total_seconds() < 120
    except Exception:
        return False
