"""Вспомогательные функции слоя данных дашборда:
Redis, SQLite, healthcheck, Prometheus."""

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

g_running = Gauge("bitbot_running", "1 если бот жив (healthcheck < 120 с)")
g_balance = Gauge("bitbot_balance_usdt", "Свободный баланс в USDT")
g_pnl = Gauge("bitbot_total_pnl_usdt", "Суммарный PnL в USDT")
g_win_rate = Gauge("bitbot_win_rate_percent", "Доля прибыльных закрытых сделок")
g_trades = Gauge("bitbot_total_trades", "Всего сделок в базе данных")

# ── Redis ─────────────────────────────────────────────────────────────────────


def get_redis():
    """Возвращает подключённый клиент redis.Redis или None при недоступности."""
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
    """Получает и декодирует JSON из Redis по ключу;
    возвращает None при промахе или ошибке."""
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
    """Возвращает до *limit* самых свежих сделок из базы данных SQLite."""
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
    """Возвращает агрегированную статистику win-rate / PnL из SQLite."""
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
    """Возвращает True если data/healthcheck.txt обновлялся в последние 120 с."""
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
