"""
SQLite-based trade history tracker.
Stores open/closed trades and computes win rate + expected value.
DB path: data/trades.db
"""

import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Dict, Optional

_BACKTEST_JSON = os.path.join("data", "backtest_results.json")


def get_backtest_stats(strategy: str) -> Dict:
    """
    Read per-strategy stats from data/backtest_results.json.
    Returns zeros when file is missing or strategy not found.
    Run backtest.py once to generate the file.
    """
    try:
        with open(_BACKTEST_JSON, encoding="utf-8") as f:
            data = json.load(f)
        for r in data.get("results", []):
            if r.get("strategy") == strategy:
                return {
                    "win_rate": float(r.get("win_rate", 0.0)),
                    "total_trades": int(r.get("total_trades", 0)),
                    "ev": float(r.get("expected_value", 0.0)),
                    "total_return_pct": float(r.get("total_return_pct", 0.0)),
                }
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    return {
        "win_rate": 0.0,
        "total_trades": 0,
        "ev": 0.0,
        "total_return_pct": 0.0,
    }


logger = logging.getLogger(__name__)

_DB_PATH = os.path.join("data", "trades.db")
_DDL = """
CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    strategy    TEXT    NOT NULL,
    action      TEXT    NOT NULL,
    entry_price REAL    NOT NULL,
    exit_price  REAL,
    quantity    REAL    NOT NULL,
    confidence  REAL    NOT NULL DEFAULT 0.0,
    pnl         REAL,
    pnl_pct     REAL,
    commission  REAL    NOT NULL DEFAULT 0.0,
    entry_time  TEXT    NOT NULL,
    exit_time   TEXT,
    status      TEXT    NOT NULL DEFAULT 'open'
);
"""


class TradeHistory:
    """Persists trades to SQLite; computes win rate and EV."""

    def __init__(self, db_path: str = _DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = asyncio.Lock()
        with self._conn:
            self._conn.execute(_DDL)

    # ── Write ─────────────────────────────────────────────

    async def record_open(
        self,
        symbol: str,
        strategy: str,
        action: str,
        entry_price: float,
        quantity: float,
        confidence: float,
        commission: float = 0.0,
    ) -> int:
        """Insert an open trade; return its row id."""
        async with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO trades
                  (symbol, strategy, action, entry_price,
                   quantity, confidence, commission,
                   entry_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    strategy,
                    action,
                    entry_price,
                    quantity,
                    confidence,
                    commission,
                    datetime.now().isoformat(),
                ),
            )
            self._conn.commit()
            return cur.lastrowid

    async def record_close(
        self,
        trade_id: int,
        exit_price: float,
        commission: float = 0.0,
    ) -> None:
        """Close a trade; compute PnL."""
        async with self._lock:
            row = self._conn.execute(
                "SELECT action, entry_price, quantity, "
                "commission FROM trades WHERE id = ?",
                (trade_id,),
            ).fetchone()
            if row is None:
                logger.warning(f"trade_id={trade_id} not found")
                return
            action, entry_price, qty, entry_comm = row
            total_comm = entry_comm + commission
            if action == "buy":
                pnl = (exit_price - entry_price) * qty - total_comm
            else:
                pnl = (entry_price - exit_price) * qty - total_comm
            pnl_pct = pnl / (entry_price * qty) if entry_price else 0
            self._conn.execute(
                """
                UPDATE trades
                SET exit_price=?, commission=commission+?,
                    pnl=?, pnl_pct=?, exit_time=?,
                    status='closed'
                WHERE id=?
                """,
                (
                    exit_price,
                    commission,
                    pnl,
                    pnl_pct,
                    datetime.now().isoformat(),
                    trade_id,
                ),
            )
            self._conn.commit()

    # ── Read ──────────────────────────────────────────────

    async def get_win_rate(
        self,
        strategy: Optional[str] = None,
        lookback: int = 50,
    ) -> float:
        """Fraction of profitable closed trades (0.0-1.0)."""
        where = (
            "WHERE status='closed' AND strategy=?"
            if strategy
            else "WHERE status='closed'"
        )
        params = (strategy, lookback) if strategy else (lookback,)
        async with self._lock:
            row = self._conn.execute(
                f"""
                SELECT
                  COUNT(*) AS total,
                  SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)
                      AS wins
                FROM (
                  SELECT pnl, strategy FROM trades
                  {where}
                  ORDER BY id DESC LIMIT ?
                )
                """,
                params,
            ).fetchone()
        if not row or not row[0]:
            return 0.5  # default assumption: 50%
        return row[1] / row[0]

    async def get_expected_value(
        self,
        strategy: Optional[str] = None,
        lookback: int = 50,
    ) -> float:
        """
        EV as fraction: win_rate * avg_win_pct
                      - loss_rate * abs(avg_loss_pct)
        """
        where = (
            "WHERE status='closed' AND strategy=?"
            if strategy
            else "WHERE status='closed'"
        )
        params = (strategy, lookback) if strategy else (lookback,)
        async with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT pnl_pct FROM trades
                {where}
                ORDER BY id DESC LIMIT ?
                """,
                params,
            ).fetchall()
        if not rows:
            return 0.0
        wins = [r[0] for r in rows if r[0] is not None and r[0] > 0]
        losses = [r[0] for r in rows if r[0] is not None and r[0] <= 0]
        n = len(rows)
        if n == 0:
            return 0.0
        wr = len(wins) / n
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
        return wr * avg_win - (1 - wr) * avg_loss

    async def get_trade_count(
        self,
        strategy: Optional[str] = None,
        lookback: int = 50,
    ) -> int:
        """Count of closed trades used for win rate."""
        where = (
            "WHERE status='closed' AND strategy=?"
            if strategy
            else "WHERE status='closed'"
        )
        params = (strategy, lookback) if strategy else (lookback,)
        async with self._lock:
            row = self._conn.execute(
                f"SELECT COUNT(*) FROM "
                f"(SELECT id FROM trades {where} "
                f"ORDER BY id DESC LIMIT ?)",
                params,
            ).fetchone()
        return row[0] if row else 0

    async def get_summary(self) -> Dict:
        """Overall stats dict for display."""
        async with self._lock:
            row = self._conn.execute(
                """
                SELECT
                  COUNT(*) AS total,
                  SUM(CASE WHEN status='closed' AND pnl>0
                      THEN 1 ELSE 0 END) AS wins,
                  SUM(CASE WHEN status='closed'
                      THEN 1 ELSE 0 END) AS closed,
                  SUM(CASE WHEN status='closed'
                      THEN pnl ELSE 0 END) AS total_pnl,
                  SUM(commission) AS total_comm
                FROM trades
                """
            ).fetchone()
        total, wins, closed, total_pnl, total_comm = row
        win_rate = (wins / closed) if closed else 0.0
        return {
            "total_trades": total or 0,
            "closed_trades": closed or 0,
            "win_rate": win_rate,
            "total_pnl": total_pnl or 0.0,
            "total_commissions": total_comm or 0.0,
        }
