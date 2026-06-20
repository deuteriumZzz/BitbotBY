"""
SQLite-backed история сделок: открытие/закрытие, win-rate и EV.

Сохраняет все сделки в локальную SQLite БД с WAL-режимом для безопасного
параллельного чтения дашбордом. Поддерживает расчёт win rate, EV и сводную статистику.

УЛУЧШЕНИЕ 1: все SQLite-операции выполняются через run_in_executor,
чтобы не блокировать asyncio event loop.
УЛУЧШЕНИЕ 2: pnl_pct считается от маржи (entry_price * qty / leverage),
а не от notional, что даёт корректные данные для Kelly criterion при плече.
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
    Возвращает статистику бэктеста для стратегии из data/backtest_results.json.

    Возвращает нулевой словарь если файл отсутствует или стратегия не найдена.
    Запустите backtest.py один раз для генерации файла.

    :param strategy: Название стратегии.
    :return: Словарь с win_rate, total_trades, ev, total_return_pct.
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
    """
    Хранилище сделок на SQLite с расчётом win rate и EV (expected value).

    WAL-режим позволяет дашборду читать данные одновременно с записью бота.
    Все DB-операции выполняются через run_in_executor чтобы не блокировать event loop.
    """

    def __init__(self, db_path: str = _DB_PATH):
        """
        Инициализирует хранилище, создаёт таблицу если не существует.

        :param db_path: Путь к файлу SQLite БД.
        """
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        # WAL mode allows concurrent readers (dashboard) alongside the writer (bot)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = asyncio.Lock()
        with self._conn:
            self._conn.execute(_DDL)

    # ── Executor helper ───────────────────────────────────────────────────────

    async def _run_db(self, func):
        """Запускает синхронную SQLite-операцию в thread pool executor.

        Предотвращает блокировку asyncio event loop при операциях с диском.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func)

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
        """
        Записывает открытие сделки в БД.

        :param symbol: Символ торговой пары.
        :param strategy: Название стратегии.
        :param action: Направление ("buy" или "sell").
        :param entry_price: Цена входа.
        :param quantity: Количество актива.
        :param confidence: Уверенность сигнала [0, 1].
        :param commission: Комиссия при открытии.
        :return: ID новой записи в таблице.
        """
        now_iso = datetime.now().isoformat()
        async with self._lock:
            def _write():
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
                        now_iso,
                    ),
                )
                self._conn.commit()
                return cur.lastrowid

            return await self._run_db(_write)

    async def record_close(
        self,
        trade_id: int,
        exit_price: float,
        commission: float = 0.0,
        leverage: int = 1,
    ) -> None:
        """
        Закрывает сделку и рассчитывает PnL из данных входа.

        PnL% считается от маржи (entry_price * qty / leverage), а не от
        notional, чтобы Kelly criterion получал корректные данные при плече.

        :param trade_id: ID записи в таблице.
        :param exit_price: Цена выхода.
        :param commission: Комиссия при закрытии.
        :param leverage: Плечо позиции (1 = без плеча).
        """
        now_iso = datetime.now().isoformat()

        async with self._lock:
            def _read():
                return self._conn.execute(
                    "SELECT action, entry_price, quantity, "
                    "commission FROM trades WHERE id = ?",
                    (trade_id,),
                ).fetchone()

            row = await self._run_db(_read)
            if row is None:
                logger.warning("trade_id=%s not found", trade_id)
                return
            action, entry_price, qty, entry_comm = row
            total_comm = entry_comm + commission
            if action == "buy":
                pnl = (exit_price - entry_price) * qty - total_comm
            else:
                pnl = (entry_price - exit_price) * qty - total_comm
            # УЛУЧШЕНИЕ 2: pnl_pct от маржи, а не от notional.
            # Маржа = notional / leverage → при leverage=3 результат корректен.
            lev = max(1, int(leverage))
            margin = entry_price * qty / lev
            pnl_pct = pnl / margin if margin > 0 else 0.0

            def _write():
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
                        now_iso,
                        trade_id,
                    ),
                )
                self._conn.commit()

            await self._run_db(_write)

    # ── Read ──────────────────────────────────────────────

    async def get_win_rate(
        self,
        strategy: Optional[str] = None,
        lookback: int = 50,
    ) -> float:
        """
        Возвращает долю прибыльных закрытых сделок [0.0–1.0].

        По умолчанию 0.5 при отсутствии данных.

        :param strategy: Фильтр по названию стратегии (None = все стратегии).
        :param lookback: Количество последних сделок для расчёта.
        :return: Win rate от 0.0 до 1.0.

        """
        where = (
            "WHERE status='closed' AND strategy=?"
            if strategy
            else "WHERE status='closed'"
        )
        params = (strategy, lookback) if strategy else (lookback,)
        sql = f"""
            SELECT
              COUNT(*) AS total,
              SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)
                  AS wins
            FROM (
              SELECT pnl, strategy FROM trades
              {where}
              ORDER BY id DESC LIMIT ?
            )
            """
        async with self._lock:
            def _read():
                return self._conn.execute(sql, params).fetchone()

            row = await self._run_db(_read)
        if not row or not row[0]:
            return 0.5  # default assumption: 50%
        return row[1] / row[0]

    async def get_expected_value(
        self,
        strategy: Optional[str] = None,
        lookback: int = 50,
    ) -> float:
        """
        Возвращает EV как долю: win_rate × avg_win_pct − loss_rate × |avg_loss_pct|.

        :param strategy: Фильтр по названию стратегии (None = все стратегии).
        :param lookback: Количество последних сделок для расчёта.
        :return: Expected value (может быть отрицательным).
        """
        where = (
            "WHERE status='closed' AND strategy=?"
            if strategy
            else "WHERE status='closed'"
        )
        params = (strategy, lookback) if strategy else (lookback,)
        sql = f"""
            SELECT pnl_pct FROM trades
            {where}
            ORDER BY id DESC LIMIT ?
            """
        async with self._lock:
            def _read():
                return self._conn.execute(sql, params).fetchall()

            rows = await self._run_db(_read)
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
        """
        Возвращает количество закрытых сделок в окне lookback.

        :param strategy: Фильтр по названию стратегии.
        :param lookback: Максимальное количество учитываемых сделок.
        :return: Количество закрытых сделок.
        """
        where = (
            "WHERE status='closed' AND strategy=?"
            if strategy
            else "WHERE status='closed'"
        )
        params = (strategy, lookback) if strategy else (lookback,)
        sql = (
            f"SELECT COUNT(*) FROM "
            f"(SELECT id FROM trades {where} "
            f"ORDER BY id DESC LIMIT ?)"
        )
        async with self._lock:
            def _read():
                return self._conn.execute(sql, params).fetchone()

            row = await self._run_db(_read)
        return row[0] if row else 0

    async def get_summary(self) -> Dict:
        """
        Возвращает агрегированную статистику по всем сделкам для дашборда.

        :return: Словарь с total_trades, closed_trades, win_rate,
            total_pnl, total_commissions.
        """
        sql = """
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
        async with self._lock:
            def _read():
                return self._conn.execute(sql).fetchone()

            row = await self._run_db(_read)
        total, wins, closed, total_pnl, total_comm = row
        win_rate = (wins / closed) if closed else 0.0
        return {
            "total_trades": total or 0,
            "closed_trades": closed or 0,
            "win_rate": win_rate,
            "total_pnl": total_pnl or 0.0,
            "total_commissions": total_comm or 0.0,
        }
