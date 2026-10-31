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
from typing import Any, Dict, List, Optional, Tuple

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


def _db_path() -> str:
    from config import Config  # noqa: PLC0415

    suffix = "paper" if Config.PAPER_TRADING else "live"
    return os.path.join("data", f"trades_{suffix}.db")


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
    status      TEXT    NOT NULL DEFAULT 'open',
    context_flags TEXT  NOT NULL DEFAULT ''
);
"""


class TradeHistory:
    """
    Хранилище сделок на SQLite с расчётом win rate и EV (expected value).

    WAL-режим позволяет дашборду читать данные одновременно с записью бота.
    Все DB-операции выполняются через run_in_executor чтобы не блокировать event loop.
    """

    def __init__(self, db_path: str = ""):
        """
        Инициализирует хранилище, создаёт таблицу если не существует.

        :param db_path: Путь к файлу SQLite БД.
        """
        if not db_path:
            db_path = _db_path()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        # WAL mode allows concurrent readers (dashboard) alongside the writer (bot)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = asyncio.Lock()
        with self._conn:
            self._conn.execute(_DDL)
            # Колонки, добавленные после первых версий: в старых БД их нет.
            # reasoning — почему открыта (текст AI или стратегии),
            # exit_reason — как закрыта (tp, sl, trail, manual_app, …).
            # trace — шаги решения по узлам конвейера (JSON, core/decision_trace.py).
            for column in ("context_flags", "reasoning", "exit_reason", "trace"):
                try:
                    self._conn.execute(
                        f"ALTER TABLE trades ADD COLUMN {column} "
                        "TEXT NOT NULL DEFAULT ''"
                    )
                except sqlite3.OperationalError:
                    pass  # column already exists
            # Частичные закрытия: сколько уже закрыто, какой результат зафиксирован
            # и сколько комиссий ушло на эти закрытия.
            for column in ("closed_qty", "realized_pnl", "partial_fees"):
                try:
                    self._conn.execute(
                        f"ALTER TABLE trades ADD COLUMN {column} "
                        "REAL NOT NULL DEFAULT 0"
                    )
                except sqlite3.OperationalError:
                    pass  # column already exists

            # Качество исполнения: цена решения, цена исполнения и время
            # отправки/исполнения ордера (epoch, сек). У старых строк — NULL.
            for column in (
                "signal_price",
                "fill_price",
                "order_sent_at",
                "order_filled_at",
            ):
                try:
                    self._conn.execute(f"ALTER TABLE trades ADD COLUMN {column} REAL")
                except sqlite3.OperationalError:
                    pass  # column already exists

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
        context_flags: str = "",
        reasoning: str = "",
        signal_price: Optional[float] = None,
        fill_price: Optional[float] = None,
        order_sent_at: Optional[float] = None,
        order_filled_at: Optional[float] = None,
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
        :param context_flags: JSON со сработавшими контекстными фильтрами
            (funding/OI/liquidation/ETF/etc.) на момент входа — для
            последующей корреляции с исходом сделки (win/loss).
        :param signal_price: Цена в момент решения (до поправок).
        :param fill_price: Цена исполнения (paper — симулированная).
        :param order_sent_at: Время отправки ордера, epoch сек.
        :param order_filled_at: Время исполнения, epoch сек.
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
                       entry_time, context_flags, reasoning,
                       signal_price, fill_price,
                       order_sent_at, order_filled_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        context_flags,
                        reasoning[:1000],
                        signal_price,
                        fill_price,
                        order_sent_at,
                        order_filled_at,
                    ),
                )
                self._conn.commit()
                return cur.lastrowid

            return await self._run_db(_write)

    async def set_trace(self, trade_id: int, steps: List[Dict[str, Any]]) -> None:
        """Сохраняет шаги решения («Почему открыта» в журнале) как JSON."""
        payload = json.dumps(steps, ensure_ascii=False)
        async with self._lock:

            def _write():
                self._conn.execute(
                    "UPDATE trades SET trace = ? WHERE id = ?", (payload, trade_id)
                )
                self._conn.commit()

            await self._run_db(_write)

    async def record_partial_close(
        self, trade_id: int, qty: float, exit_price: float, commission: float = 0.0
    ) -> Optional[float]:
        """Фиксирует результат частичного закрытия, сделка остаётся открытой.

        Доля комиссии за вход, приходящаяся на закрытый объём, списывается сразу;
        итог сделки при полном закрытии складывается из частей.

        :return: зафиксированный результат части или None, если сделки нет.
        """
        async with self._lock:

            def _read():
                return self._conn.execute(
                    "SELECT action, entry_price, quantity, commission, "
                    "closed_qty, partial_fees FROM trades "
                    "WHERE id = ? AND status = 'open'",
                    (trade_id,),
                ).fetchone()

            row = await self._run_db(_read)
            if row is None:
                logger.warning("partial close: open trade_id=%s not found", trade_id)
                return None
            action, entry_price, total_qty, comm, closed, fees = row
            qty = min(qty, max(0.0, total_qty - closed))
            entry_comm = comm - fees
            share = entry_comm * qty / total_qty if total_qty else 0.0
            move = (exit_price - entry_price) * qty
            pnl = move if action == "buy" else -move
            pnl -= share + commission

            def _write():
                self._conn.execute(
                    "UPDATE trades SET closed_qty = closed_qty + ?, "
                    "realized_pnl = realized_pnl + ?, "
                    "partial_fees = partial_fees + ?, "
                    "commission = commission + ? WHERE id = ?",
                    (qty, pnl, commission, commission, trade_id),
                )
                self._conn.commit()

            await self._run_db(_write)
            return pnl

    async def record_close(
        self,
        trade_id: int,
        exit_price: float,
        commission: float = 0.0,
        leverage: int = 1,
        reason: str = "",
    ) -> Optional[float]:
        """
        Закрывает сделку и рассчитывает PnL из данных входа.

        PnL% считается от маржи (entry_price * qty / leverage), а не от
        notional, чтобы Kelly criterion получал корректные данные при плече.

        :param trade_id: ID записи в таблице.
        :param exit_price: Цена выхода.
        :param commission: Комиссия при закрытии.
        :param leverage: Плечо позиции (1 = без плеча).
        :param reason: Как закрыта: tp, sl, trail, manual_app, …
        :return: PnL за вычетом комиссий или None, если сделки нет.
        """
        now_iso = datetime.now().isoformat()

        async with self._lock:

            def _read():
                return self._conn.execute(
                    "SELECT action, entry_price, quantity, commission, "
                    "closed_qty, realized_pnl, partial_fees "
                    "FROM trades WHERE id = ?",
                    (trade_id,),
                ).fetchone()

            row = await self._run_db(_read)
            if row is None:
                logger.warning("trade_id=%s not found", trade_id)
                return None
            action, entry_price, qty, comm, closed, realized, partial_fees = row
            # Остаток после частичных закрытий: их результат уже в realized_pnl,
            # комиссия за вход делится пропорционально объёму.
            remaining = max(0.0, qty - closed)
            entry_comm = (comm - partial_fees) * (remaining / qty if qty else 1.0)
            move = (exit_price - entry_price) * remaining
            pnl = realized + (move if action == "buy" else -move)
            pnl -= entry_comm + commission
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
                        status='closed', exit_reason=?
                    WHERE id=?
                    """,
                    (
                        exit_price,
                        commission,
                        pnl,
                        pnl_pct,
                        now_iso,
                        reason,
                        trade_id,
                    ),
                )
                self._conn.commit()

            await self._run_db(_write)
            return pnl

    async def get_open_trades(self) -> list[Dict]:
        """
        Возвращает все сделки со статусом 'open' — используется для сверки
        _monitored с БД после рестарта бота в paper-режиме.

        :return: Список словарей с id, symbol, action, entry_price,
            quantity, confidence, entry_time, strategy, reasoning.
        """
        async with self._lock:

            def _read():
                cur = self._conn.execute(
                    "SELECT id, symbol, action, entry_price, quantity, "
                    "confidence, entry_time, strategy, reasoning "
                    "FROM trades WHERE status='open' "
                    "ORDER BY entry_time ASC"
                )
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]

            return await self._run_db(_read)

    async def mark_orphaned(self, trade_id: int) -> None:
        """
        Помечает сделку как потерянную (не найдена в _monitored после
        рестарта, либо дубликат по тому же символу).

        Статус 'orphaned' исключён из win-rate/EV/summary — те фильтруют
        status='closed' — поэтому запись не искажает статистику, но и не
        зависает в 'open' навсегда без выдуманного PnL.

        :param trade_id: ID записи в таблице.
        """
        now_iso = datetime.now().isoformat()
        async with self._lock:

            def _write():
                self._conn.execute(
                    "UPDATE trades SET status='orphaned', exit_time=? WHERE id=?",
                    (now_iso, trade_id),
                )
                self._conn.commit()

            await self._run_db(_write)

    # ── Read ──────────────────────────────────────────────

    async def _fetch_recent_closed(
        self,
        strategy: Optional[str],
        market_type: str,
        lookback: int,
    ) -> list:
        """Возвращает [(pnl, pnl_pct), ...] последних закрытых сделок.

        При заданном market_type фильтрует по context_flags["market_type"]
        (тег пишет signal_combiner.py при входе) — сделки другого типа
        рынка (spot/linear) не должны влиять на live win_rate/EV текущего
        режима, механика сделки принципиально разная (плечо/шорт/funding
        только на linear). Записи без тега (до появления инструментации)
        не отбрасываются. Тег живёт в TEXT-колонке, а не в отдельном
        столбце — фильтруем в Python, поэтому при активном фильтре берём
        запас сверх lookback, чтобы после фильтрации осталось достаточно.

        :param strategy: Фильтр по названию стратегии (None = все).
        :param market_type: "spot"|"linear"; "" = без фильтра (как раньше).
        :param lookback: Сколько последних (уже отфильтрованных) сделок вернуть.
        :return: [(pnl, pnl_pct), ...] от новых к старым, максимум lookback.
        """
        where = (
            "WHERE status='closed' AND strategy=?"
            if strategy
            else "WHERE status='closed'"
        )
        fetch_limit = max(lookback * 5, 200) if market_type else lookback
        params = (strategy, fetch_limit) if strategy else (fetch_limit,)
        sql = f"""
            SELECT pnl, pnl_pct, context_flags FROM trades
            {where}
            ORDER BY id DESC LIMIT ?
            """
        async with self._lock:

            def _read():
                return self._conn.execute(sql, params).fetchall()

            rows = await self._run_db(_read)

        if not market_type:
            return [(r[0], r[1]) for r in rows]

        result = []
        for pnl, pnl_pct, context_flags in rows:
            row_mtype = ""
            if context_flags:
                try:
                    row_mtype = json.loads(context_flags).get("market_type", "")
                except (json.JSONDecodeError, AttributeError):
                    pass
            if not row_mtype or row_mtype == market_type:
                result.append((pnl, pnl_pct))
                if len(result) >= lookback:
                    break
        return result

    async def recent_results(
        self, strategy: str, n: int, after_id: int = 0
    ) -> List[Tuple[int, float]]:
        """(id, pnl) последних n закрытых сделок стратегии новее after_id."""
        sql = (
            "SELECT id, pnl FROM trades WHERE status='closed' AND strategy=? "
            "AND id>? ORDER BY id DESC LIMIT ?"
        )
        async with self._lock:

            def _read():
                return self._conn.execute(sql, (strategy, after_id, n)).fetchall()

            rows = await self._run_db(_read)
        return [(int(i), float(p or 0.0)) for i, p in rows]

    async def get_win_rate(
        self,
        strategy: Optional[str] = None,
        lookback: int = 50,
        market_type: str = "",
    ) -> float:
        """
        Возвращает долю прибыльных закрытых сделок [0.0–1.0].

        По умолчанию 0.5 при отсутствии данных.

        :param strategy: Фильтр по названию стратегии (None = все стратегии).
        :param lookback: Количество последних сделок для расчёта.
        :param market_type: "spot"|"linear" — фильтр по типу рынка на момент
            входа; "" = без фильтра (обратная совместимость).
        :return: Win rate от 0.0 до 1.0.
        """
        rows = await self._fetch_recent_closed(strategy, market_type, lookback)
        if not rows:
            return 0.5  # default assumption: 50%
        wins = sum(1 for pnl, _ in rows if pnl is not None and pnl > 0)
        return wins / len(rows)

    async def get_expected_value(
        self,
        strategy: Optional[str] = None,
        lookback: int = 50,
        market_type: str = "",
    ) -> float:
        """
        Возвращает EV как долю: win_rate × avg_win_pct − loss_rate × |avg_loss_pct|.

        :param strategy: Фильтр по названию стратегии (None = все стратегии).
        :param lookback: Количество последних сделок для расчёта.
        :param market_type: "spot"|"linear" — фильтр по типу рынка на момент
            входа; "" = без фильтра (обратная совместимость).
        :return: Expected value (может быть отрицательным).
        """
        rows = await self._fetch_recent_closed(strategy, market_type, lookback)
        if not rows:
            return 0.0
        pnl_pcts = [pnl_pct for _, pnl_pct in rows]
        wins = [p for p in pnl_pcts if p is not None and p > 0]
        losses = [p for p in pnl_pcts if p is not None and p <= 0]
        n = len(pnl_pcts)
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
        market_type: str = "",
    ) -> int:
        """
        Возвращает количество закрытых сделок в окне lookback.

        :param strategy: Фильтр по названию стратегии.
        :param lookback: Максимальное количество учитываемых сделок.
        :param market_type: "spot"|"linear" — фильтр по типу рынка на момент
            входа; "" = без фильтра (обратная совместимость).
        :return: Количество закрытых сделок.
        """
        rows = await self._fetch_recent_closed(strategy, market_type, lookback)
        return len(rows)

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
