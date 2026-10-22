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

_DB_PATH = os.path.join(
    "data",
    (
        "trades_paper.db"
        if os.getenv("PAPER_TRADING", "").lower() == "true"
        else "trades_live.db"
    ),
)

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


# ── Режим счёта и состояние бота (для MiniApp / Desktop API) ─────────────────


def paper_mode() -> bool:
    """Paper/Live с учётом переключения из Telegram или приложения.

    Бот хранит выбор в Redis (`bot:paper_trading_override`), а env дашборда
    его не видит — поэтому сначала Redis, потом env.
    """
    r = get_redis()
    if r:
        try:
            v = r.get("bot:paper_trading_override")
            if v in ("1", "0"):
                return v == "1"
        except Exception:
            pass
    return os.getenv("PAPER_TRADING", "").lower() == "true"


def _mode_suffix() -> str:
    return "paper" if paper_mode() else "live"


def portfolio_key() -> str:
    """Бот пишет портфель под portfolio_state_{paper|live} (portfolio_manager)."""
    return f"portfolio_state_{_mode_suffix()}"


def db_path() -> str:
    return os.path.join("data", f"trades_{_mode_suffix()}.db")


def get_open_positions() -> List[dict]:
    """Открытые позиции из снимка бота `monitored_positions_{paper|live}`."""
    raw = redis_get(f"monitored_positions_{_mode_suffix()}") or {}
    out = []
    for sym, p in raw.items():
        if not p:
            continue
        out.append(
            {
                "id": str(p.get("trade_id") or sym),
                "trade_id": p.get("trade_id"),
                "symbol": sym,
                "side": "long" if p.get("side", "buy") == "buy" else "short",
                "entry": p.get("entry"),
                "qty": p.get("qty"),
                "sl": p.get("stop_loss"),
                "tp": p.get("take_profit"),
                "pnl_pct": p.get("pnl_pct"),
                "partial_tp_done": bool(p.get("partial_tp_triggered")),
                "strategy": p.get("strategy"),
                "confidence": p.get("confidence"),
                "reasoning": p.get("reasoning", ""),
                "opened_at": p.get("opened_at"),
                "trace": p.get("trace") or [],
            }
        )
    return out


def _aware(iso: Optional[str]) -> Optional[str]:
    """Время сделки с часовым поясом сервера.

    В базе оно без пояса (datetime.now() бота), а браузер прочитал бы его как
    своё местное — в Docker (UTC) это сдвиг на разницу поясов.
    """
    if not iso:
        return iso
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    return (dt if dt.tzinfo else dt.astimezone()).isoformat()


def _steps(raw: object) -> List[dict]:
    """Шаги решения из колонки trace (JSON); у старых сделок — пусто."""
    try:
        val = json.loads(str(raw or "[]"))
    except ValueError:
        return []
    return val if isinstance(val, list) else []


def get_closed_trades(since: Optional[datetime] = None) -> List[dict]:
    """Закрытые сделки по возрастанию времени выхода.

    `pnl` в базе — уже за вычетом комиссий.
    """
    path = db_path()
    if not os.path.exists(path):
        return []
    sql = (
        "SELECT symbol, strategy, pnl, commission, exit_time FROM trades "
        "WHERE status='closed' AND pnl IS NOT NULL AND exit_time IS NOT NULL"
    )
    params: tuple = ()
    if since is not None:
        sql += " AND exit_time >= ?"
        params = (since.isoformat(),)
    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql + " ORDER BY exit_time", params).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error("Closed trades query error: %s", e)
        return []


def get_journal(since: Optional[datetime] = None, limit: int = 1000) -> List[dict]:
    """Журнал закрытых сделок — свежие сверху.

    `SELECT *`: база могла ещё не получить колонки reasoning/exit_reason
    (их добавляет бот при старте) — тогда поля просто пустые.
    """
    path = db_path()
    if not os.path.exists(path):
        return []
    sql = "SELECT * FROM trades WHERE status='closed' AND pnl IS NOT NULL"
    params: list = []
    if since is not None:
        sql += " AND exit_time >= ?"
        params.append(since.isoformat())
    params.append(limit)
    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql + " ORDER BY exit_time DESC LIMIT ?", params)
        out = [dict(r) for r in rows.fetchall()]
        conn.close()
    except Exception as e:
        logger.error("Journal query error: %s", e)
        return []
    return [
        {
            "id": r["id"],
            "symbol": r["symbol"],
            "side": "long" if r["action"] == "buy" else "short",
            "strategy": r["strategy"],
            "confidence": r["confidence"],
            "entry": r["entry_price"],
            "exit": r["exit_price"],
            "qty": r["quantity"],
            "pnl": r["pnl"],  # уже за вычетом комиссий
            "pnl_pct": r["pnl_pct"],  # от маржи
            "fees": r["commission"],
            "opened": _aware(r["entry_time"]),
            "closed": _aware(r["exit_time"]),
            "reason": r.get("exit_reason") or "",
            "reasoning": r.get("reasoning") or "",
            "trace": _steps(r.get("trace")),
        }
        for r in out
    ]


EXEC_MIN_TRADES = 20


def get_execution_rows(since: Optional[datetime] = None) -> List[dict]:
    """Сделки с записанными данными исполнения (старые строки с NULL пропускаются)."""
    path = db_path()
    if not os.path.exists(path):
        return []
    sql = (
        "SELECT action, signal_price, fill_price, order_sent_at, order_filled_at "
        "FROM trades WHERE signal_price > 0 AND fill_price > 0 "
        "AND order_sent_at IS NOT NULL AND order_filled_at IS NOT NULL"
    )
    params: tuple = ()
    if since is not None:
        sql += " AND entry_time >= ?"
        params = (since.isoformat(),)
    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []  # база ещё без колонок исполнения — бот добавит их при старте
    except Exception as e:
        logger.error("Execution rows query error: %s", e)
        return []


def compute_execution(rows: List[dict]) -> dict:
    """Задержка (отправка → исполнение, мс) и проскальзывание (исполнение против
    цены решения, %, со знаком по направлению: плюс — хуже для сделки).

    Средние отдаются только от EXEC_MIN_TRADES сделок: на меньшей выборке они случайны.
    """
    n = len(rows)
    out: dict = {
        "n": n,
        "min_n": EXEC_MIN_TRADES,
        "latency_ms_avg": None,
        "slippage_pct_avg": None,
        "slippage_pct_worst": None,
    }
    if n < EXEC_MIN_TRADES:
        return out
    lat = [(r["order_filled_at"] - r["order_sent_at"]) * 1000 for r in rows]
    slip = [
        (1 if r["action"] == "buy" else -1)
        * (r["fill_price"] - r["signal_price"])
        / r["signal_price"]
        * 100
        for r in rows
    ]
    out["latency_ms_avg"] = round(sum(lat) / n, 1)
    out["slippage_pct_avg"] = round(sum(slip) / n, 4)
    out["slippage_pct_worst"] = round(max(slip), 4)
    return out


SHARPE_MIN_TRADES = 50


def compute_metrics(trades: List[dict], start_balance: float) -> dict:
    """Метрики эффективности по закрытым сделкам (порядок — по времени выхода).

    `pnl` в базе уже за вычетом комиссий, поэтому `net` = сумма pnl,
    а `realized` (до комиссий) = net + fees. Sharpe — по дневной доходности
    и только от SHARPE_MIN_TRADES сделок: на меньшей выборке он случаен.
    """
    n = len(trades)
    pnls = [float(t["pnl"]) for t in trades]
    fees = sum(float(t.get("commission") or 0) for t in trades)
    net = sum(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [-p for p in pnls if p < 0]
    avg_win = sum(wins) / len(wins) if wins else None
    avg_loss = sum(losses) / len(losses) if losses else None

    equity = peak = float(start_balance)
    max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)

    sharpe = None
    if n >= SHARPE_MIN_TRADES and start_balance > 0:
        daily: dict = {}
        for t in trades:
            day = str(t["exit_time"])[:10]
            daily[day] = daily.get(day, 0.0) + float(t["pnl"])
        rets = [v / start_balance for v in daily.values()]
        if len(rets) > 1:
            mean = sum(rets) / len(rets)
            sd = (sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)) ** 0.5
            sharpe = round(mean / sd * 365**0.5, 3) if sd > 0 else None

    return {
        "n": n,
        "net": round(net, 2),
        "fees": round(fees, 2),
        "realized": round(net + fees, 2),
        "win_rate": round(len(wins) / n, 4) if n else None,
        "profit_factor": round(sum(wins) / sum(losses), 3) if losses else None,
        "avg_win": round(avg_win, 2) if avg_win is not None else None,
        "avg_loss": round(avg_loss, 2) if avg_loss is not None else None,
        "rr": round(avg_win / avg_loss, 3) if avg_win and avg_loss else None,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe": sharpe,
    }


def daily_pnl(trades: List[dict]) -> List[dict]:
    """P&L по дням выхода: [{date, net, fees, trades}] по возрастанию даты."""
    days: dict = {}
    for t in trades:
        d = days.setdefault(
            str(t["exit_time"])[:10], {"net": 0.0, "fees": 0.0, "trades": 0}
        )
        d["net"] += float(t["pnl"])
        d["fees"] += float(t.get("commission") or 0)
        d["trades"] += 1
    return [
        {
            "date": k,
            "net": round(v["net"], 2),
            "fees": round(v["fees"], 2),
            "trades": v["trades"],
        }
        for k, v in sorted(days.items())
    ]
