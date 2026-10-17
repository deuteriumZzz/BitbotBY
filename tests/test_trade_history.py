import os
import tempfile

import pytest

from src.trade_history import TradeHistory


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    th = TradeHistory(db_path=path)
    yield th
    os.unlink(path)


async def test_record_open_returns_id(db):
    tid = await db.record_open(
        symbol="BTC/USDT",
        strategy="ema_crossover",
        action="buy",
        entry_price=60000.0,
        quantity=0.01,
        confidence=0.8,
        commission=0.6,
    )
    assert isinstance(tid, int)
    assert tid > 0


async def test_record_close_computes_pnl(db):
    tid = await db.record_open(
        "BTC/USDT",
        "ema_crossover",
        "buy",
        60000.0,
        0.01,
        0.8,
        0.6,
    )
    await db.record_close(tid, exit_price=62000.0, commission=0.62)
    summary = await db.get_summary()
    assert summary["closed_trades"] == 1
    assert summary["total_pnl"] > 0  # profitable trade


async def test_win_rate_empty(db):
    assert await db.get_win_rate() == 0.5  # default


async def test_win_rate_calculation(db):
    # 3 wins, 1 loss
    for price_out in [61000, 62000, 63000]:
        tid = await db.record_open(
            "BTC/USDT",
            "ema_crossover",
            "buy",
            60000.0,
            0.01,
            0.8,
        )
        await db.record_close(tid, exit_price=price_out)
    tid = await db.record_open(
        "BTC/USDT",
        "ema_crossover",
        "buy",
        60000.0,
        0.01,
        0.8,
    )
    await db.record_close(tid, exit_price=59000.0)  # loss
    wr = await db.get_win_rate("ema_crossover")
    assert wr == pytest.approx(0.75, abs=0.01)


async def test_get_trade_count(db):
    for _ in range(3):
        tid = await db.record_open(
            "BTC/USDT",
            "macd_crossover",
            "buy",
            60000.0,
            0.01,
            0.8,
        )
        await db.record_close(tid, exit_price=61000.0)
    assert await db.get_trade_count("macd_crossover") == 3
    assert await db.get_trade_count("ema_crossover") == 0


async def test_reasoning_and_exit_reason_are_kept(db):
    tid = await db.record_open(
        "BTC/USDT",
        "ema_crossover",
        "buy",
        60000.0,
        0.01,
        0.8,
        0.6,
        reasoning="EMA 9 пересекла EMA 21 снизу",
    )
    (opened,) = await db.get_open_trades()
    assert opened["reasoning"] == "EMA 9 пересекла EMA 21 снизу"
    pnl = await db.record_close(tid, exit_price=62000.0, commission=0.62, reason="tp")
    assert pnl is not None and pnl > 0
    row = db._conn.execute("SELECT exit_reason FROM trades WHERE id=?", (tid,))
    assert row.fetchone()[0] == "tp"
    # Повторное закрытие несуществующей сделки — None, без исключения.
    assert await db.record_close(9999, exit_price=1.0) is None


async def test_partial_closes_add_up_to_the_same_result_as_one_close(db):
    # Покупка 10 по 100, комиссия входа 1.0; закрыли 4 по 110 (комиссия 0.44),
    # остаток 6 по 120 (комиссия 0.72).
    tid = await db.record_open("BTC/USDT", "x", "buy", 100.0, 10.0, 0.8, 1.0)
    part = await db.record_partial_close(tid, 4.0, 110.0, 0.44)
    final = await db.record_close(tid, exit_price=120.0, commission=0.72)
    # (110-100)*4 - 0.4 - 0.44 = 39.16;  (120-100)*6 - 0.6 - 0.72 = 118.68
    assert part == pytest.approx(39.16)
    assert final == pytest.approx(39.16 + 118.68)
    row = db._conn.execute(
        "SELECT commission, closed_qty, status FROM trades WHERE id=?", (tid,)
    ).fetchone()
    assert row[0] == pytest.approx(1.0 + 0.44 + 0.72)  # все комиссии учтены
    assert row[1] == 4.0 and row[2] == "closed"


async def test_short_partial_close(db):
    tid = await db.record_open("ETH/USDT", "x", "sell", 100.0, 10.0, 0.8, 1.0)
    part = await db.record_partial_close(tid, 5.0, 90.0, 0.0)
    final = await db.record_close(tid, exit_price=95.0, commission=0.0)
    assert part == pytest.approx(50.0 - 0.5)
    assert final == pytest.approx(49.5 + 25.0 - 0.5)


async def test_partial_close_of_missing_trade_is_none(db):
    assert await db.record_partial_close(999, 1.0, 100.0) is None


async def test_execution_fields_roundtrip_and_old_rows_are_null(db):
    new = await db.record_open(
        "BTC/USDT", "s", "buy", 101.0, 1.0, 0.5, 0.0,
        signal_price=100.0, fill_price=101.0, order_sent_at=10.0, order_filled_at=10.2,
    )  # fmt: skip
    old = await db.record_open("BTC/USDT", "s", "buy", 101.0, 1.0, 0.5, 0.0)
    rows = {
        r[0]: r[1:]
        for r in db._conn.execute(
            "SELECT id, signal_price, fill_price, order_sent_at, order_filled_at "
            "FROM trades"
        )
    }
    assert rows[new] == (100.0, 101.0, 10.0, 10.2)
    assert rows[old] == (None, None, None, None)


def test_migration_adds_execution_columns_to_legacy_db(tmp_path):
    import sqlite3

    path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE trades (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT "
        "NOT NULL, strategy TEXT NOT NULL, action TEXT NOT NULL, entry_price REAL "
        "NOT NULL, exit_price REAL, quantity REAL NOT NULL, confidence REAL NOT NULL "
        "DEFAULT 0.0, pnl REAL, pnl_pct REAL, commission REAL NOT NULL DEFAULT 0.0, "
        "entry_time TEXT NOT NULL, exit_time TEXT, status TEXT NOT NULL DEFAULT "
        "'open', context_flags TEXT NOT NULL DEFAULT '')"
    )
    conn.execute(
        "INSERT INTO trades (symbol,strategy,action,entry_price,quantity,entry_time)"
        " VALUES ('A','s','buy',1,1,'2026-01-01')"
    )
    conn.commit()
    conn.close()
    th = TradeHistory(db_path=path)
    row = th._conn.execute(
        "SELECT signal_price, order_filled_at FROM trades"
    ).fetchone()
    assert row == (None, None)
