import os
import tempfile
import pytest
from src.trade_history import TradeHistory


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(
        suffix=".db", delete=False
    ) as f:
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
        "BTC/USDT", "ema_crossover", "buy",
        60000.0, 0.01, 0.8, 0.6,
    )
    await db.record_close(
        tid, exit_price=62000.0, commission=0.62
    )
    summary = await db.get_summary()
    assert summary["closed_trades"] == 1
    assert summary["total_pnl"] > 0  # profitable trade


async def test_win_rate_empty(db):
    assert await db.get_win_rate() == 0.5  # default


async def test_win_rate_calculation(db):
    # 3 wins, 1 loss
    for price_out in [61000, 62000, 63000]:
        tid = await db.record_open(
            "BTC/USDT", "ema_crossover", "buy",
            60000.0, 0.01, 0.8,
        )
        await db.record_close(tid, exit_price=price_out)
    tid = await db.record_open(
        "BTC/USDT", "ema_crossover", "buy",
        60000.0, 0.01, 0.8,
    )
    await db.record_close(tid, exit_price=59000.0)  # loss
    wr = await db.get_win_rate("ema_crossover")
    assert wr == pytest.approx(0.75, abs=0.01)


async def test_get_trade_count(db):
    for _ in range(3):
        tid = await db.record_open(
            "BTC/USDT", "macd_crossover", "buy",
            60000.0, 0.01, 0.8,
        )
        await db.record_close(tid, exit_price=61000.0)
    assert await db.get_trade_count("macd_crossover") == 3
    assert await db.get_trade_count("ema_crossover") == 0
