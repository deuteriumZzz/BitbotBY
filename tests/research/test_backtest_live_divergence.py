"""Тесты для backtest_live_divergence.py — п.5 роадмапа (live vs backtest parity)."""

import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.research.backtest_live_divergence import (
    build_report,
    compute_divergence,
    find_reference,
    load_cache,
    load_closed_trades,
)


def test_load_cache_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_cache("BTC/USDT", "15m") is None


def _write_cache(tmp_path, symbol, timeframe, start_ms, step_ms, n, price=100.0):
    cache_dir = tmp_path / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = symbol.replace("/", "_")
    rows = []
    for i in range(n):
        rows.append(
            {
                "timestamp": start_ms + i * step_ms,
                "open": price,
                "high": price * 1.001,
                "low": price * 0.999,
                "close": price,
                "volume": 100.0,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(cache_dir / f"{safe}_{timeframe}.csv")
    return df


def test_load_cache_reads_and_sorts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_cache(tmp_path, "BTC/USDT", "15m", start_ms=1_000_000, step_ms=900_000, n=5)
    df = load_cache("BTC/USDT", "15m")
    assert df is not None
    assert list(df["timestamp"]) == sorted(df["timestamp"])
    assert len(df) == 5


def test_find_reference_uses_utc_offset_and_picks_closest_timeframe(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    # Свеча закрылась в UTC; entry_time пишется как datetime.now() (локальное).
    utc_now = datetime.utcnow()
    local_now = datetime.now()
    candle_utc = utc_now - timedelta(minutes=20)
    start_ms = int((candle_utc - timedelta(minutes=45)).timestamp() * 1000)
    _write_cache(
        tmp_path,
        "BTC/USDT",
        "15m",
        start_ms=start_ms,
        step_ms=900_000,
        n=4,
        price=100.0,
    )

    entry_time_iso = local_now.isoformat()
    ref = find_reference("BTC/USDT", entry_time_iso)
    assert ref is not None
    close_price, window, timeframe = ref
    assert timeframe == "15m"
    assert close_price == pytest.approx(100.0)
    assert not window.empty


def test_find_reference_no_cache_returns_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert find_reference("BTC/USDT", datetime.now().isoformat()) is None


def test_compute_divergence_matches_reference_price(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    utc_now = datetime.utcnow()
    candle_utc = utc_now - timedelta(minutes=10)
    start_ms = int((candle_utc - timedelta(minutes=45 * 15)).timestamp() * 1000)
    _write_cache(
        tmp_path,
        "BTC/USDT",
        "15m",
        start_ms=start_ms,
        step_ms=900_000,
        n=60,
        price=100.0,
    )

    trade = {
        "symbol": "BTC/USDT",
        "strategy": "ema_crossover",
        "action": "buy",
        "entry_price": 100.5,
        "quantity": 1.0,
        "commission": 0.1,
        "entry_time": datetime.now().isoformat(),
    }
    record = compute_divergence(trade)
    assert record is not None
    # implied_price ~ 100 * (1 + small impact) — entry 100.5 должен дать
    # положительное расхождение (реальная цена выше того, что предсказал бы бэктест)
    assert record.entry_divergence_pct > 0
    assert record.commission_divergence_pct is not None


def test_compute_divergence_no_data_returns_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trade = {
        "symbol": "BTC/USDT",
        "strategy": "ema_crossover",
        "action": "buy",
        "entry_price": 100.5,
        "quantity": 1.0,
        "commission": 0.1,
        "entry_time": datetime.now().isoformat(),
    }
    assert compute_divergence(trade) is None


def test_load_closed_returns_missing_file_returns_empty(tmp_path):
    missing = tmp_path / "does_not_exist.db"
    assert load_closed_trades(str(missing)) == []


@pytest.fixture
def trades_db(tmp_path):
    db_path = tmp_path / "trades_paper.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE trades (
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
            context_flags TEXT NOT NULL DEFAULT ''
        )
        """
    )
    rows = [
        (
            "BTC/USDT",
            "ema_crossover",
            "buy",
            100,
            101,
            1,
            0.6,
            1.0,
            0.01,
            0.1,
            "2026-01-01T00:00:00",
            "2026-01-01T01:00:00",
            "closed",
        ),
        # open trade — must be excluded (no entry_price check needed, but status filters it)
        (
            "BTC/USDT",
            "ema_crossover",
            "buy",
            100,
            None,
            1,
            0.6,
            None,
            None,
            0.1,
            "2026-01-01T02:00:00",
            None,
            "open",
        ),
    ]
    conn.executemany(
        "INSERT INTO trades (symbol, strategy, action, entry_price, exit_price, "
        "quantity, confidence, pnl, pnl_pct, commission, entry_time, exit_time, status) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return str(db_path)


def test_load_closed_trades_reads_only_closed(trades_db):
    result = load_closed_trades(trades_db)
    assert len(result) == 1
    assert result[0]["symbol"] == "BTC/USDT"
    assert result[0]["entry_time"] == "2026-01-01T00:00:00"


def test_build_report_empty_returns_message():
    report = build_report([], total_trades=0)
    assert "Нет сопоставленных сделок" in report


def test_build_report_skipped_count_and_warning_threshold():
    from src.research.backtest_live_divergence import DivergenceRecord

    records = [
        DivergenceRecord(
            symbol="BTC/USDT",
            strategy="ema_crossover",
            entry_time="2026-01-01T00:00:00",
            entry_divergence_pct=0.01,
            commission_divergence_pct=0.02,
            timeframe_used="15m",
        )
    ]
    report = build_report(records, total_trades=3)
    assert "1/3" in report
    assert "2 пропущено" in report
    assert "⚠" in report  # 1% > 0.5% warning threshold
    assert "BTC/USDT" in report
