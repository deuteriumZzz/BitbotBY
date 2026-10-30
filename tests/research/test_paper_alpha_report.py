"""Тесты для paper_alpha_report.py — значимость альфы на реальных сделках."""

import sqlite3

import pytest

from src.research.paper_alpha_report import (
    build_profile_report,
    build_report,
    group_returns_by_profile,
    group_returns_by_strategy,
    load_closed_returns,
    load_closed_returns_by_profile,
)


def test_group_returns_by_strategy_groups_and_preserves_order():
    rows = [
        ("ema_crossover", 0.01),
        ("rsi_momentum", -0.02),
        ("ema_crossover", 0.03),
    ]
    grouped = group_returns_by_strategy(rows)
    assert list(grouped.keys()) == ["ema_crossover", "rsi_momentum"]
    assert grouped["ema_crossover"] == [0.01, 0.03]
    assert grouped["rsi_momentum"] == [-0.02]


def test_group_returns_by_strategy_empty():
    assert group_returns_by_strategy([]) == {}


def test_load_closed_returns_missing_file_returns_empty(tmp_path):
    missing = tmp_path / "does_not_exist.db"
    assert load_closed_returns(str(missing)) == []


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
        # symbol, strategy, action, entry, exit, qty, conf, pnl, pnl_pct, comm, entry_t, exit_t, status, context_flags
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
            "t1",
            "t2",
            "closed",
            '{"profile": "bluechip"}',
        ),
        (
            "BTC/USDT",
            "ema_crossover",
            "buy",
            100,
            98,
            1,
            0.6,
            -2.0,
            -0.02,
            0.1,
            "t3",
            "t4",
            "closed",
            '{"profile": "bluechip"}',
        ),
        (
            "ETH/USDT",
            "rsi_momentum",
            "sell",
            50,
            49,
            1,
            0.7,
            1.0,
            0.02,
            0.1,
            "t5",
            "t6",
            "closed",
            '{"profile": "meme"}',
        ),
        # no context_flags — pre-dates profile tagging, must land in "unknown"
        (
            "SOL/USDT",
            "breakout",
            "buy",
            20,
            21,
            1,
            0.6,
            1.0,
            0.05,
            0.1,
            "t8",
            "t9",
            "closed",
            "",
        ),
        # open trade — no pnl_pct yet, must be excluded
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
            "t7",
            None,
            "open",
            "",
        ),
    ]
    conn.executemany(
        "INSERT INTO trades (symbol, strategy, action, entry_price, exit_price, "
        "quantity, confidence, pnl, pnl_pct, commission, entry_time, exit_time, status, context_flags) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return str(db_path)


def test_load_closed_returns_reads_only_closed_with_pnl(trades_db):
    result = load_closed_returns(trades_db)
    assert result == [
        ("ema_crossover", 0.01),
        ("ema_crossover", -0.02),
        ("rsi_momentum", 0.02),
        ("breakout", 0.05),
    ]


def test_build_report_empty_returns_message():
    report = build_report({})
    assert "Нет закрытых сделок" in report


def test_build_report_insufficient_data_marked():
    report = build_report({"ema_crossover": [0.01, -0.01, 0.02]})
    assert "INSUFFICIENT DATA" in report


def test_build_report_includes_bonferroni_and_portfolio_section():
    report = build_report(
        {
            "ema_crossover": [0.01, -0.01, 0.02, 0.01, -0.005, 0.015],
            "rsi_momentum": [0.02, -0.01, 0.03, -0.02, 0.01, 0.02],
        }
    )
    assert "Bonferroni-поправки" in report
    assert "ПОРТФЕЛЬ ЦЕЛИКОМ" in report
    assert "ema_crossover" in report
    assert "rsi_momentum" in report


# ── group/report by market profile (п.6 роадмапа) ───────────────────────────


def test_group_returns_by_profile_groups_and_preserves_order():
    rows = [
        ("meme", 0.01),
        ("bluechip", -0.02),
        ("meme", 0.03),
    ]
    grouped = group_returns_by_profile(rows)
    assert list(grouped.keys()) == ["meme", "bluechip"]
    assert grouped["meme"] == [0.01, 0.03]
    assert grouped["bluechip"] == [-0.02]


def test_group_returns_by_profile_empty():
    assert group_returns_by_profile([]) == {}


def test_load_closed_returns_by_profile_missing_file_returns_empty(tmp_path):
    missing = tmp_path / "does_not_exist.db"
    assert load_closed_returns_by_profile(str(missing)) == []


def test_load_closed_returns_by_profile_reads_profile_and_defaults_unknown(trades_db):
    result = load_closed_returns_by_profile(trades_db)
    assert result == [
        ("bluechip", 0.01),
        ("bluechip", -0.02),
        ("meme", 0.02),
        ("unknown", 0.05),
    ]


def test_load_closed_returns_by_profile_handles_malformed_json(tmp_path):
    db_path = tmp_path / "trades_paper.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE trades (
            symbol TEXT, strategy TEXT, action TEXT, entry_price REAL,
            exit_price REAL, quantity REAL, confidence REAL, pnl REAL,
            pnl_pct REAL, commission REAL, entry_time TEXT, exit_time TEXT,
            status TEXT, context_flags TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO trades VALUES "
        "('BTC/USDT','ema_crossover','buy',100,101,1,0.6,1.0,0.01,0.1,"
        "'t1','t2','closed','not valid json')"
    )
    conn.commit()
    conn.close()
    assert load_closed_returns_by_profile(str(db_path)) == [("unknown", 0.01)]


def test_build_profile_report_empty_returns_message():
    report = build_profile_report({})
    assert "Нет закрытых сделок" in report


def test_build_profile_report_includes_bonferroni_and_portfolio_section():
    report = build_profile_report(
        {
            "meme": [0.02, -0.01, 0.03, -0.02, 0.01, 0.02],
            "bluechip": [0.01, -0.01, 0.02, 0.01, -0.005, 0.015],
        }
    )
    assert "Bonferroni-поправки" in report
    assert "ПОРТФЕЛЬ ЦЕЛИКОМ" in report
    assert "meme" in report
    assert "bluechip" in report
