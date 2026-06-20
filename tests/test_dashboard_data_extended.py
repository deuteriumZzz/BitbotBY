"""
Тесты src/dashboard/data.py: Redis-хелперы, SQLite, healthcheck.

Покрывают строки 32-44, 55-56, 83-85, 114-116, 127-134:
- get_redis(): успешный пинг и ошибка пинга
- redis_get(): успешный hit / json-ошибка
- get_trades(): DB exception
- get_stats(): DB exception
- check_healthcheck(): файл существует, stale timestamp, content > 50 chars, bad ISO
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.dashboard.data import (
    check_healthcheck,
    get_redis,
    get_stats,
    get_trades,
    redis_get,
)


# ---------------------------------------------------------------------------
# get_redis
# ---------------------------------------------------------------------------

class TestGetRedis:
    def test_returns_client_when_ping_succeeds(self):
        mock_redis_lib = MagicMock()
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis_lib.Redis.return_value = mock_client

        with patch.dict("sys.modules", {"redis": mock_redis_lib}):
            result = get_redis()

        assert result is mock_client

    def test_returns_none_when_ping_raises(self):
        mock_redis_lib = MagicMock()
        mock_client = MagicMock()
        mock_client.ping.side_effect = ConnectionRefusedError("refused")
        mock_redis_lib.Redis.return_value = mock_client

        with patch.dict("sys.modules", {"redis": mock_redis_lib}):
            result = get_redis()

        assert result is None

    def test_returns_none_when_import_fails(self):
        with patch.dict("sys.modules", {"redis": None}):
            result = get_redis()
        assert result is None


# ---------------------------------------------------------------------------
# redis_get
# ---------------------------------------------------------------------------

class TestRedisGet:
    def test_returns_dict_on_hit(self):
        mock_client = MagicMock()
        mock_client.get.return_value = json.dumps({"balance": 1234.5})

        with patch("src.dashboard.data.get_redis", return_value=mock_client):
            result = redis_get("portfolio_state")

        assert result == {"balance": 1234.5}

    def test_returns_none_on_miss(self):
        mock_client = MagicMock()
        mock_client.get.return_value = None

        with patch("src.dashboard.data.get_redis", return_value=mock_client):
            result = redis_get("portfolio_state")

        assert result is None

    def test_returns_none_on_json_error(self):
        mock_client = MagicMock()
        mock_client.get.return_value = "{invalid json"

        with patch("src.dashboard.data.get_redis", return_value=mock_client):
            result = redis_get("portfolio_state")

        assert result is None

    def test_returns_none_when_no_redis(self):
        with patch("src.dashboard.data.get_redis", return_value=None):
            result = redis_get("any_key")
        assert result is None


# ---------------------------------------------------------------------------
# get_trades: exception и успешный путь через реальную БД
# ---------------------------------------------------------------------------

class TestGetTradesException:
    def test_returns_empty_on_db_error(self, tmp_path):
        fake_db = tmp_path / "trades.db"
        fake_db.write_bytes(b"not a sqlite db")

        with patch("src.dashboard.data._DB_PATH", str(fake_db)):
            result = get_trades()

        assert result == []

    def test_returns_empty_when_no_db(self, tmp_path):
        with patch("src.dashboard.data._DB_PATH", str(tmp_path / "missing.db")):
            result = get_trades()
        assert result == []

    def test_returns_rows_from_real_db(self, tmp_path):
        db = tmp_path / "trades.db"
        conn = sqlite3.connect(str(db))
        conn.execute("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY,
                symbol TEXT, strategy TEXT, action TEXT,
                entry_price REAL, exit_price REAL,
                quantity REAL, pnl REAL, pnl_pct REAL,
                confidence REAL, commission REAL,
                entry_time TEXT, exit_time TEXT, status TEXT
            )
        """)
        conn.execute(
            "INSERT INTO trades VALUES "
            "(1,'BTC/USDT','ema','buy',60000,61000,0.1,100,0.016,0.8,5,"
            "'2026-01-01','2026-01-02','closed')"
        )
        conn.commit()
        conn.close()

        with patch("src.dashboard.data._DB_PATH", str(db)):
            result = get_trades(limit=10)

        assert len(result) == 1
        assert result[0]["symbol"] == "BTC/USDT"
        assert result[0]["status"] == "closed"


# ---------------------------------------------------------------------------
# get_stats: exception и корректный расчёт
# ---------------------------------------------------------------------------

class TestGetStatsException:
    def test_returns_empty_on_db_error(self, tmp_path):
        fake_db = tmp_path / "trades.db"
        fake_db.write_bytes(b"garbage")

        with patch("src.dashboard.data._DB_PATH", str(fake_db)):
            result = get_stats()

        assert result == {}

    def test_returns_empty_when_no_db(self, tmp_path):
        with patch("src.dashboard.data._DB_PATH", str(tmp_path / "missing.db")):
            result = get_stats()
        assert result == {}

    def test_win_rate_computed_correctly(self, tmp_path):
        db = tmp_path / "trades.db"
        conn = sqlite3.connect(str(db))
        conn.execute("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY,
                symbol TEXT, strategy TEXT, action TEXT,
                entry_price REAL, exit_price REAL,
                quantity REAL, pnl REAL, pnl_pct REAL,
                confidence REAL, commission REAL,
                entry_time TEXT, exit_time TEXT, status TEXT
            )
        """)
        conn.execute(
            "INSERT INTO trades VALUES "
            "(1,'BTC/USDT','ema','buy',60000,61000,0.1,100,0.016,0.8,5,'t1','t2','closed')"
        )
        conn.execute(
            "INSERT INTO trades VALUES "
            "(2,'ETH/USDT','ema','buy',3000,2900,0.5,-50,-0.033,0.6,3,'t3','t4','closed')"
        )
        conn.commit()
        conn.close()

        with patch("src.dashboard.data._DB_PATH", str(db)):
            stats = get_stats()

        assert stats["total_trades"] == 2
        assert stats["closed_trades"] == 2
        assert stats["win_rate"] == pytest.approx(50.0)
        assert stats["total_pnl"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# check_healthcheck
# ---------------------------------------------------------------------------

class TestCheckHealthcheck:
    def test_returns_false_when_no_file(self, tmp_path):
        missing = tmp_path / "healthcheck.txt"
        with patch("src.dashboard.data.Path", return_value=missing):
            result = check_healthcheck()
        assert result is False

    def test_returns_true_for_fresh_timestamp(self, tmp_path):
        hc = tmp_path / "healthcheck.txt"
        hc.write_text(datetime.now().isoformat())

        with patch("src.dashboard.data.Path", return_value=hc):
            result = check_healthcheck()

        assert result is True

    def test_returns_false_for_stale_timestamp(self, tmp_path):
        hc = tmp_path / "healthcheck.txt"
        old_ts = (datetime.now() - timedelta(seconds=200)).isoformat()
        hc.write_text(old_ts)

        with patch("src.dashboard.data.Path", return_value=hc):
            result = check_healthcheck()

        assert result is False

    def test_returns_false_for_content_too_long(self, tmp_path):
        hc = tmp_path / "healthcheck.txt"
        hc.write_text("x" * 60)

        with patch("src.dashboard.data.Path", return_value=hc):
            result = check_healthcheck()

        assert result is False

    def test_returns_false_for_invalid_iso(self, tmp_path):
        hc = tmp_path / "healthcheck.txt"
        hc.write_text("not-a-date")

        with patch("src.dashboard.data.Path", return_value=hc):
            result = check_healthcheck()

        assert result is False
