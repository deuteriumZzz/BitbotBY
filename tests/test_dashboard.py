"""Тесты FastAPI dashboard приложения."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# App fixture — isolate env between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_api_key_env(monkeypatch):
    """Remove DASHBOARD_API_KEY from env so tests start clean."""
    monkeypatch.delenv("DASHBOARD_API_KEY", raising=False)


@pytest.fixture()
def client():
    from src.dashboard import app

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def auth_client(monkeypatch):
    """Client with DASHBOARD_API_KEY='secret' set."""
    monkeypatch.setenv("DASHBOARD_API_KEY", "secret")
    from src.dashboard import app

    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------


class TestAuthMiddleware:
    def test_health_path_no_auth_required(self, monkeypatch):
        """GET /health bypasses auth even when key is set."""
        monkeypatch.setenv("DASHBOARD_API_KEY", "secret")
        from src.dashboard import app

        c = TestClient(app, raise_server_exceptions=False)
        resp = c.get("/health")
        assert resp.status_code != 401

    def test_no_key_configured_allows_all(self, client):
        """When DASHBOARD_API_KEY is empty, all paths are public."""
        with patch("src.dashboard.data.get_redis", return_value=None), patch(
            "src.dashboard.data.check_healthcheck", return_value=False
        ), patch("src.dashboard.data.get_stats", return_value={}):
            resp = client.get("/api/status")
        assert resp.status_code != 401

    def test_protected_path_without_key_returns_401(self, monkeypatch):
        """Missing X-API-Key header → 401."""
        monkeypatch.setenv("DASHBOARD_API_KEY", "secret")
        from src.dashboard import app

        c = TestClient(app, raise_server_exceptions=False)
        resp = c.get("/api/status")
        assert resp.status_code == 401

    def test_protected_path_with_wrong_key_returns_401(self, monkeypatch):
        """Wrong X-API-Key → 401."""
        monkeypatch.setenv("DASHBOARD_API_KEY", "secret")
        from src.dashboard import app

        c = TestClient(app, raise_server_exceptions=False)
        resp = c.get("/api/status", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401

    def test_protected_path_with_correct_key_passes(self, monkeypatch):
        """Correct X-API-Key → not 401."""
        monkeypatch.setenv("DASHBOARD_API_KEY", "secret")
        from src.dashboard import app

        c = TestClient(app, raise_server_exceptions=False)
        with patch("src.dashboard.data.get_redis", return_value=None), patch(
            "src.dashboard.data.check_healthcheck", return_value=False
        ), patch("src.dashboard.data.get_stats", return_value={}):
            resp = c.get("/api/status", headers={"X-API-Key": "secret"})
        assert resp.status_code != 401

    def test_metrics_path_no_auth_required(self, monkeypatch):
        """/metrics bypasses auth (in _NO_AUTH_PATHS)."""
        monkeypatch.setenv("DASHBOARD_API_KEY", "secret")
        from src.dashboard import app

        c = TestClient(app, raise_server_exceptions=False)
        with patch("src.dashboard.data.get_redis", return_value=None), patch(
            "src.dashboard.data.check_healthcheck", return_value=False
        ), patch("src.dashboard.data.get_stats", return_value={}), patch(
            "src.dashboard.routers.ops.generate_latest", return_value=b""
        ), patch(
            "src.dashboard.routers.ops.CONTENT_TYPE_LATEST", "text/plain"
        ):
            resp = c.get("/metrics")
        assert resp.status_code != 401


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /api/status
# ---------------------------------------------------------------------------


class TestApiStatus:
    def test_status_bot_not_running(self, client):
        with patch("src.dashboard.data.get_redis", return_value=None), patch(
            "src.dashboard.data.check_healthcheck", return_value=False
        ), patch("src.dashboard.data.get_stats", return_value={}):
            resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["bot_running"] is False

    def test_status_bot_running(self, client):
        # api.py импортирует check_healthcheck через `from ... import`, поэтому
        # патчим ссылку в пространстве имён роутера, а не в data-модуле.
        with patch("src.dashboard.data.get_redis", return_value=None), patch(
            "src.dashboard.routers.api.check_healthcheck", return_value=True
        ), patch(
            "src.dashboard.routers.api.get_stats", return_value={"total_pnl": 100.0}
        ):
            resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["bot_running"] is True
        assert "timestamp" in data

    def test_status_with_portfolio_state(self, client):
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.get.return_value = json.dumps(
            {"balance": 12345.0, "positions": {"BTC/USDT": {"qty": 0.01}}}
        )
        with patch("src.dashboard.data.get_redis", return_value=mock_redis), patch(
            "src.dashboard.data.check_healthcheck", return_value=True
        ), patch("src.dashboard.data.get_stats", return_value={}):
            resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["balance"] == pytest.approx(12345.0)

    def test_status_response_has_required_keys(self, client):
        with patch("src.dashboard.data.get_redis", return_value=None), patch(
            "src.dashboard.data.check_healthcheck", return_value=False
        ), patch("src.dashboard.data.get_stats", return_value={}):
            resp = client.get("/api/status")
        data = resp.json()
        for key in ("bot_running", "mode", "paper_trading", "balance", "timestamp"):
            assert key in data, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# /api/trades
# ---------------------------------------------------------------------------


class TestApiTrades:
    def test_trades_returns_list(self, client):
        with patch("src.dashboard.data.get_trades", return_value=[]):
            resp = client.get("/api/trades")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_trades_returns_trade_data(self, client):
        sample = [
            {"symbol": "BTC/USDT", "pnl": 50.0, "status": "closed"},
            {"symbol": "ETH/USDT", "pnl": -10.0, "status": "closed"},
        ]
        with patch("src.dashboard.routers.api.get_trades", return_value=sample):
            resp = client.get("/api/trades")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["symbol"] == "BTC/USDT"

    def test_trades_calls_with_limit_100(self, client):
        with patch("src.dashboard.routers.api.get_trades", return_value=[]) as mock_gt:
            client.get("/api/trades")
        mock_gt.assert_called_once_with(limit=100)


# ---------------------------------------------------------------------------
# /api/backtest
# ---------------------------------------------------------------------------


class TestApiBacktest:
    def test_backtest_no_file_returns_empty(self, client, tmp_path):
        """When backtest file doesn't exist → {results: [], generated_at: None}."""
        with patch("src.dashboard.routers.api.Path") as mock_path_cls:
            mock_p = MagicMock()
            mock_p.exists.return_value = False
            mock_path_cls.return_value = mock_p
            resp = client.get("/api/backtest")
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"] == []
        assert data["generated_at"] is None

    def test_backtest_with_valid_file(self, client, tmp_path):
        """Valid backtest file returns parsed results."""
        bt_data = {
            "generated_at": "2024-01-01T00:00:00",
            "results": [
                {
                    "strategy": "ema_crossover",
                    "win_rate": 0.65,
                    "total_trades": 100,
                    "expected_value": 1.5,
                    "total_return_pct": 12.3,
                    "max_drawdown_pct": 5.0,
                    "sharpe_ratio": 1.8,
                }
            ],
        }
        bt_file = tmp_path / "backtest_results.json"
        bt_file.write_text(json.dumps(bt_data))

        with patch("src.dashboard.routers.api.Path") as mock_path_cls:
            mock_p = MagicMock()
            mock_p.exists.return_value = True
            mock_p.read_text.return_value = json.dumps(bt_data)
            mock_path_cls.return_value = mock_p
            resp = client.get("/api/backtest")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["strategy"] == "ema_crossover"
        assert data["results"][0]["win_rate"] == pytest.approx(0.65)

    def test_backtest_malformed_file_returns_empty(self, client):
        """Malformed JSON → {results: []} without crashing."""
        with patch("src.dashboard.routers.api.Path") as mock_path_cls:
            mock_p = MagicMock()
            mock_p.exists.return_value = True
            mock_p.read_text.return_value = "NOT JSON {"
            mock_path_cls.return_value = mock_p
            resp = client.get("/api/backtest")
        assert resp.status_code == 200
        assert resp.json()["results"] == []


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        with patch("src.dashboard.data.get_redis", return_value=None), patch(
            "src.dashboard.data.check_healthcheck", return_value=False
        ), patch("src.dashboard.data.get_stats", return_value={}), patch(
            "src.dashboard.routers.ops.generate_latest", return_value=b"# metrics\n"
        ), patch(
            "src.dashboard.routers.ops.CONTENT_TYPE_LATEST", "text/plain; version=0.0.4"
        ):
            resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_sets_gauges(self, client):
        """Metrics endpoint calls g_running, g_balance, g_pnl, etc."""
        with patch("src.dashboard.data.get_redis", return_value=None), patch(
            "src.dashboard.routers.ops.check_healthcheck", return_value=True
        ), patch(
            "src.dashboard.routers.ops.get_stats",
            return_value={"total_pnl": 500.0, "win_rate": 60.0, "total_trades": 42},
        ), patch(
            "src.dashboard.routers.ops.generate_latest", return_value=b""
        ), patch(
            "src.dashboard.routers.ops.CONTENT_TYPE_LATEST", "text/plain"
        ), patch(
            "src.dashboard.routers.ops.g_running"
        ) as mock_gr, patch(
            "src.dashboard.routers.ops.g_balance"
        ), patch(
            "src.dashboard.routers.ops.g_pnl"
        ) as mock_gpnl:
            resp = client.get("/metrics")
        assert resp.status_code == 200
        mock_gr.set.assert_called_once_with(1)
        mock_gpnl.set.assert_called_once_with(500.0)


# ---------------------------------------------------------------------------
# /webhook/alerts
# ---------------------------------------------------------------------------


class TestWebhookAlerts:
    def test_webhook_no_alerts_returns_no_alerts_status(self, client):
        resp = client.post("/webhook/alerts", json={"alerts": []})
        assert resp.status_code == 200
        assert resp.json()["status"] == "no_alerts"

    def test_webhook_with_alert_forwarded(self, client):
        payload = {
            "alerts": [
                {
                    "status": "firing",
                    "labels": {"alertname": "HighPnlLoss", "severity": "critical"},
                    "annotations": {
                        "summary": "Big loss",
                        "description": "PnL dropped",
                    },
                }
            ]
        }
        with patch("src.dashboard.routers.ops._send_telegram_sync") as mock_tg:
            resp = client.post("/webhook/alerts", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["forwarded"] == 1
        mock_tg.assert_called_once()

    def test_webhook_secret_mismatch_returns_403(self, monkeypatch):
        monkeypatch.setenv("ALERTMANAGER_WEBHOOK_SECRET", "mysecret")
        from src.dashboard import app

        c = TestClient(app, raise_server_exceptions=False)
        resp = c.post(
            "/webhook/alerts",
            json={"alerts": []},
            headers={"X-Webhook-Secret": "wrong"},
        )
        assert resp.status_code == 403

    def test_webhook_correct_secret_passes(self, monkeypatch):
        monkeypatch.setenv("ALERTMANAGER_WEBHOOK_SECRET", "mysecret")
        from src.dashboard import app

        c = TestClient(app, raise_server_exceptions=False)
        with patch("src.dashboard.routers.ops._send_telegram_sync"):
            resp = c.post(
                "/webhook/alerts",
                json={"alerts": []},
                headers={"X-Webhook-Secret": "mysecret"},
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "no_alerts"

    def test_webhook_resolved_alert_uses_checkmark_icon(self, client):
        payload = {
            "alerts": [
                {
                    "status": "resolved",
                    "labels": {"alertname": "TestAlert", "severity": "warning"},
                    "annotations": {},
                }
            ]
        }
        captured = []
        with patch(
            "src.dashboard.routers.ops._send_telegram_sync",
            side_effect=lambda text: captured.append(text),
        ):
            client.post("/webhook/alerts", json=payload)
        assert captured
        assert "✅" in captured[0]

    def test_webhook_firing_alert_uses_red_circle_icon(self, client):
        payload = {
            "alerts": [
                {
                    "status": "firing",
                    "labels": {"alertname": "TestAlert", "severity": "warning"},
                    "annotations": {},
                }
            ]
        }
        captured = []
        with patch(
            "src.dashboard.routers.ops._send_telegram_sync",
            side_effect=lambda text: captured.append(text),
        ):
            client.post("/webhook/alerts", json=payload)
        assert captured
        assert "\U0001f534" in captured[0]


# ---------------------------------------------------------------------------
# data.py helpers
# ---------------------------------------------------------------------------


class TestDataHelpers:
    def test_redis_get_returns_none_when_no_redis(self):
        from src.dashboard.data import redis_get

        with patch("src.dashboard.data.get_redis", return_value=None):
            result = redis_get("any_key")
        assert result is None

    def test_redis_get_returns_dict_on_hit(self):
        from src.dashboard.data import redis_get

        mock_r = MagicMock()
        mock_r.get.return_value = json.dumps({"balance": 1000.0})
        with patch("src.dashboard.data.get_redis", return_value=mock_r):
            result = redis_get("portfolio_state")
        assert result == {"balance": 1000.0}

    def test_redis_get_returns_none_on_miss(self):
        from src.dashboard.data import redis_get

        mock_r = MagicMock()
        mock_r.get.return_value = None
        with patch("src.dashboard.data.get_redis", return_value=mock_r):
            result = redis_get("missing_key")
        assert result is None

    def test_check_healthcheck_no_file(self, tmp_path):
        from src.dashboard.data import check_healthcheck

        with patch("src.dashboard.data.Path") as mock_path_cls:
            mock_p = MagicMock()
            mock_p.exists.return_value = False
            mock_path_cls.return_value = mock_p
            result = check_healthcheck()
        assert result is False

    def test_get_trades_no_db(self, tmp_path):
        from src.dashboard.data import get_trades

        with patch("src.dashboard.data._DB_PATH", str(tmp_path / "missing.db")):
            result = get_trades()
        assert result == []

    def test_get_stats_no_db(self, tmp_path):
        from src.dashboard.data import get_stats

        with patch("src.dashboard.data._DB_PATH", str(tmp_path / "missing.db")):
            result = get_stats()
        assert result == {}

    def test_get_trades_with_db(self, tmp_path):
        import sqlite3

        from src.dashboard.data import get_trades

        db_path = str(tmp_path / "trades.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY,
                symbol TEXT, strategy TEXT, action TEXT,
                entry_price REAL, exit_price REAL,
                quantity REAL, pnl REAL, pnl_pct REAL,
                confidence REAL, commission REAL,
                entry_time TEXT, exit_time TEXT, status TEXT
            )
        """
        )
        conn.execute(
            """
            INSERT INTO trades VALUES (1,'BTC/USDT','ema','buy',
            30000,31000,0.01,10.0,0.033,0.9,3.0,
            '2024-01-01','2024-01-02','closed')
        """
        )
        conn.commit()
        conn.close()

        with patch("src.dashboard.data._DB_PATH", db_path):
            result = get_trades(limit=10)

        assert len(result) == 1
        assert result[0]["symbol"] == "BTC/USDT"
        assert result[0]["pnl"] == pytest.approx(10.0)

    def test_get_stats_with_db(self, tmp_path):
        import sqlite3

        from src.dashboard.data import get_stats

        db_path = str(tmp_path / "trades.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY,
                symbol TEXT, strategy TEXT, action TEXT,
                entry_price REAL, exit_price REAL,
                quantity REAL, pnl REAL, pnl_pct REAL,
                confidence REAL, commission REAL,
                entry_time TEXT, exit_time TEXT, status TEXT
            )
        """
        )
        conn.executemany(
            """INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    1,
                    "BTC/USDT",
                    "ema",
                    "buy",
                    30000,
                    31000,
                    0.01,
                    100.0,
                    0.033,
                    0.9,
                    3.0,
                    "2024-01-01",
                    "2024-01-02",
                    "closed",
                ),
                (
                    2,
                    "ETH/USDT",
                    "rsi",
                    "buy",
                    2000,
                    1900,
                    0.1,
                    -10.0,
                    -0.05,
                    0.7,
                    2.0,
                    "2024-01-02",
                    "2024-01-03",
                    "closed",
                ),
            ],
        )
        conn.commit()
        conn.close()

        with patch("src.dashboard.data._DB_PATH", db_path):
            result = get_stats()

        assert result["total_trades"] == 2
        assert result["closed_trades"] == 2
        assert result["win_rate"] == pytest.approx(50.0)
        assert result["total_pnl"] == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# ops helpers
# ---------------------------------------------------------------------------


class TestTgEscape:
    def test_strips_markdown_chars(self):
        from src.dashboard.routers.ops import _tg_escape

        result = _tg_escape("*bold* _italic_ `code` [link")
        assert "*" not in result
        assert "_" not in result
        assert "`" not in result
        assert "[" not in result

    def test_safe_string_unchanged(self):
        from src.dashboard.routers.ops import _tg_escape

        assert _tg_escape("Hello world 123") == "Hello world 123"
