"""API для MiniApp / Desktop (src/dashboard/routers/control.py) и метрики (data.py)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src import bot_bus
from src.dashboard import data
from src.dashboard.routers import control
from tests.test_bot_bus import FakeRedis


class StrRedis(FakeRedis):
    """FakeRedis как у дашборда: decode_responses=True + множества."""

    def smembers(self, key):
        return set(self.kv.get(key) or set())

    def sadd(self, key, *values):
        self.kv.setdefault(key, set()).update(values)

    def srem(self, key, *values):
        (self.kv.get(key) or set()).difference_update(values)


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    monkeypatch.delenv("DASHBOARD_API_KEY", raising=False)


@pytest.fixture()
def client():
    from src.dashboard import app

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def keyed(monkeypatch):
    monkeypatch.setenv("DASHBOARD_API_KEY", "secret")
    from src.dashboard import app

    return TestClient(app, raise_server_exceptions=False)


def _trade(pnl, day, fee=1.0, strategy="rsi_momentum"):
    return {
        "pnl": pnl,
        "commission": fee,
        "exit_time": f"{day}T12:00:00",
        "strategy": strategy,
    }


# ── метрики ───────────────────────────────────────────────────────────────────


def test_compute_metrics_basic():
    trades = [
        _trade(10, "2026-10-01"),
        _trade(-5, "2026-10-01"),
        _trade(20, "2026-10-02"),
        _trade(-5, "2026-10-03"),
    ]
    m = data.compute_metrics(trades, 1000)
    assert m["n"] == 4
    assert m["net"] == 20 and m["fees"] == 4 and m["realized"] == 24
    assert m["win_rate"] == 0.5
    assert m["profit_factor"] == 3.0
    assert m["avg_win"] == 15 and m["avg_loss"] == 5 and m["rr"] == 3.0
    assert m["max_drawdown_pct"] == pytest.approx(0.5, abs=0.01)
    assert m["sharpe"] is None  # меньше 50 сделок


def test_compute_metrics_empty():
    m = data.compute_metrics([], 1000)
    assert m["n"] == 0 and m["win_rate"] is None and m["profit_factor"] is None


def test_sharpe_appears_from_50_trades():
    start = datetime(2026, 9, 1)
    trades = [
        _trade(5 if i % 3 else -4, (start + timedelta(days=i % 20)).date().isoformat())
        for i in range(60)
    ]
    trades.sort(key=lambda t: t["exit_time"])
    assert data.compute_metrics(trades, 1000)["sharpe"] is not None


def test_daily_pnl_groups_by_day():
    rows = data.daily_pnl(
        [_trade(10, "2026-10-01"), _trade(-4, "2026-10-01"), _trade(3, "2026-10-02")]
    )
    assert rows == [
        {"date": "2026-10-01", "net": 6.0, "fees": 2.0, "trades": 2},
        {"date": "2026-10-02", "net": 3.0, "fees": 1.0, "trades": 1},
    ]


def test_get_closed_trades_reads_paper_db(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    conn = sqlite3.connect(tmp_path / "data" / "trades_paper.db")
    conn.execute(
        "CREATE TABLE trades (symbol TEXT, strategy TEXT, pnl REAL, commission REAL,"
        " exit_time TEXT, status TEXT)"
    )
    conn.executemany(
        "INSERT INTO trades VALUES (?,?,?,?,?,?)",
        [
            ("SOL/USDT", "breakout", 12.0, 1.0, "2026-10-10T10:00:00", "closed"),
            ("ETH/USDT", "breakout", None, 1.0, None, "open"),
        ],
    )
    conn.commit()
    conn.close()
    with patch.object(data, "paper_mode", return_value=True):
        rows = data.get_closed_trades()
    assert [r["symbol"] for r in rows] == ["SOL/USDT"]


def test_portfolio_key_follows_mode():
    with patch.object(data, "paper_mode", return_value=True):
        assert data.portfolio_key() == "portfolio_state_paper"
    with patch.object(data, "paper_mode", return_value=False):
        assert data.portfolio_key() == "portfolio_state_live"


# ── чтение ────────────────────────────────────────────────────────────────────


def test_state_without_redis(client):
    with patch.object(data, "get_redis", return_value=None), patch.object(
        data, "check_healthcheck", return_value=False
    ):
        resp = client.get("/api/state")
    assert resp.status_code == 200
    body = resp.json()
    assert body["bot"]["alive"] is False and body["positions"] == []


def test_metrics_rejects_unknown_period(client):
    assert client.get("/api/metrics?period=year").status_code == 400


# ── команды ───────────────────────────────────────────────────────────────────


def test_commands_disabled_without_api_key(client):
    with patch(
        "src.dashboard.auth.telegram_auth_configured", return_value=False
    ), patch.object(data, "get_redis", return_value=None):
        resp = client.post("/api/bot/panic", json={"request_id": "req-00000001"})
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "control_disabled"


def test_commands_need_api_key_header(keyed):
    resp = keyed.post("/api/bot/pause", json={"request_id": "req-00000001"})
    assert resp.status_code == 401


def test_commands_need_request_id(keyed):
    resp = keyed.post("/api/bot/pause", json={}, headers={"X-API-Key": "secret"})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "request_id_required"


def test_command_is_queued(keyed, monkeypatch):
    r = StrRedis()
    monkeypatch.setattr(control, "COMMAND_WAIT_SEC", 0.2)
    with patch.object(data, "get_redis", return_value=r):
        resp = keyed.post(
            "/api/bot/pause",
            json={"request_id": "req-00000002"},
            headers={"X-API-Key": "secret", "X-Client": "miniapp"},
        )
    assert resp.status_code == 202
    queued = r.lists[bot_bus.COMMANDS_KEY]
    assert len(queued) == 1 and '"source": "miniapp"' in queued[0]


def test_repeated_request_returns_stored_result(keyed):
    r = StrRedis()
    bot_bus.enqueue(r, "pause", {}, "req-00000003", "app")
    r.set("bot:cmdres:req-00000003", '{"ok": true, "paused": true}')
    with patch.object(data, "get_redis", return_value=r):
        resp = keyed.post(
            "/api/bot/pause",
            json={"request_id": "req-00000003"},
            headers={"X-API-Key": "secret"},
        )
    assert resp.status_code == 200
    assert resp.json()["result"] == {"ok": True, "paused": True}
    assert len(r.lists[bot_bus.COMMANDS_KEY]) == 1  # повтор не поставлен в очередь


def test_account_mode_validates_body(keyed):
    resp = keyed.post(
        "/api/account/mode",
        json={"request_id": "req-00000004", "paper": "no"},
        headers={"X-API-Key": "secret"},
    )
    assert resp.status_code == 400


# ── WebSocket ─────────────────────────────────────────────────────────────────


def test_ws_rejects_wrong_token(keyed):
    with keyed.websocket_connect("/ws") as ws:
        ws.send_json({"type": "auth", "token": "nope"})
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
    assert closed.value.code == 4401  # приложение по этому коду идёт на вход


def test_update_position_needs_fields(keyed):
    resp = keyed.patch(
        "/api/positions/7",
        json={"request_id": "req-00000020"},
        headers={"X-API-Key": "secret"},
    )
    assert resp.status_code == 400


def test_pending_signals_survive_reconnect(client):
    r = StrRedis()
    r.set("bot:pending_signal:sig7", '{"id": "sig7", "symbol": "SOL/USDT"}')
    with patch.object(data, "get_redis", return_value=r):
        resp = client.get("/api/signals/pending")
    assert resp.json() == [{"id": "sig7", "symbol": "SOL/USDT"}]


H = {"X-API-Key": "secret"}


def test_candles_from_cache(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "cache").mkdir(parents=True)
    (tmp_path / "data" / "cache" / "SOL_USDT_15m.csv").write_text(
        ",timestamp,open,high,low,close,volume\n"
        "0,1758592800000,216.61,217.86,216.35,217.45,12739.0\n"
    )
    resp = client.get("/api/candles?symbol=SOL/USDT&tf=15m")
    assert resp.json() == [
        {
            "time": 1758592800,
            "open": 216.61,
            "high": 217.86,
            "low": 216.35,
            "close": 217.45,
        }
    ]
    assert client.get("/api/candles?symbol=../etc&tf=15m").status_code == 400


def test_symbols_normalised_and_conflicts_rejected(keyed):
    r = StrRedis()
    with patch.object(data, "get_redis", return_value=r):
        ok = keyed.put(
            "/api/symbols",
            json={"forced": ["sol", "LINK/USDT"], "excluded": ["hype"]},
            headers=H,
        )
        assert ok.json() == {
            "forced": ["LINK/USDT", "SOL/USDT"],
            "excluded": ["HYPE/USDT"],
        }
        assert keyed.get("/api/symbols", headers=H).json()["excluded"] == ["HYPE/USDT"]
        bad = keyed.put(
            "/api/symbols", json={"forced": ["SOL"], "excluded": ["SOL"]}, headers=H
        )
        assert bad.status_code == 400


def test_alert_cooldown_validated(keyed):
    with patch.object(data, "get_redis", return_value=StrRedis()):
        bad = keyed.patch(
            "/api/alerts/drawdown", json={"cooldown_min": 5000}, headers=H
        )
        assert bad.status_code == 400
        unknown = keyed.patch("/api/alerts/nope", json={"enabled": False}, headers=H)
        assert unknown.status_code == 404


def test_profile_and_strategy_validation(keyed):
    with patch.object(data, "get_redis", return_value=StrRedis()):
        assert (
            keyed.post("/api/profile", json={"name": "moon"}, headers=H).status_code
            == 400
        )
        resp = keyed.patch("/api/strategies/nope", json={"enabled": False}, headers=H)
        assert resp.status_code == 404


# ── AI-провайдеры ─────────────────────────────────────────────────────────────

_KEY = "k" * 20


def _keys(*envs):
    """Ключи «сохранены» только для перечисленных переменных."""
    return patch.object(
        control.secrets_store,
        "get_value",
        side_effect=lambda env: _KEY if env in envs else None,
    )


def test_ai_view_free_chain_by_default(client):
    with patch.object(data, "get_redis", return_value=StrRedis()), _keys(
        "GROQ_API_KEY", "DEEPSEEK_API_KEY"
    ):
        body = client.get("/api/ai").json()
    assert body["mode"] == "free"
    assert [c["provider"] for c in body["chain"]] == ["groq"]  # DeepSeek платный
    groq = next(p for p in body["providers"] if p["id"] == "groq")
    assert groq["key_set"] and groq["usable"] and groq["free"]
    assert len(body["providers"]) == 16
    assert _KEY not in json.dumps(body)  # значения ключей не отдаются


def test_put_ai_chain_applies_and_validates(keyed):
    r = StrRedis()
    h = {"X-API-Key": "secret"}
    with patch.object(data, "get_redis", return_value=r), _keys(
        "GROQ_API_KEY", "DEEPSEEK_API_KEY"
    ):
        bad = keyed.put("/api/ai/chain", json={"mode": "cheap"}, headers=h)
        ok = keyed.put(
            "/api/ai/chain",
            json={"mode": "auto", "models": {"groq": "openai/gpt-oss-20b"}},
            headers=h,
        )
    assert bad.status_code == 400
    assert ok.status_code == 200
    chain = ok.json()["chain"]
    assert [c["provider"] for c in chain] == ["groq", "deepseek"]
    assert chain[0]["model"] == "openai/gpt-oss-20b"
    assert json.loads(r.get("bot:ai_chain"))["mode"] == "auto"
    assert r.published[-1][0] == bot_bus.EVENTS_CHANNEL
    assert r.published[-1][1]["data"] == {"keys": ["AI_CHAIN"]}


def test_put_ai_chain_needs_login(client):
    with patch.object(data, "get_redis", return_value=StrRedis()):
        resp = client.put("/api/ai/chain", json={"mode": "auto"})
    assert resp.status_code in (401, 403)


def test_ai_models_live_list(keyed, monkeypatch):
    class Resp:
        status_code = 200

        def json(self):
            return {"data": [{"id": "models/gemini-x"}, {"id": "b"}]}

    class Client:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, headers=None):
            assert url.endswith("/openai/models")
            assert headers == {"Authorization": f"Bearer {_KEY}"}
            return Resp()

    monkeypatch.setattr("httpx.AsyncClient", Client)
    with _keys("GEMINI_API_KEY"):
        resp = keyed.get(
            "/api/ai/providers/gemini/models", headers={"X-API-Key": "secret"}
        )
    assert resp.json() == {"provider": "gemini", "models": ["b", "gemini-x"]}


def test_single_provider_mode_via_api(keyed):
    h = {"X-API-Key": "secret"}
    with patch.object(data, "get_redis", return_value=StrRedis()), _keys(
        "GROQ_API_KEY", "GEMINI_API_KEY"
    ):
        both = keyed.get("/api/ai", headers=h).json()
        one = keyed.put("/api/ai/chain", json={"single": True}, headers=h).json()
    assert [c["provider"] for c in both["chain"]] == ["groq", "gemini"]
    assert one["single"] is True
    assert [c["provider"] for c in one["chain"]] == ["groq"]
