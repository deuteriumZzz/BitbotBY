"""Локальный вход лаунчера: /api/auth/local и src/dashboard/auth.check_local_login."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.dashboard import auth, data
from tests.test_bot_bus import FakeRedis

TOKEN = "t" * 40


@pytest.fixture()
def local(monkeypatch):
    monkeypatch.setenv("BITBOT_LOCAL_MODE", "1")
    monkeypatch.setenv("BITBOT_LOCAL_LOGIN_TOKEN", TOKEN)
    monkeypatch.delenv("DASHBOARD_API_KEY", raising=False)


@pytest.fixture()
def client():
    from src.dashboard import app

    return TestClient(app, raise_server_exceptions=False, base_url="http://localhost")


def test_disabled_without_flag_or_with_short_token(monkeypatch):
    monkeypatch.delenv("BITBOT_LOCAL_MODE", raising=False)
    monkeypatch.setenv("BITBOT_LOCAL_LOGIN_TOKEN", TOKEN)
    assert auth.check_local_login(FakeRedis(), TOKEN, "localhost:8080") == "disabled"
    monkeypatch.setenv("BITBOT_LOCAL_MODE", "1")
    monkeypatch.setenv("BITBOT_LOCAL_LOGIN_TOKEN", "short")
    assert auth.check_local_login(FakeRedis(), "short", "localhost") == "disabled"


def test_right_token_once_then_used(local):
    r = FakeRedis()
    assert auth.check_local_login(r, TOKEN, "127.0.0.1:8080") == "ok"
    assert auth.check_local_login(r, TOKEN, "127.0.0.1:8080") == "used"


@pytest.mark.parametrize(
    "host", ["evil.example", "192.168.1.5:8080", "localhost.evil.com"]
)
def test_other_hosts_are_refused(local, host):
    assert auth.check_local_login(FakeRedis(), TOKEN, host) == "host"


def test_ipv6_loopback_host_is_accepted(local):
    assert auth.check_local_login(FakeRedis(), TOKEN, "[::1]:8080") == "ok"


def test_five_wrong_tries_lock_even_the_right_token(local):
    r = FakeRedis()
    for _ in range(5):
        assert auth.check_local_login(r, "x" * 40, "localhost") == "wrong"
    assert auth.check_local_login(r, TOKEN, "localhost") == "locked"


def test_route_gives_a_session_that_controls_the_bot(local, client):
    r = FakeRedis()
    with patch.object(data, "get_redis", return_value=r):
        resp = client.post("/api/auth/local", json={"token": TOKEN})
        assert resp.status_code == 200
        session = resp.json()["token"]
        assert auth.check_session(r, session)["src"] == "local"
        again = client.post("/api/auth/local", json={"token": TOKEN})
    assert again.status_code == 401 and again.json()["error"]["code"] == "local_used"


def test_route_is_404_when_disabled_and_rejects_foreign_host(monkeypatch, client):
    monkeypatch.delenv("BITBOT_LOCAL_MODE", raising=False)
    with patch.object(data, "get_redis", return_value=FakeRedis()):
        off = client.post("/api/auth/local", json={"token": TOKEN})
    assert off.status_code == 404
    monkeypatch.setenv("BITBOT_LOCAL_MODE", "1")
    monkeypatch.setenv("BITBOT_LOCAL_LOGIN_TOKEN", TOKEN)
    with patch.object(data, "get_redis", return_value=FakeRedis()):
        foreign = client.post(
            "/api/auth/local", json={"token": TOKEN}, headers={"Host": "evil.example"}
        )
    assert (
        foreign.status_code == 401 and foreign.json()["error"]["code"] == "local_host"
    )


def test_denied_message_in_local_mode_tells_how_to_sign_in(local, client):
    with patch.object(data, "get_redis", return_value=FakeRedis()):
        resp = client.post("/api/bot/pause", json={"request_id": "12345678"})
    assert resp.status_code == 403
    msg = resp.json()["error"]["message"]
    assert "BitbotBY.command" in msg and "DASHBOARD_API_KEY" not in msg


def test_denied_message_on_a_server_is_unchanged(monkeypatch, client):
    monkeypatch.delenv("BITBOT_LOCAL_MODE", raising=False)
    monkeypatch.delenv("DASHBOARD_API_KEY", raising=False)
    with patch.object(data, "get_redis", return_value=FakeRedis()):
        resp = client.post("/api/bot/pause", json={"request_id": "12345678"})
    assert "DASHBOARD_API_KEY" in resp.json()["error"]["message"]


def test_telegram_link_is_one_time_and_local_only(local):
    from src import login_link

    r = FakeRedis()
    token = login_link.issue(r)
    assert all(token not in k for k in r.kv)  # в Redis только хеш
    assert auth.check_local_login(r, token, "evil.example:8080") == "host"
    assert auth.check_local_login(r, token, "localhost:8080") == "ok"
    assert auth.check_local_login(r, token, "localhost:8080") == "wrong"
    # Токен лаунчера по-прежнему работает отдельно от ссылок из Telegram.
    assert auth.check_local_login(r, TOKEN, "localhost:8080") == "ok"
