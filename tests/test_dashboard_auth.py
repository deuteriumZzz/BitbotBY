"""Вход в приложение: Telegram initData и код для Desktop (src/dashboard/auth.py)."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.parse
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.dashboard import auth, data
from src.dashboard.routers import control
from tests.test_bot_bus import FakeRedis

BOT_TOKEN = "123456:TEST-token-for-unit-tests-only-abcdef"
OWNER = "424242"


def make_init_data(user_id=OWNER, auth_date=None, token=BOT_TOKEN):
    fields = {
        "auth_date": str(int(auth_date or time.time())),
        "query_id": "AAH-test",
        "user": json.dumps({"id": int(user_id), "first_name": "Test"}),
    }
    check = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
    secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    fields["hash"] = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    return urllib.parse.urlencode(fields)


def test_valid_init_data():
    user, err = auth.verify_init_data(make_init_data(), BOT_TOKEN, OWNER)
    assert err == "" and user["id"] == int(OWNER)


def test_tampered_init_data():
    tampered = make_init_data().replace("Test", "Evil")
    assert auth.verify_init_data(tampered, BOT_TOKEN, OWNER) == (None, "bad_signature")


def test_signed_by_other_bot():
    other = make_init_data(token="999:other-bot")
    assert auth.verify_init_data(other, BOT_TOKEN, OWNER)[1] == "bad_signature"


def test_expired_init_data():
    old = make_init_data(auth_date=time.time() - 2 * 86400)
    assert auth.verify_init_data(old, BOT_TOKEN, OWNER)[1] == "expired"


def test_not_owner():
    stranger = make_init_data(user_id="777")
    assert auth.verify_init_data(stranger, BOT_TOKEN, OWNER)[1] == "not_owner"


def test_session_roundtrip_stores_only_hash():
    r = FakeRedis()
    token = auth.create_session(r, OWNER, "miniapp")
    assert all(token not in k for k in r.kv)
    assert auth.check_session(r, token)["sub"] == OWNER
    auth.drop_session(r, token)
    assert auth.check_session(r, token) is None


def _wrong(code: str) -> str:
    return "000000" if code != "000000" else "111111"


def test_desktop_code_flow():
    r = FakeRedis()
    code = auth.issue_desktop_code(r)
    assert code and len(code) == 6
    assert auth.issue_desktop_code(r) is None  # повтор раньше 30 с
    assert auth.check_desktop_code(r, _wrong(code)) == "wrong"
    assert auth.check_desktop_code(r, code) == "ok"
    assert auth.check_desktop_code(r, code) == "expired"  # одноразовый


def test_desktop_code_locks_after_attempts():
    r = FakeRedis()
    code = auth.issue_desktop_code(r)
    for _ in range(auth.CODE_MAX_ATTEMPTS):
        auth.check_desktop_code(r, _wrong(code))
    assert auth.check_desktop_code(r, code) == "locked"


# ── endpoints ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("DASHBOARD_API_KEY", raising=False)
    from src.dashboard import app

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def owner():
    with patch.object(auth, "owner_credentials", return_value=(BOT_TOKEN, OWNER)):
        yield


def test_login_then_command_with_bearer(client, owner, monkeypatch):
    r = FakeRedis()
    monkeypatch.setattr(control, "COMMAND_WAIT_SEC", 0.2)
    with patch.object(data, "get_redis", return_value=r):
        resp = client.post("/api/auth/telegram", json={"initData": make_init_data()})
        assert resp.status_code == 200, resp.text
        token = resp.json()["token"]

        anon = client.post("/api/bot/pause", json={"request_id": "req-00000010"})
        assert anon.status_code == 401

        ok = client.post(
            "/api/bot/pause",
            json={"request_id": "req-00000011"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert ok.status_code == 202


def test_login_rejects_stranger(client, owner):
    with patch.object(data, "get_redis", return_value=FakeRedis()):
        resp = client.post(
            "/api/auth/telegram", json={"initData": make_init_data(user_id="777")}
        )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "not_owner"


def test_desktop_login(client, owner):
    r = FakeRedis()
    sent = {}

    async def fake_send(text):
        sent["text"] = text
        return True

    with patch.object(data, "get_redis", return_value=r), patch.object(
        auth, "send_owner", fake_send
    ):
        assert client.post("/api/auth/desktop/request").status_code == 200
        code = sent["text"].split(": ")[1][:6]
        bad = client.post("/api/auth/desktop/verify", json={"code": "abcdef"})
        assert bad.status_code == 401
        good = client.post("/api/auth/desktop/verify", json={"code": code})
    assert good.status_code == 200 and good.json()["token"]


def test_bearer_passes_api_key_middleware(monkeypatch, owner):
    monkeypatch.setenv("DASHBOARD_API_KEY", "secret")
    from src.dashboard import app

    c = TestClient(app, raise_server_exceptions=False)
    r = FakeRedis()
    token = auth.create_session(r, OWNER, "miniapp")
    with patch.object(data, "get_redis", return_value=r):
        assert c.get("/api/state").status_code == 401
        headers = {"Authorization": f"Bearer {token}"}
        assert c.get("/api/state", headers=headers).status_code == 200


def test_webapp_static_skips_api_key(monkeypatch):
    # MiniApp грузит /app/ без заголовков; данные по-прежнему только с ключом.
    monkeypatch.setenv("DASHBOARD_API_KEY", "secret")
    from src.dashboard import app

    c = TestClient(app, raise_server_exceptions=False)
    with patch.object(data, "get_redis", return_value=FakeRedis()):
        assert c.get("/app/").status_code != 401
        assert c.get("/app/assets/none.js").status_code != 401
        assert c.get("/apple").status_code == 401
        assert c.get("/api/state").status_code == 401
