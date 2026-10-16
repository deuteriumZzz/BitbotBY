"""Хранилище ключей API (src/secrets_store.py) и /api/secrets."""

from __future__ import annotations

import json
import os
import stat
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from src import secrets_store as ss
from src.dashboard import auth, data
from tests.test_bot_bus import FakeRedis

GROQ_KEY = "gsk_test_" + "x" * 30


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("BITBOT_SECRETS_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("SECRETS_PATH", str(tmp_path / "secrets.enc.json"))
    return tmp_path / "secrets.enc.json"


def test_disabled_without_master_key(monkeypatch):
    monkeypatch.delenv("BITBOT_SECRETS_KEY", raising=False)
    assert not ss.enabled()
    with pytest.raises(ss.SecretsDisabled):
        ss.load_values()


def test_save_encrypts_and_never_exposes_value(store):
    ss.save(
        ss.BY_ID["groq"], {"GROQ_API_KEY": GROQ_KEY}, "llama", "ok", "Работает.", "test"
    )
    assert GROQ_KEY not in store.read_text()
    assert stat.S_IMODE(os.stat(store).st_mode) == 0o600
    assert ss.load_values()["GROQ_API_KEY"] == GROQ_KEY
    assert GROQ_KEY not in json.dumps(ss.public_view(), ensure_ascii=False)
    groq = next(i for i in ss.public_view() if i["service"] == "groq")
    assert groq["fields"][0]["last4"] == GROQ_KEY[-4:]
    assert groq["model"]["value"] == "llama"


def test_passwords_show_no_last4(store):
    ss.save(ss.BY_ID["server"], {"TRADING_MODE_PIN": "4821"}, None, "ok", "", "test")
    server = next(i for i in ss.public_view() if i["service"] == "server")
    pin = next(f for f in server["fields"] if f["env"] == "TRADING_MODE_PIN")
    assert pin["set"] is True and pin["last4"] is None and pin["value"] is None


def test_apply_secrets_overrides_config(store, monkeypatch):
    ss.save(ss.BY_ID["groq"], {"GROQ_API_KEY": GROQ_KEY}, None, "ok", "", "test")
    cfg = type("Cfg", (), {"GROQ_API_KEY": "from-env"})
    assert ss.apply_secrets(cfg) == ["GROQ_API_KEY"]
    assert cfg.GROQ_API_KEY == GROQ_KEY
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


def test_check_input():
    errors = ss.check_input(
        ss.BY_ID["telegram"], {"TELEGRAM_BOT_TOKEN": "short", "TELEGRAM_CHAT_ID": "abc"}
    )
    assert set(errors) == {"TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"}
    assert ss.check_input(ss.BY_ID["groq"], {"NOPE": "x"}) == {
        "NOPE": "Нет такого поля."
    }


@pytest.mark.parametrize(
    "code,status",
    [(200, "ok"), (401, "bad"), (402, "warn"), (429, "warn"), (500, "warn")],
)
async def test_validate_maps_http_codes(code, status):
    transport = httpx.MockTransport(lambda req: httpx.Response(code, json={}))
    async with httpx.AsyncClient(transport=transport) as client:
        got, _ = await ss.validate(ss.BY_ID["groq"], {"GROQ_API_KEY": GROQ_KEY}, client)
    assert got == status


async def test_validate_network_error_is_warn():
    def boom(req):
        raise httpx.ConnectError("down")

    async with httpx.AsyncClient(transport=httpx.MockTransport(boom)) as client:
        got, _ = await ss.validate(ss.BY_ID["groq"], {"GROQ_API_KEY": GROQ_KEY}, client)
    assert got == "warn"


def test_bybit_key_with_withdraw_is_rejected():
    status, msg = ss.bybit_permissions_result(
        {"permissions": {"Wallet": ["AccountTransfer", "Withdraw"]}, "ips": ["1.2.3.4"]}
    )
    assert status == "bad" and "вывод" in msg
    status, _ = ss.bybit_permissions_result(
        {"permissions": {"ContractTrade": ["Order"]}, "ips": ["1.2.3.4"]}
    )
    assert status == "ok"


# ── /api/secrets ──────────────────────────────────────────────────────────────


@pytest.fixture()
def keyed(store, monkeypatch):
    monkeypatch.setenv("DASHBOARD_API_KEY", "secret")
    from src.dashboard import app

    return TestClient(app, raise_server_exceptions=False)


H = {"X-API-Key": "secret"}


def test_put_rejected_key_is_not_saved(keyed):
    with patch.object(
        ss, "validate", AsyncMock(return_value=("bad", "Ключ не подходит."))
    ), patch.object(data, "get_redis", return_value=FakeRedis()):
        resp = keyed.put(
            "/api/secrets/groq", json={"fields": {"GROQ_API_KEY": GROQ_KEY}}, headers=H
        )
    assert resp.status_code == 422
    assert "GROQ_API_KEY" not in ss.load_values()


def test_put_valid_key_saved_without_echo(keyed):
    with patch.object(
        ss, "validate", AsyncMock(return_value=("ok", "Ключ работает."))
    ), patch.object(data, "get_redis", return_value=FakeRedis()):
        resp = keyed.put(
            "/api/secrets/groq", json={"fields": {"GROQ_API_KEY": GROQ_KEY}}, headers=H
        )
    assert resp.status_code == 200
    assert GROQ_KEY not in resp.text
    body = resp.json()
    assert body["status"] == "ok" and body["restart_required"] is True


def test_secrets_need_auth(keyed):
    with patch.object(data, "get_redis", return_value=None), patch.object(
        auth, "telegram_auth_configured", return_value=True
    ):
        assert keyed.get("/api/secrets").status_code == 401
