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
    assert groq["model"] is None  # модель AI выбирается в разделе «AI», не в ключах


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


def test_twscrape_accounts_format_is_checked():
    svc = ss.BY_ID["twitter"]
    ok = "user:pass:me@mail.com:mailpass,user2:p2:b@mail.com:mp2"
    assert ss.check_input(svc, {"TWSCRAPE_ACCOUNTS": ok}) == {}
    for bad in ("user:pass", "user:pass:mail:mp:extra", "user::mail:mp", "а,б"):
        assert "TWSCRAPE_ACCOUNTS" in ss.check_input(svc, {"TWSCRAPE_ACCOUNTS": bad})


def test_twitter_fields_explain_themselves(store):
    fields = {
        f["env"]: f
        for f in next(i for i in ss.public_view() if i["service"] == "twitter")[
            "fields"
        ]
    }
    assert "платный" in fields["TWITTER_BEARER_TOKEN"]["hint"].lower()
    assert "почта" in fields["TWSCRAPE_ACCOUNTS"]["hint"].lower()
    assert fields["TWSCRAPE_ACCOUNTS"]["placeholder"]


def test_x_accounts_survive_awkward_passwords_and_hide_secrets(store):
    ss.x_add_account("alice", "p:a,s%s", "al@mail.com", "m:p", "test")
    ss.x_add_account("bob", "pw2", "bob@mail.com", "mp2", "test")
    raw = ss.get_value(ss.X_ENV)
    assert raw.count(",") == 1 and raw.count(":") == 6  # формат не сломан
    assert ss.x_parse(raw)[0] == ("alice", "p:a,s%s", "al@mail.com", "m:p")
    # то же разбирает и сам бот
    from src.twitter_analyzer import _parse_twscrape_accounts

    assert _parse_twscrape_accounts(raw)[0] == (
        "alice",
        "p:a,s%s",
        "al@mail.com",
        "m:p",
    )
    # в приложение уходят только логин и спрятанная почта
    twitter = next(i for i in ss.public_view() if i["service"] == "twitter")
    assert twitter["accounts"] == [
        {"login": "alice", "email": "a***@mail.com"},
        {"login": "bob", "email": "b***@mail.com"},
    ]
    assert "pw2" not in json.dumps(ss.public_view(), ensure_ascii=False)


def test_x_account_replace_and_remove(store):
    ss.x_add_account("alice", "one", "a@m.com", "x", "test")
    ss.x_add_account("alice", "two", "a@m.com", "x", "test")  # тот же логин — замена
    assert [a[1] for a in ss.x_parse(ss.get_value(ss.X_ENV))] == ["two"]
    assert ss.x_remove_account("nobody", "test") is False
    assert ss.x_remove_account("alice", "test") is True
    assert ss.x_parse(ss.get_value(ss.X_ENV)) == []


def test_x_account_routes(keyed):
    body = {
        "login": "alice",
        "password": "pw",
        "email": "a@m.com",
        "email_password": "mp",
    }
    with patch.object(data, "get_redis", return_value=FakeRedis()):
        bad = keyed.post(
            "/api/secrets/twitter/accounts", json={**body, "email": "no-at"}, headers=H
        )
        ok = keyed.post("/api/secrets/twitter/accounts", json=body, headers=H)
        gone = keyed.delete("/api/secrets/twitter/accounts/alice", headers=H)
        again = keyed.delete("/api/secrets/twitter/accounts/alice", headers=H)
    assert bad.status_code == 422 and "email" in bad.json()["error"]["fields"]
    assert ok.status_code == 200 and "pw" not in ok.text and "mp" not in ok.text
    assert ok.json()["accounts"] == [{"login": "alice", "email": "a***@m.com"}]
    assert gone.json()["accounts"] == [] and again.status_code == 404


def test_telegram_chats_only_private_and_deduplicated():
    payload = {
        "result": [
            {
                "message": {
                    "chat": {
                        "id": 42,
                        "type": "private",
                        "first_name": "Дмитрий",
                        "last_name": "В",
                    }
                }
            },
            {
                "message": {
                    "chat": {"id": 42, "type": "private", "first_name": "Дмитрий"}
                }
            },
            {
                "message": {
                    "chat": {"id": -100, "type": "supergroup", "title": "группа"}
                }
            },
            {
                "edited_message": {
                    "chat": {"id": 7, "type": "private", "username": "bob"}
                }
            },
            {"callback_query": {"id": "x"}},
        ]
    }
    assert ss.telegram_chats(payload) == [
        {"id": "42", "name": "Дмитрий"},
        {"id": "7", "name": "bob"},
    ]
    assert ss.telegram_chats({}) == []


def test_telegram_chats_by_link_code_only_the_sender_of_the_code():
    def msg(chat_id, text):
        return {"message": {"chat": {"id": chat_id, "type": "private"}, "text": text}}

    payload = {"result": [msg(7, "/start"), msg(42, " 123456 "), msg(9, "123457")]}
    assert [c["id"] for c in ss.telegram_chats(payload, "123456")] == ["42"]
    assert ss.telegram_chats(payload, "000000") == []
