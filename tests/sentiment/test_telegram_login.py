"""Вход в Telegram из приложения и подключение бота с сохранённой сессией."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest
import telethon
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from telethon import errors
from telethon.crypto import AuthKey
from telethon.sessions import StringSession

from src.dashboard import data
from src.sentiment import telegram_login
from src.sentiment.telegram_sentiment import STATUS_KEY, TelegramSentiment
from src.settings import secrets_store as ss
from tests.infra.test_bot_bus import FakeRedis


def _session() -> str:
    s = StringSession()
    s.set_dc(2, "149.154.167.51", 443)
    s.auth_key = AuthKey(b"\x01" * 256)
    return s.save()


SESSION = _session()
H = {"X-API-Key": "secret"}


class FakeClient:
    """Telethon-клиент: код 11111 — вход, 22222 — нужен облачный пароль."""

    authorized = True
    instances: list = []

    def __init__(self, session, api_id, api_hash):
        self.session = type("S", (), {"save": lambda _self: SESSION})()
        self.connected = False
        FakeClient.instances.append(self)

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.connected = False

    def is_connected(self):
        return self.connected

    def add_event_handler(self, callback, event):
        self.handler = callback

    async def is_user_authorized(self):
        return FakeClient.authorized

    async def send_code_request(self, phone):
        from telethon.tl.types.auth import SentCode, SentCodeTypeApp

        return SentCode(type=SentCodeTypeApp(length=5), phone_code_hash="h")

    async def sign_in(self, phone=None, code=None, *, password=None, **kw):
        if password is not None:
            if password != "cloud":
                raise errors.PasswordHashInvalidError(request=None)
            return
        if code == "22222":
            raise errors.SessionPasswordNeededError(request=None)
        if code != "11111":
            raise errors.PhoneCodeInvalidError(request=None)

    async def get_me(self):
        return type("U", (), {"username": "me"})()


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("BITBOT_SECRETS_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("SECRETS_PATH", str(tmp_path / "secrets.enc.json"))
    for k in (*ss.TG_KEYS, ss.TG_SESSION_ENV):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(telethon, "TelegramClient", FakeClient)
    FakeClient.authorized = True
    FakeClient.instances = []
    telegram_login._pending.clear()


def _keys():
    ss.save(
        ss.BY_ID["telethon"],
        {
            "TELEGRAM_SENTIMENT_API_ID": "12345",
            "TELEGRAM_SENTIMENT_API_HASH": "a" * 32,
            "TELEGRAM_SENTIMENT_PHONE": "+375291234567",
        },
        None,
        "warn",
        "",
        "test",
    )


@pytest.fixture()
def app(store, monkeypatch):
    monkeypatch.setenv("DASHBOARD_API_KEY", "secret")
    from src.dashboard import app

    with patch.object(data, "get_redis", return_value=FakeRedis()):
        yield TestClient(app, raise_server_exceptions=False)


# ── хранилище ─────────────────────────────────────────────────────────────────


def test_session_stays_out_of_environment_and_api(store, monkeypatch):
    _keys()
    ss.tg_save_session(SESSION, "@me", "test")
    for k in ss.TG_KEYS:
        monkeypatch.setenv(k, "")  # apply_secrets пишет в окружение — откатим
    ss.apply_secrets(type("C", (), {}))
    assert ss.TG_SESSION_ENV not in os.environ
    assert os.environ["TELEGRAM_SENTIMENT_API_ID"] == "12345"
    view = json.dumps(ss.public_view(), ensure_ascii=False)
    assert SESSION not in view and SESSION[-4:] not in view
    item = next(i for i in ss.public_view() if i["service"] == "telethon")
    assert ss.TG_SESSION_ENV not in [f["env"] for f in item["fields"]]
    assert item["signed_in"] is True
    assert item["status"] == "ok" and "@me" in item["message"]
    ss.delete(ss.BY_ID["telethon"], "test")  # «Удалить» стирает и вход
    assert ss.get_value(ss.TG_SESSION_ENV) is None


# ── вход из приложения ────────────────────────────────────────────────────────


def test_login_needs_saved_keys(app):
    resp = app.post("/api/secrets/telethon/login", headers=H)
    assert resp.status_code == 400 and resp.json()["error"]["code"] == "no_keys"


def test_code_without_login_is_explained(app):
    resp = app.post("/api/secrets/telethon/code", json={"code": "11111"}, headers=H)
    assert resp.status_code == 409


def test_login_with_code(app):
    _keys()
    sent = app.post("/api/secrets/telethon/login", headers=H)
    assert sent.json() == {"step": "code", "via": "app"}
    bad = app.post("/api/secrets/telethon/code", json={"code": "99999"}, headers=H)
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "bad_code"
    ok = app.post("/api/secrets/telethon/code", json={"code": "11 111"}, headers=H)
    assert ok.status_code == 200 and ok.json()["step"] == "done"
    assert SESSION not in ok.text
    assert ss.get_value(ss.TG_SESSION_ENV) == SESSION
    assert not FakeClient.instances[0].connected  # клиент входа отпущен
    assert telegram_login._pending == {}


def test_login_with_cloud_password(app):
    _keys()
    app.post("/api/secrets/telethon/login", headers=H)
    step = app.post("/api/secrets/telethon/code", json={"code": "22222"}, headers=H)
    assert step.json() == {"step": "password"}
    bad = app.post(
        "/api/secrets/telethon/password", json={"password": "nope"}, headers=H
    )
    assert bad.status_code == 422 and ss.get_value(ss.TG_SESSION_ENV) is None
    ok = app.post(
        "/api/secrets/telethon/password", json={"password": "cloud"}, headers=H
    )
    assert ok.json()["step"] == "done" and "cloud" not in ok.text
    assert ss.get_value(ss.TG_SESSION_ENV) == SESSION


def test_status_endpoint_reads_what_the_bot_wrote(app):
    r = FakeRedis()
    r.set(STATUS_KEY, json.dumps({"state": "ok", "message": "Подключено"}))
    with patch.object(data, "get_redis", return_value=r):
        resp = app.get("/api/secrets/telethon/status", headers=H)
    assert resp.json()["state"] == "ok"
    unknown = app.get("/api/secrets/telethon/status", headers=H)
    assert unknown.json()["state"] == "unknown"


# ── бот: только сохранённая сессия, код не просит никогда ─────────────────────


class Redis:
    def __init__(self):
        self.redis_client = FakeRedis()

    def status(self):
        return json.loads(self.redis_client.get(STATUS_KEY))["state"]


async def test_bot_without_login_reports_it_and_never_asks_for_code(store):
    _keys()
    r = Redis()
    tg = TelegramSentiment(r)
    await tg.start()
    assert r.status() == "no_session" and FakeClient.instances == []


async def test_bot_connects_with_saved_session(store):
    _keys()
    ss.tg_save_session(SESSION, "", "test")
    r = Redis()
    tg = TelegramSentiment(r)
    await tg.start()
    assert r.status() == "ok" and FakeClient.instances[0].connected
    tg._retry_at = 0
    await tg.start()  # уже подключён с той же сессией — второго клиента нет
    assert len(FakeClient.instances) == 1
    ss.delete(ss.BY_ID["telethon"], "test")
    tg._retry_at = 0
    await tg.start()  # ключи удалили в приложении — отключаемся
    assert not FakeClient.instances[0].connected and r.status() == "off"


async def test_revoked_session_is_reported_once_and_not_retried(store):
    _keys()
    ss.tg_save_session(SESSION, "", "test")
    FakeClient.authorized = False
    r = Redis()
    tg = TelegramSentiment(r)
    await tg.start()
    assert r.status() == "revoked" and not FakeClient.instances[0].connected
    tg._retry_at = 0
    await tg.start()  # та же отозванная сессия — не переподключаемся
    assert len(FakeClient.instances) == 1


async def test_corrupted_session_is_treated_as_revoked(store):
    _keys()
    ss.tg_save_session("1" + "A" * 300, "", "test")
    r = Redis()
    await TelegramSentiment(r).start()
    assert r.status() == "revoked" and FakeClient.instances == []
