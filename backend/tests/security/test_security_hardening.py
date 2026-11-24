"""Регрессии аудита безопасности (2026-11-24): маскировка секретов, вход в
дашборд, WebSocket, вебхук, код входа, плечо, ответы LLM."""

from __future__ import annotations

import io
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from bitbot.infra.logger import (
    JSONFormatter,
    _SecretFilter,
    install_masking,
    mask_secrets,
)
from tests.infra.test_bot_bus import FakeRedis

BOT_TOKEN = "123456789:AAH" + "x" * 32  # синтетический, не настоящий


@pytest.fixture(autouse=True)
def _keep_secrets():
    saved = list(_SecretFilter._secrets)
    yield
    _SecretFilter._secrets = saved


# ── Маскировка ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text, leaked",
    [
        (f"POST https://api.telegram.org/bot{BOT_TOKEN}/sendMessage 401", BOT_TOKEN),
        ("headers={'Authorization': 'Bearer abcdefghijklmnop'}", "abcdefghijklmnop"),
        ('{"api_key": "AKIA1234567890XYZ"}', "AKIA1234567890XYZ"),
        (
            "X-BAPI-SIGN: 9f86d081884c7d659a2feaa0c55ad015",
            "9f86d081884c7d659a2feaa0c55ad015",
        ),
        ("password=hunter2hunter2", "hunter2hunter2"),
    ],
)
def test_mask_secrets_by_shape(text, leaked):
    assert leaked not in mask_secrets(text)


def test_mask_keeps_ordinary_text():
    text = "tokens_today=1200 signal=buy BTC/USDT conf=0.81"
    assert mask_secrets(text) == text


def test_register_is_additive():
    _SecretFilter.register("first-secret-value")
    _SecretFilter.register("second-secret-value")
    out = mask_secrets("first-secret-value second-secret-value")
    assert "first-secret-value" not in out and "second-secret-value" not in out


def test_traceback_and_extra_fields_are_masked():
    secret = "live-bybit-secret-0123456789"
    _SecretFilter.register(secret)
    stream = io.StringIO()
    h = logging.StreamHandler(stream)
    h.setFormatter(JSONFormatter())
    install_masking(h)
    log = logging.getLogger("test.masking")
    log.handlers, log.propagate, log.level = [h], False, logging.INFO
    try:
        raise RuntimeError(f"bybit rejected key {secret}")
    except RuntimeError:
        log.exception("order failed", extra={"payload": f"key {secret}"})
    out = stream.getvalue()
    assert "order failed" in out and "RuntimeError" in out
    assert secret not in out


@pytest.mark.asyncio
async def test_telegram_notify_masks_text():
    from bitbot.adapters.telegram.notifier import TelegramNotifier

    n = TelegramNotifier("123:token", "1")
    resp = MagicMock()
    resp.json.return_value = {"ok": True, "result": {"message_id": 7}}
    n._client = SimpleNamespace(post=AsyncMock(return_value=resp))
    await n.notify(f"❌ Бэктест упал: https://api.telegram.org/bot{BOT_TOKEN}/x")
    sent = n._client.post.await_args.kwargs["json"]["text"]
    assert BOT_TOKEN not in sent


# ── Дашборд: Host, размер тела, WebSocket ─────────────────────────────────────


@pytest.fixture
def open_app(monkeypatch):
    """Режим без ключа и без Telegram — данные открыты только для localhost."""
    monkeypatch.delenv("DASHBOARD_API_KEY", raising=False)
    monkeypatch.delenv("DASHBOARD_ALLOWED_HOSTS", raising=False)
    from bitbot.api import app, auth

    monkeypatch.setattr(auth, "telegram_auth_configured", lambda: False)
    return app


def test_foreign_host_is_rejected_without_auth(open_app):
    c = TestClient(open_app, base_url="http://evil.example:8080")
    assert c.get("/api/trades").status_code == 403


def test_localhost_is_allowed_without_auth(open_app):
    c = TestClient(open_app, base_url="http://127.0.0.1:8080")
    with patch("bitbot.api.routers.api.get_trades", return_value=[]):
        assert c.get("/api/trades").status_code == 200


def test_oversized_body_is_rejected(open_app):
    c = TestClient(open_app, base_url="http://localhost")
    big = json.dumps({"initData": "x" * (2 * 1024 * 1024)})
    resp = c.post(
        "/api/auth/telegram", content=big, headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 413


def _ws(headers):
    from bitbot.api.routers.control import _ws_origin_ok

    return _ws_origin_ok(SimpleNamespace(headers=headers))


def test_ws_origin_check():
    assert _ws({"host": "localhost:8080", "origin": "http://localhost:8080"})
    assert _ws({"host": "bot.example.com", "origin": "https://bot.example.com"})
    assert _ws({"host": "127.0.0.1:8080", "origin": "tauri://localhost"})
    assert _ws({"host": "127.0.0.1:8080"})  # не браузер — Origin нет
    assert not _ws({"host": "127.0.0.1:8080", "origin": "https://evil.example"})


# ── Код входа Desktop ─────────────────────────────────────────────────────────


def test_desktop_code_attempts_counted_atomically():
    from bitbot.api import auth

    r = FakeRedis()
    code = auth.issue_desktop_code(r)
    wrong = "000000" if code != "000000" else "111111"
    results = [auth.check_desktop_code(r, wrong) for _ in range(auth.CODE_MAX_ATTEMPTS)]
    assert results[-1] == "locked"
    assert auth.check_desktop_code(r, code) == "locked"  # верный код уже не поможет


def test_desktop_code_single_use_under_race():
    from bitbot.api import auth

    r = FakeRedis()
    code = auth.issue_desktop_code(r)
    assert auth.check_desktop_code(r, code) == "ok"
    # Второй одновременный запрос прочитал код до удаления: delete вернёт 0.
    r.set(auth._CODE_KEY, json.dumps({"h": auth._sha(code)}))
    r.delete = lambda *k: 0
    assert auth.check_desktop_code(r, code) == "expired"


# ── Плечо ─────────────────────────────────────────────────────────────────────


def _api(exc):
    from bitbot.adapters.exchange.bybit import BybitAPI

    api = object.__new__(BybitAPI)
    api._leverage_cache, api._runtime_config = {}, None
    api.logger = logging.getLogger("test.lev")
    api.exchange = MagicMock()
    api.exchange.set_leverage = AsyncMock(side_effect=exc)
    return api


@pytest.mark.asyncio
async def test_set_leverage_real_failure_returns_false():
    with patch("bitbot.adapters.exchange.bybit.Config") as cfg:
        cfg.PAPER_TRADING, cfg.MARKET_TYPE = False, "linear"
        api = _api(Exception("bybit 110013 cannot set leverage gt max"))
        assert await api.set_leverage("BTC/USDT", 10) is False
        assert "BTC/USDT" not in api._leverage_cache


@pytest.mark.asyncio
async def test_set_leverage_not_modified_is_success():
    with patch("bitbot.adapters.exchange.bybit.Config") as cfg:
        cfg.PAPER_TRADING, cfg.MARKET_TYPE = False, "linear"
        api = _api(
            Exception('bybit {"retCode":110043,"retMsg":"leverage not modified"}')
        )
        assert await api.set_leverage("BTC/USDT", 3) is True


# ── LLM и Telegram ────────────────────────────────────────────────────────────


def test_ai_rec_for_symbol_not_sent_to_model_is_dropped():
    from bitbot.ai.analyzer import AIAnalyzer

    a = object.__new__(AIAnalyzer)
    a.logger = logging.getLogger("test.ai")
    recs = [{"symbol": "BTC/USDT"}, {"symbol": "SCAMCOIN/USDT"}]
    kept = a._only_known_symbols(recs, [{"symbol": "BTC/USDT", "price": 100}])
    assert [r["symbol"] for r in kept] == ["BTC/USDT"]


async def test_telegram_answers_only_the_owner():
    from bitbot.adapters.telegram.notifier import TelegramNotifier

    assert (
        TelegramNotifier("123:token", "")._enabled is False
    )  # нет владельца — выключен
    n = TelegramNotifier("123:token", "42")
    n._send = AsyncMock()
    await n._handle_message({"chat": {"id": 999}, "text": "/app"})
    n._send.assert_not_awaited()  # чужой чат — молчим
    await n._handle_message({"chat": {"id": 42}, "text": "/start"})
    n._send.assert_awaited_once()
