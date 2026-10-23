"""Telegram: /app (ссылка входа) и управление ботом обычными словами."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src import login_link
from src import telegram_commander as tc
from tests.test_bot_bus import FakeRedis

pytestmark = pytest.mark.skipif(
    not getattr(tc, "TELEGRAM_AVAILABLE", True), reason="нет python-telegram-bot"
)

OWNER = "42"


def _commander(r=None):
    c = object.__new__(tc.TelegramCommander)
    c._notifier = SimpleNamespace(_chat_id=OWNER, notify=AsyncMock())
    c._rc = MagicMock()
    c._rc._r.redis_client = r if r is not None else FakeRedis()
    c._reply = AsyncMock()
    return c


def _update(text="", chat=OWNER):
    return SimpleNamespace(
        effective_chat=SimpleNamespace(id=chat),
        message=SimpleNamespace(text=text, reply_text=AsyncMock()),
    )


async def test_app_link_is_local_one_time_and_owner_only(monkeypatch):
    monkeypatch.setenv("APP_PORT", "8080")
    r = FakeRedis()
    c = _commander(r)
    await c._cmd_app(_update(chat="7"), None)  # чужой чат — молчим
    c._reply.assert_not_awaited()

    await c._cmd_app(_update(), None)
    text = c._reply.await_args.args[1]
    assert "http://localhost:8080/app/#local=" in text
    token = text.split("#local=", 1)[1].split("`", 1)[0]
    assert login_link.consume(r, token) and not login_link.consume(r, token)


async def test_app_link_refuses_without_local_port(monkeypatch):
    monkeypatch.delenv("APP_PORT", raising=False)
    c = _commander()
    await c._cmd_app(_update(), None)
    assert "#local=" not in c._reply.await_args.args[1]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Пауза", ("pause", "")),
        ("бот, продолжи торговлю!", ("resume", "")),
        ("Поставь риск 1,5%", ("risk", "1,5")),
        ("сделок 5", ("slots", "5")),
        ("режим гибрид", ("mode", "гибрид")),
        ("добавь PEPE", ("add", "pepe")),
        ("закрой всё", ("closeall", "")),
        ("выключи стратегию BB", ("strat_off", "bb")),
        ("стоп-лосс 3%", None),  # не «стоп» и не пауза
        ("переключи на live", None),  # Paper/Live словами нельзя
    ],
)
def test_parse_phrase(text, expected):
    assert tc.parse_phrase(text) == expected


async def _say(c, text, chat=OWNER):
    c._rc.get_awaiting_alert_cooldown.return_value = None
    c._rc.is_awaiting_mode_pin.return_value = False
    await c._handle_text(_update(text, chat), None)


async def test_words_call_existing_commands_and_check_bounds():
    c = _commander()
    c._cmd_pause = AsyncMock()
    c._cmd_mode = AsyncMock()
    await _say(c, "пауза")
    c._cmd_pause.assert_awaited_once()
    await _say(c, "режим локальный")
    assert c._cmd_mode.await_args.args[1].args == ["local"]

    await _say(c, "риск 1.5%")
    c._rc.set_risk_per_trade.assert_called_once_with(0.015)
    await _say(c, "риск 25%")
    assert c._rc.set_risk_per_trade.call_count == 1
    assert "от 0.1% до 10%" in c._reply.await_args.args[1]

    await _say(c, "выключи стратегию rsi")
    c._rc.disable_strategy.assert_called_once_with("rsi_momentum")


async def test_words_from_a_stranger_are_ignored():
    c = _commander()
    c._cmd_pause = AsyncMock()
    await _say(c, "пауза", chat="7")
    c._cmd_pause.assert_not_awaited()
    c._reply.assert_not_awaited()


async def test_unknown_phrase_gets_examples():
    c = _commander()
    await _say(c, "привет")
    assert "Можно писать словами" in c._reply.await_args.args[1]


async def test_sac_means_dqn_in_mode_command_and_words():
    c = _commander()
    c._rc.set_mode.return_value = True
    await c._cmd_mode(_update(), SimpleNamespace(args=["sac"]))
    c._rc.set_mode.assert_called_with("dqn")
    await _say(c, "режим sac")
    c._rc.set_mode.assert_called_with("dqn")


def test_telegram_preset_lines_come_from_the_shared_presets():
    from src.runtime_config import RISK_PRESETS

    text = tc._presets_text()
    for p in RISK_PRESETS.values():
        assert f"{p['MAX_POSITIONS']} поз, {p['RISK_PER_TRADE'] * 100:g}%" in text
    assert "плечо фикс" in text


async def test_notify_respects_gate_for_marked_messages_only():
    from src.telegram_notifier import TelegramNotifier

    n = TelegramNotifier("", "")
    n._enabled = True
    n._app = MagicMock()
    n._app.bot.send_message = AsyncMock()
    n.gate = lambda kind: kind != "position_open"
    await n.notify("открыта", kind="position_open")
    n._app.bot.send_message.assert_not_awaited()
    await n.notify("позиция без SL")  # без вида — приходит всегда
    await n.notify("закрыта", kind="position_close")
    assert n._app.bot.send_message.await_count == 2


def test_settings_menu_offers_the_miniapp_only_with_https(monkeypatch):
    monkeypatch.setenv("WEBAPP_URL", "https://example.test/app/")
    first = tc._kb_settings(paused=False, auto_exec=False).inline_keyboard[0][0]
    assert first.web_app is not None and "приложении" in first.text
    monkeypatch.setenv("WEBAPP_URL", "")
    first = tc._kb_settings(paused=False, auto_exec=False).inline_keyboard[0][0]
    assert first.web_app is None
