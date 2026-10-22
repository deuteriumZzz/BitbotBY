"""Шина бот ↔ дашборд и команды из приложения (src/bot_bus.py, src/bot_commands.py)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src import bot_bus
from src.bot_commands import BotCommands


class FakeRedis:
    """Минимальный redis.Redis для очереди, результатов и событий."""

    def __init__(self) -> None:
        self.kv: dict = {}
        self.lists: dict = {}
        self.published: list = []

    def set(self, key, value, nx=False, ex=None, keepttl=False):
        if nx and key in self.kv:
            return None
        self.kv[key] = value
        return True

    def hset(self, key, field, value):
        self.kv.setdefault(key, {})[field] = value

    def hgetall(self, key):
        return dict(self.kv.get(key) or {})

    def hdel(self, key, field):
        (self.kv.get(key) or {}).pop(field, None)

    def get(self, key):
        return self.kv.get(key)

    def delete(self, key):
        return int(self.kv.pop(key, None) is not None)

    def incr(self, key):
        self.kv[key] = str(int(self.kv.get(key) or 0) + 1)
        return int(self.kv[key])

    def expire(self, key, seconds):
        return True

    def lpush(self, key, value):
        self.lists.setdefault(key, []).insert(0, value)

    def ltrim(self, key, start, stop):
        items = self.lists.get(key) or []
        self.lists[key] = items[start : stop + 1]

    def lrange(self, key, start, stop):
        items = self.lists.get(key) or []
        return items[start : (None if stop == -1 else stop + 1)]

    def brpop(self, key, timeout=0):
        items = self.lists.get(key) or []
        return (key, items.pop()) if items else None

    def scan_iter(self, pattern):
        prefix = pattern.rstrip("*")
        return [k for k in list(self.kv) if k.startswith(prefix)]

    def publish(self, channel, message):
        self.published.append((channel, json.loads(message)))
        return 1


def _event_types(r: FakeRedis) -> list:
    return [m["type"] for _, m in r.published]


# ── bus ───────────────────────────────────────────────────────────────────────


def test_enqueue_is_idempotent():
    r = FakeRedis()
    assert bot_bus.enqueue(r, "pause", {}, "req-00000001", "miniapp") is True
    assert bot_bus.enqueue(r, "pause", {}, "req-00000001", "miniapp") is False
    assert len(r.lists[bot_bus.COMMANDS_KEY]) == 1


async def test_consumer_runs_handler_and_stores_result():
    r = FakeRedis()
    handler = AsyncMock(return_value={"paused": True})
    bot_bus.enqueue(r, "pause", {"x": 1}, "req-00000002", "desktop")
    consumer = bot_bus.CommandConsumer(r, {"pause": handler})

    raw = r.brpop(bot_bus.COMMANDS_KEY)[1]
    result = await consumer.handle(raw)

    handler.assert_awaited_once_with({"x": 1})
    assert result == {"ok": True, "paused": True}
    assert bot_bus.get_result(r, "req-00000002") == result
    assert "command.done" in _event_types(r)


async def test_handler_error_result_is_kept_as_is():
    r = FakeRedis()
    handler = AsyncMock(return_value=bot_bus.error("not_found", "Нет такой позиции."))
    bot_bus.enqueue(r, "close_position", {}, "req-00000003", "app")
    consumer = bot_bus.CommandConsumer(r, {"close_position": handler})
    result = await consumer.handle(r.brpop(bot_bus.COMMANDS_KEY)[1])
    assert result["ok"] is False
    assert result["error"]["code"] == "not_found"


async def test_unknown_command_is_reported():
    r = FakeRedis()
    bot_bus.enqueue(r, "launch_rockets", {}, "req-00000004", "app")
    consumer = bot_bus.CommandConsumer(r, {})
    result = await consumer.handle(r.brpop(bot_bus.COMMANDS_KEY)[1])
    assert result["error"]["code"] == "unknown_command"


async def test_handler_exception_does_not_leak_details():
    r = FakeRedis()
    handler = AsyncMock(side_effect=ValueError("api-secret-123"))
    bot_bus.enqueue(r, "panic", {}, "req-00000005", "app")
    consumer = bot_bus.CommandConsumer(r, {"panic": handler})
    result = await consumer.handle(r.brpop(bot_bus.COMMANDS_KEY)[1])
    assert result["error"]["code"] == "failed"
    assert "api-secret-123" not in json.dumps(result, ensure_ascii=False)


async def test_bad_payload_is_dropped():
    consumer = bot_bus.CommandConsumer(FakeRedis(), {})
    assert await consumer.handle(b"not json") is None


def test_emit_never_raises():
    broken = MagicMock()
    broken.publish.side_effect = ConnectionError("redis down")
    bot_bus.emit(broken, "tick", {"symbol": "SOL/USDT"})
    bot_bus.emit(None, "tick")


# ── commands ──────────────────────────────────────────────────────────────────


@pytest.fixture()
async def fake_bot():
    pos = {
        "trade_id": 7,
        "side": "buy",
        "qty": 1.0,
        "entry": 100.0,
        "stop_loss": 98.0,
        "take_profit": 104.0,
    }
    return SimpleNamespace(
        redis=SimpleNamespace(redis_client=FakeRedis()),
        _runtime_config=MagicMock(),
        telegram=SimpleNamespace(
            notify=AsyncMock(), resolve_pending=MagicMock(return_value=True)
        ),
        _handle_close_all=AsyncMock(return_value=2),
        _handle_mode_switch=AsyncMock(),
        _handle_restart=AsyncMock(),
        _monitored={"SOL/USDT": pos},
        _monitored_lock=asyncio.Lock(),
        api=SimpleNamespace(get_current_price=AsyncMock(return_value=101.5)),
        _position_monitor=SimpleNamespace(
            _force_close_at_market=AsyncMock(),
            close_fraction=AsyncMock(
                side_effect=lambda sym, pos, fraction, *a, **k: (
                    {**pos, "qty": pos["qty"] * (1 - fraction)},
                    False,
                )
            ),
            update_levels=AsyncMock(
                side_effect=lambda sym, pos, sl, tp, *a: {
                    **pos,
                    "stop_loss": sl,
                    "take_profit": tp,
                }
            ),
        ),
        _save_monitored_state=MagicMock(),
    )


@pytest.fixture()
def paper_with_pin(monkeypatch):
    from config import Config

    monkeypatch.setattr(Config, "PAPER_TRADING", True)
    monkeypatch.setattr(Config, "TRADING_MODE_PIN", "4821")


async def test_panic_closes_everything_and_pauses(fake_bot):
    result = await BotCommands(fake_bot).panic({})
    assert result == {"closed": 2, "paused": True}
    fake_bot._runtime_config.set_paused.assert_called_once_with(True)
    fake_bot.telegram.notify.assert_awaited_once()


async def test_live_requires_pin_to_be_configured(fake_bot, monkeypatch):
    from config import Config

    monkeypatch.setattr(Config, "PAPER_TRADING", True)
    monkeypatch.setattr(Config, "TRADING_MODE_PIN", "")
    result = await BotCommands(fake_bot).account_mode({"paper": False})
    assert result["error"]["code"] == "pin_required"
    fake_bot._runtime_config.apply_paper_trading.assert_not_called()


async def test_live_rejects_wrong_pin(fake_bot, paper_with_pin):
    result = await BotCommands(fake_bot).account_mode({"paper": False, "pin": "0000"})
    assert result["error"]["code"] == "wrong_pin"
    fake_bot._runtime_config.apply_paper_trading.assert_not_called()


async def test_live_locks_after_three_wrong_pins(fake_bot, paper_with_pin):
    cmds = BotCommands(fake_bot)
    for _ in range(3):
        await cmds.account_mode({"paper": False, "pin": "0000"})
    result = await cmds.account_mode({"paper": False, "pin": "4821"})
    assert result["error"]["code"] == "pin_locked"
    fake_bot._runtime_config.apply_paper_trading.assert_not_called()


async def test_live_with_correct_pin_switches_mode(fake_bot, paper_with_pin):
    result = await BotCommands(fake_bot).account_mode({"paper": False, "pin": "4821"})
    assert result == {"paper": False, "changed": True}
    fake_bot._runtime_config.apply_paper_trading.assert_called_once_with(False)
    fake_bot._handle_mode_switch.assert_awaited_once()


async def test_back_to_paper_needs_no_pin(fake_bot, monkeypatch):
    from config import Config

    monkeypatch.setattr(Config, "PAPER_TRADING", False)
    monkeypatch.setattr(Config, "TRADING_MODE_PIN", "4821")
    result = await BotCommands(fake_bot).account_mode({"paper": True})
    assert result == {"paper": True, "changed": True}


async def test_close_position_by_trade_id(fake_bot):
    result = await BotCommands(fake_bot).close_position({"trade_id": "7"})
    assert result == {"symbol": "SOL/USDT", "price": 101.5, "fraction": 1.0}
    args = fake_bot._position_monitor._force_close_at_market.await_args.args
    assert args[0] == "SOL/USDT" and args[2] == 101.5
    fake_bot._save_monitored_state.assert_called_once()


async def test_close_unknown_position(fake_bot):
    result = await BotCommands(fake_bot).close_position({"trade_id": "999"})
    assert result["error"]["code"] == "not_found"


async def test_partial_close_half(fake_bot):
    result = await BotCommands(fake_bot).close_position(
        {"trade_id": "7", "fraction": 0.5}
    )
    assert result["fraction"] == 0.5
    fake_bot._position_monitor.close_fraction.assert_awaited_once()
    fake_bot._position_monitor._force_close_at_market.assert_not_awaited()
    types = _event_types(fake_bot.redis.redis_client)
    assert "position.updated" in types


async def test_bad_fraction_rejected(fake_bot):
    result = await BotCommands(fake_bot).close_position(
        {"trade_id": "7", "fraction": 0.99}
    )
    assert result["error"]["code"] == "bad_fraction"


async def test_update_levels_validates_sides(fake_bot):
    cmds = BotCommands(fake_bot)
    # LONG по цене 101.5: стоп выше цены — нельзя.
    bad = await cmds.update_position({"trade_id": "7", "sl": 102, "tp": 110})
    assert bad["error"]["code"] == "bad_levels"
    fake_bot._position_monitor.update_levels.assert_not_awaited()
    ok = await cmds.update_position({"trade_id": "7", "sl": 99, "tp": 110})
    assert ok == {"symbol": "SOL/USDT", "sl": 99.0, "tp": 110.0}


async def test_update_levels_exchange_failure_keeps_old(fake_bot):
    fake_bot._position_monitor.update_levels = AsyncMock(side_effect=RuntimeError("x"))
    res = await BotCommands(fake_bot).update_position(
        {"trade_id": "7", "sl": 99, "tp": 110}
    )
    assert res["error"]["code"] == "order_failed"


async def test_trailing_toggle(fake_bot):
    res = await BotCommands(fake_bot).update_position(
        {"trade_id": "7", "trailing": False}
    )
    assert res["trailing"] is False
    assert fake_bot._monitored["SOL/USDT"]["trailing"] is False


async def test_signal_answer(fake_bot):
    res = await BotCommands(fake_bot).signal_answer(
        {"signal_id": "sig42", "confirm": True}
    )
    assert res == {"confirmed": True}
    fake_bot.telegram.resolve_pending.assert_called_once_with("sig42", True)
    fake_bot.telegram.resolve_pending.return_value = False
    late = await BotCommands(fake_bot).signal_answer({"signal_id": "sig42"})
    assert late["error"]["code"] == "expired"
