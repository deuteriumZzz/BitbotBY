"""Операционные задачи бота: /emergency, бэкапы БД, WS-shadow, метрики."""

import asyncio
import json
import os
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core import ops_tasks as ot
from src.infra import health_server
from src.ops.emergency import EmergencyControl


def make_bot(tmp_path):
    bot = SimpleNamespace(
        _runtime_config=MagicMock(),
        _monitored={"BTC/USDT": {"side": "buy"}, "ETH/USDT": None},
        _monitored_lock=asyncio.Lock(),
        api=SimpleNamespace(get_current_price=AsyncMock(return_value=100.0)),
        _position_monitor=SimpleNamespace(close_losers_tighten_winners=AsyncMock()),
        _close_all_open_positions=AsyncMock(),
        _save_monitored_state=MagicMock(),
        telegram=SimpleNamespace(notify=AsyncMock()),
        _ledger_write=MagicMock(),
        _emergency_ctl=None,
    )
    bot._emergency_ctl = ot.build_emergency(bot, str(tmp_path / "em.json"))
    return bot


# ── аварийный уровень из приложения ───────────────────────────────────────────
async def test_invalid_level_is_rejected(tmp_path):
    bot = make_bot(tmp_path)
    for bad in ("9", "abc", ""):
        res = await ot.emergency_apply(bot, bad, True, "owner")
        assert res["ok"] is False and res["error"]["code"] == "bad_level"


async def test_level_requires_confirmation_then_pauses_entries(tmp_path):
    bot = make_bot(tmp_path)
    ask = await ot.emergency_apply(bot, "1", False, "owner")
    assert ask["preview"] and "пауза" in ask["steps"][0]
    bot._runtime_config.set_paused.assert_not_called()  # без подтверждения ничего
    done = await ot.emergency_apply(bot, "1", True, "owner")
    assert done["executed"] == ["block_entries"] and done["level"] == 1
    bot._runtime_config.set_paused.assert_called_once_with(True)
    assert bot._emergency_ctl.level == 1


async def test_level_three_runs_levels_in_order_and_never_cancels_exchange_orders(
    tmp_path,
):
    bot = make_bot(tmp_path)
    out = await ot.emergency_apply(bot, "3", True, "owner")
    assert out["executed"] == ["block_entries", "cancel_orders", "reduce_positions"]
    args = bot._position_monitor.close_losers_tighten_winners.await_args
    assert args.args[2] == {"BTC/USDT": 100.0}  # цены только по открытым позициям
    assert args.kwargs["reason"] == "emergency"
    assert not hasattr(bot.api, "cancel_all_orders")  # биржевые SL/TP не снимаются
    bot._close_all_open_positions.assert_not_awaited()


async def test_level_four_closes_everything_and_keeps_pause(tmp_path):
    bot = make_bot(tmp_path)
    await ot.emergency_apply(bot, "4", True, "owner")
    bot._close_all_open_positions.assert_awaited_once_with("emergency_full_stop")
    bot._save_monitored_state.assert_called_once()
    assert bot._runtime_config.set_paused.call_args_list[-1].args == (True,)


async def test_lower_level_needs_reset_and_reset_needs_confirm_and_keeps_pause(
    tmp_path,
):
    bot = make_bot(tmp_path)
    await ot.emergency_apply(bot, "2", True, "owner")
    lower = await ot.emergency_apply(bot, "1", True, "owner")
    assert lower["ok"] is False and lower["error"]["code"] == "level_active"
    preview = await ot.emergency_apply(bot, "reset", False, "owner")
    assert preview["preview"] and bot._emergency_ctl.level == 2  # без confirm не снято
    out = await ot.emergency_apply(bot, "reset", True, "owner")
    assert out["reset"] and bot._emergency_ctl.level == 0
    assert all(
        c.args != (False,) for c in bot._runtime_config.set_paused.call_args_list
    )


async def test_failed_action_is_reported_and_level_is_still_persisted(tmp_path):
    bot = make_bot(tmp_path)
    bot._position_monitor.close_losers_tighten_winners.side_effect = RuntimeError(
        "биржа"
    )
    out = await ot.emergency_apply(bot, "3", True, "owner")
    assert "reduce_positions" in out["errors"]
    assert bot._emergency_ctl.level == 3
    bot._runtime_config.set_paused.assert_called_once_with(
        True
    )  # уровень 1 сработал до сбоя


async def test_restart_keeps_trading_paused_while_emergency_is_active(tmp_path):
    bot = make_bot(tmp_path)
    await ot.apply_emergency_on_start(bot)
    bot._runtime_config.set_paused.assert_not_called()  # норма — ничего не трогаем
    bot._emergency_ctl.escalate(2, "owner", "тест")
    await ot.apply_emergency_on_start(bot)
    bot._runtime_config.set_paused.assert_called_with(True)
    assert "остаётся на паузе" in bot.telegram.notify.await_args.args[0]


async def test_aescalate_supports_sync_and_async_actions(tmp_path):
    calls = []
    ec = EmergencyControl(
        str(tmp_path / "e.json"),
        {
            "block_entries": lambda: calls.append("a"),
            "cancel_orders": AsyncMock(side_effect=lambda: calls.append("b")),
            "reduce_positions": lambda: calls.append("c"),
            "full_stop": lambda: calls.append("d"),
        },
    )
    res = await ec.aescalate(2, "o", "r")
    assert calls == ["a", "b"] and res["executed"] == ["block_entries", "cancel_orders"]
    assert (await ec.aescalate(2, "o", "r"))["noop"]


# ── бэкапы ────────────────────────────────────────────────────────────────────
def make_db(path, rows=20):
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE trades (id INTEGER PRIMARY KEY, x TEXT)")
    c.executemany("INSERT INTO trades(x) VALUES (?)", [("v" * 30,)] * rows)
    c.commit()
    c.close()


async def test_backup_once_creates_verified_copy_and_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(ot, "BACKUP_DIR", str(tmp_path / "bk"))
    db = str(tmp_path / "trades_paper.db")
    make_db(db)
    bot = make_bot(tmp_path)
    res = await ot.backup_once(bot, db)
    assert res["ok"] and os.path.exists(res["backup"])
    assert (
        bot._ledger_write.call_args.args[0] == "db_backup"
        and bot._ledger_write.call_args.kwargs["ok"]
    )
    bot.telegram.notify.assert_not_awaited()


async def test_backup_skips_empty_or_missing_and_alerts_on_corruption(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(ot, "BACKUP_DIR", str(tmp_path / "bk"))
    bot = make_bot(tmp_path)
    empty = tmp_path / "live.db"
    empty.write_bytes(b"")
    assert (await ot.backup_once(bot, str(empty)))["skipped"]
    assert (await ot.backup_once(bot, str(tmp_path / "none.db")))["skipped"]
    bad = str(tmp_path / "bad.db")
    make_db(bad, rows=3000)
    with open(bad, "r+b") as f:
        f.seek(4096 * 4)
        f.write(b"\xff" * 600)
    res = await ot.backup_once(bot, bad)
    assert not res["ok"] and "не создана" in bot.telegram.notify.await_args.args[0]


async def test_backup_loop_survives_failures_and_uses_configured_interval(monkeypatch):
    sleeps = []

    async def sleep(d):
        sleeps.append(d)
        if len(sleeps) >= 3:
            raise asyncio.CancelledError

    monkeypatch.setattr(
        ot, "backup_once", AsyncMock(side_effect=[RuntimeError("x"), {"ok": True}])
    )
    monkeypatch.setattr(ot.Config, "DB_BACKUP_INTERVAL_H", 6.0)
    with pytest.raises(asyncio.CancelledError):
        await ot.db_backup_loop(SimpleNamespace(), sleep=sleep, first_delay=5.0)
    assert sleeps == [5.0, 6 * 3600, 6 * 3600] and ot.backup_once.await_count == 2


# ── WS-shadow и метрики ───────────────────────────────────────────────────────
class FakeWS:
    def __init__(self):
        self.sent = []
        self.msgs = [
            json.dumps(
                {
                    "topic": "orderbook.50.BTCUSDT",
                    "type": "snapshot",
                    "ts": 1,
                    "data": {"u": 1, "b": [["100", "1"]], "a": [["101", "1"]]},
                }
            )
        ]

    async def send(self, s):
        self.sent.append(s)

    async def recv(self):
        if self.msgs:
            return self.msgs.pop(0)
        await asyncio.sleep(3600)
        return None

    async def close(self):
        pass


async def test_ws_shadow_publishes_metrics_and_alerts_on_silence(monkeypatch):
    monkeypatch.setattr(ot.Config, "WS_SHADOW_SYMBOLS", "BTC/USDT")
    monkeypatch.setattr(ot, "STALE_ALERT_S", -1.0)  # всё считаем молчащим
    published = []
    monkeypatch.setattr(health_server, "publish_ws_metrics", published.append)
    bot = SimpleNamespace(telegram=SimpleNamespace(notify=AsyncMock()), ws_stream=None)
    stop = asyncio.Event()
    ticks = []

    async def sleep(_):
        ticks.append(1)
        await asyncio.sleep(0.05)  # дать потоку прочитать сообщение
        if len(ticks) >= 3:
            stop.set()

    ws = FakeWS()

    async def connect():
        return ws

    await asyncio.wait_for(
        ot.ws_shadow_loop(bot, connect=connect, sleep=sleep, stop=stop), 10
    )
    assert bot.ws_stream.books["BTCUSDT"].ready and published
    assert bot.telegram.notify.await_count == 1  # кулдаун: повторов за три такта нет
    assert "замолчали" in bot.telegram.notify.await_args.args[0]


def test_publish_ws_metrics_exposes_gauges(monkeypatch):
    # prometheus_client в тестах подменён заглушкой (tests/conftest.py) — проверяем вызовы gauge
    g = {
        n: MagicMock()
        for n in ("_g_ws_gaps", "_g_ws_lag_p99", "_g_ws_silence", "_g_ws_reconnects")
    }
    for name, mock in g.items():
        monkeypatch.setattr(health_server, name, mock)
    health_server.publish_ws_metrics(
        {
            "reconnects": 2,
            "streams": {
                "orderbook.BTCUSDT": {
                    "gaps": 3,
                    "seconds_since_last": 1.5,
                    "lag_ms": {"p99": 42.0},
                },
                "kline.BTCUSDT": {
                    "gaps": 0,
                    "seconds_since_last": None,
                    "lag_ms": {"p99": None},
                },
            },
        }
    )
    g["_g_ws_reconnects"].set.assert_called_once_with(2)
    g["_g_ws_gaps"].labels.assert_any_call("orderbook.BTCUSDT")
    g["_g_ws_gaps"].labels.return_value.set.assert_any_call(3)
    g["_g_ws_lag_p99"].labels.assert_called_once_with(
        "orderbook.BTCUSDT"
    )  # None не публикуется
    g["_g_ws_lag_p99"].labels.return_value.set.assert_called_once_with(42.0)
    g["_g_ws_silence"].labels.assert_called_once_with("orderbook.BTCUSDT")
