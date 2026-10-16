"""События для приложения и действия с одной позицией.

radar.stage (причины отказов в OrderExecutor), tick и трейлинг в PositionMonitor,
перевыставление SL/TP, ответ на предложение сделки из приложения (TelegramNotifier).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.telegram_notifier import TelegramNotifier
from tests.test_order_executor import _TOP_BUY
from tests.test_order_executor import _make_cfg as _exec_cfg
from tests.test_order_executor import _make_executor
from tests.test_position_monitor_new import _make_cfg as _mon_cfg
from tests.test_position_monitor_new import _make_monitor, _open_pos

# ── radar.stage ───────────────────────────────────────────────────────────────


async def test_rejection_reason_is_published():
    monitored = {
        f"SYM{i}/USDT": {"qty": 0.1, "side": "buy", "entry": 100.0} for i in range(3)
    }
    executor, _ = _make_executor(monitored=monitored)
    events = []
    executor._emit = lambda t, d: events.append((t, d))
    with patch("src.order_executor.Config", _exec_cfg(max_pos=3)):
        await executor.execute(_TOP_BUY, {}, 10_000.0)
    assert ("radar.stage", {"symbol": "BTC/USDT", "stage": "guards"}) in events
    assert (
        "radar.stage",
        {"symbol": "BTC/USDT", "stage": "rejected", "reason": "все слоты заняты"},
    ) in events


# ── PositionMonitor ───────────────────────────────────────────────────────────


def test_trailing_can_be_disabled_per_position():
    cfg = _mon_cfg(trailing_mult=1.0)
    monitor = _make_monitor(cfg=cfg)
    pos = _open_pos(side="buy", sl=49000, atr=500.0)
    with patch("src.position_monitor.Config", cfg):
        moved = monitor._apply_trailing_stop("BTC/USDT", pos, 51000.0)
        kept = monitor._apply_trailing_stop(
            "BTC/USDT", {**pos, "trailing": False}, 51000.0
        )
    assert moved["stop_loss"] > 49000
    assert kept["stop_loss"] == 49000


def test_tick_carries_price_and_levels():
    monitor = _make_monitor()
    events = []
    monitor._emit = lambda t, d: events.append((t, d))
    monitor._publish_tick("BTC/USDT", _open_pos(trade_id=9), 50500.0)
    ((type_, data),) = events
    assert type_ == "tick"
    assert data["price"] == 50500.0 and data["sl"] == 49000.0
    assert data["trade_id"] == 9 and data["side"] == "long"
    assert data["trailing"] is True


async def test_update_levels_places_new_before_cancelling_old():
    cfg = _mon_cfg(paper=False)
    monitor = _make_monitor(cfg=cfg)
    order = []

    async def place(*a):
        order.append("place")
        return ("sl2", "tp2")

    async def cancel(**k):
        order.append(("cancel", k["order_id"]))

    monitor._api.place_exchange_sl_tp = AsyncMock(side_effect=place)
    monitor._api.cancel_order = AsyncMock(side_effect=cancel)
    pos = _open_pos(exchange_sl_id="sl1", exchange_tp_id="tp1")
    monitored = {"BTC/USDT": pos}
    with patch("src.position_monitor.Config", cfg):
        new = await monitor.update_levels(
            "BTC/USDT", pos, 49500.0, 52500.0, monitored, asyncio.Lock()
        )
    assert order == ["place", ("cancel", "sl1"), ("cancel", "tp1")]
    assert new["exchange_sl_id"] == "sl2"
    assert monitored["BTC/USDT"]["stop_loss"] == 49500.0


async def test_update_levels_failure_keeps_old_orders():
    cfg = _mon_cfg(paper=False)
    monitor = _make_monitor(cfg=cfg)
    monitor._api.place_exchange_sl_tp = AsyncMock(side_effect=RuntimeError("no"))
    pos = _open_pos(exchange_sl_id="sl1", exchange_tp_id="tp1")
    monitored = {"BTC/USDT": pos}
    with patch("src.position_monitor.Config", cfg), pytest.raises(RuntimeError):
        await monitor.update_levels(
            "BTC/USDT", pos, 49500.0, 52500.0, monitored, asyncio.Lock()
        )
    monitor._api.cancel_order.assert_not_awaited()
    assert monitored["BTC/USDT"]["stop_loss"] == 49000.0


async def test_close_fraction_paper_keeps_sl_by_default():
    cfg = _mon_cfg(paper=True)
    monitor = _make_monitor(cfg=cfg)
    pos = _open_pos(qty=1.0, sl=49000.0)
    monitored = {"BTC/USDT": pos}
    with patch("src.position_monitor.Config", cfg):
        updated, unprotected = await monitor.close_fraction(
            "BTC/USDT", pos, 0.5, monitored, asyncio.Lock()
        )
    assert updated["qty"] == 0.5 and updated["stop_loss"] == 49000.0
    assert monitored["BTC/USDT"]["qty"] == 0.5 and unprotected is False


# ── Ответ на предложение сделки из приложения ─────────────────────────────────


def _notifier() -> TelegramNotifier:
    n = TelegramNotifier.__new__(TelegramNotifier)
    n._pending, n._decisions, n._decided_by = {}, {}, {}
    n._emit = None
    return n


def test_resolve_pending_from_app():
    n = _notifier()
    event = asyncio.Event()
    n._pending[5] = event
    n._decisions[5] = True
    assert n.resolve_pending("sig5", False) is True
    assert event.is_set() and n._decisions[5] is False
    assert n._decided_by[5] == "app"
    assert n.resolve_pending("sig5", True) is False  # уже отвечено
    assert n.resolve_pending("sig404", True) is False
    assert n.resolve_pending("garbage", True) is False


async def test_ask_confirm_publishes_and_accepts_app_answer():
    n = _notifier()
    n._enabled = True
    n._chat_id = "1"
    n._app = SimpleNamespace(
        bot=SimpleNamespace(
            send_message=AsyncMock(return_value=SimpleNamespace(message_id=7)),
            edit_message_text=AsyncMock(),
        )
    )
    events = []
    n.set_event_emitter(lambda t, d: events.append((t, d)))

    async def answer_later():
        await asyncio.sleep(0.05)
        assert n.resolve_pending("sig7", False)

    rec = {"symbol": "SOL/USDT", "action": "buy", "confidence": 0.7, "entry": 100.0}
    result, _ = await asyncio.gather(n.ask_confirm(rec, timeout=5), answer_later())

    assert result is False
    pending = next(d for t, d in events if t == "signal.pending")
    assert pending["id"] == "sig7" and pending["side"] == "long"
    resolved = next(d for t, d in events if t == "signal.resolved")
    assert resolved == {
        "id": "sig7",
        "symbol": "SOL/USDT",
        "confirmed": False,
        "by": "app",
    }
    footer = n._app.bot.edit_message_text.await_args.kwargs["text"]
    assert "в приложении" in footer
