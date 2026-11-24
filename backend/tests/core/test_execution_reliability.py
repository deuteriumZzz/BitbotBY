"""Phase 5: машина состояний ордера, сверка позиций, ордера с неизвестным исходом."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import ccxt.async_support as ccxt
import pytest

from bitbot.domain import order_states as OS
from bitbot.execution import reliability as ER
from bitbot.storage.decision_ledger import DecisionLedger
from tests.core.test_reconciliation import make_bot
from tests.exchange.test_bybit_api_errors import _make_api

S = OS.OrderState

# ── Машина состояний ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "order,state",
    [
        ({"status": "open", "filled": 0}, S.SUBMITTED),
        ({"status": "open", "filled": 0.4}, S.PARTIALLY_FILLED),
        ({"status": "closed", "filled": 1}, S.FILLED),
        ({"status": "canceled"}, S.CANCELED),
        ({"status": "cancelled"}, S.CANCELED),
        ({"status": "rejected"}, S.REJECTED),
        ({"status": "expired"}, S.EXPIRED),
        ({"status": None}, S.UNKNOWN),
        ({"status": "open", "filled": "bad"}, S.SUBMITTED),
        (None, S.UNKNOWN),
        ({}, S.UNKNOWN),
    ],
)
def test_classify_order(order, state) -> None:
    assert OS.classify_order(order) == state


def test_transitions_accept_forward_events_only() -> None:
    assert OS.next_state(S.CREATED, S.SUBMITTED) == (S.SUBMITTED, True)
    assert OS.next_state(S.SUBMITTED, S.PARTIALLY_FILLED) == (S.PARTIALLY_FILLED, True)
    assert OS.next_state(S.PARTIALLY_FILLED, S.PARTIALLY_FILLED)[1] is True
    assert OS.next_state(S.PARTIALLY_FILLED, S.FILLED) == (S.FILLED, True)


def test_late_or_duplicate_events_do_not_change_terminal_orders() -> None:
    for terminal in OS.TERMINAL:
        for incoming in S:
            state, accepted = OS.next_state(terminal, incoming)
            assert (state, accepted) == (terminal, False)
    # повтор «исполнен» и откат к частичному — отброшены
    assert OS.next_state(S.FILLED, S.FILLED) == (S.FILLED, False)
    assert OS.next_state(S.FILLED, S.PARTIALLY_FILLED) == (S.FILLED, False)
    assert OS.next_state(S.SUBMITTED, S.CREATED) == (S.SUBMITTED, False)


# ── Сверка позиций ────────────────────────────────────────────────────────────


def _ex(symbol, qty=1.0, side="long"):
    return {"symbol": symbol, "contracts": qty, "side": side}


def _pos(qty=1.0, side="buy"):
    return {"qty": qty, "side": side}


def test_diff_detects_untracked_stale_and_mismatches() -> None:
    monitored = {
        "BTC/USDT": _pos(1.0, "buy"),
        "ETH/USDT": _pos(2.0, "buy"),  # на бирже нет
        "SOL/USDT": _pos(1.0, "buy"),  # объём другой
        "XRP/USDT": _pos(1.0, "buy"),  # сторона другая
    }
    exchange = [
        _ex("BTC/USDT:USDT", 1.0, "long"),
        _ex("SOL/USDT:USDT", 1.5, "long"),
        _ex("XRP/USDT:USDT", 1.0, "short"),
        _ex("DOGE/USDT:USDT", 100.0, "long"),  # бот не знает
    ]
    d = ER.diff_positions(monitored, exchange)
    assert d.untracked == ("DOGE/USDT",)
    assert d.stale == ("ETH/USDT",)
    assert d.qty_mismatch == (("SOL/USDT", 1.0, 1.5),)
    assert d.side_mismatch == ("XRP/USDT",)
    assert d.keys() == {
        "untracked:DOGE/USDT",
        "stale:ETH/USDT",
        "qty:SOL/USDT",
        "side:XRP/USDT",
    }


def test_diff_small_qty_difference_is_tolerated() -> None:
    d = ER.diff_positions({"BTC/USDT": _pos(1.0)}, [_ex("BTC/USDT", 1.01)])
    assert d.empty


def test_diff_ignores_in_flight_orders_and_empty_positions() -> None:
    # None — плейсхолдер: ордер уходит прямо сейчас, исполнитель не закончил
    assert ER.diff_positions({"BTC/USDT": None}, [_ex("BTC/USDT", 1.0)]).empty
    assert ER.diff_positions({"BTC/USDT": None}, []).empty
    junk = [_ex("BTC/USDT", 0.0), {"symbol": "X", "contracts": "bad"}]
    assert ER.diff_positions({}, junk).empty


def test_confirm_tracker_requires_two_consecutive_checks() -> None:
    t = ER.ConfirmTracker()
    assert t.update({"untracked:A"}) == set()
    assert t.update({"untracked:A", "stale:B"}) == {"untracked:A"}
    assert t.update({"stale:B"}) == {"stale:B"}  # A исчезла, B подтверждена
    assert t.update(set()) == set()
    assert t.update({"stale:B"}) == set()  # мигнула — подтверждение сброшено


# ── BybitAPI: ордера с неизвестным исходом и честные позиции ──────────────────


def _chaos_api():
    api = _make_api()
    api.exchange.fetch_open_orders.return_value = []
    api.exchange.fetch_closed_orders.return_value = []
    return api


@pytest.mark.asyncio
async def test_network_failure_after_all_retries_marks_order_ambiguous() -> None:
    api = _chaos_api()
    api.exchange.create_order.side_effect = ccxt.RequestTimeout("t/o")
    with patch("asyncio.sleep", new_callable=AsyncMock):
        assert await api.create_order("BTC/USDT", "market", "buy", 0.1) is None
    pending = api.ambiguous_orders
    assert len(pending) == 1
    info = next(iter(pending.values()))
    assert info["symbol"] == "BTC/USDT" and info["side"] == "buy"


@pytest.mark.asyncio
async def test_definitive_rejections_are_not_ambiguous() -> None:
    for exc in (ccxt.InsufficientFunds("x"), ccxt.AuthenticationError("x")):
        api = _chaos_api()
        api.exchange.create_order.side_effect = exc
        try:
            await api.create_order("BTC/USDT", "market", "buy", 0.1)
        except ccxt.AuthenticationError:
            pass
        assert not api.__dict__.get("ambiguous_orders")


@pytest.mark.asyncio
async def test_resolve_finds_order_that_was_executed_after_all() -> None:
    api = _chaos_api()
    api.exchange.create_order.side_effect = ccxt.NetworkError("down")
    with patch("asyncio.sleep", new_callable=AsyncMock):
        await api.create_order("BTC/USDT", "market", "buy", 0.1)
    cid = next(iter(api.ambiguous_orders))
    # позже биржа показывает: ордер исполнился
    api.exchange.fetch_closed_orders.return_value = [
        {"clientOrderId": cid, "id": "X1", "status": "closed", "filled": 0.1}
    ]
    results = await api.resolve_ambiguous()
    assert results == [
        {
            "client_id": cid,
            "symbol": "BTC/USDT",
            "state": "FILLED",
            "filled": 0.1,
            "order_id": "X1",
        }
    ]
    assert api.ambiguous_orders == {}


@pytest.mark.asyncio
async def test_unresolved_order_stays_pending_then_expires_without_claiming_absent():
    api = _chaos_api()
    api._note_ambiguous("BTC/USDT", "bb-1", "buy", 0.1)
    api.exchange.fetch_open_orders.side_effect = ccxt.NetworkError("still down")
    assert await api.resolve_ambiguous() == []  # поиск не удался — ждём
    assert "bb-1" in api.ambiguous_orders
    api.ambiguous_orders["bb-1"]["ts"] -= 10_000
    out = await api.resolve_ambiguous()
    assert out[0]["state"] == "UNRESOLVED"  # не «ордера нет», а «не выяснено»
    assert api.ambiguous_orders == {}


@pytest.mark.asyncio
async def test_accepted_order_with_lost_response_is_not_duplicated() -> None:
    api = _chaos_api()
    existing = {"clientOrderId": None, "id": "E1", "status": "closed", "filled": 0.1}
    calls = []

    async def create(*args, **kwargs):
        calls.append(kwargs["params"]["clientOrderId"])
        existing["clientOrderId"] = calls[-1]
        raise ccxt.RequestTimeout("response lost")

    api.exchange.create_order.side_effect = create
    api.exchange.fetch_closed_orders.return_value = [existing]
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await api.create_order("BTC/USDT", "market", "buy", 0.1)
    assert result is existing and len(calls) == 1  # повторной отправки нет
    assert not api.__dict__.get("ambiguous_orders")


@pytest.mark.asyncio
async def test_fetch_positions_checked_raises_instead_of_lying() -> None:
    api = _chaos_api()
    for exc in (ccxt.NetworkError("n"), RuntimeError("boom"), ccxt.NotSupported("s")):
        api.exchange.fetch_positions.side_effect = exc
        with pytest.raises(type(exc)):
            await api.fetch_positions_checked()
    api.exchange.fetch_positions.side_effect = None
    api.exchange.fetch_positions.return_value = None
    with pytest.raises(RuntimeError):
        await api.fetch_positions_checked()
    api.exchange.fetch_positions.return_value = [
        {"symbol": "A", "contracts": 0},
        {"symbol": "B", "contracts": 2},
    ]
    assert [p["symbol"] for p in await api.fetch_positions_checked()] == ["B"]


# ── Бот: стартовая и периодическая сверка ─────────────────────────────────────


def _live_bot(tmp_path, positions, monitored=None, market="linear", autofix=False):
    bot = make_bot()
    bot.ledger = DecisionLedger(path=str(tmp_path / "l.jsonl"), enabled=True)
    bot.api.market_type = MagicMock(return_value=market)
    bot.api.resolve_ambiguous = AsyncMock(return_value=[])
    bot.api.fetch_positions_checked = AsyncMock(return_value=positions)
    bot.telegram.notify = AsyncMock()
    bot._monitored.clear()
    bot._monitored.update(monitored or {})
    cfg = MagicMock()
    cfg.PAPER_TRADING = False
    cfg.RECONCILE_AUTOFIX = autofix
    return bot, cfg


def _ledger(tmp_path):
    p = tmp_path / "l.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


@pytest.mark.asyncio
async def test_untracked_alerts_only_after_two_checks_and_once(tmp_path) -> None:
    bot, cfg = _live_bot(tmp_path, [_ex("DOGE/USDT:USDT", 5.0)])
    with patch("bitbot.workers.bot.Config", cfg):
        await bot._reconcile_tick()
        bot.telegram.notify.assert_not_called()  # первая проверка — не подтверждена
        await bot._reconcile_tick()
        assert bot.telegram.notify.await_count == 1
        text = bot.telegram.notify.await_args.args[0]
        assert "DOGE/USDT" in text and "НЕ наблюдает" in text
        assert "ничего не менял" in text
        await bot._reconcile_tick()
        assert bot.telegram.notify.await_count == 1  # кулдаун: без спама
    rec = next(r for r in _ledger(tmp_path) if r["kind"] == "reconcile")
    assert rec["untracked"] == ["DOGE/USDT"]


@pytest.mark.asyncio
async def test_in_flight_order_never_triggers_alert(tmp_path) -> None:
    bot, cfg = _live_bot(
        tmp_path, [_ex("BTC/USDT:USDT", 1.0)], monitored={"BTC/USDT": None}
    )
    with patch("bitbot.workers.bot.Config", cfg):
        for _ in range(3):
            await bot._reconcile_tick()
    bot.telegram.notify.assert_not_called()


@pytest.mark.asyncio
async def test_exchange_failure_is_not_read_as_no_positions(tmp_path) -> None:
    bot, cfg = _live_bot(tmp_path, [], monitored={"BTC/USDT": _pos()})
    bot.api.fetch_positions_checked = AsyncMock(side_effect=ccxt.NetworkError("down"))
    with patch("bitbot.workers.bot.Config", cfg):
        for _ in range(4):
            await bot._reconcile_tick()
    assert "BTC/USDT" in bot._monitored  # позиция бота не тронута
    assert bot.telegram.notify.await_count == 1  # одно сообщение на 3-й сбой
    assert "3 проверки подряд" in bot.telegram.notify.await_args.args[0]
    assert not [r for r in _ledger(tmp_path) if r["kind"] == "reconcile"]


@pytest.mark.asyncio
async def test_paper_and_spot_do_nothing(tmp_path) -> None:
    bot, cfg = _live_bot(tmp_path, [_ex("DOGE/USDT:USDT")])
    cfg.PAPER_TRADING = True
    with patch("bitbot.workers.bot.Config", cfg):
        await bot._reconcile_tick()
    bot.api.fetch_positions_checked.assert_not_called()

    bot, cfg = _live_bot(tmp_path, [_ex("DOGE/USDT:USDT")], market="spot")
    with patch("bitbot.workers.bot.Config", cfg):
        await bot._reconcile_tick()
    bot.api.fetch_positions_checked.assert_not_called()


@pytest.mark.asyncio
async def test_autofix_runs_startup_reconcile_only_when_enabled(tmp_path) -> None:
    for autofix, expected_calls in ((False, 0), (True, 1)):
        bot, cfg = _live_bot(tmp_path, [_ex("DOGE/USDT:USDT")], autofix=autofix)
        bot._reconcile_positions = AsyncMock()
        with patch("bitbot.workers.bot.Config", cfg):
            await bot._reconcile_tick()
            assert bot._reconcile_positions.await_count == 0  # не подтверждено
            await bot._reconcile_tick()
        assert bot._reconcile_positions.await_count == expected_calls


@pytest.mark.asyncio
async def test_ambiguous_order_that_filled_is_reported(tmp_path) -> None:
    bot, cfg = _live_bot(tmp_path, [])
    bot.api.resolve_ambiguous = AsyncMock(
        return_value=[
            {
                "client_id": "bb-1",
                "symbol": "BTC/USDT",
                "state": "FILLED",
                "filled": 0.1,
                "order_id": "X",
            },
            {
                "client_id": "bb-2",
                "symbol": "ETH/USDT",
                "state": "UNRESOLVED",
                "filled": None,
                "order_id": None,
            },
        ]
    )
    with patch("bitbot.workers.bot.Config", cfg):
        await bot._reconcile_tick()
    assert bot.telegram.notify.await_count == 1
    assert "BTC/USDT" in bot.telegram.notify.await_args.args[0]
    kinds = [r["kind"] for r in _ledger(tmp_path)]
    assert kinds.count("reconcile_order") == 2


@pytest.mark.asyncio
async def test_startup_reconcile_skips_spot_and_keeps_positions(tmp_path) -> None:
    bot, cfg = _live_bot(tmp_path, [], monitored={"BTC/USDT": _pos()}, market="spot")
    with patch("bitbot.workers.bot.Config", cfg):
        await bot._reconcile_positions()
    assert "BTC/USDT" in bot._monitored
    bot.api.fetch_positions_checked.assert_not_called()


@pytest.mark.asyncio
async def test_startup_reconcile_keeps_positions_on_exchange_error(tmp_path) -> None:
    bot, cfg = _live_bot(tmp_path, [], monitored={"BTC/USDT": _pos()})
    bot.api.fetch_positions_checked = AsyncMock(side_effect=ccxt.NetworkError("down"))
    with patch("bitbot.workers.bot.Config", cfg):
        await bot._reconcile_positions()
    assert "BTC/USDT" in bot._monitored
    assert bot.telegram.notify.await_count == 1  # «сверка провалилась, проверь вручную»
