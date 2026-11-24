"""Phase 4b: стакан, метрики потока, свечи, приватные события, соединение (без сети)."""

import asyncio
import json

import pytest

from src.core.execution_reliability import OrderState
from src.market.ws.book import IGNORED, OK, RESYNC, OrderBook
from src.market.ws.candles import CandleAssembler, compare_with_rest
from src.market.ws.client import (
    MarketStream,
    backoff_delay,
    run_stream,
    to_bybit,
    topics,
)
from src.market.ws.health import StreamHealth
from src.market.ws.private import ExecutionDedup, OrderEvents


def snap(u=10, b=(("100", "5"), ("99", "3")), a=(("101", "4"), ("102", "2"))):
    return {"u": u, "b": [list(x) for x in b], "a": [list(x) for x in a]}


# ── стакан ────────────────────────────────────────────────────────────────────
def test_snapshot_then_deltas_apply_and_remove():
    ob = OrderBook("BTCUSDT")
    assert ob.apply("snapshot", snap()) == OK and ob.ready
    assert (
        ob.apply(
            "delta", {"u": 11, "b": [["100", "0"], ["98", "7"]], "a": [["101", "9"]]}
        )
        == OK
    )
    assert ob.best_bid == 99 and ob.bids[98] == 7 and ob.asks[101] == 9
    assert ob.spread_bps() == pytest.approx(2 / 100 * 1e4, rel=0.01)


def test_gap_requires_resync_and_blocks_until_snapshot():
    ob = OrderBook("BTCUSDT")
    ob.apply("snapshot", snap(u=10))
    assert ob.apply("delta", {"u": 12, "b": [], "a": []}) == RESYNC
    assert not ob.ready and ob.gaps == 1
    assert (
        ob.apply("delta", {"u": 13, "b": [], "a": []}) == IGNORED
    )  # до нового snapshot
    assert ob.imbalance() is None
    assert ob.apply("snapshot", snap(u=20)) == OK and ob.ready


def test_service_restart_u1_is_treated_as_snapshot():
    ob = OrderBook("BTCUSDT")
    ob.apply("snapshot", snap(u=500))
    assert ob.apply("delta", snap(u=1, b=(("50", "1"),), a=(("51", "1"),))) == OK
    assert ob.bids == {50.0: 1.0} and ob.last_u == 1


def test_crossed_book_is_invalid():
    ob = OrderBook("BTCUSDT")
    ob.apply("snapshot", snap())
    assert ob.apply("delta", {"u": 11, "b": [["105", "1"]], "a": []}) == RESYNC
    assert ob.crossed == 1 and not ob.ready


def test_delta_before_snapshot_ignored_and_reset():
    ob = OrderBook("BTCUSDT")
    assert ob.apply("delta", {"u": 2, "b": [], "a": []}) == IGNORED
    ob.apply("snapshot", snap())
    ob.reset()
    assert not ob.ready and not ob.bids


def test_imbalance_and_trim():
    ob = OrderBook("X", depth=2)
    ob.apply(
        "snapshot", snap(b=(("100", "9"), ("99", "1")), a=(("101", "1"), ("102", "1")))
    )
    assert ob.imbalance(2) == pytest.approx((10 - 2) / 12)
    ob.apply("delta", {"u": 11, "b": [[str(90 - i), "1"] for i in range(10)], "a": []})
    assert len(ob.bids) <= 4  # depth*2


# ── метрики потока ────────────────────────────────────────────────────────────
def test_lag_percentiles_relative_to_offset():
    now = [1_000_000.0]
    h = StreamHealth(wall_ms=lambda: now[0], mono=lambda: now[0] / 1000)
    for lag in [500 + i for i in range(100)]:  # смещение часов ≈ 500 мс + сеть 0..99
        h.on_message("orderbook.X", now[0] - lag)
    m = h.lag("orderbook.X")
    assert m["offset_ms"] == pytest.approx(505, abs=1)
    assert 40 <= m["p50"] <= 60 and m["p99"] >= 90


def test_silent_stream_detected_and_snapshot_shape():
    t = [0.0]
    h = StreamHealth(wall_ms=lambda: 1000.0, mono=lambda: t[0])
    h.on_message("kline.X", 1000)
    h.on_message("tickers.X", 1000)
    t[0] = 10
    h.on_message("tickers.X", 1000)
    t[0] = 60
    assert h.silent_streams(30) == ["kline.X", "tickers.X"]
    h.on_gap("kline.X")
    h.on_reconnect()
    s = h.snapshot()
    assert s["reconnects"] == 1 and s["streams"]["kline.X"]["gaps"] == 1


# ── свечи ─────────────────────────────────────────────────────────────────────
def k(start, confirm=True, close=1.0, vol=10):
    return {
        "start": start,
        "open": 1,
        "high": 2,
        "low": 0.5,
        "close": close,
        "volume": vol,
        "confirm": confirm,
    }


def test_only_confirmed_candles_enter_history():
    c = CandleAssembler(900_000)
    assert (
        not c.on_kline(k(0, confirm=False)) and c.confirmed == [] and c.forming[0] == 0
    )
    assert c.on_kline(k(0)) and c.forming is None
    assert not c.on_kline(k(0))  # дубликат
    assert c.on_kline(k(900_000))
    assert not c.on_kline(k(0))  # старее последней
    assert (c.duplicates, c.out_of_order, c.gaps) == (1, 1, 0)
    c.on_kline(k(900_000 * 4))
    assert c.gaps == 1 and c.ohlcv()[-1][0] == 3_600_000


def test_compare_with_rest_detects_mismatch_and_missing():
    ws = [
        [0, 1, 2, 0.5, 1.0, 10],
        [900, 1, 2, 0.5, 1.0, 10],
        [3600, 1, 2, 0.5, 1.0, 10],
    ]
    rest = [
        [0, 1, 2, 0.5, 1.0, 10],
        [900, 1, 2, 0.5, 1.1, 10],
        [1800, 1, 2, 0.5, 1.0, 10],
        [3600, 1, 2, 0.5, 1.0, 10],
    ]
    r = compare_with_rest(ws, rest)
    assert r["mismatched"] == [900] and r["missing_in_ws"] == [1800] and not r["ok"]
    assert compare_with_rest(ws[:1], rest[:1])["ok"]


# ── приватные события ─────────────────────────────────────────────────────────
def ev(status, t, oid="o1"):
    return {"orderId": oid, "orderStatus": status, "updatedTime": t}


def test_order_events_dedupe_reorder_and_terminal():
    oe = OrderEvents()
    assert oe.on_event(ev("PartiallyFilled", 2)) == (
        OrderState.PARTIALLY_FILLED,
        "accepted",
    )
    assert oe.on_event(ev("PartiallyFilled", 2))[1] == "duplicate"
    assert oe.on_event(ev("Filled", 4))[0] == OrderState.FILLED
    assert (
        oe.on_event(ev("PartiallyFilled", 3))[1] == "out_of_order"
    )  # пришло позже, чем Filled
    assert (
        oe.on_event(ev("Cancelled", 5))[1] == "invalid_transition"
    )  # из финального не выходим
    assert oe.state_of("o1") == OrderState.FILLED
    assert oe.dropped["duplicate"] == 1 and oe.dropped["out_of_order"] == 1


def test_first_event_filled_without_prior_new_is_accepted():
    oe = OrderEvents()
    assert oe.on_event(ev("Filled", 9, "m1")) == (OrderState.FILLED, "accepted")


def test_unknown_status_and_bounded_memory():
    oe = OrderEvents(max_orders=3)
    assert oe.on_event(ev("Weird", 1))[1] == "unknown_status"
    for i in range(10):
        oe.on_event(ev("New", i, f"o{i}"))
    assert oe.state_of("o0") is None and oe.state_of("o9") is not None


def test_execution_dedup_bounded():
    d = ExecutionDedup(capacity=3)
    assert d.is_new("a") and not d.is_new("a") and d.duplicates == 1
    for x in "bcd":
        d.is_new(x)
    assert d.is_new("a")  # вытеснен


# ── клиент ────────────────────────────────────────────────────────────────────
def test_topics_and_symbol_mapping():
    assert to_bybit("BTC/USDT:USDT") == "BTCUSDT"
    assert topics(["BTC/USDT"], "15m", 50) == [
        "kline.15.BTCUSDT",
        "orderbook.50.BTCUSDT",
        "tickers.BTCUSDT",
        "publicTrade.BTCUSDT",
    ]


def test_backoff_has_jitter_and_cap():
    assert backoff_delay(3, rng=lambda: 1.0) == 8.0
    assert backoff_delay(3, rng=lambda: 0.0) == 0.0
    assert backoff_delay(30, rng=lambda: 1.0) == 60.0


def ob_msg(typ, u, b=(), a=()):
    return json.dumps(
        {
            "topic": "orderbook.50.BTCUSDT",
            "type": typ,
            "ts": 1,
            "data": {
                "s": "BTCUSDT",
                "u": u,
                "b": [list(x) for x in b],
                "a": [list(x) for x in a],
            },
        }
    )


def test_stream_routes_and_requests_resubscribe_on_gap():
    ms = MarketStream(["BTC/USDT"])
    assert ms.handle("not json") == [] and ms.handle(json.dumps({"op": "pong"})) == []
    assert ms.handle(ob_msg("snapshot", 1, [("100", "1")], [("101", "1")])) == []
    assert ms.handle(ob_msg("delta", 5)) == [("resubscribe", "orderbook.50.BTCUSDT")]
    assert ms.health.gaps["orderbook.BTCUSDT"] == 1
    ms.handle(json.dumps({"topic": "kline.15.BTCUSDT", "ts": 1, "data": [k(0)]}))
    assert ms.candles["BTCUSDT"].ohlcv()[0][0] == 0
    ms.handle(
        json.dumps(
            {
                "topic": "tickers.BTCUSDT",
                "ts": 1,
                "data": {"fundingRate": "0.0001", "x": ""},
            }
        )
    )
    assert ms.tickers["BTCUSDT"] == {"fundingRate": "0.0001"}


class FakeWS:
    def __init__(self, script, stop=None):
        self.script, self.sent, self.closed, self.stop = list(script), [], False, stop

    async def send(self, s):
        self.sent.append(json.loads(s))

    async def recv(self):
        if not self.script:
            await asyncio.sleep(3600)  # молчим
        item = self.script.pop(0)
        if item == "STOP":
            self.stop.set()
            await asyncio.sleep(3600)
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self):
        self.closed = True


async def test_reconnects_with_backoff_and_resets_book():
    stop = asyncio.Event()
    ws1 = FakeWS(
        [ob_msg("snapshot", 1, [("100", "1")], [("101", "1")]), ConnectionError("drop")]
    )
    ws2 = FakeWS([ob_msg("snapshot", 7, [("100", "2")], [("101", "1")]), "STOP"], stop)
    conns, sleeps = [ws1, ws2], []

    async def connect():
        return conns.pop(0)

    async def sleep(d):
        sleeps.append(d)

    ms = MarketStream(["BTC/USDT"])
    await asyncio.wait_for(
        run_stream(
            connect, ms, ["orderbook.50.BTCUSDT"], stop, sleep=sleep, rng=lambda: 1.0
        ),
        5,
    )
    assert sleeps == [2.0] and ms.health.reconnects == 1 and ws1.closed and ws2.closed
    assert (
        ws1.sent[0]["op"] == "subscribe" and ws2.sent[0]["op"] == "subscribe"
    )  # повторная подписка
    assert ms.books["BTCUSDT"].ready and ms.books["BTCUSDT"].last_u == 7


async def test_gap_triggers_unsubscribe_subscribe_pair():
    stop = asyncio.Event()
    ws = FakeWS(
        [
            ob_msg("snapshot", 1, [("100", "1")], [("101", "1")]),
            ob_msg("delta", 9),
            "STOP",
        ],
        stop,
    )

    async def connect():
        return ws

    await asyncio.wait_for(
        run_stream(connect, MarketStream(["BTC/USDT"]), ["t"], stop), 5
    )
    assert [m["op"] for m in ws.sent][:3] == ["subscribe", "unsubscribe", "subscribe"]


async def test_silent_connection_is_dropped_and_ping_sent():
    stop = asyncio.Event()
    ws1 = FakeWS([])  # вечная тишина
    ws2 = FakeWS(["STOP"], stop)
    conns = [ws1, ws2]

    async def connect():
        return conns.pop(0)

    async def sleep(_):
        return None

    await asyncio.wait_for(
        run_stream(
            connect,
            MarketStream(["BTC/USDT"]),
            ["t"],
            stop,
            sleep=sleep,
            stale_after=0.06,
            ping_every=0.02,
        ),
        5,
    )
    assert {
        "op": "ping"
    } in ws1.sent and ws1.closed  # пинговали, затем бросили молчащее соединение
    assert not conns


async def test_connect_failure_is_retried():
    stop = asyncio.Event()
    ws = FakeWS(["STOP"], stop)
    attempts = []

    async def connect():
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError("no route")
        return ws

    async def sleep(_):
        return None

    await asyncio.wait_for(
        run_stream(connect, MarketStream(["BTC/USDT"]), ["t"], stop, sleep=sleep), 5
    )
    assert len(attempts) == 3
