"""Шина событий (Redis Streams), быстрый JSON и приёмник Ledger → поток."""

import datetime
import json

import pytest

from bitbot.infra import fastjson
from bitbot.storage.decision_ledger import DecisionLedger
from bitbot.storage.event_stream import EventStream


class FakeRedis:
    """Минимум Redis Streams: xadd с maxlen, xread, группы, xack."""

    def __init__(self):
        self.entries, self.groups, self.n = [], {}, 0

    def xadd(self, key, fields, maxlen=None, approximate=True):
        self.n += 1
        eid = f"{self.n}-0"
        self.entries.append(
            (eid.encode(), {k.encode(): v.encode() for k, v in fields.items()})
        )
        if maxlen:
            del self.entries[:-maxlen]
        return eid.encode()

    def xread(self, streams, count=None, block=None):
        ((_, last),) = streams.items()
        after = int(last.split("-")[0])
        out = [e for e in self.entries if int(e[0].decode().split("-")[0]) > after][
            :count
        ]
        return [(b"k", out)] if out else []

    def xgroup_create(self, key, group, id="0", mkstream=False):
        if group in self.groups:
            raise RuntimeError("BUSYGROUP Consumer Group name already exists")
        self.groups[group] = {"pos": 0, "acked": set()}

    def xreadgroup(self, group, consumer, streams, count=None, block=None):
        g = self.groups[group]
        out = self.entries[g["pos"] : g["pos"] + count]
        g["pos"] += len(out)
        return [(b"k", out)] if out else []

    def xack(self, key, group, *ids):
        self.groups[group]["acked"].update(ids)
        return len(ids)


def test_publish_and_read_roundtrip_with_unicode_and_cursor():
    es = EventStream(FakeRedis())
    a = es.publish("signal.open", {"symbol": "BTC/USDT", "note": "вход"})
    b = es.publish("signal.close", {"pnl": 1.5})
    got = es.read("0-0")
    assert [g[0] for g in got] == [a, b] and got[0][1]["type"] == "signal.open"
    assert got[0][1]["data"]["note"] == "вход" and got[1][1]["ts"] > 0
    assert [g[0] for g in es.read(a)] == [b]  # догоняющее чтение с курсора
    assert es.read(b) == []


def test_stream_is_trimmed_to_maxlen():
    es = EventStream(FakeRedis(), maxlen=3)
    for i in range(7):
        es.publish("e", {"i": i})
    assert [g[1]["data"]["i"] for g in es.read("0-0")] == [4, 5, 6]


def test_consumer_group_delivers_once_and_acks():
    es = EventStream(FakeRedis())
    es.ensure_group("g")
    es.ensure_group("g")  # повтор штатный (BUSYGROUP)
    ids = [es.publish("e", {"i": i}) for i in range(3)]
    first = es.consume("g", "c1", count=2)
    second = es.consume("g", "c1", count=2)
    assert [x[0] for x in first] == ids[:2] and [x[0] for x in second] == ids[2:]
    assert es.consume("g", "c1") == []
    assert es.ack("g", [x[0] for x in first]) == 2 and es.ack("g", []) == 0


def test_ensure_group_reraises_real_errors():
    class Broken(FakeRedis):
        def xgroup_create(self, *a, **k):
            raise RuntimeError("WRONGTYPE")

    with pytest.raises(RuntimeError):
        EventStream(Broken()).ensure_group("g")


def test_publish_never_raises_and_none_redis_is_noop():
    class Down:
        def xadd(self, *a, **k):
            raise ConnectionError("redis down")

    assert EventStream(Down()).publish("e") is None
    assert EventStream(None).publish("e") is None


def test_ledger_sink_publishes_compact_marker_and_never_breaks_the_journal(tmp_path):
    es = EventStream(FakeRedis())
    led = DecisionLedger(path=str(tmp_path / "l.jsonl"), enabled=True)
    led.sink = es.ledger_sink
    led.write(
        "signals", decision_id="d1", symbol="BTC/USDT", decisions=[{"big": "x" * 1000}]
    )
    ev = es.read("0-0")[0][1]
    assert (
        ev["type"] == "ledger.signals"
        and ev["data"]["decision_id"] == "d1"
        and "decisions" not in ev["data"]
    )

    def broken(kind, rec):
        raise RuntimeError("sink down")

    led.sink = broken
    led.write("execution", decision_id="d2")
    lines = (tmp_path / "l.jsonl").read_text().splitlines()
    assert (
        len(lines) == 2 and json.loads(lines[1])["decision_id"] == "d2"
    )  # запись в файл не пострадала


def test_ledger_without_sink_is_unchanged(tmp_path):
    led = DecisionLedger(path=str(tmp_path / "l.jsonl"), enabled=True)
    assert led.sink is None
    led.write("x", a=1)
    assert json.loads((tmp_path / "l.jsonl").read_text())["a"] == 1


def test_fastjson_roundtrip_unicode_and_unknown_types():
    s = fastjson.dumps(
        {"текст": "значение", "when": datetime.date(2026, 11, 24), "n": 1.5}
    )
    back = fastjson.loads(s)
    assert (
        back["текст"] == "значение"
        and back["when"] == "2026-11-24"
        and back["n"] == 1.5
    )
    assert fastjson.backend() in ("orjson", "json")
    assert fastjson.loads(b'{"a": 1}') == {"a": 1}
