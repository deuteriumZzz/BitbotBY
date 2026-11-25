"""Табло честности: прогноз сверяется со свечами после него, без заглядывания вперёд."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from bitbot.research import ai_scoreboard as sb
from bitbot.storage.decision_ledger import DecisionLedger


def bar(ts, high, low):
    return [ts, 0, high, low, 0, 0]


def test_first_touch_order_and_ambiguity():
    assert sb.first_touch([bar(0, 101, 99.5), bar(1, 103, 100)], 102, 98) == "up"
    assert sb.first_touch([bar(0, 101, 97)], 102, 98) == "down"
    assert (
        sb.first_touch([bar(0, 103, 97)], 102, 98) == "both"
    )  # в одной свече — неизвестно
    assert sb.first_touch([bar(0, 101, 99)], 102, 98) == "none"


def test_resolve_forecast_and_short_trade():
    rec = {
        "ts": "t",
        "provider": "groq",
        "atr_mult": 1.0,
        "items": [{"symbol": "A", "price": 100, "atr": 2, "p_up": 0.7}],
        "trades": [
            {
                "symbol": "A",
                "action": "sell",
                "stop_loss": 104,
                "take_profit": 96,
                "p_win": 0.6,
            }
        ],
    }
    out = sb.resolve(rec, {"A": [bar(0, 101, 97.5), bar(1, 100, 95)]})
    assert out["t|A|up"]["y"] == 0  # −ATR (98) раньше +ATR (102)
    assert out["t|A|trade"]["y"] == 1  # шорт: тейк 96 раньше стопа 104


def test_score_verdicts_against_coin_flip():
    good = [
        {"provider": "g", "kind": "forecast", "p": 0.8 if y else 0.2, "y": y}
        for y in [1, 0] * 150
    ]
    flat = [
        {"provider": "f", "kind": "forecast", "p": 0.5, "y": y} for y in [1, 0] * 150
    ]
    few = [{"provider": "x", "kind": "forecast", "p": 0.9, "y": 1}] * 10
    s = sb.score(good + flat + few)
    assert (
        s["g"]["forecast"]["verdict"] == "лучше монетки"
        and s["g"]["forecast"]["skill"] > 0.5
    )
    assert s["f"]["forecast"]["verdict"] == "не лучше монетки"
    assert s["x"]["forecast"]["verdict"] == "мало данных"
    assert sb.untrusted(s) == ["f"]


@pytest.mark.asyncio
async def test_update_reads_ledger_fetches_once_and_saves(tmp_path):
    ledger = str(tmp_path / "l.jsonl")
    old = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    rec = {
        "ts": old,
        "kind": "ai_forecast",
        "provider": "groq",
        "timeframe": "15m",
        "horizon": 4,
        "atr_mult": 1.0,
        "items": [{"symbol": "A", "price": 100, "atr": 2, "p_up": 0.6}],
        "trades": [],
    }
    (tmp_path / "l.jsonl").write_text(json.dumps(rec) + "\n")
    at_ms = int(datetime.fromisoformat(old).timestamp() * 1000)
    calls = []

    async def fetch(sym, tf, since, limit):
        calls.append((sym, tf, since, limit))
        # свеча до прогноза (open < since) не должна учитываться: в ней было бы −ATR
        return [bar(at_ms - 1, 100, 90), bar(at_ms, 103, 99)]

    board = str(tmp_path / "board.json")
    s = await sb.update(ledger, fetch, board)
    assert calls == [("A", "15m", at_ms, 5)]
    assert s["groq"]["forecast"]["n"] == 1 and s["groq"]["forecast"]["hit_rate"] == 1.0
    await sb.update(ledger, fetch, board)  # уже проверен — биржу не спрашиваем снова
    assert len(calls) == 1
    assert sb.read(board)["providers"]["groq"]["forecast"]["n"] == 1
    DecisionLedger  # импорт — тот же формат строк журнала, что пишет бот
