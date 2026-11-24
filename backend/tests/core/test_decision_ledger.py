"""Блок 1: Decision Ledger, provenance AI, трасса confidence, статусы данных."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from bitbot.market.market_context import MarketContext
from bitbot.market.market_scanner import MarketScanner
from bitbot.signals.combiner import SignalCombiner
from bitbot.storage.decision_ledger import DecisionLedger
from tests.ai.test_ai_analyzer import TestAnalyze, _http_error, _make_config
from tests.core.test_signal_combiner import make_rec, make_snapshot

# ── DecisionLedger ────────────────────────────────────────────────────────────


def _lines(path) -> list:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def test_ledger_writes_record_with_trace_id(tmp_path) -> None:
    path = tmp_path / "l.jsonl"
    led = DecisionLedger(path=str(path), enabled=True)
    tid = led.begin_cycle(7)
    led.write("signals", n=2, note="кириллица")
    rec = _lines(path)[0]
    assert rec["kind"] == "signals"
    assert rec["trace_id"] == tid and tid.endswith("-7")
    assert rec["n"] == 2 and rec["note"] == "кириллица"
    assert rec["ts"].endswith("+00:00")


def test_ledger_disabled_writes_nothing(tmp_path) -> None:
    path = tmp_path / "l.jsonl"
    DecisionLedger(path=str(path), enabled=False).write("x")
    assert not path.exists()


def test_ledger_truncates_raw(tmp_path) -> None:
    path = tmp_path / "l.jsonl"
    led = DecisionLedger(path=str(path), enabled=True)
    led.write("ai_call", raw="a" * 20_000)
    rec = _lines(path)[0]
    assert len(rec["raw"]) == 8_000 and rec["raw_truncated"] is True


def test_ledger_rotates_by_size(tmp_path) -> None:
    path = tmp_path / "l.jsonl"
    led = DecisionLedger(path=str(path), max_bytes=200, enabled=True)
    for i in range(10):
        led.write("x", i=i, pad="p" * 50)
    assert (tmp_path / "l.jsonl.1").exists()
    assert path.exists()


def test_ledger_never_raises_on_bad_path(tmp_path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    led = DecisionLedger(path=str(blocker / "sub" / "l.jsonl"), enabled=True)
    led.write("x", a=1)  # путь «внутри файла» — не должно бросить


# ── AI provenance (H-2) ───────────────────────────────────────────────────────

_REPLY = json.dumps(
    [{"symbol": "BTC/USDT", "action": "buy", "strategy": "s", "confidence": 0.9}]
)


async def _analyze(chain, responses, tmp_path):
    ai = TestAnalyze()._make_ai(chain)
    ai._ledger = DecisionLedger(path=str(tmp_path / "l.jsonl"), enabled=True)

    async def call(provider, model, prompt):
        r = responses[provider]
        if isinstance(r, Exception):
            raise r
        return r

    ai._call = AsyncMock(side_effect=call)
    with patch("bitbot.ai.analyzer.Config", _make_config(MIN_SIGNAL_CONFIDENCE=0.65)):
        recs = await ai.analyze([{"symbol": "BTC/USDT"}], 10_000)
    return recs, _lines(tmp_path / "l.jsonl")


@pytest.mark.asyncio
async def test_recs_carry_provider_that_actually_answered(tmp_path) -> None:
    chain = [("groq", "m1"), ("gemini", "m2")]
    recs, ledger = await _analyze(
        chain, {"groq": _http_error(429, "limit"), "gemini": _REPLY}, tmp_path
    )
    assert recs[0]["ai_provider"] == "gemini" and recs[0]["ai_model"] == "m2"
    call = ledger[0]
    assert call["kind"] == "ai_call" and call["status"] == "ok"
    assert call["provider"] == "gemini" and call["raw"] == _REPLY
    assert call["attempts"] == [
        {"provider": "groq", "model": "m1", "status": "rate_limit"}
    ]
    assert len(call["prompt_sha256"]) == 16 and call["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_exhausted_chain_is_logged(tmp_path) -> None:
    chain = [("groq", "m1"), ("gemini", "m2")]
    recs, ledger = await _analyze(
        chain,
        {"groq": _http_error(429, "limit"), "gemini": _http_error(402, "money")},
        tmp_path,
    )
    assert recs == []
    assert ledger[0]["status"] == "exhausted"
    assert [a["status"] for a in ledger[0]["attempts"]] == ["rate_limit", "no_balance"]


@pytest.mark.asyncio
async def test_analyzer_works_without_ledger() -> None:
    ai = TestAnalyze()._make_ai([("groq", "m1")])
    ai._call = AsyncMock(return_value=_REPLY)
    with patch("bitbot.ai.analyzer.Config", _make_config(MIN_SIGNAL_CONFIDENCE=0.65)):
        recs = await ai.analyze([{"symbol": "BTC/USDT"}], 10_000)
    assert recs[0]["ai_provider"] == "groq"


# ── Трасса confidence (C-1/C-2, только измерение) ─────────────────────────────


def _rc(min_conf: float) -> MagicMock:
    rc = MagicMock()
    rc.get_mode.return_value = "ai"
    rc.get_signal_confidence.return_value = min_conf
    rc.get_market_type.return_value = "linear"
    rc.get_market_profile.return_value = ""
    return rc


@pytest.mark.asyncio
async def test_conf_trail_records_each_stage_and_merge_boost() -> None:
    ai = MagicMock()
    ai.analyze = AsyncMock(return_value=[make_rec(confidence=0.8)])
    ctx = {"BTC/USDT": {"funding_signal": "short_overheated", "funding_rate": -0.002}}
    with patch("bitbot.signals.combiner.SACSignal"):
        result = await SignalCombiner(ai=ai, rc=_rc(0.5)).combine(
            [make_snapshot()], balance=10_000, market_context=ctx
        )
    rec = next(r for r in result if r["strategy"] != "context_signal")
    stages = [s for s, _ in rec["conf_trail"]]
    assert stages == ["source:ai", "context_filter", "context_merge"]
    assert rec["context_boost"] is True
    # confidence считается так же, как раньше: trail только наблюдает
    assert rec["confidence"] == rec["conf_trail"][-1][1]


@pytest.mark.asyncio
async def test_context_penalty_is_recorded_raw_and_capped() -> None:
    ai = MagicMock()
    ai.analyze = AsyncMock(return_value=[make_rec(confidence=0.95)])
    ctx = {
        "BTC/USDT": {
            "funding_signal": "long_overheated",
            "oi_signal": "oi_bearish",
            "liquidation_pressure": "long_liquidation",
            "basis_signal": "greed_premium",
        }
    }
    with patch("bitbot.signals.combiner.SACSignal"):
        result = await SignalCombiner(ai=ai, rc=_rc(0.1)).combine(
            [make_snapshot()], balance=10_000, market_context=ctx
        )
    rec = next(r for r in result if r["strategy"] != "context_signal")
    assert rec["ctx_penalty_raw"] > rec["ctx_penalty"] == 0.25


# ── H-8: статусы данных ───────────────────────────────────────────────────────


def test_failed_sources_expire_after_ttl() -> None:
    mc = MarketContext()
    mc._mark_failed("funding")
    assert mc._failed_sources() == ["funding"]
    with patch("bitbot.market.market_context.time.monotonic", return_value=1e9):
        assert mc._failed_sources() == []


@pytest.mark.asyncio
async def test_full_context_failure_is_flagged() -> None:
    mc = MarketContext()
    mc.get_context = AsyncMock(side_effect=RuntimeError("boom"))
    out = await mc.get_context_for_symbols(["BTC/USDT"], {"BTC/USDT": 1.0})
    assert out["BTC/USDT"]["data_status"] == {"failed_sources": ["all"]}
    assert out["BTC/USDT"]["funding_signal"] == "neutral"


# ── H-13: метка свечи ─────────────────────────────────────────────────────────


def test_candle_time_from_datetime_and_ms_index() -> None:
    df = pd.DataFrame({"close": [1.0]}, index=pd.to_datetime(["2026-11-23 10:15"]))
    assert MarketScanner._candle_time(df).startswith("2026-11-23T10:15")
    df_ms = pd.DataFrame({"close": [1.0]}, index=[1_700_000_000_000])
    assert MarketScanner._candle_time(df_ms).startswith("2023-11-14")


def test_candle_time_none_for_non_time_index() -> None:
    assert MarketScanner._candle_time(pd.DataFrame({"close": [1.0, 2.0]})) is None
