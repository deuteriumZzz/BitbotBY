"""Phase 1: контракты решений, таблица режимов, запись решений в журнал."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bitbot.domain import contracts as C
from bitbot.domain.signals.combiner import SignalCombiner
from bitbot.domain.trading_modes import (
    REASON_AI_UNAVAILABLE,
    REASON_SAC_MISSING,
    REASON_UNKNOWN_MODE,
    all_readiness,
    readiness,
    requires_sac,
)
from bitbot.storage.decision_ledger import DecisionLedger
from tests.core.test_reconciliation import make_bot
from tests.core.test_signal_combiner import make_rec, make_snapshot

# ── Контракты ─────────────────────────────────────────────────────────────────


def _ai_rec(**kw):
    rec = make_rec(confidence=0.8, strategy="rsi")
    rec.update(ai_provider="gemini", ai_model="flash", _regime="ranging")
    rec.update(kw)
    return rec


def test_ai_recommendation_becomes_long_decision_with_provenance() -> None:
    rec = _ai_rec(conf_trail=[["source:ai", 0.8]])
    d = C.decision_from_rec(rec, "ai", "t-1")
    assert d.status == C.LONG and d.mode == "ai" and d.regime == "ranging"
    out = d.outputs[0]
    assert (out.kind, out.provider, out.model_version) == ("ai", "gemini", "flash")
    assert out.confidence_kind == C.CK_LLM and out.confidence == 0.8
    assert d.confidence_kind == C.CK_LLM


def test_sell_is_short_and_unknown_action_is_no_trade() -> None:
    assert C.decision_from_rec(_ai_rec(action="sell"), "ai").status == C.SHORT
    assert C.decision_from_rec(_ai_rec(action="hold"), "ai").status == C.NO_TRADE


def test_decision_id_is_deterministic_and_scoped_by_trace() -> None:
    rec = _ai_rec()
    a = C.decision_from_rec(rec, "ai", "t-1").decision_id
    assert a == C.decision_from_rec(rec, "ai", "t-1").decision_id
    assert a != C.decision_from_rec(rec, "ai", "t-2").decision_id


def test_sac_and_context_and_local_outputs_have_distinct_confidence_kinds() -> None:
    sac = C.decision_from_rec(
        make_rec(strategy="sac", confidence=0.7, reasoning="SAC conf 70%"), "dqn"
    )
    ctx = C.decision_from_rec(
        make_rec(strategy="context_signal", confidence=0.68, reasoning="F&G"), "ai"
    )
    loc = C.decision_from_rec(
        make_rec(
            strategy="rsi_momentum",
            reasoning="Локальная стратегия «rsi_momentum»: сигнал",
        ),
        "local",
    )
    assert sac.outputs[0].confidence_kind == C.CK_SAC
    assert ctx.outputs[0].confidence_kind == C.CK_HEURISTIC
    assert loc.outputs[0].model_id == "local:rsi_momentum"
    assert loc.outputs[0].confidence_kind == C.CK_STRATEGY


def test_reason_codes_expose_context_effects() -> None:
    rec = _ai_rec(
        context_boost=True,
        ctx_penalty=0.25,
        ctx_penalty_raw=0.4,
        context_flags=json.dumps({"funding": "long_overheated"}),
    )
    codes = C.decision_from_rec(rec, "ai").reason_codes
    assert "context_boost" in codes and "ctx_penalty_capped" in codes
    assert "ctx:funding=long_overheated" in codes


def test_adapter_does_not_mutate_recommendation_and_is_json_serializable() -> None:
    rec = _ai_rec(conf_trail=[["source:ai", 0.8]])
    snapshot = json.dumps(rec, sort_keys=True)
    d = C.decision_from_rec(rec, "ai", "t")
    assert json.dumps(rec, sort_keys=True) == snapshot
    json.dumps(d.to_dict())


@pytest.mark.asyncio
async def test_real_hybrid_recommendation_maps_to_ai_and_sac_outputs() -> None:
    ai = MagicMock()
    ai.analyze = AsyncMock(return_value=[_ai_rec(confidence=0.75)])
    rc = MagicMock()
    rc.get_mode.return_value = "hybrid"
    rc.get_signal_confidence.return_value = 0.5
    rc.get_chronos_enabled.return_value = False
    with patch("bitbot.domain.signals.combiner.SACSignal") as sac_cls:
        sac = sac_cls.return_value
        sac.loaded = True
        sac.get_signal.return_value = {"action": "buy", "confidence": 0.75}
        recs = await SignalCombiner(ai=ai, rc=rc).combine(
            [make_snapshot()], balance=10_000
        )
    d = C.decision_from_rec(recs[0], "hybrid", "t")
    assert [o.kind for o in d.outputs] == ["ai", "sac"]
    assert d.outputs[0].provider == "gemini"
    assert d.confidence_kind == C.CK_COMBINED
    assert [s for s, _ in d.conf_trail][:3] == ["ai", "sac", "hybrid"]


# ── Режимы ────────────────────────────────────────────────────────────────────


def test_readiness_matrix() -> None:
    assert readiness("local", sac_loaded=False, ai_available=False) == (True, "")
    assert readiness("ai", sac_loaded=False) == (True, "")
    assert readiness("ai", sac_loaded=False, ai_available=False) == (
        False,
        REASON_AI_UNAVAILABLE,
    )
    assert readiness("dqn", sac_loaded=False) == (False, REASON_SAC_MISSING)
    assert readiness("hybrid", sac_loaded=False) == (False, REASON_SAC_MISSING)
    assert readiness("hybrid", sac_loaded=True) == (True, "")
    assert readiness("nope", sac_loaded=True) == (False, REASON_UNKNOWN_MODE)


def test_requires_sac_only_for_sac_modes() -> None:
    assert {m for m in ("local", "ai", "dqn", "hybrid") if requires_sac(m)} == {
        "dqn",
        "hybrid",
    }
    assert requires_sac("unknown") is False


def test_all_readiness_covers_every_mode() -> None:
    state = all_readiness(sac_loaded=False)
    assert set(state) == {"local", "ai", "dqn", "hybrid"}
    assert state["hybrid"] == {"ready": False, "reason": REASON_SAC_MISSING}


# ── Подключение к боту и журналу ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_signals_writes_decisions_and_links_execution_id(
    tmp_path,
) -> None:
    bot = make_bot()
    bot.ledger = DecisionLedger(path=str(tmp_path / "l.jsonl"), enabled=True)
    trace = bot.ledger.begin_cycle(3)
    rec = _ai_rec(conf_trail=[["source:ai", 0.8]])
    bot.combiner.combine = AsyncMock(return_value=[rec])
    bot.combiner.decider_unavailable = MagicMock(return_value=False)
    bot._cycle.optimize_allocation = MagicMock(side_effect=lambda r, md: r)
    bot._runtime_config = MagicMock()
    bot._runtime_config.get_mode.return_value = "ai"

    out = await bot._generate_signals([make_snapshot()], 10_000, {}, {}, None)

    assert out[0]["decision_id"]
    line = json.loads((tmp_path / "l.jsonl").read_text().splitlines()[0])
    assert line["kind"] == "signals" and line["trace_id"] == trace
    decision = line["decisions"][0]
    assert decision["decision_id"] == out[0]["decision_id"]
    assert decision["outputs"][0]["provider"] == "gemini"


def test_bot_state_exposes_mode_readiness() -> None:
    bot = make_bot()
    bot.combiner.sac.loaded = False
    bot.ai.enabled = True
    bot._runtime_config = MagicMock()
    bot._runtime_config.get_disabled_strategies.return_value = set()
    state = {"mode_readiness": bot._mode_readiness()}
    assert state["mode_readiness"]["hybrid"] == {
        "ready": False,
        "reason": REASON_SAC_MISSING,
    }
    assert state["mode_readiness"]["ai"]["ready"] is True
