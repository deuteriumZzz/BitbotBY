"""Phase 6: стратегии объединения, причины отказов, теневые вердикты, объяснение."""

from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from src.core import decision_engine as DE
from src.core import explain as EX
from src.core import signal_combiner as SC
from src.core.contracts import decision_from_rec
from src.core.decision_engine import Proposal, Views
from src.core.decision_ledger import DecisionLedger
from src.core.signal_combiner import SignalCombiner
from src.research import signal_intelligence_report as R
from tests.core.test_reconciliation import make_bot
from tests.core.test_replay import _AI, _SAC, _Fc, _rc
from tests.core.test_signal_combiner import make_rec, make_snapshot

MIN = 0.5


def V(ai=None, sac=None, regime="ranging", min_conf=MIN) -> Views:
    return Views(
        Proposal(*ai) if ai else None, Proposal(*sac) if sac else None, regime, min_conf
    )


# ── Эталон совпадает с боевым _hybrid ─────────────────────────────────────────


def test_constants_are_not_duplicated_out_of_sync() -> None:
    assert DE.REGIME_WEIGHTS == SC._REGIME_WEIGHTS
    assert DE.SAC_SOLO_MIN == SignalCombiner._SAC_SOLO_MIN
    assert DE.DEFAULT_WEIGHTS == (SignalCombiner._W_SAC, SignalCombiner._W_AI)


@pytest.mark.asyncio
async def test_current_strategy_equals_real_hybrid_over_input_grid() -> None:
    ai_opts = [None] + [
        (a, c) for a in ("buy", "sell", "hold") for c in (0.55, 0.7, 0.9)
    ]
    sac_opts = [("hold", 0.0)] + [
        (a, c) for a in ("buy", "sell") for c in (0.4, 0.7, 0.85)
    ]
    regimes = [*DE.REGIME_WEIGHTS, "not_a_regime"]
    checked = 0
    ai_stub, sac_stub = _AI([]), _SAC({})
    combiner = SignalCombiner(
        ai=ai_stub, rc=_rc("hybrid", min_conf=MIN, chronos=False), sac=sac_stub
    )
    for regime in regimes:
        for ai in ai_opts:
            for sac in sac_opts:
                # один комбинатор на всю сетку: подменяем только входы политик
                ai_stub._recs = [] if ai is None else [make_rec("X/USDT", ai[0], ai[1])]
                sac_stub._s = {"X/USDT": {"action": sac[0], "confidence": sac[1]}}
                out = await combiner._hybrid(
                    [make_snapshot("X/USDT")], 1000.0, regime, {"X/USDT": regime}, {}
                )
                verdict = DE.current(V(ai, sac, regime))
                where = (regime, ai, sac)
                if out:
                    assert (out[0]["action"], out[0]["confidence"]) == (
                        verdict.action,
                        verdict.confidence,
                    ), where
                else:
                    assert verdict.action == "hold", where
                checked += 1
    assert checked == len(regimes) * len(ai_opts) * len(sac_opts) == 350


# ── Альтернативные стратегии ──────────────────────────────────────────────────


def test_agreement_uses_conservative_confidence_and_requires_both() -> None:
    assert DE.agreement(V(("buy", 0.8), ("buy", 0.6))).confidence == 0.6
    assert DE.agreement(V(("buy", 0.8), ("sell", 0.7))).action == "hold"
    assert DE.agreement(V(None, ("buy", 0.95))).action == "hold"  # solo запрещён
    assert DE.agreement(V(("buy", 0.8), ("buy", 0.4))).reasons == ("below_min_conf",)


def test_weighted_lets_disagreement_cancel_and_hold_dilute() -> None:
    assert DE.weighted(V(("buy", 0.8), ("sell", 0.7))).action == "hold"
    diluted = DE.weighted(V(("buy", 0.9), ("hold", 0.0)))  # (0.6*0.9)/1.0
    assert (diluted.action, diluted.confidence) == ("buy", 0.54)
    solo = DE.weighted(V(None, ("buy", 0.9)))  # единственная политика — вес нормируется
    assert (solo.action, solo.confidence) == ("buy", 0.9)
    assert DE.weighted(V()).reasons == ("no_inputs",)


def test_contextual_lets_sac_lead_only_in_uptrend() -> None:
    sac_only = V(None, ("buy", 0.85), regime="trending_up")
    assert DE.contextual(sac_only).reasons == ("trend_sac_lead",)
    assert DE.contextual(V(None, ("buy", 0.85), regime="ranging")).action == "hold"
    agree = V(("buy", 0.7), ("buy", 0.7), regime="trending_up")
    assert DE.contextual(agree) == DE.current(agree)
    other = V(("buy", 0.8), ("buy", 0.6), regime="ranging")
    assert DE.contextual(other) == DE.agreement(other)


def test_risk_aware_subtracts_half_of_the_policy_gap() -> None:
    v = DE.risk_aware(V(("buy", 0.9), ("buy", 0.6)))
    # current = 0.4*0.6 + 0.6*0.9 = 0.78; gap 0.3 → 0.78 - 0.15
    assert (v.action, v.confidence) == ("buy", 0.63)
    assert "gap_penalty" in v.reasons
    harsh = DE.risk_aware(V(("buy", 0.95), ("buy", 0.55), min_conf=0.7))
    assert harsh.reasons == ("uncertainty_penalty",)  # 0.79 - 0.2 = 0.59 < 0.7
    assert DE.risk_aware(V(None, ("buy", 0.9))).confidence == 0.9  # solo без вычета


def test_evaluate_all_returns_every_strategy() -> None:
    out = DE.evaluate_all(V(("buy", 0.8), ("buy", 0.7)))
    assert set(out) == {"current", "agreement", "weighted", "contextual", "risk_aware"}


# ── Захват причин отказов и теневых вердиктов ─────────────────────────────────

SNAP = [make_snapshot("BTC/USDT", 100.0, 2.0)]


async def _hybrid_run(ai, sac, chronos=None, min_conf=MIN, regime="ranging"):
    ai_recs = [] if ai is None else [make_rec("BTC/USDT", ai[0], ai[1])]
    combiner = SignalCombiner(
        ai=_AI(ai_recs),
        rc=_rc("hybrid", min_conf=min_conf, chronos=chronos is not None),
        sac=_SAC({"BTC/USDT": {"action": sac[0], "confidence": sac[1]}}),
        forecaster=_Fc({1.5: chronos}) if chronos else None,
    )
    md = {"BTC/USDT": pd.DataFrame({"close": [1.5] * 5})}
    recs = await combiner.combine(
        copy.deepcopy(SNAP),
        1000.0,
        regime=regime,
        regimes={"BTC/USDT": regime},
        market_data=md,
    )
    return combiner, recs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ai,sac,kwargs,reason",
    [
        (("buy", 0.9), ("hold", 0.0), {}, "sac_hold"),
        (("buy", 0.9), ("sell", 0.7), {}, "disagree"),
        (None, ("buy", 0.6), {}, "ai_silent_sac_below_solo"),
        (("buy", 0.55), ("buy", 0.4), {"min_conf": 0.9}, "below_min_conf"),
        (
            ("buy", 0.6),
            ("buy", 0.6),
            {"chronos": "down"},
            "below_min_conf_after_chronos",
        ),
    ],
)
async def test_every_veto_path_records_its_reason(ai, sac, kwargs, reason) -> None:
    combiner, recs = await _hybrid_run(ai, sac, **kwargs)
    assert recs == []
    veto = combiner.last_inputs["vetoes"]["BTC/USDT"]
    assert veto["reason"] == reason and veto["regime"] == "ranging"
    assert veto["sac"]["action"] == sac[0]


@pytest.mark.asyncio
async def test_no_veto_when_there_was_no_proposal_or_when_it_traded() -> None:
    combiner, recs = await _hybrid_run(None, ("hold", 0.0))
    assert recs == [] and "BTC/USDT" not in combiner.last_inputs.get("vetoes", {})
    assert "BTC/USDT" not in combiner.last_inputs.get("shadow", {})
    combiner, recs = await _hybrid_run(("buy", 0.8), ("buy", 0.7))
    assert recs and "vetoes" not in combiner.last_inputs


@pytest.mark.asyncio
async def test_shadow_verdicts_recorded_and_current_matches_real_decision() -> None:
    combiner, recs = await _hybrid_run(("buy", 0.8), ("buy", 0.7))
    entry = combiner.last_inputs["shadow"]["BTC/USDT"]
    assert set(entry["verdicts"]) == {
        "current",
        "agreement",
        "weighted",
        "contextual",
        "risk_aware",
    }
    cur = entry["verdicts"]["current"]
    assert (cur["action"], cur["confidence"]) == (
        recs[0]["action"],
        recs[0]["confidence"],
    )
    json.dumps(entry)  # журнал сериализуем


@pytest.mark.asyncio
async def test_capture_does_not_change_decisions() -> None:
    _, first = await _hybrid_run(("buy", 0.8), ("buy", 0.7))
    _, second = await _hybrid_run(("buy", 0.8), ("buy", 0.7))
    assert first == second


# ── Бот пишет диагностику в журнал ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bot_writes_rejections_and_shadow_to_ledger(tmp_path) -> None:
    bot = make_bot()
    path = tmp_path / "l.jsonl"
    bot.ledger = DecisionLedger(path=str(path), enabled=True)
    bot.ledger.begin_cycle(1)
    snaps = [
        make_snapshot("BTC/USDT", 100.0, 2.0),
        make_snapshot("ETH/USDT", 50.0, 1.0),
    ]
    for s in snaps:
        s["snapshot_id"] = "sid-" + s["symbol"][:3]
    bot.combiner = SignalCombiner(
        ai=_AI([make_rec("BTC/USDT", "buy", 0.9), make_rec("ETH/USDT", "buy", 0.8)]),
        rc=_rc("hybrid", chronos=False),
        sac=_SAC(
            {
                "BTC/USDT": {"action": "hold", "confidence": 0.0},  # вето
                "ETH/USDT": {"action": "buy", "confidence": 0.7},  # сделка
            }
        ),
    )
    bot._cycle.optimize_allocation = lambda r, md: r
    bot._runtime_config = _rc("hybrid", chronos=False)
    regimes = {"BTC/USDT": "ranging", "ETH/USDT": "ranging"}

    recs = await bot._generate_signals(snaps, 1000.0, regimes, {}, None)
    assert [r["symbol"] for r in recs] == ["ETH/USDT"]

    lines = [json.loads(x) for x in path.read_text().splitlines()]
    rej = next(x for x in lines if x["kind"] == "rejections")["items"]
    assert rej[0]["symbol"] == "BTC/USDT" and rej[0]["reason"] == "sac_hold"
    assert rej[0]["snapshot_id"] == "sid-BTC" and rej[0]["ai"]["confidence"] == 0.9
    sh = next(x for x in lines if x["kind"] == "shadow_decisions")["items"]
    by_sym = {i["symbol"]: i for i in sh}
    assert by_sym["ETH/USDT"]["real"] == "buy"
    # BTC: AI предложил, SAC hold — стратегии расходятся (weighted входит)
    assert by_sym["BTC/USDT"]["real"] is None
    assert by_sym["BTC/USDT"]["verdicts"]["current"]["action"] == "hold"
    assert by_sym["BTC/USDT"]["verdicts"]["weighted"]["action"] == "buy"


# ── Объяснение решения ────────────────────────────────────────────────────────


def _decision_dict() -> dict:
    rec = make_rec(strategy="hybrid(rsi+sac+chronos)", confidence=0.7)
    rec.update(
        ai_provider="gemini",
        ai_model="flash",
        chronos_signal="down",
        conf_trail=[["ai", 0.8], ["sac", 0.6], ["hybrid", 0.7]],
        context_flags=json.dumps({"funding": "long_overheated"}),
        _regime="ranging",
    )
    return decision_from_rec(rec, "hybrid", "t-9").to_dict()


def test_explanation_covers_participants_conflicts_and_redundancy() -> None:
    text = EX.explain_decision(_decision_dict())
    for needle in (
        "ЛОНГ",
        "BTC/USDT",
        "ai (gemini/flash)",
        "самооценка LLM, не вероятность",
        "величина действия SAC",
        "ai 0.80 → sac 0.60 → hybrid 0.70",
        "против forecast:chronos",
        "риск повторного учёта",
        "funding: sac_obs + filter",
        "не вероятность прибыли сделки",
    ):
        assert needle in text, needle


def test_explanation_tolerates_sparse_records() -> None:
    text = EX.explain_decision({"decision_id": "x", "status": "NO_TRADE"})
    assert "без сделки" in text and "Участники:" in text


def test_find_decision_and_cli(tmp_path, capsys) -> None:
    d = _decision_dict()
    path = tmp_path / "l.jsonl"
    rows = [
        {"kind": "signals", "decisions": [d]},
        {
            "kind": "execution",
            "decision_id": d["decision_id"],
            "opened": False,
            "reject": "все слоты заняты",
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert EX.main([d["decision_id"], str(path)]) == 0
    out = capsys.readouterr().out
    assert "ЛОНГ" in out and "не открыто — все слоты заняты" in out
    assert EX.main(["nope", str(path)]) == 1
    assert EX.main([]) == 2


# ── Разделы отчёта ────────────────────────────────────────────────────────────


def test_report_summarises_rejections_and_shadow_strategies() -> None:
    records = [
        {
            "kind": "rejections",
            "items": [
                {"reason": "sac_hold", "ai": {"confidence": 0.9}},
                {"reason": "sac_hold", "ai": {"confidence": 0.6}},
                {"reason": "disagree", "ai": {"confidence": 0.8}},
            ],
        },
        {
            "kind": "shadow_decisions",
            "items": [
                {
                    "real": "buy",
                    "verdicts": {
                        "current": {"action": "buy"},
                        "weighted": {"action": "buy"},
                        "agreement": {"action": "hold"},
                    },
                },
                {
                    "real": None,
                    "verdicts": {
                        "current": {"action": "hold"},
                        "weighted": {"action": "buy"},
                        "agreement": {"action": "hold"},
                    },
                },
            ],
        },
    ]
    rep = R.build_report(records)
    assert rep["rejections"] == {
        "n": 3,
        "by_reason": {"sac_hold": 2, "disagree": 1},
        "strong_ai_overruled_by_sac_hold": 1,
    }
    sh = rep["shadow"]
    assert sh["n"] == 2 and sh["current_vs_real_mismatch"] == 0
    assert sh["per_strategy"]["weighted"] == {"trades": 2, "differs_from_current": 1}
    assert sh["per_strategy"]["agreement"] == {"trades": 0, "differs_from_current": 1}
    text = R.format_report(rep)
    assert "ОТКЛОНЁННЫЕ ВОЗМОЖНОСТИ: 3" in text
    assert "СТРАТЕГИИ ОБЪЕДИНЕНИЯ" in text and "по этим данным сказать нельзя" in text


def test_report_flags_a_drifted_reference_strategy() -> None:
    rec = {
        "kind": "shadow_decisions",
        "items": [{"real": "buy", "verdicts": {"current": {"action": "hold"}}}],
    }
    assert R.build_report([rec])["shadow"]["current_vs_real_mismatch"] == 1
