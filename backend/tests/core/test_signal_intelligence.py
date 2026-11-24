"""Phase 3: группы независимости, избыточные каналы, конфликты, отчёт по журналу."""

from __future__ import annotations

import json

import pytest

from bitbot.domain import contracts as C
from bitbot.domain.signals import intelligence as SI
from bitbot.domain.strategies import STRATEGY_REGISTRY
from bitbot.research import signal_intelligence_report as R
from tests.core.test_signal_combiner import make_rec

# ── Семейства и группы ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,family",
    [
        ("rsi_momentum", "momentum"),
        ("stoch_rsi", "momentum"),
        ("ema_crossover", "trend"),
        ("vw_macd", "trend"),
        ("mean_reversion", "mean_reversion"),
        ("vwap_reversion", "mean_reversion"),
        ("volume_spike", "volume"),
        ("mfi_trend", "volume"),
        ("support_resistance", "structure"),
        ("momentum_burst", "volatility"),
        ("sentiment", "sentiment"),
        ("shooting_star", "pattern"),
        ("totally_unknown", "other"),
    ],
)
def test_strategy_family(name: str, family: str) -> None:
    assert SI.strategy_family(name) == family


def test_every_registered_strategy_has_a_family() -> None:
    other = [n for n in STRATEGY_REGISTRY if SI.strategy_family(n) == "other"]
    assert other == [], f"добавьте семейство для: {other}"


def test_context_signal_group_from_reason_text() -> None:
    assert (
        SI.context_signal_group("Funding overheated 0.1% → contrarian sell")
        == "funding"
    )
    assert SI.context_signal_group("Orderbook bid dominant") == "order_flow"
    assert SI.context_signal_group("Fear&Greed=8 (extreme panic)") == "sentiment"
    assert SI.context_signal_group("something else") == "context"


# ── analyze_signal ────────────────────────────────────────────────────────────

AI = ("ai", "buy", "ai", "")
SAC = ("sac", "buy", "sac", "")


def _flags(**kw) -> dict:
    return {"context_flags": json.dumps(kw)}


def test_groups_channels_and_redundancy_for_hybrid_with_context() -> None:
    rec = _flags(funding="long_overheated", fng="greed", profile="bluechip")
    a = SI.analyze_signal(rec, [AI, SAC])
    assert set(a.independent_groups) == {"llm", "rl", "funding", "sentiment"}
    assert a.channels["funding"] == ("sac_obs", "filter")
    # профиль — тег, а не фактор
    assert "profile" not in json.dumps(a.channels)
    assert set(a.redundant_groups) == {"funding", "sentiment"}


def test_filter_only_is_not_redundant() -> None:
    a = SI.analyze_signal(_flags(funding="x"), [AI])
    assert a.redundant_groups == () and a.channels["funding"] == ("filter",)


def test_generated_context_signal_plus_filter_is_redundant() -> None:
    ctx = ("context", "sell", "context", "Funding overheated 0.1% → contrarian sell")
    a = SI.analyze_signal(_flags(funding="long_overheated"), [ctx])
    assert a.redundant_groups == ("funding",)
    assert a.channels["funding"] == ("generated", "filter")


def test_merge_boost_is_recorded_but_not_counted_as_redundant() -> None:
    a = SI.analyze_signal({"context_boost": True}, [AI])
    assert a.channels == {"context": ("merge",)} and a.redundant_groups == ()


def test_local_output_maps_to_strategy_family() -> None:
    a = SI.analyze_signal({}, [("local", "buy", "local:rsi_momentum", "")])
    assert a.independent_groups == ("momentum",)


def test_direction_conflict_is_reported_as_object() -> None:
    forecast = ("forecast", "sell", "forecast:chronos", "")
    a = SI.analyze_signal({}, [AI, SAC, forecast])
    assert a.conflicts == (
        {"a": "ai", "b": "forecast:chronos", "kind": "direction"},
        {"a": "sac", "b": "forecast:chronos", "kind": "direction"},
    )


def test_bad_context_flags_do_not_crash() -> None:
    for bad in ("{not json", "[1,2]", 5, None):
        a = SI.analyze_signal({"context_flags": bad}, [AI])
        assert a.independent_groups == ("llm",)


# ── Интеграция с контрактом ───────────────────────────────────────────────────


def test_decision_carries_analysis_and_confidence_is_untouched() -> None:
    rec = make_rec(strategy="hybrid(rsi+sac+chronos)", confidence=0.7)
    rec.update(
        ai_provider="gemini",
        chronos_signal="down",
        conf_trail=[["ai", 0.8], ["sac", 0.6], ["hybrid", 0.7]],
        context_flags=json.dumps({"funding": "long_overheated"}),
    )
    d = C.decision_from_rec(rec, "hybrid", "t")
    assert d.confidence == 0.7
    assert {"llm", "rl", "forecasting", "funding"} <= set(d.independent_groups)
    assert "funding" in d.redundant_groups
    assert len(d.conflicts) == 2  # chronos «вниз» против AI и SAC
    json.dumps(d.to_dict())


# ── Отчёт по журналу ──────────────────────────────────────────────────────────


def _dec(trail, **kw) -> dict:
    base = {
        "status": "LONG",
        "mode": "hybrid",
        "confidence_kind": "combined_heuristic",
        "outputs": [{"kind": "ai"}, {"kind": "sac"}],
        "conf_trail": trail,
        "independent_groups": ["llm", "rl"],
        "redundant_groups": [],
        "conflicts": [],
        "reason_codes": [],
    }
    base.update(kw)
    return base


RECORDS = [
    {
        "kind": "signals",
        "decisions": [
            _dec(
                [["ai", 0.8], ["hybrid", 0.7], ["context_filter", 0.55]],
                redundant_groups=["funding"],
                reason_codes=["ctx_penalty_capped"],
            ),
            _dec(
                [["ai", 0.8], ["hybrid", 0.7], ["context_filter", 0.75]],
                independent_groups=["llm"],
                conflicts=[{"a": "ai", "b": "forecast:chronos", "kind": "direction"}],
                reason_codes=["context_boost"],
            ),
        ],
    },
    {
        "kind": "ai_call",
        "status": "ok",
        "provider": "gemini",
        "latency_ms": 100,
        "attempts": [{"provider": "groq", "status": "rate_limit"}],
    },
    {
        "kind": "ai_call",
        "status": "ok",
        "provider": "groq",
        "latency_ms": 300,
        "attempts": [],
    },
    {"kind": "ai_call", "status": "exhausted", "attempts": [{"status": "no_balance"}]},
    {"kind": "data_quality", "counts": {"VALID": 8, "STALE": 2}},
    {"kind": "execution", "opened": True},
    {"kind": "execution", "opened": False, "reject": "все слоты заняты"},
]


def test_report_aggregates_stage_deltas_redundancy_and_conflicts() -> None:
    s = R.build_report(RECORDS)["signals"]
    assert s["n_decisions"] == 2 and s["by_status"] == {"LONG": 2}
    fil = s["stage_deltas"]["hybrid→context_filter"]
    assert fil["n"] == 2 and fil["mean"] == pytest.approx(-0.05)
    assert fil["share_up"] == 0.5 and fil["share_down"] == 0.5
    assert s["stage_deltas"]["ai→hybrid"]["mean"] == pytest.approx(-0.1)
    assert s["with_redundant_channels"] == 1
    assert s["redundant_by_group"] == {"funding": 1}
    assert s["with_capped_penalty"] == 1 and s["with_context_boost"] == 1
    assert s["conflicts"] == {"ai vs forecast:chronos": 1}
    assert s["mean_independent_groups"] == 1.5 and s["single_group_share"] == 0.5
    assert s["by_source"] == {"ai+sac": 2}


def test_report_ai_dataquality_execution_sections() -> None:
    rep = R.build_report(RECORDS)
    ai = rep["ai"]
    assert (ai["n_calls"], ai["ok"], ai["exhausted"]) == (3, 2, 1)
    assert ai["answered_by"] == {"gemini": 1, "groq": 1}
    assert ai["fallback_share"] == 0.5
    assert ai["failed_attempts_by_status"] == {"rate_limit": 1, "no_balance": 1}
    assert ai["latency_ms_mean"] == 200.0
    assert rep["data_quality"] == {
        "cycles": 1,
        "snapshots_by_status": {"VALID": 8, "STALE": 2},
    }
    assert rep["execution"] == {
        "n": 2,
        "opened": 1,
        "top_rejects": {"все слоты заняты": 1},
    }


def test_report_formats_and_handles_empty_input() -> None:
    text = R.format_report(R.build_report(RECORDS))
    assert "hybrid→context_filter" in text and "gemini" in text
    assert "не следует ни наличие, ни отсутствие" in text
    empty = R.format_report(R.build_report([]))
    assert "РЕШЕНИЯ: 0" in empty


def test_load_records_reads_rotated_file_and_skips_garbage(tmp_path) -> None:
    path = tmp_path / "l.jsonl"
    (tmp_path / "l.jsonl.1").write_text('{"kind":"old"}\n', encoding="utf-8")
    path.write_text('{"kind":"new"}\nnot json\n\n{"kind":"newer"}\n', encoding="utf-8")
    assert [r["kind"] for r in R.load_records(str(path))] == ["old", "new", "newer"]


def test_main_exit_codes(tmp_path, capsys) -> None:
    assert R.main([str(tmp_path / "missing.jsonl")]) == 1
    path = tmp_path / "l.jsonl"
    path.write_text(json.dumps(RECORDS[1]) + "\n", encoding="utf-8")
    assert R.main([str(path)]) == 0
    assert "AI-ВЫЗОВЫ" in capsys.readouterr().out
