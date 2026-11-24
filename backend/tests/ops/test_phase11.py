"""Phase 11: аудит решения, дневная сводка, аварийное управление, стадии выкатки."""

from datetime import datetime, timedelta, timezone

import pytest

from bitbot.ops.daily_digest import build_digest, format_digest, in_window
from bitbot.ops.decision_audit import audit_decision, format_audit
from bitbot.ops.emergency import LEVELS, EmergencyControl
from bitbot.ops.rollout import Rollout, RolloutError, capabilities

NOW = datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc)


def ts(h_ago):
    return (NOW - timedelta(hours=h_ago)).isoformat()


# ── Decision Audit ────────────────────────────────────────────────────────────
def ledger(opened=True):
    return [
        {"kind": "data_quality", "trace_id": "T1", "counts": {"VALID": 3}},
        {"kind": "ai_call", "trace_id": "T1", "status": "ok"},
        {"kind": "data_quality", "trace_id": "T2"},
        {
            "kind": "signals",
            "trace_id": "T1",
            "decisions": [{"decision_id": "d1", "symbol": "BTC/USDT"}],
        },
        {
            "kind": "execution",
            "trace_id": "T1",
            "decision_id": "d1",
            "opened": opened,
            "reject": None if opened else "max_pos",
        },
        {"kind": "execution", "trace_id": "T1", "decision_id": "other", "opened": True},
    ]


def test_audit_chain_and_gaps():
    a = audit_decision(ledger(), "d1")
    assert a["trace_id"] == "T1" and a["symbol"] == "BTC/USDT"
    assert a["stages"] == ["data_quality", "ai_call", "signals", "execution"]
    assert a["gaps"] == ["outcome"] and not a["complete"]  # открыта, исхода нет
    assert audit_decision(ledger(), "d1", outcome={"pnl": 5.0})["complete"]


def test_audit_not_opened_needs_no_outcome():
    a = audit_decision(ledger(opened=False), "d1")
    assert a["complete"] and "max_pos" in format_audit(a)


def test_audit_ignores_other_cycles_and_unknown_id():
    a = audit_decision(ledger(), "d1")
    assert a["chain"]["data_quality"]["trace_id"] == "T1"
    assert audit_decision(ledger(), "nope") is None


def test_audit_reports_missing_required_stage():
    recs = [r for r in ledger() if r["kind"] != "data_quality"]
    assert "data_quality" in audit_decision(recs, "d1")["gaps"]


# ── Daily Digest ──────────────────────────────────────────────────────────────
def test_window_filters_by_time_and_handles_bad_ts():
    recs = [
        {"ts": ts(1), "kind": "x"},
        {"ts": ts(30), "kind": "x"},
        {"ts": "junk", "kind": "x"},
        {"ts": NOW.replace(tzinfo=None).isoformat(), "kind": "x"},
    ]  # без таймзоны = UTC
    assert len(in_window(recs, 24, NOW)) == 2


def test_digest_counts_and_alerts():
    recs = [
        {
            "ts": ts(1),
            "kind": "ai_call",
            "trace_id": "a",
            "status": "exhausted",
            "attempts": [{"provider": "p", "status": "no_credit"}],
        },
        {
            "ts": ts(2),
            "kind": "ai_call",
            "trace_id": "b",
            "status": "exhausted",
            "attempts": [],
        },
        {
            "ts": ts(2),
            "kind": "ai_call",
            "trace_id": "c",
            "status": "ok",
            "provider": "q",
            "latency_ms": 100,
            "attempts": [],
        },
        {
            "ts": ts(3),
            "kind": "data_quality",
            "trace_id": "a",
            "counts": {"VALID": 2, "STALE": 2},
        },
        {"ts": ts(3), "kind": "reconcile", "trace_id": "a"},
        {
            "ts": ts(40),
            "kind": "ai_call",
            "status": "ok",
            "provider": "old",
            "attempts": [],
        },  # вне окна
    ]
    d = build_digest(recs, 24, NOW)
    assert (
        d["ai"]["n_calls"] == 3
        and d["ai"]["exhausted"] == 2
        and "old" not in d["providers"]
    )
    assert d["cycles"] == 3 and d["reconcile_events"] == 1
    text = " ".join(d["alerts"])
    assert "AI" in text and "Данные" in text and "Биржа" in text
    assert "Тревоги" in format_digest(d)


def test_digest_empty_window_is_an_alert_not_a_crash():
    d = build_digest([], 24, NOW)
    assert d["records"] == 0 and any("пуст" in a for a in d["alerts"])


def test_digest_healthy_has_no_alerts():
    recs = [
        {
            "ts": ts(1),
            "kind": "ai_call",
            "trace_id": "a",
            "status": "ok",
            "provider": "p",
            "latency_ms": 50,
            "attempts": [],
        },
        {"ts": ts(1), "kind": "data_quality", "trace_id": "a", "counts": {"VALID": 5}},
    ]
    assert build_digest(recs, 24, NOW)["alerts"] == []


# ── Emergency Control ─────────────────────────────────────────────────────────
def make_control(tmp_path, fail=None):
    calls = []

    def act(name):
        def f():
            calls.append(name)
            if name == fail:
                raise RuntimeError("boom")

        return f

    return (
        EmergencyControl(
            str(tmp_path / "em.json"), {n: act(n) for n in LEVELS.values()}
        ),
        calls,
    )


def test_escalate_runs_new_levels_in_order_only_once(tmp_path):
    ec, calls = make_control(tmp_path)
    assert ec.entries_allowed() and ec.level == 0
    r = ec.escalate(2, "owner", "test")
    assert (
        calls == ["block_entries", "cancel_orders"]
        and r["level"] == 2
        and not ec.entries_allowed()
    )
    assert (
        ec.escalate(2, "owner", "again")["noop"]
        and ec.escalate(1, "o", "lower")["noop"]
    )
    ec.escalate(4, "owner", "worse")
    assert calls[2:] == [
        "reduce_positions",
        "full_stop",
    ]  # прежние уровни не повторяются


def test_failed_action_does_not_stop_others_and_level_is_persisted(tmp_path):
    ec, calls = make_control(tmp_path, fail="cancel_orders")
    r = ec.escalate(3, "owner", "x")
    assert "cancel_orders" in r["errors"] and calls == [
        "block_entries",
        "cancel_orders",
        "reduce_positions",
    ]
    assert EmergencyControl(ec.path, ec.actions).level == 3  # пережило «перезапуск»


def test_reset_is_explicit_and_validation(tmp_path):
    ec, _ = make_control(tmp_path)
    ec.escalate(1, "o", "r")
    with pytest.raises(ValueError):
        ec.reset("", "x")
    with pytest.raises(ValueError):
        ec.escalate(5, "o", "r")
    with pytest.raises(ValueError):
        ec.escalate(2, "", "r")
    ec.reset("owner", "всё в порядке")
    assert ec.entries_allowed() and ec.state()["by"] == "owner"


def test_missing_actions_rejected(tmp_path):
    with pytest.raises(ValueError):
        EmergencyControl(str(tmp_path / "e.json"), {"block_entries": lambda: None})


def test_corrupt_state_file_means_level_zero(tmp_path):
    ec, _ = make_control(tmp_path)
    (tmp_path / "em.json").write_text("{not json")
    assert ec.level == 0


# ── Rollout ───────────────────────────────────────────────────────────────────
def test_default_stage_is_shadow_without_real_orders(tmp_path):
    r = Rollout(str(tmp_path / "r.json"))
    assert r.stage == "shadow" and not capabilities(r.stage)["real_orders"]


def test_promotion_is_one_step_with_evidence(tmp_path):
    r = Rollout(str(tmp_path / "r.json"))
    with pytest.raises(RolloutError):
        r.promote("owner", {})
    with pytest.raises(RolloutError):
        r.promote("", {"x": 1})
    assert r.promote("owner", {"paper_days": 14}) == "paper"
    assert r.promote("owner", {"ok": True}) == "canary"
    caps = capabilities("canary")
    assert caps["real_orders"] and 0 < caps["capital_fraction"] < 1
    r.promote("owner", {"ok": True})
    with pytest.raises(RolloutError):
        r.promote("owner", {"ok": True})
    assert [h["to"] for h in Rollout(r.path).history()] == [
        "paper",
        "canary",
        "production",
    ]


def test_demote_to_earlier_stage_only_with_reason(tmp_path):
    r = Rollout(str(tmp_path / "r.json"))
    r.promote("o", {"e": 1})
    r.promote("o", {"e": 1})
    with pytest.raises(RolloutError):
        r.demote("production", "o", "r")  # не вверх
    with pytest.raises(RolloutError):
        r.demote("canary", "o", "r")  # не на ту же стадию
    with pytest.raises(RolloutError):
        r.demote("shadow", "o", "")
    assert r.demote("shadow", "o", "дрейф") == "shadow"
    assert not capabilities(r.stage)["real_orders"]


def test_unknown_stage_and_corrupt_file(tmp_path):
    with pytest.raises(RolloutError):
        capabilities("live")
    p = tmp_path / "r.json"
    p.write_text('{"stage": "production-ish"}')
    assert Rollout(str(p)).stage == "shadow"  # мусор → самая безопасная стадия
