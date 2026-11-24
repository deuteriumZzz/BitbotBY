"""Теневой Risk Gate: сборка входов и вердикт по фактическому входу."""

import numpy as np
import pytest

from src.risk.risk_gate import GateLimits
from src.risk.shadow import DEFAULT_BETA, DEFAULT_CVAR, asset_risk, shadow_verdict

rng = np.random.default_rng(1)
BTC = rng.normal(0, 0.01, 50)
SERIES = {"BTC/USDT": BTC, "ETH/USDT": 1.5 * BTC + rng.normal(0, 0.002, 50)}


def returns(sym):
    return SERIES.get(sym)


def call(**kw):
    base = dict(
        symbol="ETH/USDT",
        side="buy",
        qty=1.0,
        entry=1000.0,
        positions_before={},
        equity=10_000.0,
        peak_equity=10_000.0,
        day_start_equity=10_000.0,
        consecutive_losses=0,
        returns=returns,
        max_corr=lambda s, o: 0.0,
        limits=GateLimits(),
    )
    base.update(kw)
    return shadow_verdict(**base)


def test_asset_risk_computed_and_default():
    r = asset_risk("ETH/USDT", returns)
    assert (
        r["source"] == "computed"
        and r["beta"] == pytest.approx(1.5, abs=0.15)
        and r["cvar"] > 0
    )
    assert asset_risk("BTC/USDT", returns)["beta"] == 1.0
    d = asset_risk("NEW/USDT", returns)
    assert (d["beta"], d["cvar"], d["source"]) == (
        DEFAULT_BETA,
        DEFAULT_CVAR,
        "default",
    )
    partial = asset_risk(
        "ETH/USDT", lambda s: SERIES["ETH/USDT"] if s == "ETH/USDT" else None
    )
    assert partial["source"] == "computed_cvar_default_beta"


def test_small_entry_would_be_allowed():
    out = call()
    assert out["verdict"]["action"] == "allow" and not out["would_block"]
    assert (
        out["proposal"]["notional"] == 1000.0 and out["inputs"]["source"] == "computed"
    )


def test_oversized_entry_would_be_reduced_and_flagged():
    out = call(qty=5.0)  # 5000 при капитале 10 000 → лимит на символ 2500
    assert out["verdict"]["action"] == "reduce" and out["would_block"]
    assert out["verdict"]["approved_notional"] == pytest.approx(2500.0)
    assert "symbol_exposure" in out["verdict"]["reasons"]


def test_short_side_sign_and_existing_positions_counted():
    held = {"SOL/USDT": {"side": "buy", "qty": 20.0, "entry": 100.0}}
    out = call(side="sell", positions_before=held)
    assert out["proposal"]["notional"] == -1000.0 and out["inputs"]["n_open"] == 1


def test_breaker_and_correlation_denials():
    assert call(consecutive_losses=3)["verdict"]["reasons"][0].startswith(
        "circuit_breaker"
    )
    held = {"SOL/USDT": {"side": "buy", "qty": 1.0, "entry": 100.0}}
    out = call(positions_before=held, max_corr=lambda s, o: 0.95)
    assert out["verdict"]["action"] == "deny" and out["verdict"]["reasons"] == [
        "correlation"
    ]
    assert (
        call(day_start_equity=10_600.0, equity=10_000.0)["verdict"]["action"] == "deny"
    )  # дневной убыток ≥ 5%


def _bot(written, returns_fn=returns):
    from types import SimpleNamespace

    return SimpleNamespace(
        _live_equity_cache=0.0,
        _peak_balance=10_000.0,
        risk_manager=SimpleNamespace(_day_start_balance=10_000.0),
        _position_monitor=SimpleNamespace(_consecutive_losses=0),
        corr_filter=SimpleNamespace(
            returns=returns_fn, max_correlation=lambda s, o: 0.0
        ),
        _ledger_write=lambda kind, **f: written.append((kind, f)),
    )


def test_bot_method_writes_ledger_record_and_respects_flag(monkeypatch):
    from config import Config
    from src.core.trading_bot import TradingBot

    monkeypatch.setattr(Config, "PAPER_TRADING", True)
    written = []
    rec, new = {"symbol": "ETH/USDT", "decision_id": "d1"}, {
        "side": "buy",
        "qty": 1.0,
        "entry": 1000.0,
    }
    monkeypatch.setattr(Config, "RISK_GATE_SHADOW", True)
    TradingBot._shadow_risk_gate(_bot(written), rec, {}, new, 10_000.0)
    kind, f = written[0]
    assert (
        kind == "risk_gate_shadow"
        and f["decision_id"] == "d1"
        and f["verdict"]["action"] == "allow"
    )
    monkeypatch.setattr(Config, "RISK_GATE_SHADOW", False)
    TradingBot._shadow_risk_gate(_bot(written), rec, {}, new, 10_000.0)
    assert len(written) == 1  # выключено — ничего не пишет


def test_bot_method_swallows_failures(monkeypatch):
    from config import Config
    from src.core.trading_bot import TradingBot

    monkeypatch.setattr(Config, "RISK_GATE_SHADOW", True)

    def boom(_s):
        raise RuntimeError("no data")

    written = []
    TradingBot._shadow_risk_gate(
        _bot(written, boom),
        {"symbol": "X"},
        {},
        {"side": "buy", "qty": 1, "entry": 1},
        100.0,
    )
    assert written == []  # сбой тени не доходит до торгового цикла


def test_report_section_counts_actions_and_reasons():
    from src.research.signal_intelligence_report import build_report, format_report

    recs = [
        {
            "kind": "risk_gate_shadow",
            "verdict": {"action": "allow", "reasons": []},
            "would_block": False,
            "inputs": {"source": "computed"},
        },
        {
            "kind": "risk_gate_shadow",
            "verdict": {"action": "reduce", "reasons": ["symbol_exposure"]},
            "would_block": True,
            "inputs": {"source": "mixed_or_default"},
        },
    ]
    rg = build_report(recs)["risk_gate"]
    assert (
        rg["n"] == 2
        and rg["by_action"] == {"allow": 1, "reduce": 1}
        and rg["would_block_share"] == 0.5
    )
    assert rg["by_reason"] == {"symbol_exposure": 1} and rg["inputs_computed"] == 1
    assert "ТЕНЕВОЙ RISK GATE" in format_report(build_report(recs))
    assert build_report([])["risk_gate"]["n"] == 0


def test_max_corr_receives_only_other_open_symbols():
    seen = []
    held = {
        "ETH/USDT": {"side": "buy", "qty": 1.0, "entry": 10.0},
        "SOL/USDT": {"side": "buy", "qty": 1.0, "entry": 10.0},
    }
    call(positions_before=held, max_corr=lambda s, o: seen.append(o) or 0.0)
    assert seen == [["SOL/USDT"]]
