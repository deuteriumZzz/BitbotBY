"""Phase 9: метрики риска, Risk Gate, распределение капитала."""

import numpy as np
import pytest

from src.risk import allocation as al
from src.risk import metrics as mt
from src.risk.risk_gate import (
    GateLimits,
    Portfolio,
    Position,
    Proposal,
    breaker_levels,
    evaluate,
)

rng = np.random.default_rng(0)
LIM = GateLimits()


def pf(**kw):
    base = dict(equity=10_000.0, peak_equity=10_000.0, day_start_equity=10_000.0)
    base.update(kw)
    return Portfolio(**base)


# ── метрики ───────────────────────────────────────────────────────────────────
def test_var_cvar_known_and_ordered():
    r = -np.arange(1, 101) / 100.0  # потери 1%..100%
    var, cvar = mt.hist_var_cvar(r, 0.95)
    assert var == pytest.approx(0.9505, abs=1e-3) and cvar >= var
    with pytest.raises(ValueError):
        mt.hist_var_cvar(np.array([0.1]))


def test_beta_known_values():
    m = rng.normal(0, 0.01, 500)
    assert mt.beta(2 * m, m) == pytest.approx(2.0)
    assert mt.beta(-m, m) == pytest.approx(-1.0)
    assert mt.beta(np.ones(10), np.ones(10)) == 0.0


def test_hedge_reduces_portfolio_cvar_and_stress_is_upper_bound():
    m = rng.normal(0, 0.02, (1000, 1))
    R = np.hstack([m, m + rng.normal(0, 0.005, (1000, 1))])  # два сильно коррелированных актива
    same = mt.portfolio_cvar(R, [1000, 1000])
    hedge = mt.portfolio_cvar(R, [1000, -1000])
    assert hedge < same
    cv = [mt.hist_var_cvar(R[:, i])[1] for i in range(2)]
    assert mt.stress_cvar([1000, 1000], cv) >= same - 1e-9  # субаддитивность CVaR
    assert mt.net_beta_exposure([1000, -500], [1.0, 2.0]) == 0.0


# ── Risk Gate ─────────────────────────────────────────────────────────────────
def test_allow_normal_entry():
    v = evaluate(Proposal("ETH/USDT", 1000.0), pf(), LIM)
    assert (v.action, v.approved_notional, v.reasons) == ("allow", 1000.0, [])


def test_reduce_only_never_blocked():
    v = evaluate(Proposal("ETH/USDT", -500.0, reduce_only=True), pf(kill_switch=True), LIM)
    assert v.action == "allow" and v.reasons == ["reduce_only"]


@pytest.mark.parametrize(
    "kw",
    [
        {"kill_switch": True},
        {"equity": 9_400.0},  # дневной убыток 6% ≥ 5%
        {"consecutive_losses": 3},
    ],
)
def test_breaker_denies_entries(kw):
    v = evaluate(Proposal("ETH/USDT", 1000.0), pf(**kw), LIM)
    assert v.action == "deny" and v.level >= 1 and v.reasons[0].startswith("circuit_breaker")


def test_ood_blocks_entries_only_when_enabled():
    p = Proposal("ETH/USDT", 1000.0, ood=True)
    assert evaluate(p, pf(), LIM).action == "deny"
    assert "ood" in evaluate(p, pf(), LIM).reasons
    assert evaluate(p, pf(), GateLimits(block_on_ood=False)).action == "allow"


def test_breaker_levels_are_independent_and_graded():
    def L(**kw):
        return breaker_levels(pf(**kw), LIM)

    assert not any(L().values())
    day = L(equity=9_400.0, peak_equity=9_400.0)  # дневной убыток без просадки от пика
    assert day[1] and not (day[2] or day[3] or day[4])
    assert L(equity=8_400.0, day_start_equity=8_400.0) == {1: True, 2: True, 3: False, 4: False}
    lv3 = L(equity=7_900.0, day_start_equity=7_900.0)
    assert lv3[3] and not lv3[4]
    assert all(L(equity=7_400.0, day_start_equity=7_400.0).values())
    assert all(L(kill_switch=True).values())


def test_correlation_denied():
    v = evaluate(Proposal("ETH/USDT", 1000.0, max_corr_with_open=0.9), pf(), LIM)
    assert v.action == "deny" and v.reasons == ["correlation"]


def test_symbol_exposure_reduces_to_cap():
    v = evaluate(Proposal("ETH/USDT", 4000.0), pf(), LIM)  # лимит 25% = 2500
    assert v.action == "reduce" and v.approved_notional == pytest.approx(2500.0)
    assert "symbol_exposure" in v.reasons
    held = pf(positions=[Position("ETH/USDT", 2000.0, cvar=0.0)])
    got = evaluate(Proposal("ETH/USDT", 2000.0, cvar=0.0), held, LIM)
    assert got.approved_notional == pytest.approx(500.0)


def test_gross_exposure_cap():
    held = pf(positions=[Position(s, 2400.0, beta=0.0, cvar=0.0) for s in "ABCD"])  # 9600 из 10000
    v = evaluate(Proposal("E", 2000.0, beta=0.0, cvar=0.0), held, LIM)
    assert v.action == "reduce" and v.approved_notional == pytest.approx(400.0)


def test_net_beta_blocks_longs_but_allows_offsetting_short():
    held = pf(positions=[Position(s, 2400.0, cvar=0.0) for s in "ABC"])  # бета-экспозиция 7200 при лимите 7500
    long_ = evaluate(Proposal("D", 2000.0, cvar=0.0), held, LIM)
    assert long_.approved_notional == pytest.approx(300.0) and "net_beta" in long_.reasons
    short = evaluate(Proposal("D", -2000.0, cvar=0.0), held, LIM)
    assert short.action == "reduce" or short.action == "allow"
    assert short.approved_notional < 0 and "net_beta" not in short.reasons  # бету снижает


def test_stress_cvar_scales_position():
    held = pf(positions=[Position("A", 1500.0, beta=0.0, cvar=0.4)])  # стресс-CVaR 600 из лимита 1000
    v = evaluate(Proposal("B", 2000.0, beta=0.0, cvar=0.4), held, LIM)
    assert v.action == "reduce" and v.approved_notional == pytest.approx(1000.0)
    assert "stress_cvar" in v.reasons


def test_below_min_notional_denied():
    held = pf(positions=[Position("ETH/USDT", 2495.0, cvar=0.0)])
    v = evaluate(Proposal("ETH/USDT", 1000.0, cvar=0.0), held, LIM)
    assert v.action == "deny" and v.approved_notional == 0.0


def test_short_proposal_keeps_sign_when_reduced():
    v = evaluate(Proposal("ETH/USDT", -4000.0), pf(), LIM)
    assert v.action == "reduce" and v.approved_notional < 0


def test_from_config_reads_existing_thresholds():
    class C:
        DAILY_LOSS_LIMIT = 0.03
        MAX_DRAWDOWN_PERCENT = 0.1
        CIRCUIT_BREAKER_LOSSES = 5
        MAX_CORRELATION = 0.6

    lim = GateLimits.from_config(C)
    assert (lim.daily_loss_limit, lim.max_drawdown, lim.max_consecutive_losses, lim.max_correlation) == (
        0.03,
        0.1,
        5,
        0.6,
    )


# ── распределение капитала ────────────────────────────────────────────────────
PREV = {"a": 0.5, "b": 0.3, "c": 0.2}


def test_allocation_unchanged_without_oos():
    assert al.exp_weights({"a": 0, "b": 5, "c": 0}, PREV) == PREV


def test_allocation_rate_limited_floor_and_sums_to_one():
    w = al.exp_weights({"a": -5, "b": 5, "c": 0}, PREV, oos_validated=True, max_step=0.1)
    assert sum(w.values()) == pytest.approx(1.0)
    assert sum(abs(w[k] - PREV[k]) for k in PREV) <= 0.1 + 1e-9  # скорость перелива ограничена
    assert w["b"] > PREV["b"] and w["a"] < PREV["a"]
    for _ in range(100):  # длительное давление не опускает вес ниже пола
        w = al.exp_weights({"a": -50, "b": 50, "c": 0}, w, oos_validated=True, floor=0.05)
    assert min(w.values()) >= 0.05 - 1e-9 and sum(w.values()) == pytest.approx(1.0)


def test_allocation_input_checks():
    with pytest.raises(ValueError):
        al.exp_weights({"a": 1}, PREV, oos_validated=True)
    with pytest.raises(ValueError):
        al.exp_weights({"a": 1, "b": 1, "c": 1}, {"a": 0.5, "b": 0.3, "c": 0.3}, oos_validated=True)
