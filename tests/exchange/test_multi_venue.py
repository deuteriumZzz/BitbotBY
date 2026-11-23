"""Phase 12: здоровье площадки и план хеджа (чистая логика)."""

import pytest

from src.exchange.multi_venue import DEGRADED, DOWN, HEALTHY, HedgeLimits, VenueHealth, hedge_plan


def feed(h, outcomes, latency=100.0):
    for ok in outcomes:
        h.record(ok, latency)
    return h


def test_few_calls_is_healthy_but_streak_is_down():
    assert feed(VenueHealth(), [False, True]).state() == HEALTHY  # мало данных
    h = feed(VenueHealth(), [False] * 5)
    assert h.state() == DOWN and h.policy()["block_entries"]  # серия подряд


def test_error_rate_thresholds():
    assert feed(VenueHealth(), [True] * 20).state() == HEALTHY
    assert feed(VenueHealth(), [True] * 17 + [False, True, False]).state() == DEGRADED  # 10%
    mixed = [True, False] * 10  # 50% без длинной серии
    assert feed(VenueHealth(), mixed).state() == DOWN


def test_slow_p95_is_degraded_and_success_resets_streak():
    h = feed(VenueHealth(), [True] * 20, latency=4000.0)
    assert h.state() == DEGRADED and h.policy()["widen_caution"] and not h.policy()["block_entries"]
    h2 = feed(VenueHealth(), [False] * 4 + [True])
    assert h2.consecutive_errors == 0 and h2.stats()["calls"] == 5


def test_policy_never_auto_failover_and_window_bounded():
    h = feed(VenueHealth(window=20), [False] * 50)
    assert h.policy()["auto_failover"] is False and h.stats()["calls"] == 20


def test_hedge_is_opposite_sign_and_full_when_unconstrained():
    p = hedge_plan(10_000, 0.5, 100_000, 5, 10_000_000, 10, 5.5)
    assert p["viable"] and p["hedge_notional"] == -5000.0 and p["capped_by"] == []
    assert p["expected_cost_bps"] == 11.0 and p["margin_required"] == 1000.0
    assert p["residual_basis_risk"] == pytest.approx(5000 * 10 / 1e4)
    assert hedge_plan(-4000, 1.0, 100_000, 5, 10_000_000, 10, 5)["hedge_notional"] == 4000.0


def test_capital_and_depth_caps():
    cap = hedge_plan(100_000, 1.0, 5_000, 5, 10_000_000, 10, 5)  # 20% × 5 000 × 5 = 5 000
    assert cap["hedge_notional"] == -5000.0 and cap["capped_by"] == ["capital"]
    depth = hedge_plan(100_000, 1.0, 1_000_000, 5, 20_000, 10, 5)  # 10% глубины = 2 000
    assert depth["hedge_notional"] == -2000.0 and depth["capped_by"] == ["depth"]
    assert cap["coverage"] == 0.05


def test_basis_risk_and_min_notional_make_hedge_not_viable():
    risky = hedge_plan(10_000, 1.0, 100_000, 5, 10_000_000, 80, 5, HedgeLimits(max_basis_bps=50))
    assert not risky["viable"] and risky["hedge_notional"] == 0.0 and "basis_risk_too_high" in risky["reasons"]
    tiny = hedge_plan(8, 1.0, 100_000, 5, 10_000_000, 10, 5)
    assert not tiny["viable"] and "below_min_notional" in tiny["reasons"]


def test_validation_and_zero_exposure():
    with pytest.raises(ValueError):
        hedge_plan(1000, 1.5, 1000, 5, 1000, 10, 5)
    with pytest.raises(ValueError):
        hedge_plan(1000, 0.5, 1000, 0, 1000, 10, 5)
    assert not hedge_plan(0, 1.0, 1000, 5, 1000, 10, 5)["viable"]
