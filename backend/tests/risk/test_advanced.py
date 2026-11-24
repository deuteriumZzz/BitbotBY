"""Динамический CVaR, стресс BTC-беты, Black–Litterman, Kelly от калиброванной вероятности, Thompson."""

import numpy as np
import pytest

from bitbot.risk import advanced as ad
from bitbot.risk import allocation as al
from bitbot.risk.metrics import hist_var_cvar

rng = np.random.default_rng(7)


# ── динамический CVaR ─────────────────────────────────────────────────────────
def test_dynamic_cvar_reacts_to_volatility_regime_unlike_static_window():
    calm_then_storm = np.concatenate(
        [rng.normal(0, 0.005, 300), rng.normal(0, 0.03, 20)]
    )
    storm_then_calm = np.concatenate(
        [rng.normal(0, 0.03, 200), rng.normal(0, 0.003, 200)]
    )
    up = ad.dynamic_cvar(calm_then_storm)
    down = ad.dynamic_cvar(storm_then_calm)
    assert (
        up["cvar"] > 1.5 * hist_var_cvar(calm_then_storm)[1]
    )  # статическое окно занижает риск после скачка
    assert (
        down["cvar"] < 0.6 * hist_var_cvar(storm_then_calm)[1]
    )  # и завышает после успокоения
    assert up["cvar"] >= up["var"] > 0


def test_dynamic_cvar_needs_enough_data_and_ewma_shape():
    with pytest.raises(ValueError):
        ad.dynamic_cvar(rng.normal(0, 0.01, 10))
    r = rng.normal(0, 0.01, 100)
    assert len(ad.ewma_sigma(r)) == 101 and (ad.ewma_sigma(r) > 0).all()


# ── стресс беты ───────────────────────────────────────────────────────────────
def test_btc_shock_systematic_and_idiosyncratic_parts():
    pos = [
        ad.StressPosition("ETH", 1000, 1.5, 0.02),
        ad.StressPosition("BTC", -500, 1.0, 0.02),
    ]
    r = ad.stress_btc_shock(pos, 10_000, shock=-0.20, max_loss=0.02)
    assert r["systematic"] == pytest.approx(
        -300 + 100
    )  # лонг теряет, шорт зарабатывает
    assert r["idio"] == pytest.approx(-30.0) and r["pnl"] == pytest.approx(-230.0)
    assert r["loss_pct"] == pytest.approx(0.023) and r["breach"] is True
    assert ad.stress_btc_shock(pos, 10_000, max_loss=0.5)["breach"] is False
    assert ad.stress_btc_shock([], 1000)["loss_pct"] == 0.0
    with pytest.raises(ValueError):
        ad.stress_btc_shock(pos, 0)


def test_hedged_book_loses_less_than_one_sided_book():
    long_only = [ad.StressPosition("A", 1000, 1.0), ad.StressPosition("B", 1000, 1.0)]
    hedged = [ad.StressPosition("A", 1000, 1.0), ad.StressPosition("B", -1000, 1.0)]
    assert (
        ad.stress_btc_shock(hedged, 10_000)["loss_pct"]
        < ad.stress_btc_shock(long_only, 10_000)["loss_pct"]
    )


# ── Black–Litterman ───────────────────────────────────────────────────────────
U = ["BTC", "ETH", "SOL"]
COV = np.array([[0.04, 0.03, 0.02], [0.03, 0.09, 0.04], [0.02, 0.04, 0.16]])
W = [0.6, 0.3, 0.1]


def test_no_views_returns_equilibrium():
    r = ad.black_litterman(U, COV, W, [])
    assert np.allclose(r["mu"], r["pi"]) and np.allclose(r["cov"], COV)


def test_view_moves_posterior_toward_view_more_with_confidence():
    pi = ad.black_litterman(U, COV, W, [])["pi"]
    low = ad.black_litterman(U, COV, W, [ad.View({"ETH": 1.0}, pi[1] + 0.10, 0.1)])[
        "mu"
    ]
    high = ad.black_litterman(U, COV, W, [ad.View({"ETH": 1.0}, pi[1] + 0.10, 0.9)])[
        "mu"
    ]
    assert pi[1] < low[1] < high[1] <= pi[1] + 0.10 + 1e-9
    r = ad.black_litterman(U, COV, W, [ad.View({"ETH": 1.0, "BTC": -1.0}, 0.05, 0.5)])
    assert np.all(np.linalg.eigvalsh(r["cov"]) > 0) and np.all(
        np.linalg.eigvalsh(r["cov"] - COV) >= -1e-9
    )


def test_validate_views_accepts_good_and_rejects_bad_structured_input():
    good = [
        {"assets": {"ETH": 1, "BTC": -1}, "expected_return": 0.03, "confidence": 0.6}
    ]
    assert ad.validate_views(good, U)[0].confidence == 0.6
    bad = [
        "text",
        [
            {"assets": {"DOGE": 1}, "expected_return": 0.01, "confidence": 0.5}
        ],  # актив вне universe
        [{"assets": {"ETH": 1}, "expected_return": float("nan"), "confidence": 0.5}],
        [{"assets": {"ETH": 1}, "expected_return": 0.01, "confidence": 0.0}],
        [{"assets": {"ETH": 1}, "expected_return": 0.01, "confidence": 1.5}],
        [
            {"assets": {"ETH": 1}, "expected_return": 0.9, "confidence": 0.5}
        ],  # слишком большой q
        [{"assets": {"ETH": True}, "expected_return": 0.01, "confidence": 0.5}],
        [
            {
                "assets": {"ETH": 1},
                "expected_return": 0.01,
                "confidence": 0.5,
                "note": "купи всё",
            }
        ],
        [{"assets": {}, "expected_return": 0.01, "confidence": 0.5}],
    ]
    for b in bad:
        with pytest.raises(ValueError):
            ad.validate_views(b, U)


def test_mv_weights_respect_caps_and_long_only():
    mu = np.array([0.10, 0.08, -0.02])
    w = ad.mv_weights(mu, COV, cap=0.3)
    assert (w >= 0).all() and (w <= 0.3 + 1e-12).all() and w.sum() <= 1 + 1e-9


# ── Kelly от Meta-Model ───────────────────────────────────────────────────────
def test_kelly_zero_unless_meta_model_is_usable():
    assert ad.kelly_from_meta(0.7, 2.0, usable=False, n_cal=1000) == {
        "fraction": 0.0,
        "p_eff": None,
        "reason": "meta_model_not_usable",
    }


def test_kelly_shrinks_with_small_calibration_sample_and_is_capped():
    big = ad.kelly_from_meta(0.6, 2.0, usable=True, n_cal=5000)
    small = ad.kelly_from_meta(0.6, 2.0, usable=True, n_cal=50)
    assert (
        0 < small["fraction"] < big["fraction"] <= 0.20
        and small["p_eff"] < big["p_eff"]
    )
    assert (
        ad.kelly_from_meta(0.95, 5.0, usable=True, n_cal=5000)["fraction"] == 0.20
    )  # потолок


def test_kelly_no_edge_and_bad_input():
    no_edge = ad.kelly_from_meta(0.3, 1.0, usable=True, n_cal=1000)
    assert no_edge["reason"] == "no_edge" and no_edge["fraction"] == 0.0
    for kw in (dict(p=1.2, reward_risk=2), dict(p=0.5, reward_risk=0)):
        assert ad.kelly_from_meta(usable=True, n_cal=100, **kw)["reason"] == "bad_input"
    assert ad.kelly_from_meta(0.6, 2.0, usable=True, n_cal=0)["reason"] == "bad_input"


# ── Thompson Sampling ─────────────────────────────────────────────────────────
PREV = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
STATS = {"a": (80, 20), "b": (20, 80), "c": (50, 50)}


def test_thompson_is_gated_by_oos_and_sample_size():
    assert al.thompson_weights(STATS, PREV) == PREV  # без OOS
    assert (
        al.thompson_weights({**STATS, "c": (5, 5)}, PREV, oos_validated=True) == PREV
    )  # мало исходов
    with pytest.raises(ValueError):
        al.thompson_weights({"a": (1, 1)}, PREV, oos_validated=True)


def test_thompson_moves_toward_best_with_rate_limit_floor_and_determinism():
    w = al.thompson_weights(STATS, PREV, oos_validated=True, max_step=0.10)
    assert (
        sum(w.values()) == pytest.approx(1.0)
        and w["a"] > PREV["a"]
        and w["b"] < PREV["b"]
    )
    assert sum(abs(w[k] - PREV[k]) for k in PREV) <= 0.10 + 1e-9
    assert w == al.thompson_weights(
        STATS, PREV, oos_validated=True, max_step=0.10
    )  # тот же seed
    cur = dict(PREV)
    for _ in range(60):
        cur = al.thompson_weights(
            STATS, cur, oos_validated=True, floor=0.05, max_step=0.2
        )
    assert min(cur.values()) >= 0.05 - 1e-9 and cur["a"] > 0.8
