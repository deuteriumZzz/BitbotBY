"""Phase 8: калибровка, дрейф/OOD/conformal, оценка прогнозов, мета-модель (синтетические данные)."""

import numpy as np
import pytest

from src.reliability import calibration as cal
from src.reliability import drift as dr
from src.reliability import forecast_eval as fe
from src.reliability.meta_model import MetaModel

rng = np.random.default_rng(0)


# ── калибровка ────────────────────────────────────────────────────────────────
def test_brier_ece_known_values():
    assert cal.brier(np.array([1.0, 0.0]), np.array([1, 0])) == 0
    assert cal.brier(np.array([0.5, 0.5]), np.array([1, 0])) == pytest.approx(0.25)
    p = np.full(1000, 0.7)
    y = (np.arange(1000) % 10 < 7).astype(float)  # частота ровно 0.7
    assert cal.ece(p, y) == pytest.approx(0, abs=1e-9)
    assert cal.ece(np.full(1000, 0.9), y) == pytest.approx(0.2, abs=1e-9)


@pytest.mark.parametrize("cls", [cal.PlattCalibrator, cal.IsotonicCalibrator])
def test_calibrators_fix_overconfidence(cls):
    true_p = rng.uniform(0.2, 0.8, 4000)
    y = (rng.uniform(size=4000) < true_p).astype(float)
    overconf = np.clip(0.5 + (true_p - 0.5) * 2.0, 0.01, 0.99)  # раздуваем уверенность
    fit, test = slice(0, 2000), slice(2000, 4000)
    c = cls().fit(overconf[fit], y[fit])
    assert cal.ece(c.predict(overconf[test]), y[test]) < cal.ece(
        overconf[test], y[test]
    )


def test_isotonic_is_monotone():
    p = rng.uniform(size=500)
    y = (rng.uniform(size=500) < p).astype(float)
    out = cal.IsotonicCalibrator().fit(p, y).predict(np.linspace(0, 1, 100))
    assert np.all(np.diff(out) >= -1e-12)


# ── дрейф / OOD ───────────────────────────────────────────────────────────────
def test_psi_and_ks_detect_shift_not_noise():
    a = rng.normal(0, 1, 5000)
    same = rng.normal(0, 1, 5000)
    shifted = rng.normal(1.0, 1, 5000)
    assert dr.psi(a, same) < dr.PSI_MODERATE
    assert dr.psi(a, shifted) > dr.PSI_STRONG
    assert dr.ks(a, same)[1] > 0.01 > dr.ks(a, shifted)[1]


def test_mahalanobis_flags_outliers_and_keeps_inliers():
    X = rng.normal(size=(2000, 3))
    ood = dr.MahalanobisOOD(0.99).fit(X)
    assert ood.is_ood(rng.normal(size=(2000, 3))).mean() < 0.03
    assert ood.is_ood(np.full((5, 3), 8.0)).all()


def test_cusum_alarm_after_shift_only():
    flat = rng.normal(0, 1, 300)
    assert dr.cusum(flat, 0.0, k=0.5, h=8) is None
    shifted = np.concatenate([flat, rng.normal(2, 1, 100)])
    i = dr.cusum(shifted, 0.0, k=0.5, h=8)
    assert i is not None and i >= 300  # тревога не раньше сдвига


# ── conformal ─────────────────────────────────────────────────────────────────
def test_conformal_coverage_iid_close_to_nominal():
    res = rng.normal(0, 1, 2000)
    q = dr.conformal_halfwidth(res, 0.1)
    new = rng.normal(0, 1, 20000)
    assert dr.empirical_coverage(new, -q, q) == pytest.approx(0.9, abs=0.02)


def test_conformal_undercovers_under_shift_and_we_measure_it():
    q = dr.conformal_halfwidth(rng.normal(0, 1, 2000), 0.1)
    shifted = rng.normal(0, 2.0, 20000)  # волатильность выросла вдвое
    assert dr.empirical_coverage(shifted, -q, q) < 0.8  # номинал 0.9 не держится


def test_conformal_too_few_samples_is_infinite():
    assert dr.conformal_halfwidth(np.array([1.0, 2.0]), 0.1) == float("inf")


# ── прогнозы ──────────────────────────────────────────────────────────────────
def test_metrics_known_values():
    assert fe.mae([1, 2], [2, 4]) == 1.5
    assert fe.rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))
    assert fe.mase([3], [2], np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)
    assert fe.pinball(np.array([1.0]), np.array([0.0]), 0.9) == pytest.approx(0.9)
    assert (
        fe.directional_accuracy(np.array([2.0]), np.array([3.0]), np.array([1.0]))
        == 1.0
    )


def test_evaluate_uses_only_past_and_same_origins():
    seen = []

    def spy(history, h):
        seen.append(len(history))
        return {"median": np.full(h, history[-1])}

    s = np.arange(100.0)
    res = fe.evaluate({"spy": spy, "naive": fe.naive}, s, [50, 60], 5)
    assert seen == [50, 60]  # видел только префикс до origin
    assert res["spy"]["mae"] == res["naive"]["mae"]
    with pytest.raises(ValueError):
        fe.evaluate({"naive": fe.naive}, s, [98], 5)


def test_seasonal_naive_wins_on_seasonal_series():
    t = np.arange(400)
    s = np.sin(2 * np.pi * t / 24)
    res = fe.evaluate(
        {
            "naive": fe.naive,
            "snaive": fe.seasonal_naive(24),
            "ma": fe.moving_average(24),
        },
        s,
        [200, 250, 300],
        12,
        season=24,
    )
    assert res["snaive"]["mae"] < 1e-9 < res["naive"]["mae"]


def test_interval_coverage_and_pinball_reported():
    s = rng.normal(0, 1, 600)

    def banded(history, h):
        return {
            "median": np.zeros(h),
            "lo": np.full(h, -1.645),
            "hi": np.full(h, 1.645),
        }

    r = fe.evaluate({"b": banded}, s, list(range(100, 500, 10)), 10)["b"]
    assert (
        r["coverage"] == pytest.approx(0.9, abs=0.03) and r["nominal_coverage"] == 0.9
    )
    assert r["pinball"] > 0


# ── мета-модель ───────────────────────────────────────────────────────────────
def _data(n, signal=1.5, seed=0):
    g = np.random.default_rng(seed)
    X = g.normal(size=(n, 4))
    p = 1 / (1 + np.exp(-signal * X[:, 0]))
    return X, (g.uniform(size=n) < p).astype(float)


def test_meta_model_with_signal_is_usable_and_calibrated():
    Xtr, ytr = _data(3000, seed=1)
    Xv, yv = _data(1500, seed=2)
    Xt, yt = _data(1500, seed=3)
    r = MetaModel(method="platt").fit(Xtr, ytr, Xv, yv).evaluate(Xt, yt)
    assert r["usable"] and r["brier_calibrated"] < r["brier_base_rate"]
    assert r["ece_calibrated"] < 0.06


def test_meta_model_on_noise_is_not_usable():
    Xtr, ytr = _data(3000, signal=0.0, seed=1)
    Xv, yv = _data(1500, signal=0.0, seed=2)
    Xt, yt = _data(1500, signal=0.0, seed=3)
    r = MetaModel(method="isotonic").fit(Xtr, ytr, Xv, yv).evaluate(Xt, yt)
    assert not r["usable"]  # нет сигнала → вероятность нельзя использовать


def test_meta_model_too_small_calibration_set_is_not_usable():
    Xtr, ytr = _data(3000, seed=1)
    Xv, yv = _data(100, seed=2)
    Xt, yt = _data(1500, seed=3)
    assert not MetaModel().fit(Xtr, ytr, Xv, yv).evaluate(Xt, yt)["usable"]


def test_meta_model_guards():
    with pytest.raises(RuntimeError):
        MetaModel().predict_proba(np.zeros((1, 4)))
    with pytest.raises(ValueError):
        MetaModel(method="bogus")
