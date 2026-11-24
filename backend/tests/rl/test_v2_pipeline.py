"""SAC v2: причинные признаки, walk-forward обучение (малый прогон), кандидат в реестр, инференс."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from bitbot.ml import features_v2 as fv
from bitbot.ml import train_sac_v2 as tr
from bitbot.ml.model_registry import ModelRegistry
from bitbot.ml.sac_v2_policy import SACV2Policy, exposure_to_signal


def synthetic(n=1500, seed=0, funding=False):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0008, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0008, n)))
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.lognormal(3, 0.4, n),
        },
        index=pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"),
    )
    if funding:
        df["funding_rate"] = 0.0001
    return df


# ── признаки ──────────────────────────────────────────────────────────────────
def test_features_shape_names_and_bounds():
    df = synthetic()
    f = fv.build_features(df)
    assert list(f.columns) == fv.FEATURE_NAMES and len(f) == len(df) - fv.WARMUP
    assert np.isfinite(f.to_numpy()).all() and np.abs(f.to_numpy()).max() <= fv.CLIP
    assert f.dtypes.eq(np.float32).all()


def test_no_lookahead_truncating_the_future_changes_nothing():
    df = synthetic()
    full = fv.build_features(df)
    for cut in (400, 900, 1300):
        part = fv.build_features(df.iloc[:cut])
        pd.testing.assert_frame_equal(
            full.loc[part.index], part, check_exact=False, atol=1e-5
        )


def test_tail_window_matches_full_history_for_inference():
    df = synthetic(2000)
    full_last = fv.build_features(df).iloc[-1].to_numpy()
    tail_last = fv.latest_features(df.iloc[-fv.MIN_TAIL :])
    assert np.allclose(full_last, tail_last, atol=1e-3)
    assert fv.latest_features(df.iloc[: fv.WARMUP]) is None


def test_funding_feature_and_validation():
    assert np.allclose(
        fv.build_features(synthetic(funding=True))["funding"].to_numpy(), 0.01
    )
    assert (fv.build_features(synthetic())["funding"] == 0).all()
    with pytest.raises(ValueError):
        fv.build_features(synthetic().drop(columns=["volume"]))
    bad = synthetic()
    bad.iloc[10, bad.columns.get_loc("close")] = 0.0
    with pytest.raises(ValueError):
        fv.build_features(bad)


def test_prepare_alignment():
    feats, o, c, f = fv.prepare(synthetic(funding=True))
    assert len(feats) == len(o) == len(c) == len(f) and (f == 0.0001).all()


def test_load_ohlcv_sorts_dedupes_and_parses_ms(tmp_path):
    d = synthetic(5).reset_index(drop=True)
    ts = (pd.Timestamp("2026-01-01", tz="UTC").value // 10**6) + np.array(
        [2, 0, 1, 1, 3]
    ) * 900_000
    d.insert(0, "Timestamp", ts)
    p = tmp_path / "x.csv"
    d.to_csv(p, index=False)
    out = tr.load_ohlcv(str(p))
    assert (
        out.index.is_monotonic_increasing
        and not out.index.has_duplicates
        and len(out) == 4
    )


# ── обучение и инференс (малый реальный прогон SB3) ───────────────────────────
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("models") / "sac_v2.zip")
    res = tr.run(
        synthetic(1500),
        timesteps=300,
        n_folds=3,
        min_train_frac=0.5,
        embargo=8,
        seed=1,
        out_path=out,
        symbol="TEST/USDT",
        timeframe="15m",
    )
    return out, res


def test_training_produces_artifacts_and_registers_candidate_only(trained):
    out, res = trained
    base = out[:-4]
    for p in (base + ".zip", base + "_norm_stats.json", base + ".meta.json"):
        assert os.path.exists(p)
    meta = json.load(open(base + ".meta.json"))
    assert (
        meta["env"] == "v2"
        and meta["feature_names"] == fv.FEATURE_NAMES
        and meta["obs_dim"] == 15
    )
    reg = ModelRegistry(os.path.join(os.path.dirname(out), "registry.json"))
    entry = reg.get(res["model_id"])
    assert (
        entry["stage"] == "candidate"
        and entry["dataset_version"] == res["dataset_version"]
    )
    assert not reg.is_executable(base + ".zip", paper=True)  # не допущена к торговле
    assert len(res["folds"]) == 3


def test_random_walk_does_not_pass_promotion_gates(trained):
    _, res = trained
    assert res["verdict"]["passed"] is False  # эджа на случайном блуждании нет


def test_second_run_same_data_does_not_overwrite_registry_entry(trained):
    out, res = trained
    again = tr.run(
        synthetic(1500),
        timesteps=200,
        n_folds=3,
        embargo=8,
        seed=1,
        out_path=out,
        symbol="TEST/USDT",
        timeframe="15m",
    )
    assert again["model_id"] != res["model_id"]


def test_policy_inference_matches_training_observation(trained):
    out, _ = trained
    pol = SACV2Policy(out)
    assert pol.loaded
    df = synthetic(1500)
    e = pol.target_exposure(df.iloc[-400:], exposure=0.0, equity_pnl=0.0)
    assert e is not None and -1.0 <= e <= 1.0
    obs = pol.observation(df.iloc[-400:], 0.5, 0.02)
    assert (
        obs.shape == (15,) and obs[-2] == 0.5 and obs[-1] == pytest.approx(0.02)
    )  # позиция передаётся
    assert pol.target_exposure(df.iloc[:30]) is None  # мало истории


def test_policy_refuses_mismatched_or_missing_model(trained, tmp_path):
    out, _ = trained
    assert SACV2Policy(str(tmp_path / "nope.zip")).loaded is False
    for suffix in (".zip", "_norm_stats.json"):
        (tmp_path / ("bad" + suffix)).write_bytes(open(out[:-4] + suffix, "rb").read())
    meta = json.load(open(out[:-4] + ".meta.json"))
    meta["feature_names"] = meta["feature_names"][:-1]  # другой набор признаков
    json.dump(meta, open(str(tmp_path / "bad.meta.json"), "w"))
    pol = SACV2Policy(str(tmp_path / "bad.zip"))
    assert pol.loaded is False and pol.target_exposure(synthetic(400)) is None


def test_exposure_to_signal_band():
    assert exposure_to_signal(0.5, 0.45) == "hold"
    assert (
        exposure_to_signal(0.8, 0.0) == "buy"
        and exposure_to_signal(-0.6, 0.0) == "sell"
    )
    assert exposure_to_signal(0.0, 0.5) == "sell"  # сокращение лонга к нулю
