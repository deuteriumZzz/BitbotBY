"""Phase 7: реестр моделей, допуск SAC, среда v2, оценка, датасет."""

import numpy as np
import pandas as pd
import pytest

from reinforcement_learning import dataset as ds
from reinforcement_learning import evaluation as ev
from reinforcement_learning.env_v2 import ExposureSim, simulate
from reinforcement_learning.model_registry import ModelRegistry, PromotionError
from src.core import trading_modes as tm

WF_OK = {"walk_forward": {"passed": True, "n_folds": 3}, "dataset_version": "abc"}


@pytest.fixture
def reg(tmp_path):
    return ModelRegistry(str(tmp_path / "registry.json"))


def _to(reg, mid, steps):
    """Продвигает модель на `steps` шагов с достаточными доказательствами."""
    evid = [
        WF_OK,
        {"approved_by": "owner"},
        {"approved_by": "owner", "paper_days": 14},
        {"approved_by": "owner", "shadow_days": 14},
        {"approved_by": "owner", "canary_days": 14},
    ]
    for e in evid[:steps]:
        reg.promote(mid, e, "t")


# ── реестр ────────────────────────────────────────────────────────────────────
def test_unregistered_cannot_execute(reg, tmp_path):
    assert reg.stage_of_path(str(tmp_path / "m.zip")) == "unregistered"
    assert not reg.is_executable(str(tmp_path / "m.zip"), paper=True)


def test_cannot_skip_or_promote_without_evidence(reg, tmp_path):
    reg.register("m1", str(tmp_path / "m.zip"))
    with pytest.raises(PromotionError):
        reg.promote("m1", {}, "t")  # нет walk-forward
    with pytest.raises(PromotionError):
        reg.promote("m1", {"walk_forward": {"passed": True, "n_folds": 1}, "dataset_version": "x"}, "t")
    reg.promote("m1", WF_OK, "t")  # validated
    with pytest.raises(PromotionError):
        reg.promote("m1", {}, "t")  # нет approved_by
    reg.promote("m1", {"approved_by": "o"}, "t")  # paper
    with pytest.raises(PromotionError):
        reg.promote("m1", {"approved_by": "o", "paper_days": 3}, "t")  # мало дней на paper


def test_executable_stages_paper_vs_live(reg, tmp_path):
    p = str(tmp_path / "m.zip")
    reg.register("m1", p)
    _to(reg, "m1", 2)
    assert reg.stage_of_path(p) == "paper"
    assert reg.is_executable(p, paper=True) and not reg.is_executable(p, paper=False)
    for e in [
        {"approved_by": "o", "paper_days": 14},
        {"approved_by": "o", "shadow_days": 14},
        {"approved_by": "o", "canary_days": 14},
    ]:
        reg.promote("m1", e, "t")
    assert reg.stage_of_path(p) == "production" and reg.is_executable(p, paper=False)


def test_new_production_retires_old_and_rollback(reg, tmp_path):
    for mid in ("a", "b"):
        reg.register(mid, str(tmp_path / f"{mid}.zip"))
    _to(reg, "a", 5)
    _to(reg, "b", 5)
    assert reg.get("a")["stage"] == "retired" and reg.get("b")["stage"] == "production"
    back = reg.rollback("плохой дрейф", "owner")
    assert back["model_id"] == "a" and reg.get("a")["stage"] == "production"
    assert reg.get("b")["stage"] == "retired"
    assert reg.get("a")["history"][-1]["evidence"]["reason"].startswith("откат")


def test_demote_requires_reason_and_persists(reg, tmp_path):
    reg.register("m1", str(tmp_path / "m.zip"))
    with pytest.raises(PromotionError):
        reg.demote("m1", "rejected", "", "t")
    reg.demote("m1", "rejected", "нет эджа", "t")
    assert ModelRegistry(reg.path).get("m1")["stage"] == "rejected"


# ── допуск в SACSignal / режимы ───────────────────────────────────────────────
def test_readiness_sac_not_approved():
    assert tm.readiness("hybrid", sac_loaded=True, sac_approved=False) == (
        False,
        tm.REASON_SAC_NOT_APPROVED,
    )
    assert tm.readiness("ai", sac_loaded=False, sac_approved=False)[0]  # без SAC не затронут
    assert tm.readiness("dqn", sac_loaded=True)[0]  # по умолчанию поведение прежнее


def test_sac_tradable_requires_registry(reg, tmp_path, monkeypatch):
    from config import Config
    from src.ai.dqn_signal import SACSignal

    path = str(tmp_path / "sac.zip")
    monkeypatch.setattr(Config, "SAC_MODEL_PATH", path)
    monkeypatch.setattr(Config, "PAPER_TRADING", True)
    monkeypatch.setattr(Config, "SAC_REQUIRE_APPROVAL", True)
    sac = SACSignal()
    sac.loaded = True
    assert not sac.tradable  # файл не в реестре
    reg.register("m1", path)
    assert not sac.tradable  # candidate
    _to(reg, "m1", 2)
    assert sac.tradable  # paper допущен в Paper
    monkeypatch.setattr(Config, "PAPER_TRADING", False)
    assert not sac.tradable  # в Live стадии paper мало
    monkeypatch.setattr(Config, "SAC_REQUIRE_APPROVAL", False)
    assert sac.tradable


# ── среда v2 ──────────────────────────────────────────────────────────────────
def _flat(n=10, p=100.0):
    return np.full(n, p), np.full(n, p)


def test_costs_on_turnover_only():
    o, c = _flat()
    r = simulate(o, c, np.array([1.0] * 9), fee=0.001, slippage=0.0)
    # одна покупка на весь капитал; дальше лишь дорасчёт до цели после съеденной комиссии
    assert r["costs"] == pytest.approx(10_000 * 0.001, rel=2e-3)
    assert r["equity"][-1] == pytest.approx(10_000 - r["costs"], rel=1e-6)


def test_short_profits_when_price_falls():
    o = np.array([100, 100, 90, 80, 70.0])
    c = np.array([100, 95, 85, 75, 70.0])
    short = simulate(o, c, np.array([-1.0] * 4), fee=0, slippage=0)
    long_ = simulate(o, c, np.array([1.0] * 4), fee=0, slippage=0)
    assert short["equity"][-1] > 10_000 > long_["equity"][-1]


def test_funding_paid_by_long_received_by_short():
    o, c = _flat()
    f = np.full(10, 0.001)
    long_ = simulate(o, c, np.ones(9), f, fee=0, slippage=0)
    short = simulate(o, c, -np.ones(9), f, fee=0, slippage=0)
    assert long_["equity"][-1] < 10_000 < short["equity"][-1]
    assert long_["funding_paid"] > 0 > short["funding_paid"]


def test_no_lookahead_execution_at_next_open():
    o = np.array([100.0, 200.0, 200.0, 200.0])
    sim = ExposureSim(o, o.copy(), fee=0, slippage=0)
    sim.step(1.0)  # решение на close t=0 исполняется по open t=1
    assert sim.units == pytest.approx(10_000 / 200.0)


def test_gym_wrapper_truncated_at_end_of_data():
    from reinforcement_learning.env_v2 import TradingEnvV2

    p = np.full(4, 100.0)
    env = TradingEnvV2(np.zeros((4, 2)), p, p.copy(), fee=0, slippage=0)
    env.reset()
    flags = [env.step([0.0])[2:4] for _ in range(3)]
    assert flags[-1] == (False, True)  # конец данных — усечение, не терминация
    assert flags[0] == (False, False)


def test_gym_wrapper_terminated_on_ruin():
    from reinforcement_learning.env_v2 import TradingEnvV2

    o = np.array([100.0, 100.0, 1.0, 1.0, 1.0])
    env = TradingEnvV2(np.zeros((5, 2)), o, o.copy(), fee=0, slippage=0, max_exposure=1.0)
    env.reset()
    env.step([-1.0])  # шорт; цена падает → прибыль, не банкротство
    # шорт на росте: цена x100 → банкротство
    o2 = np.array([1.0, 1.0, 100.0, 100.0, 100.0])
    env2 = TradingEnvV2(np.zeros((5, 2)), o2, o2.copy(), fee=0, slippage=0)
    env2.reset()
    env2.step([-1.0])
    _, _, term, trunc, _ = env2.step([-1.0])
    assert term and not trunc


# ── оценка ────────────────────────────────────────────────────────────────────
def test_walk_forward_embargo_and_no_overlap():
    for tr, te in ev.walk_forward_splits(1000, 4, 400, 10):
        assert tr.max() + 10 < te.min()
        assert not set(tr) & set(te)
    with pytest.raises(ValueError):
        ev.walk_forward_splits(100, 4, 90, 10)


def test_bootstrap_deterministic_and_contains_mean():
    x = np.random.default_rng(1).normal(0.1, 1.0, 500)
    a, b = ev.block_bootstrap_ci(x), ev.block_bootstrap_ci(x)
    assert a == b and a[0] < x.mean() < a[1]


def _fold(edge, n=200, seed=0):
    base = np.random.default_rng(seed).normal(0, 0.01, n)
    model = base + edge
    return {"model": model, "baseline": base, "equity": 10_000 * np.cumprod(1 + model * 0.1)}


def test_promotion_verdict_gates():
    assert ev.promotion_verdict([_fold(0.005, seed=i) for i in range(3)])["passed"]
    no_edge = ev.promotion_verdict([_fold(0.0, seed=i) for i in range(3)])
    assert not no_edge["passed"]
    assert "folds" in ev.promotion_verdict([_fold(0.005)])["failed"]


def test_metrics_drawdown():
    m = ev.metrics(np.array([100, 120, 90, 100.0]), 365)
    assert m["max_drawdown"] == pytest.approx(0.25)


# ── датасет ───────────────────────────────────────────────────────────────────
def _df(n=50):
    idx = pd.date_range("2026-01-01", periods=n, freq="15min")
    return pd.DataFrame({"a": np.arange(n, dtype=float), "b": np.sin(np.arange(n))}, index=idx)


def test_parquet_roundtrip_and_tamper_detection(tmp_path):
    p = str(tmp_path / "d.parquet")
    man = ds.save_parquet(_df(), p, {"tf": "15m"})
    df, man2 = ds.load_parquet(p)
    assert man == man2 and len(df) == 50
    t = _df()
    t.iloc[3, 0] = 999.0
    t.to_parquet(p, compression="zstd")
    with pytest.raises(ValueError):
        ds.load_parquet(p)


def test_version_changes_with_params_and_data():
    a = ds.build_manifest(_df(), {"x": 1})["dataset_version"]
    assert a != ds.build_manifest(_df(), {"x": 2})["dataset_version"]
    assert a != ds.build_manifest(_df(60), {"x": 1})["dataset_version"]


def test_causal_checks():
    with pytest.raises(ValueError):
        ds.check_causal(_df().iloc[::-1])
    nan = _df()
    nan.iloc[2, 0] = np.nan
    with pytest.raises(ValueError):
        ds.check_causal(nan)


def test_normalizer_fit_on_train_only():
    df = _df(100)
    train, test = df.iloc[:60], df.iloc[60:]
    stats = ds.fit_normalizer(train)
    assert ds.apply_normalizer(train, stats)["a"].mean() == pytest.approx(0, abs=1e-9)
    assert ds.apply_normalizer(test, stats)["a"].mean() > 1  # тест не центрируется по себе
