"""Tests for AlphaTester — bootstrap Sharpe + Wilcoxon significance."""

import numpy as np
import pytest

from src.alpha_tester import AlphaTester, AlphaResult


@pytest.fixture
def tester():
    return AlphaTester()


def test_empty_returns_gives_safe_result(tester):
    r = tester.test([], name="empty")
    assert r.n_trades == 0
    assert r.sharpe == 0.0
    assert r.sharpe_pvalue == 1.0
    assert r.wilcoxon_pvalue is None
    assert r.verdict == "INSUFFICIENT DATA"


def test_four_returns_gives_insufficient(tester):
    r = tester.test([0.01, -0.01, 0.02, 0.01], name="tiny")
    assert r.verdict == "INSUFFICIENT DATA"


def test_strongly_positive_returns_significant(tester):
    rng = np.random.default_rng(0)
    returns = list(rng.normal(loc=0.05, scale=0.01, size=100))
    r = tester.test(returns, name="strong_edge")
    assert r.sharpe > 0
    assert r.sharpe_ci_low > 0
    assert r.sharpe_pvalue < 0.05
    if r.wilcoxon_pvalue is not None:  # scipy may not be installed in test env
        assert r.wilcoxon_pvalue < 0.05
    assert r.is_significant  # True via bootstrap when wilcoxon unavailable


def test_zero_mean_not_confidently_positive(tester):
    rng = np.random.default_rng(42)
    returns = list(rng.normal(loc=0.0, scale=0.02, size=200))
    r = tester.test(returns, name="noise")
    assert r.sharpe_ci_low < 0.3


def test_negative_returns_high_pvalue(tester):
    rng = np.random.default_rng(1)
    returns = list(rng.normal(loc=-0.03, scale=0.01, size=80))
    r = tester.test(returns, name="losing")
    assert r.sharpe < 0
    assert r.mean_return < 0
    assert r.sharpe_pvalue > 0.5


def test_result_fields_populated(tester):
    returns = [0.01] * 30 + [-0.005] * 10
    r = tester.test(returns, name="mixed")
    assert isinstance(r, AlphaResult)
    assert r.n_trades == 40
    assert r.mean_return == pytest.approx(np.mean(returns), rel=1e-5)
    assert len(r.summary()) > 0


def test_verdict_insufficient_when_few_trades(tester):
    r = tester.test([0.1, 0.2, 0.3], name="v")
    assert "INSUFFICIENT" in r.verdict


def test_verdict_significant_for_strong_edge(tester):
    rng = np.random.default_rng(7)
    returns = list(rng.normal(0.04, 0.005, 120))
    r = tester.test(returns, name="v2")
    # SIGNIFICANT when both tests pass; falls back to Sharpe-only without scipy
    assert r.is_significant
