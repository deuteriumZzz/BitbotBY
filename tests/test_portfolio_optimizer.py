"""Unit tests for PortfolioOptimizer (Markowitz + CVaR)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio_optimizer import PortfolioOptimizer


def _returns_df(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Synthetic daily log-returns for BTC, ETH, SOL."""
    rng = np.random.default_rng(seed)
    data = {
        "BTC/USDT": rng.normal(0.001, 0.02, n),
        "ETH/USDT": rng.normal(0.002, 0.025, n),
        "SOL/USDT": rng.normal(0.003, 0.04, n),
    }
    return pd.DataFrame(data)


class TestEqualWeights:
    def test_single_asset_returns_equal_weight(self):
        opt = PortfolioOptimizer()
        df = _returns_df()
        weights = opt.allocate(["BTC/USDT"], df)
        assert list(weights.keys()) == ["BTC/USDT"]
        assert abs(weights["BTC/USDT"] - 1.0) < 1e-6

    def test_too_few_rows_falls_back_to_equal(self):
        opt = PortfolioOptimizer()
        df = _returns_df(n=5)
        weights = opt.allocate(["BTC/USDT", "ETH/USDT"], df)
        for v in weights.values():
            assert abs(v - 0.5) < 1e-6

    def test_unknown_symbol_falls_back_to_equal_weights(self):
        # Only 1 valid column → < 2 → falls back to _equal_weights(symbols),
        # which includes all input symbols (even ones not in the DataFrame).
        opt = PortfolioOptimizer()
        df = _returns_df()
        weights = opt.allocate(["BTC/USDT", "UNKNOWN/USDT"], df)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert abs(weights.get("BTC/USDT", 0) - 0.5) < 1e-6


class TestMarkowitz:
    def test_weights_sum_to_one(self):
        opt = PortfolioOptimizer()
        df = _returns_df()
        weights = opt.optimize_markowitz(df)
        assert abs(sum(weights.values()) - 1.0) < 1e-5

    def test_each_weight_in_bounds(self):
        opt = PortfolioOptimizer()
        df = _returns_df()
        weights = opt.optimize_markowitz(df)
        for v in weights.values():
            assert 0.049 <= v <= 0.601

    def test_all_symbols_present(self):
        opt = PortfolioOptimizer()
        df = _returns_df()
        weights = opt.optimize_markowitz(df)
        assert set(weights.keys()) == set(df.columns)

    def test_single_asset_equal_weight_fallback(self):
        opt = PortfolioOptimizer()
        df = _returns_df()[["BTC/USDT"]]
        weights = opt.optimize_markowitz(df)
        assert abs(weights["BTC/USDT"] - 1.0) < 1e-6


class TestCVaR:
    def test_weights_sum_to_one(self):
        opt = PortfolioOptimizer()
        df = _returns_df()
        weights = opt.optimize_cvar(df)
        assert abs(sum(weights.values()) - 1.0) < 1e-5

    def test_each_weight_in_bounds(self):
        opt = PortfolioOptimizer()
        df = _returns_df()
        weights = opt.optimize_cvar(df)
        for v in weights.values():
            assert 0.049 <= v <= 0.601

    def test_all_symbols_present(self):
        opt = PortfolioOptimizer()
        df = _returns_df()
        weights = opt.optimize_cvar(df)
        assert set(weights.keys()) == set(df.columns)

    def test_high_volatility_asset_gets_lower_weight(self):
        """CVaR should underweight the asset with 4x higher volatility."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "LOW_VOL": rng.normal(0.001, 0.01, 100),
                "HIGH_VOL": rng.normal(0.001, 0.04, 100),
            }
        )
        opt = PortfolioOptimizer()
        weights = opt.optimize_cvar(df)
        assert weights["LOW_VOL"] >= weights["HIGH_VOL"]

    def test_too_few_rows_falls_back_to_equal(self):
        opt = PortfolioOptimizer()
        df = _returns_df(n=5)
        weights = opt.optimize_cvar(df[["BTC/USDT", "ETH/USDT"]])
        for v in weights.values():
            assert abs(v - 0.5) < 1e-6


class TestAllocate:
    def test_cvar_is_default_method(self):
        opt = PortfolioOptimizer()
        df = _returns_df()
        syms = list(df.columns)
        w_default = opt.allocate(syms, df)
        w_cvar = opt.allocate(syms, df, method="cvar")
        assert w_default == w_cvar

    def test_markowitz_method_accepted(self):
        opt = PortfolioOptimizer()
        df = _returns_df()
        weights = opt.allocate(list(df.columns), df, method="markowitz")
        assert abs(sum(weights.values()) - 1.0) < 1e-5

    def test_unknown_method_defaults_to_cvar(self):
        opt = PortfolioOptimizer()
        df = _returns_df()
        weights = opt.allocate(list(df.columns), df, method="unknown_method")
        assert abs(sum(weights.values()) - 1.0) < 1e-5

    def test_columns_subset_respected(self):
        opt = PortfolioOptimizer()
        df = _returns_df()
        weights = opt.allocate(["BTC/USDT", "ETH/USDT"], df)
        assert "SOL/USDT" not in weights
        assert abs(sum(weights.values()) - 1.0) < 1e-5
