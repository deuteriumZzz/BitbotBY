"""
Statistical significance testing for strategy alpha.

Two tests per strategy:
  1. Bootstrap Sharpe — 1 000 resamples → point estimate + 95% CI + p-value
     (fraction of bootstrap Sharpes ≤ 0; robust to non-normality)
  2. Wilcoxon signed-rank — non-parametric H₀: median return = 0
     (preferred over t-test for fat-tailed crypto returns)

Usage:
    from src.alpha_tester import AlphaTester
    result = AlphaTester().test(trade_returns, name="ema_crossover")
    print(result.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_N_BOOTSTRAP = 1_000
_ALPHA = 0.05  # 95% confidence level


@dataclass
class AlphaResult:
    """Statistical test results for a single strategy."""

    name: str
    n_trades: int

    # Bootstrap Sharpe (per-trade, not annualised — honest for variable holding periods)
    sharpe: float
    sharpe_ci_low: float
    sharpe_ci_high: float
    sharpe_pvalue: float  # P(Sharpe ≤ 0) under bootstrap distribution

    # Wilcoxon signed-rank
    wilcoxon_stat: Optional[float]
    wilcoxon_pvalue: Optional[float]

    mean_return: float
    std_return: float

    @property
    def is_significant(self) -> bool:
        """
        Sharpe bootstrap p < 5%. If scipy available, Wilcoxon must also pass.
        When scipy is absent (wilcoxon_pvalue is None), bootstrap alone decides.
        """
        if self.sharpe_pvalue >= _ALPHA:
            return False
        if self.wilcoxon_pvalue is None:
            return True  # scipy unavailable — bootstrap Sharpe is sufficient
        return self.wilcoxon_pvalue < _ALPHA

    @property
    def verdict(self) -> str:
        if self.n_trades < 10:
            return "INSUFFICIENT DATA"
        if self.is_significant:
            return "SIGNIFICANT"
        if self.sharpe_pvalue < _ALPHA:
            return "WEAK  (Sharpe ok, Wilcoxon fails)"
        return "NOT SIGNIFICANT"

    def summary(self) -> str:
        wp = (
            f"{self.wilcoxon_pvalue:.4f}"
            if self.wilcoxon_pvalue is not None
            else "n/a"
        )
        sig_mark = "✓" if self.is_significant else "✗"
        return (
            f"  [{self.name}]  n={self.n_trades}\n"
            f"    Sharpe:   {self.sharpe:+.3f}  "
            f"95% CI [{self.sharpe_ci_low:+.3f}, {self.sharpe_ci_high:+.3f}]  "
            f"p(Sharpe≤0)={self.sharpe_pvalue:.4f}\n"
            f"    Wilcoxon: p={wp}\n"
            f"    Mean ret: {self.mean_return:+.4f}  Std: {self.std_return:.4f}\n"
            f"    {sig_mark} {self.verdict}"
        )


class AlphaTester:
    """Bootstrap Sharpe + Wilcoxon signed-rank tests on per-trade returns."""

    def _bootstrap_sharpe(
        self,
        returns: np.ndarray,
        n_bootstrap: int = _N_BOOTSTRAP,
    ) -> tuple[float, float, float, float]:
        """
        Resample returns n_bootstrap times and compute Sharpe distribution.

        :return: (sharpe, ci_low, ci_high, p_value_sharpe_leq_0)
        """
        n = len(returns)
        std = returns.std(ddof=1)
        if std < 1e-10:
            return 0.0, 0.0, 0.0, 1.0

        sharpe = float(returns.mean() / std)

        rng = np.random.default_rng(42)
        boot = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            s = rng.choice(returns, size=n, replace=True)
            s_std = s.std(ddof=1)
            boot[i] = float(s.mean() / s_std) if s_std > 1e-10 else 0.0

        ci_low = float(np.percentile(boot, _ALPHA / 2 * 100))
        ci_high = float(np.percentile(boot, (1.0 - _ALPHA / 2) * 100))
        p_value = float((boot <= 0).mean())
        return sharpe, ci_low, ci_high, p_value

    def _wilcoxon(
        self, returns: np.ndarray
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Wilcoxon signed-rank test: H₀ = returns symmetric around zero.

        :return: (statistic, p_value) or (None, None) if scipy unavailable
                 or too few non-zero observations.
        """
        try:
            from scipy.stats import wilcoxon

            nonzero = returns[returns != 0]
            if len(nonzero) < 5:
                return None, None
            stat, pval = wilcoxon(nonzero)
            return float(stat), float(pval)
        except Exception as exc:
            logger.debug("Wilcoxon failed: %s", exc)
            return None, None

    def test(
        self,
        trade_returns: List[float],
        name: str = "strategy",
    ) -> AlphaResult:
        """
        Run bootstrap Sharpe + Wilcoxon on per-trade returns.

        :param trade_returns: Fractional return per closed trade (e.g. 0.012).
        :param name: Strategy name used in AlphaResult.summary().
        :return: AlphaResult with all statistics populated.
        """
        arr = np.array(trade_returns, dtype=float)
        n = len(arr)

        if n < 5:
            return AlphaResult(
                name=name,
                n_trades=n,
                sharpe=0.0,
                sharpe_ci_low=0.0,
                sharpe_ci_high=0.0,
                sharpe_pvalue=1.0,
                wilcoxon_stat=None,
                wilcoxon_pvalue=None,
                mean_return=float(arr.mean()) if n > 0 else 0.0,
                std_return=float(arr.std(ddof=1)) if n > 1 else 0.0,
            )

        sharpe, ci_low, ci_high, sp = self._bootstrap_sharpe(arr)
        w_stat, w_p = self._wilcoxon(arr)

        return AlphaResult(
            name=name,
            n_trades=n,
            sharpe=sharpe,
            sharpe_ci_low=ci_low,
            sharpe_ci_high=ci_high,
            sharpe_pvalue=sp,
            wilcoxon_stat=w_stat,
            wilcoxon_pvalue=w_p,
            mean_return=float(arr.mean()),
            std_return=float(arr.std(ddof=1)),
        )
