"""
Проверка статистической значимости альфы стратегий.

Два теста на каждую стратегию:
  1. Bootstrap Sharpe — 1 000 ресемплов → точечная оценка + 95% CI + p-value
     (доля bootstrap-Sharpe ≤ 0; устойчив к ненормальности распределения)
  2. Wilcoxon signed-rank — непараметрический тест H₀: медиана доходности = 0
     (предпочтительнее t-теста для крипто-доходностей с тяжёлыми хвостами)

Использование:
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
    """Результаты статистических тестов для одной стратегии."""

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
        Возвращает True если альфа статистически значима.

        p-value bootstrap Sharpe < 5%.
        Если scipy доступен — Wilcoxon тоже должен пройти.
        Если scipy недоступен (wilcoxon_pvalue=None), решает только bootstrap.
        """
        if self.sharpe_pvalue >= _ALPHA:
            return False
        if self.wilcoxon_pvalue is None:
            return True  # scipy unavailable — bootstrap Sharpe is sufficient
        return self.wilcoxon_pvalue < _ALPHA

    @property
    def verdict(self) -> str:
        """Возвращает текстовый вердикт о значимости альфы стратегии."""
        if self.n_trades < 10:
            return "INSUFFICIENT DATA"
        if self.is_significant:
            return "SIGNIFICANT"
        if self.sharpe_pvalue < _ALPHA:
            return "WEAK  (Sharpe ok, Wilcoxon fails)"
        return "NOT SIGNIFICANT"

    def summary(self) -> str:
        """Возвращает форматированную строку с результатами тестов."""
        wp = (
            f"{self.wilcoxon_pvalue:.4f}" if self.wilcoxon_pvalue is not None else "n/a"
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
    """Bootstrap Sharpe и Wilcoxon signed-rank тесты на доходностях отдельных сделок."""

    def _bootstrap_sharpe(
        self,
        returns: np.ndarray,
        n_bootstrap: int = _N_BOOTSTRAP,
    ) -> tuple[float, float, float, float]:
        """
        Ресемплирует доходности n_bootstrap раз и строит распределение Sharpe.

        :param returns: Массив доходностей сделок.
        :param n_bootstrap: Количество bootstrap-итераций.
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

    def _wilcoxon(self, returns: np.ndarray) -> tuple[Optional[float], Optional[float]]:
        """
        Критерий знаковых рангов Вилкоксона: H₀ = доходности симметричны вокруг нуля.

        :param returns: Массив доходностей сделок.
        :return: (statistic, p_value) или (None, None) если scipy недоступен
                 или слишком мало ненулевых наблюдений.
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
        Запускает bootstrap Sharpe и Wilcoxon на доходностях сделок.

        :param trade_returns: Дробная доходность каждой закрытой сделки
            (например, 0.012).
        :param name: Название стратегии для AlphaResult.summary().
        :return: AlphaResult со всеми заполненными статистиками.
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
