"""
Оптимизация портфеля: Markowitz (макс. Sharpe) и CVaR (LP по Rockafellar-Uryasev).

Использование:
    opt = PortfolioOptimizer()
    weights = opt.allocate(symbols, returns_df, method="cvar")
    # weights → {"BTC/USDT": 0.45, "ETH/USDT": 0.35, "SOL/USDT": 0.20}
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MIN_WEIGHT = 0.05  # no asset gets less than 5%
_MAX_WEIGHT = 0.60  # single-asset cap
_RISK_FREE = 0.0  # 0% risk-free rate for crypto
_CVAR_ALPHA = 0.05  # 95% CVaR (worst 5% of scenarios)


class PortfolioOptimizer:
    """
    Оптимизация портфеля методами mean-variance и CVaR.

    Оба метода возвращают веса, сумма которых равна 1, каждый в диапазоне
    [_MIN_WEIGHT, _MAX_WEIGHT]. При менее чем 2 активах или отсутствии
    scipy — возвращает равные веса.
    """

    def _equal_weights(self, symbols: List[str]) -> Dict[str, float]:
        w = round(1.0 / len(symbols), 6)
        return {s: w for s in symbols}

    def optimize_markowitz(self, returns_df: pd.DataFrame) -> Dict[str, float]:
        """
        Портфель с максимальным коэффициентом Sharpe (граница эффективности).

        Решает: max (μ·w − r_f) / √(w·Σ·w)
        через scipy.optimize.minimize (SLSQP) при sum(w)=1 и ограничениях на веса.

        :param returns_df: Доходности за период, одна колонка на актив.
        :return: Словарь весов {symbol: weight}.
        """
        symbols = list(returns_df.columns)
        n = len(symbols)
        if n < 2:
            return self._equal_weights(symbols)

        try:
            from scipy.optimize import minimize

            mu = returns_df.mean().values
            cov = returns_df.cov().values
            w0 = np.full(n, 1.0 / n)

            def neg_sharpe(w: np.ndarray) -> float:
                ret = mu @ w
                vol = np.sqrt(w @ cov @ w + 1e-12)
                return -(ret - _RISK_FREE) / vol

            result = minimize(
                neg_sharpe,
                w0,
                method="SLSQP",
                bounds=[(_MIN_WEIGHT, _MAX_WEIGHT)] * n,
                constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                options={"maxiter": 500, "ftol": 1e-9},
            )
            if not result.success:
                logger.warning("Markowitz did not converge: %s", result.message)
                return self._equal_weights(symbols)

            w = np.clip(result.x, _MIN_WEIGHT, _MAX_WEIGHT)
            w /= w.sum()
            return {s: round(float(wi), 6) for s, wi in zip(symbols, w)}

        except Exception as exc:
            logger.warning("Markowitz failed (%s) — equal weights", exc)
            return self._equal_weights(symbols)

    def optimize_cvar(
        self,
        returns_df: pd.DataFrame,
        alpha: float = _CVAR_ALPHA,
    ) -> Dict[str, float]:
        """
        Портфель с минимальным CVaR (LP-формулировка Rockafellar-Uryasev 2000).

        Переменные: [w_1…w_N, VaR, u_1…u_T]
        Минимизировать: VaR + (1/((1−α)·T)) · Σu_t
        При ограничениях:
          −R_t·w − VaR − u_t ≤ 0   для всех t  (граница потерь)
          u_t ≥ 0,  Σw = 1,  _MIN_WEIGHT ≤ w_i ≤ _MAX_WEIGHT

        :param returns_df: DataFrame доходностей за период.
        :param alpha: Вероятность хвоста (0.05 → 95% доверительный CVaR).
        :return: Словарь весов {symbol: weight}.
        """
        symbols = list(returns_df.columns)
        n = len(symbols)
        T = len(returns_df)  # noqa: N806
        if n < 2 or T < 10:
            return self._equal_weights(symbols)

        try:
            from scipy.optimize import linprog

            R = returns_df.values  # noqa: N806  # (T, N)
            total_vars = n + 1 + T

            # Objective: 0·w + 1·VaR + (1/((1-α)·T))·u
            c = np.zeros(total_vars)
            c[n] = 1.0
            c[n + 1 :] = 1.0 / ((1.0 - alpha) * T)

            # Inequality: −R_t·w − VaR − u_t ≤ 0
            A_ub = np.zeros((T, total_vars))  # noqa: N806
            A_ub[:, :n] = -R
            A_ub[:, n] = -1.0
            A_ub[np.arange(T), n + 1 + np.arange(T)] = -1.0

            # Equality: Σw = 1
            A_eq = np.zeros((1, total_vars))  # noqa: N806
            A_eq[0, :n] = 1.0

            bounds = (
                [(_MIN_WEIGHT, _MAX_WEIGHT)] * n + [(None, None)] + [(0.0, None)] * T
            )

            result = linprog(
                c,
                A_ub=A_ub,
                b_ub=np.zeros(T),
                A_eq=A_eq,
                b_eq=np.array([1.0]),
                bounds=bounds,
                method="highs",
            )

            if result.status != 0:
                logger.warning("CVaR LP status %d — equal weights", result.status)
                return self._equal_weights(symbols)

            w = np.clip(result.x[:n], _MIN_WEIGHT, _MAX_WEIGHT)
            w /= w.sum()
            return {s: round(float(wi), 6) for s, wi in zip(symbols, w)}

        except Exception as exc:
            logger.warning("CVaR failed (%s) — equal weights", exc)
            return self._equal_weights(symbols)

    def allocate(
        self,
        symbols: List[str],
        returns_df: pd.DataFrame,
        method: str = "cvar",
    ) -> Dict[str, float]:
        """
        Вычисляет оптимальные веса портфеля для заданных символов.

        :param symbols: Символы активов, соответствующие колонкам returns_df.
        :param returns_df: Доходности за период, колонки = символы.
        :param method: "cvar" (по умолчанию) или "markowitz".
        :return: Словарь весов с суммой 1.0.
        """
        cols = [s for s in symbols if s in returns_df.columns]
        if len(cols) < 2:
            return self._equal_weights(symbols)

        sub = returns_df[cols].dropna()
        if len(sub) < 10:
            logger.warning("Too few observations (%d) — equal weights", len(sub))
            return self._equal_weights(cols)

        weights = (
            self.optimize_markowitz(sub)
            if method == "markowitz"
            else self.optimize_cvar(sub)
        )
        logger.info(
            "Portfolio [%s]: %s",
            method,
            {k: f"{v:.1%}" for k, v in weights.items()},
        )
        return weights
