"""Метрики риска (Phase 9): CVaR, бета к BTC, стресс-портфель.

Только numpy. Историческая оценка CVaR занижает хвост на коротком окне и в
спокойном режиме — поэтому для лимитов используется ещё и стресс-оценка
«корреляции → 1» (верхняя граница), а динамический CVaR (implied vol /
квантильный прогноз, MASTER_SPEC §11 п.18) остаётся задачей на данных.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def hist_var_cvar(returns: np.ndarray, alpha: float = 0.95) -> Tuple[float, float]:
    """(VaR, CVaR) как положительные доли потерь на уровне alpha по историческим доходностям."""
    r = np.asarray(returns, float)
    if len(r) < 2:
        raise ValueError("мало наблюдений")
    losses = -r
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    return max(var, 0.0), max(float(tail.mean()), 0.0)


def beta(asset: np.ndarray, market: np.ndarray) -> float:
    """β = cov(asset, market) / var(market). Ряды должны быть выровнены по времени."""
    a, m = np.asarray(asset, float), np.asarray(market, float)
    if len(a) != len(m) or len(a) < 3:
        raise ValueError("ряды разной длины или короткие")
    var = float(np.var(m, ddof=1))
    return float(np.cov(a, m, ddof=1)[0, 1] / var) if var > 0 else 0.0


def portfolio_cvar(returns: np.ndarray, notionals: Sequence[float], alpha: float = 0.95) -> float:
    """CVaR портфеля историческим сценарным методом, в деньгах.

    returns — матрица [время × актив]; notionals — знаковые позиции (минус = шорт).
    """
    R, n = np.asarray(returns, float), np.asarray(notionals, float)
    if R.ndim != 2 or R.shape[1] != len(n):
        raise ValueError("форма returns не совпадает с позициями")
    pnl = R @ n  # сценарные P&L в деньгах
    losses = -pnl
    var = np.quantile(losses, alpha)
    return max(float(losses[losses >= var].mean()), 0.0)


def stress_cvar(notionals: Sequence[float], asset_cvars: Sequence[float]) -> float:
    """Верхняя граница при корреляции → 1 и одновременном ударе: Σ|n_i|·CVaR_i.

    Хеджи (противоположные знаки) не учитываются — оценка консервативна.
    """
    n, c = np.abs(np.asarray(notionals, float)), np.asarray(asset_cvars, float)
    if len(n) != len(c):
        raise ValueError("длины не совпадают")
    return float(np.sum(n * c))


def net_beta_exposure(notionals: Sequence[float], betas: Sequence[float]) -> float:
    """Суммарная знаковая BTC-бета-экспозиция в деньгах: Σ n_i·β_i."""
    return float(np.sum(np.asarray(notionals, float) * np.asarray(betas, float)))
