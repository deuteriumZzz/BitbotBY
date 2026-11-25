"""Оценка SAC: walk-forward с embargo, метрики, bootstrap, ворота допуска.

Все пороги в `PROMOTION_THRESHOLDS` — ПРЕДЛОЖЕНИЕ для утверждения владельцем.
Результат `promotion_verdict` — доказательство для реестра моделей
(`model_registry`, переход candidate → validated), а не разрешение торговать.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

PROMOTION_THRESHOLDS = {
    "min_folds": 3,
    "min_positive_fold_share": 0.6,  # доля фолдов, где модель бьёт лучший базис
    "min_ci_low_excess": 0.0,  # нижняя граница 95% CI среднего избытка над базисом
    "max_drawdown": 0.25,
}


def walk_forward_splits(
    n: int, n_folds: int, min_train: int, embargo: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Расширяющееся окно: train = [0, a), тест = [a+embargo, b). Embargo ≥ горизонта метки/решения."""
    if n_folds < 1 or min_train < 1 or embargo < 0:
        raise ValueError("некорректные параметры")
    test_len = (n - min_train) // n_folds
    if test_len <= embargo:
        raise ValueError("мало данных: тестовое окно не больше embargo")
    out = []
    for k in range(n_folds):
        a = min_train + k * test_len
        b = a + test_len
        out.append((np.arange(0, a), np.arange(a + embargo, b)))
    return out


def metrics(equity: np.ndarray, periods_per_year: float) -> Dict[str, float]:
    eq = np.asarray(equity, float)
    r = np.diff(eq) / eq[:-1]
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    peak = np.maximum.accumulate(eq)
    return {
        "total_return": float(eq[-1] / eq[0] - 1.0),
        "sharpe": float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0,
        "max_drawdown": float(((peak - eq) / peak).max()),
    }


def block_bootstrap_ci(
    x: np.ndarray,
    stat: Callable[[np.ndarray], float] = np.mean,
    *,
    n_boot: int = 2000,
    block: int = 24,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Доверительный интервал блочным bootstrap (учитывает автокорреляцию доходностей)."""
    x = np.asarray(x, float)
    n = len(x)
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)  # фиксированный seed: результат воспроизводим
    n_blocks = -(-n // block)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        sample = np.concatenate([x[s : s + block] for s in starts])[:n]
        stats[i] = stat(sample)
    return float(np.quantile(stats, alpha / 2)), float(
        np.quantile(stats, 1 - alpha / 2)
    )


def baseline_returns(close: np.ndarray) -> Dict[str, np.ndarray]:
    """Базисы: купи-и-держи и «вне рынка» (нулевая доходность)."""
    c = np.asarray(close, float)
    return {"buy_and_hold": np.diff(c) / c[:-1], "flat": np.zeros(len(c) - 1)}


def promotion_verdict(
    folds: List[Dict[str, Any]], thresholds: Dict[str, float] = PROMOTION_THRESHOLDS
) -> Dict[str, Any]:
    """folds[i] = {"model": доходности за бар, "baseline": лучший базис, "equity": капитал модели}.

    Доходности — ПОСЛЕ издержек. Возвращает `walk_forward` для реестра.
    """
    t = thresholds
    wins, excess, worst_dd = 0, [], 0.0
    for f in folds:
        m, b = np.asarray(f["model"], float), np.asarray(f["baseline"], float)
        wins += float(m.sum()) > float(b.sum())
        excess.append(m - b)
        eq = np.asarray(f["equity"], float)
        peak = np.maximum.accumulate(eq)
        worst_dd = max(worst_dd, float(((peak - eq) / peak).max()))
    ex = np.concatenate(excess) if excess else np.zeros(1)
    lo, hi = block_bootstrap_ci(ex)
    reasons = []
    if len(folds) < t["min_folds"]:
        reasons.append("folds")
    if folds and wins / len(folds) < t["min_positive_fold_share"]:
        reasons.append("fold_share")
    if lo <= t["min_ci_low_excess"]:
        reasons.append("ci_low_excess")
    if worst_dd > t["max_drawdown"]:
        reasons.append("max_drawdown")
    return {
        "passed": not reasons,
        "n_folds": len(folds),
        "fold_wins": int(wins),
        "excess_ci95": [lo, hi],
        "worst_drawdown": worst_dd,
        "failed": reasons,
        "thresholds": dict(t),
    }
