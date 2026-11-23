"""Дрейф, OOD и conformal-интервалы (Phase 8, MASTER_SPEC §11 п.13–14).

Это ИЗМЕРЕНИЯ. Решение «запретить входы / снизить риск» принимает Risk Engine
(Phase 9); при OOD закрытие позиций автоматически не выполняется.
Conformal-интервал не обещает покрытие на нестационарных данных: используйте
`empirical_coverage`, чтобы мерить фактическое.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Общепринятые ориентиры PSI (<0.1 стабильно, 0.1–0.25 умеренно, >0.25 сильно) —
# эвристика, не статистический тест.
PSI_MODERATE, PSI_STRONG = 0.10, 0.25


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index; границы корзин — квантили эталона."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def ks(expected: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
    """Двухвыборочный Колмогоров—Смирнов: (статистика, p-value)."""
    a, b = np.sort(np.asarray(expected, float)), np.sort(np.asarray(actual, float))
    grid = np.concatenate([a, b])
    d = float(
        np.max(np.abs(np.searchsorted(a, grid, "right") / len(a) - np.searchsorted(b, grid, "right") / len(b)))
    )
    en = np.sqrt(len(a) * len(b) / (len(a) + len(b)))
    lam = (en + 0.12 + 0.11 / en) * d  # асимптотика Колмогорова с поправкой Стивенса
    k = np.arange(1, 101)
    p = 2 * np.sum((-1.0) ** (k - 1) * np.exp(-2 * (k * lam) ** 2))
    return d, float(min(max(p, 0.0), 1.0))


class MahalanobisOOD:
    """Расстояние Махаланобиса до обучающего облака; порог — квантиль на обучении."""

    def __init__(self, quantile: float = 0.99) -> None:
        self.quantile = quantile

    def fit(self, X: np.ndarray) -> "MahalanobisOOD":
        X = np.asarray(X, float)
        self.mean = X.mean(axis=0)
        self.prec = np.linalg.pinv(np.atleast_2d(np.cov(X, rowvar=False)))
        self.threshold = float(np.quantile(self.score(X), self.quantile))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        d = np.asarray(X, float) - self.mean
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, self.prec, d), 0.0))

    def is_ood(self, X: np.ndarray) -> np.ndarray:
        return self.score(X) > self.threshold


def cusum(x: np.ndarray, target: float, k: float, h: float) -> Optional[int]:
    """Двусторонний CUSUM по потоку (например, ошибок модели): индекс первой тревоги.

    Только при наличии меток/исходов (concept drift). `k` — допуск, `h` — порог.
    """
    pos = neg = 0.0
    for i, v in enumerate(np.asarray(x, float)):
        pos = max(0.0, pos + v - target - k)
        neg = max(0.0, neg - v + target - k)
        if pos > h or neg > h:
            return i
    return None


def conformal_halfwidth(cal_residuals: np.ndarray, alpha: float = 0.1) -> float:
    """Split conformal: полуширина |остаток| на калибровочной выборке (уровень 1−α)."""
    r = np.sort(np.abs(np.asarray(cal_residuals, float)))
    n = len(r)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float("inf") if k > n else float(r[k - 1])


def empirical_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Фактическая доля попаданий: то, что нужно сравнивать с номиналом."""
    y = np.asarray(y, float)
    return float(np.mean((y >= np.asarray(lo, float)) & (y <= np.asarray(hi, float))))
