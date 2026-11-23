"""Калибровка вероятностей (Phase 8): Brier, ECE, Platt, isotonic.

Калибратор обучается ТОЛЬКО на валидационной выборке, не на той, где учили
модель, и не на тестовой. Вероятность без проверенной калибровки нельзя
подавать в размер позиции (MASTER_SPEC §11 п.10).
"""

from __future__ import annotations

from typing import Dict

import numpy as np

_EPS = 1e-9


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> Dict[str, np.ndarray]:
    """Средняя предсказанная вероятность и доля успехов по равным корзинам."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    n = np.bincount(idx, minlength=bins)
    conf = np.bincount(idx, weights=p, minlength=bins) / np.maximum(n, 1)
    acc = np.bincount(idx, weights=y, minlength=bins) / np.maximum(n, 1)
    return {"n": n, "confidence": conf, "accuracy": acc}


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected Calibration Error: взвешенный разрыв «уверенность — частота»."""
    r = reliability(p, y, bins)
    return float(np.sum(r["n"] * np.abs(r["confidence"] - r["accuracy"])) / max(r["n"].sum(), 1))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """p' = σ(a·logit(p) + b), параметры — максимум правдоподобия."""

    def fit(self, p: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        x, y = _logit(np.asarray(p, float)), np.asarray(y, float)
        def nll(w: np.ndarray) -> float:
            z = w[0] * x + w[1]
            return float(np.mean(np.logaddexp(0, z) - y * z))

        w = np.array([1.0, 0.0])  # демпфированный Ньютон для 2 параметров; без scipy
        for _ in range(100):
            q = 1 / (1 + np.exp(-(w[0] * x + w[1])))
            g = np.array([np.mean((q - y) * x), np.mean(q - y)])
            s = q * (1 - q) + 1e-6
            H = np.array([[np.mean(s * x * x), np.mean(s * x)], [np.mean(s * x), np.mean(s)]])
            step = np.linalg.solve(H + 1e-9 * np.eye(2), g)
            t, f0 = 1.0, nll(w)
            while t > 1e-6 and nll(w - t * step) > f0:  # откат шага, пока правдоподобие не растёт
                t /= 2
            w = w - t * step
            if np.abs(t * step).max() < 1e-8:
                break
        self.a, self.b = w
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * _logit(np.asarray(p, float)) + self.b)))


class IsotonicCalibrator:
    """Монотонная ступенчатая калибровка (PAV). Нужна выборка побольше, чем Platt."""

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        p, y = np.asarray(p, float), np.asarray(y, float)
        order = np.argsort(p, kind="stable")
        xs, ys = p[order], y[order]
        blocks = [[v, 1.0] for v in ys]  # [сумма y, вес]
        starts = list(range(len(xs)))
        i = 0
        while i < len(blocks) - 1:
            if blocks[i][0] / blocks[i][1] > blocks[i + 1][0] / blocks[i + 1][1]:
                blocks[i][0] += blocks[i + 1][0]
                blocks[i][1] += blocks[i + 1][1]
                del blocks[i + 1], starts[i + 1]
                i = max(i - 1, 0)
            else:
                i += 1
        ends = starts[1:] + [len(xs)]
        self._x = np.array([xs[(s + e - 1) // 2] for s, e in zip(starts, ends)])
        self._y = np.array([b[0] / b[1] for b in blocks])
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(p, float), self._x, self._y)
