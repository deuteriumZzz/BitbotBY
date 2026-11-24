"""Сравнение прогнозистов на общих сплитах (Phase 8, MASTER_SPEC §13.9).

Любой прогнозист — функция `f(history, horizon) -> {"median": [h], "lo": [h]?, "hi": [h]?}`.
Адаптеры Chronos/TimesFM подключаются к этому интерфейсу; здесь нет импорта
тяжёлых моделей. Одинаковые точки отсчёта (origins) и горизонты для всех
моделей, базисы — naive, seasonal naive, скользящее среднее. Метрики точности
— ещё не торговая ценность: она проверяется ablation'ом на исходах сделок.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Sequence

import numpy as np

Forecaster = Callable[[np.ndarray, int], Dict[str, np.ndarray]]


def naive(history: np.ndarray, h: int) -> Dict[str, np.ndarray]:
    return {"median": np.full(h, history[-1], float)}


def seasonal_naive(season: int) -> Forecaster:
    def f(history: np.ndarray, h: int) -> Dict[str, np.ndarray]:
        if len(history) < season:
            return naive(history, h)
        return {
            "median": np.array(
                [history[-season + (i % season)] for i in range(h)], float
            )
        }

    return f


def moving_average(window: int) -> Forecaster:
    def f(history: np.ndarray, h: int) -> Dict[str, np.ndarray]:
        return {"median": np.full(h, np.mean(history[-window:]), float)}

    return f


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mase(y: np.ndarray, yhat: np.ndarray, insample: np.ndarray, m: int = 1) -> float:
    """MAE, делённая на MAE наивного прогноза на обучающем ряду (<1 — лучше naive)."""
    scale = np.mean(np.abs(np.asarray(insample)[m:] - np.asarray(insample)[:-m]))
    return float(mae(y, yhat) / scale) if scale > 0 else float("nan")


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    d = np.asarray(y, float) - np.asarray(q, float)
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def directional_accuracy(y: np.ndarray, yhat: np.ndarray, last: np.ndarray) -> float:
    """Доля случаев, когда знак (прогноз − последнее наблюдение) совпал со знаком факта."""
    return float(
        np.mean(np.sign(np.asarray(yhat) - last) == np.sign(np.asarray(y) - last))
    )


def evaluate(
    models: Dict[str, Forecaster],
    series: np.ndarray,
    origins: Sequence[int],
    horizon: int,
    *,
    season: int = 1,
    alpha: float = 0.1,
) -> Dict[str, Dict[str, Any]]:
    """Скользящая оценка: для каждого origin прогноз на `horizon` шагов вперёд.

    Метрики считаются по всем (origin, шаг) парам; `coverage` и `pinball` — только
    у моделей, вернувших интервалы lo/hi (уровень 1−α, tau=α/2 и 1−α/2).
    """
    series = np.asarray(series, float)
    out: Dict[str, Dict[str, Any]] = {}
    for name, f in models.items():
        ys, yh, last, lo, hi, ins = [], [], [], [], [], []
        for o in origins:
            if o + horizon > len(series) or o < 2:
                raise ValueError("origin вне ряда")
            hist = series[:o]  # только прошлое: look-ahead исключён
            res = f(hist, horizon)
            ys.append(series[o : o + horizon])
            yh.append(np.asarray(res["median"], float))
            last.append(np.full(horizon, hist[-1]))
            ins.append(hist)
            if "lo" in res and "hi" in res:
                lo.append(np.asarray(res["lo"], float))
                hi.append(np.asarray(res["hi"], float))
        y, p, l = map(np.concatenate, (ys, yh, last))
        m = {
            "n": len(y),
            "mae": mae(y, p),
            "rmse": rmse(y, p),
            "mase": float(
                np.mean([mase(a, b, c, season) for a, b, c in zip(ys, yh, ins)])
            ),
            "directional_accuracy": directional_accuracy(y, p, l),
        }
        if lo and len(lo) == len(ys):
            L, H = np.concatenate(lo), np.concatenate(hi)
            m["coverage"] = float(np.mean((y >= L) & (y <= H)))
            m["nominal_coverage"] = 1 - alpha
            m["pinball"] = (pinball(y, L, alpha / 2) + pinball(y, H, 1 - alpha / 2)) / 2
        out[name] = m
    return out
