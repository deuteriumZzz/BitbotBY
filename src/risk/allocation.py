"""Распределение капитала между стратегиями (Phase 9, MASTER_SPEC §11 п.22).

Экспоненциальное взвешивание по скорингу стратегии с НИЖНИМ порогом веса и
ОГРАНИЧЕНИЕМ СКОРОСТИ перелива (сумма |Δвес| за шаг ≤ max_step). Перераспределение
включается только при `oos_validated=True` (скоринг получен вне выборки, на
которой подбирались стратегии); иначе возвращаются прежние веса. Thompson
Sampling не реализован: по ТЗ — только после OOS-подтверждения.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple


def thompson_weights(
    stats: Dict[str, Tuple[int, int]],
    prev: Dict[str, float],
    *,
    floor: float = 0.05,
    max_step: float = 0.10,
    draws: int = 4000,
    min_trials: int = 30,
    seed: int = 0,
    oos_validated: bool = False,
) -> Dict[str, float]:
    """Thompson Sampling по исходам стратегий: вес ~ вероятность быть лучшей (Beta(1+побед, 1+поражений)).

    Как и `exp_weights`: только при `oos_validated` (исходы вне выборки подбора стратегий), с нижним
    порогом веса и ограничением скорости перелива. Мало данных у ЛЮБОЙ стратегии (< `min_trials`
    исходов) — прежние веса: выборка из почти равномерных Beta только добавила бы шум.
    Детерминирован при фиксированном `seed`. Подключения к бою нет.
    """
    import numpy as np  # noqa: PLC0415

    names = list(prev)
    if set(stats) != set(prev):
        raise ValueError("stats и prev должны покрывать одни и те же стратегии")
    if floor * len(names) > 1 or abs(sum(prev.values()) - 1.0) > 1e-9:
        raise ValueError("некорректные веса или floor")
    if not oos_validated or any(w + l < min_trials for w, l in stats.values()):
        return dict(prev)
    rng = np.random.default_rng(seed)
    samples = np.column_stack(
        [rng.beta(1 + stats[k][0], 1 + stats[k][1], draws) for k in names]
    )
    p_best = np.bincount(samples.argmax(axis=1), minlength=len(names)) / draws
    free = 1.0 - floor * len(names)
    target = {k: floor + free * float(p_best[i]) for i, k in enumerate(names)}
    moved = sum(abs(target[k] - prev[k]) for k in names)
    t = 1.0 if moved <= max_step else max_step / moved
    out = {k: prev[k] + t * (target[k] - prev[k]) for k in names}
    s = sum(out.values())
    return {k: v / s for k, v in out.items()}


def exp_weights(
    scores: Dict[str, float],
    prev: Dict[str, float],
    *,
    eta: float = 1.0,
    floor: float = 0.05,
    max_step: float = 0.10,
    oos_validated: bool = False,
) -> Dict[str, float]:
    """Новые веса стратегий (сумма = 1). Параметры по умолчанию — предложения."""
    names = list(prev)
    if set(scores) != set(prev):
        raise ValueError("scores и prev должны покрывать одни и те же стратегии")
    if floor * len(names) > 1:
        raise ValueError("floor × число стратегий > 1")
    if abs(sum(prev.values()) - 1.0) > 1e-9:
        raise ValueError("прежние веса должны суммироваться в 1")
    if not oos_validated:
        return dict(prev)

    m = max(scores.values())
    raw = {k: math.exp(eta * (scores[k] - m)) for k in names}
    z = sum(raw.values())
    free = 1.0 - floor * len(names)
    target = {k: floor + free * raw[k] / z for k in names}

    moved = sum(abs(target[k] - prev[k]) for k in names)
    t = (
        1.0 if moved <= max_step else max_step / moved
    )  # масштабируем шаг, а не обрезаем веса
    out = {k: prev[k] + t * (target[k] - prev[k]) for k in names}
    s = sum(out.values())
    return {k: v / s for k, v in out.items()}
