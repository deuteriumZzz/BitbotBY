"""Ограничение нагрузки от тяжёлых внешних процессов (обучение, прогнозные модели).

Цель из исходного ТЗ: PyTorch не должен «съедать» машину и тормозить торговый
цикл. Применяется к процессам, которые бот запускает сам:
  * приоритет ниже боевого (`nice`, по умолчанию +10);
  * потолок потоков для BLAS/OpenMP/torch (по умолчанию половина ядер, но ≥ 1).
Параметры: `HEAVY_JOB_NICE`, `HEAVY_JOB_THREADS`. Это мягкая мера (CPU-приоритет и
потоки); жёсткий лимит памяти/CPU задаётся контейнером (docker `cpus`/`mem_limit`).
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional


def _threads() -> int:
    env = os.getenv("HEAVY_JOB_THREADS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, (os.cpu_count() or 2) // 2)


def nice_level() -> int:
    try:
        return max(0, min(19, int(os.getenv("HEAVY_JOB_NICE", "10"))))
    except ValueError:
        return 10


def limit_env(
    base: Optional[Dict[str, str]] = None, threads: Optional[int] = None
) -> Dict[str, str]:
    """Окружение процесса с потолком потоков (копия, исходный dict не меняется)."""
    n = str(threads if threads and threads > 0 else _threads())
    env = dict(base if base is not None else os.environ)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
        "TIMESFM_THREADS",
    ):
        env[key] = n
    return env


def lower_priority() -> Callable[[], None]:
    """`preexec_fn` для subprocess: понижает приоритет дочернего процесса."""
    level = nice_level()

    def _apply() -> None:
        try:
            os.nice(level)
        except OSError:  # нет прав/платформа без nice — не мешаем запуску
            pass

    return _apply
