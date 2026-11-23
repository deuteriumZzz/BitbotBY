"""Инференс политики SAC v2: целевая экспозиция по хвосту свечей (Phase 7, закрывает C-3).

В отличие от `SACSignal` (22 признака, `position=0` при инференсе), v2 подаёт модели
ТО ЖЕ наблюдение, на котором она училась: признаки `features_v2` + текущая экспозиция +
накопленный результат. Нормализация — статистика train из `<model>_norm_stats.json`.

В боте НЕ подключена: решение «целевая экспозиция → buy/sell/hold» (`exposure_to_signal`)
и допуск модели через реестр — отдельный шаг. Модель из `train_sac_v2` — только
`candidate` и торговать не может (см. `model_registry`).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from reinforcement_learning.features_v2 import FEATURE_NAMES, latest_features


def exposure_to_signal(target: float, current: float, band: float = 0.15) -> str:
    """Решение по разнице целевой и текущей экспозиции; внутри полосы — hold (предложение: 0.15)."""
    delta = target - current
    if abs(delta) < band:
        return "hold"
    return "buy" if delta > 0 else "sell"


class SACV2Policy:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.meta: Dict[str, Any] = {}
        self.stats: Dict[str, Dict[str, float]] = {}
        self._model: Any = None
        self.loaded = False
        self._load()

    def _load(self) -> None:
        base = self.model_path[:-4] if self.model_path.endswith(".zip") else self.model_path
        try:
            with open(base + ".meta.json", encoding="utf-8") as f:
                self.meta = json.load(f)
            with open(base + "_norm_stats.json", encoding="utf-8") as f:
                self.stats = json.load(f)
            if self.meta.get("env") != "v2" or self.meta.get("feature_names") != FEATURE_NAMES:
                return  # модель обучена на другом наборе признаков — подавать нечего
            if not os.path.exists(self.model_path):
                return
            from stable_baselines3 import SAC  # noqa: PLC0415

            self._model = SAC.load(self.model_path, device="cpu")
            self.loaded = True
        except (OSError, ValueError, ImportError):
            self.loaded = False

    def observation(self, df_tail: pd.DataFrame, exposure: float, equity_pnl: float) -> Optional[np.ndarray]:
        feats = latest_features(df_tail)
        if feats is None:
            return None
        z = np.array(
            [
                (feats[i] - self.stats[n]["mean"]) / (self.stats[n]["std"] or 1.0)
                for i, n in enumerate(FEATURE_NAMES)
            ],
            dtype=np.float32,
        )
        return np.concatenate([z, np.array([exposure, equity_pnl], dtype=np.float32)])

    def target_exposure(
        self, df_tail: pd.DataFrame, exposure: float = 0.0, equity_pnl: float = 0.0
    ) -> Optional[float]:
        """Целевая экспозиция ∈ [-1, 1] или None (модель не загружена / мало истории)."""
        if not self.loaded:
            return None
        obs = self.observation(df_tail, exposure, equity_pnl)
        if obs is None:
            return None
        action, _ = self._model.predict(obs, deterministic=True)
        return float(np.clip(np.asarray(action).ravel()[0], -1.0, 1.0))
