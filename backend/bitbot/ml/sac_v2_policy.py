"""Инференс политики SAC v2: целевая экспозиция по хвосту свечей.

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

from bitbot.ml.features_v2 import FEATURE_NAMES, latest_features


def exposure_to_signal(target: float, current: float, band: float = 0.15) -> str:
    """Решение по разнице целевой и текущей экспозиции; внутри полосы — hold (предложение: 0.15)."""
    delta = target - current
    if abs(delta) < band:
        return "hold"
    return "buy" if delta > 0 else "sell"


class SACV2Signal:
    """Замена `SACSignal` для `SignalCombiner` (`SAC_V2_ENABLED=true`): тот же интерфейс.

    Целевая экспозиция политики превращается в buy/sell/hold через `exposure_to_signal`;
    confidence = |цель − текущая| (величина действия, не вероятность). Допуск — как у v1:
    без записи в реестре с нужной стадией модель не торгует (`tradable`). Свечи и позицию
    поставляют провайдеры (комбинатор и бот); нет данных — hold.
    """

    needs_candles = True

    def __init__(self, model_path: Optional[str] = None) -> None:
        from bitbot.config.settings import Config  # noqa: PLC0415

        self._cfg = Config
        self.model_path = model_path or Config.SAC_V2_MODEL_PATH
        self.policy = SACV2Policy(self.model_path)
        self._mtime = self._file_mtime()
        self.candles_provider: Optional[Any] = None  # symbol -> DataFrame
        self.position_provider: Optional[Any] = None  # symbol -> (exposure, equity_pnl)

    def _file_mtime(self) -> float:
        try:
            return os.path.getmtime(self.model_path)
        except OSError:
            return 0.0

    @property
    def loaded(self) -> bool:
        return self.policy.loaded

    @property
    def stage(self) -> str:
        from bitbot.ml.model_registry import (
            ModelRegistry,
            default_registry_path,
        )  # noqa: PLC0415

        return ModelRegistry(default_registry_path(self.model_path)).stage_of_path(
            self.model_path
        )

    @property
    def approved(self) -> bool:
        if not getattr(self._cfg, "SAC_REQUIRE_APPROVAL", True):
            return True
        from bitbot.ml.model_registry import EXECUTABLE  # noqa: PLC0415

        return self.stage in EXECUTABLE["paper" if self._cfg.PAPER_TRADING else "live"]

    @property
    def tradable(self) -> bool:
        return bool(self.loaded and self.approved)

    def get_signal(
        self,
        snap: Dict[str, Any],
        balance: float,
        position: float = 0.0,
        entry_price: float = 0.0,
    ) -> Dict[str, Any]:
        sym = snap.get("symbol", "")
        hold = {"symbol": sym, "action": "hold", "confidence": 0.0, "source": "sac"}
        if not self.loaded or self.candles_provider is None:
            return hold
        try:
            df = self.candles_provider(sym)
            if df is None or len(df) == 0:
                return hold
            exposure, pnl = (
                self.position_provider(sym) if self.position_provider else (0.0, 0.0)
            )
            target = self.policy.target_exposure(df, exposure, pnl)
            if target is None:
                return hold
            action = exposure_to_signal(target, exposure)
            if action == "hold":
                return hold
            return {
                "symbol": sym,
                "action": action,
                "confidence": round(min(1.0, abs(target - exposure)), 3),
                "source": "sac",
                "target_exposure": round(target, 3),
            }
        except (
            Exception
        ):  # noqa: BLE001 — сбой инференса = нет сигнала, не исключение в цикле
            return hold

    def reload_if_updated(self) -> bool:
        """Горячая перезагрузка, если тренер сохранил новую версию файла модели."""
        mtime = self._file_mtime()
        if mtime and mtime > self._mtime:
            self.policy = SACV2Policy(self.model_path)
            self._mtime = mtime
            return True
        return False


class SACV2Policy:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.meta: Dict[str, Any] = {}
        self.stats: Dict[str, Dict[str, float]] = {}
        self._model: Any = None
        self.loaded = False
        self._load()

    def _load(self) -> None:
        base = (
            self.model_path[:-4]
            if self.model_path.endswith(".zip")
            else self.model_path
        )
        try:
            with open(base + ".meta.json", encoding="utf-8") as f:
                self.meta = json.load(f)
            with open(base + "_norm_stats.json", encoding="utf-8") as f:
                self.stats = json.load(f)
            if (
                self.meta.get("env") != "v2"
                or self.meta.get("feature_names") != FEATURE_NAMES
            ):
                return  # модель обучена на другом наборе признаков — подавать нечего
            if not os.path.exists(self.model_path):
                return
            from stable_baselines3 import SAC  # noqa: PLC0415

            self._model = SAC.load(self.model_path, device="cpu")
            self.loaded = True
        except (OSError, ValueError, ImportError):
            self.loaded = False

    def observation(
        self, df_tail: pd.DataFrame, exposure: float, equity_pnl: float
    ) -> Optional[np.ndarray]:
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
