"""Meta-Model (Phase 8, MASTER_SPEC §11 п.10): вероятность успеха сигнала + калибровка.

Модель принимает признаки сигнала (Local/SAC/волатильность/стакан и т.п.) и
оценивает вероятность того, что сделка закроется в плюс. По умолчанию —
логистическая регрессия на numpy: без новых зависимостей. LightGBM/CatBoost
подключаются через `estimator_factory=` (адаптер `lightgbm_estimator`), но это
новая зависимость → ADR до включения в requirements.

Правила: обучение на train, калибровка на val, оценка на test (один раз).
`usable` истинно, только если калиброванная вероятность не хуже сырой по Brier
на тесте, лучше константы «средняя частота успеха» и калибровочная выборка
достаточна. Подключение к размеру позиции — не здесь (Phase 9) и только при `usable`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np

from bitbot.ai.reliability.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    brier,
    ece,
)

MIN_CALIBRATION_SAMPLES = 200  # предложение; ниже — изотоника переобучается
BRIER_TOLERANCE = 0.002  # предложение: разница на уровне шума не считается ухудшением


class LogisticEstimator:
    """L2-логистическая регрессия, градиентный спуск; признаки стандартизуются по train."""

    def __init__(self, l2: float = 1.0, iters: int = 500, lr: float = 0.1) -> None:
        self.l2, self.iters, self.lr = l2, iters, lr

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticEstimator":
        X = np.asarray(X, float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z, y = (X - self.mu) / self.sd, np.asarray(y, float)
        w, b = np.zeros(Z.shape[1]), 0.0
        for _ in range(self.iters):
            p = 1 / (1 + np.exp(-(Z @ w + b)))
            w -= self.lr * (Z.T @ (p - y) / len(y) + self.l2 * w / len(y))
            b -= self.lr * np.mean(p - y)
        self.w, self.b = w, b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, float) - self.mu) / self.sd
        return 1 / (1 + np.exp(-(Z @ self.w + self.b)))


def lightgbm_estimator(**params: Any) -> Any:
    """Адаптер LightGBM с тем же интерфейсом; зависимость необязательна."""
    try:
        import lightgbm as lgb  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "lightgbm не установлен; новая зависимость требует ADR"
        ) from e

    class _LGB:
        def fit(self, X, y):
            self.m = lgb.LGBMClassifier(verbose=-1, **params).fit(X, y)
            return self

        def predict_proba(self, X):
            return self.m.predict_proba(X)[:, 1]

    return _LGB()


class MetaModel:
    def __init__(
        self,
        estimator_factory: Callable[[], Any] = LogisticEstimator,
        method: str = "isotonic",
    ) -> None:
        if method not in ("isotonic", "platt"):
            raise ValueError("method: isotonic | platt")
        self.est = estimator_factory()
        self.cal: Any = (
            IsotonicCalibrator() if method == "isotonic" else PlattCalibrator()
        )
        self.method = method
        self.fitted = False
        self.n_cal = 0
        self.report: Dict[str, Any] = {}

    def fit(self, Xtr, ytr, Xval, yval) -> "MetaModel":
        self.est.fit(Xtr, ytr)
        self.cal.fit(self.est.predict_proba(Xval), yval)
        self.n_cal = len(yval)
        self.fitted = True
        return self

    def predict_proba(self, X: np.ndarray, calibrated: bool = True) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("модель не обучена")
        raw = self.est.predict_proba(X)
        return self.cal.predict(raw) if calibrated else raw

    def evaluate(self, Xte, yte, bins: int = 10) -> Dict[str, Any]:
        """Оценка на тесте: сырые и калиброванные Brier/ECE и вердикт `usable`."""
        yte = np.asarray(yte, float)
        raw, cal = self.predict_proba(Xte, False), self.predict_proba(Xte, True)
        base = brier(np.full(len(yte), yte.mean()), yte)  # константа = частота успеха
        r = {
            "n_test": len(yte),
            "brier_raw": brier(raw, yte),
            "brier_calibrated": brier(cal, yte),
            "brier_base_rate": base,
            "ece_raw": ece(raw, yte, bins),
            "ece_calibrated": ece(cal, yte, bins),
        }
        r["usable"] = bool(
            self.n_cal >= MIN_CALIBRATION_SAMPLES
            and r["brier_calibrated"] <= r["brier_raw"] + BRIER_TOLERANCE
            and r["brier_calibrated"]
            < base  # хуже «всегда средняя частота» — модель бесполезна
        )
        self.report = r
        return r
