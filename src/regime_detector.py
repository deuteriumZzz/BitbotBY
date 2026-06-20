"""
Определение режима рынка через Gaussian HMM (hmmlearn).

3 скрытых состояния, размечаются по среднему лог-доходности:
  - trending_up   (наибольшая средняя доходность)
  - ranging       (доходность близка к нулю)
  - trending_down (наименьшая средняя доходность)

Использование:
    detector = RegimeDetector()
    detector.fit(df)                   # обучить на OHLCV DataFrame
    regime = detector.predict(df)      # → "trending_up" | "ranging" | "trending_down"
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_N_STATES = 3
_MIN_ROWS = 100


class RegimeDetector:
    """
    Определяет режим рынка через GaussianHMM на лог-доходности и реализованной волатильности.

    Состояния размечаются после обучения по среднему лог-доходности:
    наибольшее — "trending_up", наименьшее — "trending_down",
    среднее — "ranging".
    """

    def __init__(self, n_states: int = _N_STATES) -> None:
        """
        Инициализирует детектор режима.

        :param n_states: Количество скрытых состояний HMM (по умолчанию 3).
        """
        self.n_states = n_states
        self._model: Optional[object] = None
        self._state_labels: dict[int, str] = {}
        self._fitted = False

    def _build_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Строит матрицу признаков (N, 2): [log_return, realized_vol].

        :param df: DataFrame с колонкой 'close'.
        :return: Матрица признаков или None при нехватке данных.
        """
        if "close" not in df.columns or len(df) < _MIN_ROWS:
            return None
        close = df["close"].astype(float).values
        log_ret = np.log(close[1:] / np.maximum(close[:-1], 1e-10))
        rv = pd.Series(log_ret).rolling(20, min_periods=5).std().fillna(0).values
        return np.column_stack([log_ret, rv]).astype(np.float64)

    def fit(self, df: pd.DataFrame) -> bool:
        """
        Обучает GaussianHMM на OHLCV данных.

        :param df: DataFrame с колонкой 'close'.
        :return: True если обучение прошло успешно.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.warning("hmmlearn not installed — regime detection disabled")
            return False

        X = self._build_features(df)  # noqa: N806
        if X is None:
            logger.warning("Not enough data to fit RegimeDetector (%d rows)", len(df))
            return False

        try:
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=100,
                random_state=42,
            )
            model.fit(X)
            self._model = model

            means = model.means_[:, 0]  # first feature = log_return
            order = np.argsort(means)
            labels = ["trending_down", "ranging", "trending_up"]
            self._state_labels = {
                int(state): labels[i] for i, state in enumerate(order)
            }
            self._fitted = True
            logger.info("RegimeDetector fitted: %s", self._state_labels)
            return True
        except Exception as exc:
            logger.warning("RegimeDetector fit failed: %s", exc)
            return False

    def predict(self, df: pd.DataFrame) -> str:
        """
        Определяет текущий режим рынка по последнему окну df.

        :param df: DataFrame с колонкой 'close'.
        :return: Одно из "trending_up", "ranging", "trending_down" или "unknown".
        """
        if not self._fitted or self._model is None:
            return "unknown"

        X = self._build_features(df)  # noqa: N806
        if X is None or len(X) == 0:
            return "unknown"

        try:
            states = self._model.predict(X)  # type: ignore[attr-defined]
            return self._state_labels.get(int(states[-1]), "unknown")
        except Exception as exc:
            logger.warning("RegimeDetector predict failed: %s", exc)
            return "unknown"

    def regime_weights(self, regime: str) -> dict[str, float]:
        """
        Возвращает множители весов SAC/AI для заданного режима рынка.

        :param regime: "trending_up", "trending_down", "ranging" или "unknown".
        :return: Словарь с ключами "sac_weight", "ai_weight".
        """
        if regime == "trending_up":
            return {"sac_weight": 0.5, "ai_weight": 0.5}
        if regime == "trending_down":
            return {"sac_weight": 0.3, "ai_weight": 0.7}
        if regime == "ranging":
            return {"sac_weight": 0.4, "ai_weight": 0.6}
        return {"sac_weight": 0.4, "ai_weight": 0.6}
