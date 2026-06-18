"""
Market regime detection via Gaussian HMM (hmmlearn).

3 hidden states, labeled by mean log-return:
  - trending_up   (highest mean return)
  - ranging       (near-zero mean)
  - trending_down (lowest mean return)

Usage:
    detector = RegimeDetector()
    detector.fit(df)                   # train on OHLCV DataFrame
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
    Detects market regime using GaussianHMM on log-return + realized volatility.

    States are labeled after fitting by comparing state mean log-returns:
    the highest-mean state is "trending_up", lowest is "trending_down",
    middle is "ranging".
    """

    def __init__(self, n_states: int = _N_STATES) -> None:
        self.n_states = n_states
        self._model: Optional[object] = None
        self._state_labels: dict[int, str] = {}
        self._fitted = False

    def _build_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Return (N, 2) array of [log_return, realized_vol] or None."""
        if "close" not in df.columns or len(df) < _MIN_ROWS:
            return None
        close = df["close"].astype(float).values
        log_ret = np.log(close[1:] / np.maximum(close[:-1], 1e-10))
        rv = pd.Series(log_ret).rolling(20, min_periods=5).std().fillna(0).values
        return np.column_stack([log_ret, rv]).astype(np.float64)

    def fit(self, df: pd.DataFrame) -> bool:
        """
        Fit GaussianHMM to OHLCV data.

        :param df: DataFrame with 'close' column.
        :return: True if fitting succeeded.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.warning("hmmlearn not installed — regime detection disabled")
            return False

        X = self._build_features(df)
        if X is None:
            logger.warning(
                "Not enough data to fit RegimeDetector (%d rows)", len(df)
            )
            return False

        try:
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
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
        Predict current market regime from the most recent window of df.

        :param df: DataFrame with 'close' column.
        :return: One of "trending_up", "ranging", "trending_down", or "unknown".
        """
        if not self._fitted or self._model is None:
            return "unknown"

        X = self._build_features(df)
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
        Return SAC/AI weight multipliers for the given regime.

        :param regime: "trending_up", "trending_down", "ranging", or "unknown".
        :return: Dict with keys "sac_weight", "ai_weight".
        """
        if regime == "trending_up":
            return {"sac_weight": 0.5, "ai_weight": 0.5}
        if regime == "trending_down":
            return {"sac_weight": 0.3, "ai_weight": 0.7}
        if regime == "ranging":
            return {"sac_weight": 0.4, "ai_weight": 0.6}
        return {"sac_weight": 0.4, "ai_weight": 0.6}
