"""
Инференс SAC-сигналов (SB3) для торгового бота.

Конвертирует снэпшот рынка в вектор наблюдения (OBS_DIM=14),
выполняет детерминированный инференс SAC и возвращает
{action, confidence, source="sac"}.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import numpy as np

from config import Config
from reinforcement_learning.rl_env import HOLD_ZONE, OBS_DIM

logger = logging.getLogger(__name__)

# Колонки obs[0..10] — рыночные фичи (порядок совпадает с TradingEnv)
_MARKET_COLS = [
    "open", "high", "low", "close", "volume",
    "rsi", "macd", "macd_signal",
    "bb_upper", "bb_middle", "bb_lower",
]


def _snap_to_obs(
    snap: Dict[str, Any],
    balance: float,
    norm_stats: Optional[Dict[str, list]],
) -> np.ndarray:
    """
    Строит вектор наблюдения shape=(OBS_DIM,) из снэпшота MarketScanner.

    Структура obs совпадает с TradingEnv._get_observation():
        [open, high, low, close, volume, rsi, macd, macd_signal,
         bb_upper, bb_middle, bb_lower, balance, position, current_value]

    :param snap: Снэпшот из MarketScanner.build_snapshot().
    :param balance: Свободный баланс USDT.
    :param norm_stats: Словарь {col: [mean, std]} из train_sac или None.
    :return: float32-массив shape=(OBS_DIM,).
    """
    ind = snap.get("indicators", {})
    price = float(snap.get("price", 0.0))

    # MACD в снэпшоте может быть строкой ("bullish"/"bearish") или числом
    raw_macd = ind.get("macd", 0)
    if isinstance(raw_macd, str):
        macd_val = 1.0 if raw_macd == "bullish" else -1.0
    else:
        macd_val = float(raw_macd)

    bb_w = float(ind.get("bb_width", 0.04))
    bb_upper = price * (1.0 + bb_w / 2.0)
    bb_lower = price * (1.0 - bb_w / 2.0)

    raw = np.array([
        price,                                # open (≈ close при инференсе)
        price,                                # high
        price,                                # low
        price,                                # close
        float(snap.get("volume_ratio", 1.0)), # volume
        float(ind.get("rsi", 50.0)),          # rsi
        macd_val,                             # macd
        0.0,                                  # macd_signal (нет в снэпшоте)
        bb_upper,                             # bb_upper
        price,                                # bb_middle
        bb_lower,                             # bb_lower
        balance,                              # portfolio: balance
        0.0,                                  # portfolio: position
        balance,                              # portfolio: current_value
    ], dtype=np.float32)

    if norm_stats:
        for i, col in enumerate(_MARKET_COLS):
            if col in norm_stats:
                mu, sd = norm_stats[col]
                if sd > 0:
                    raw[i] = (raw[i] - mu) / sd

    return raw


class DQNSignal:
    """
    Инференс SAC-модели (SB3) для одного снэпшота рынка.

    Сохраняет имя класса DQNSignal для обратной совместимости
    с SignalCombiner. Загружает модель из Config.SAC_MODEL_PATH.
    При отсутствии файла модели возвращает hold с conf=0.
    """

    def __init__(self) -> None:
        """Инициализирует и пытается загрузить SAC-модель."""
        self.logger = logging.getLogger(__name__)
        self._model: Any = None
        self._norm_stats: Optional[Dict[str, list]] = None
        self.loaded = False
        self._try_load()

    def _try_load(self) -> None:
        """
        Загружает SAC-модель и norm_stats из файловой системы.

        Сначала проверяет SAC_MODEL_PATH, затем DQN_MODEL_PATH
        (обратная совместимость). Если ни один файл не найден —
        логирует предупреждение и оставляет loaded=False.
        """
        try:
            from stable_baselines3 import SAC  # noqa: PLC0415
        except ImportError:
            self.logger.error(
                "stable-baselines3 не установлен. "
                "pip install stable-baselines3"
            )
            return

        candidates = [
            getattr(Config, "SAC_MODEL_PATH", ""),
            getattr(Config, "DQN_MODEL_PATH", ""),
        ]
        path = next(
            (p for p in candidates if p and os.path.exists(p)),
            None,
        )
        if path is None:
            self.logger.warning(
                "SAC-модель не найдена. "
                "Запустите: python reinforcement_learning/train_sac.py"
            )
            return

        try:
            self._model = SAC.load(path)
            self.loaded = True
            self.logger.info(f"SAC загружен из {path}")
        except Exception as e:
            self.logger.warning(f"Ошибка загрузки SAC: {e}")
            return

        norm_path = path.replace(".zip", "_norm_stats.json")
        if os.path.exists(norm_path):
            try:
                with open(norm_path, encoding="utf-8") as f:
                    self._norm_stats = json.load(f)
                self.logger.info(
                    f"Norm stats загружены из {norm_path}"
                )
            except (OSError, json.JSONDecodeError) as e:
                self._norm_stats = None
                self.logger.warning(
                    f"Ошибка загрузки norm stats: {e}"
                )
        else:
            self.logger.warning(
                "Norm stats не найдены — инференс без нормализации"
            )

    def get_signal(
        self,
        snap: Dict[str, Any],
        balance: float,
    ) -> Dict[str, Any]:
        """
        Выполняет детерминированный инференс SAC для одного снэпшота.

        SAC возвращает непрерывное действие a ∈ [-1, 1]:
            a > HOLD_ZONE  → BUY,  confidence = a
            a < -HOLD_ZONE → SELL, confidence = |a|
            else           → HOLD, confidence = 0

        :param snap: Снэпшот из MarketScanner.build_snapshot().
        :param balance: Свободный баланс USDT.
        :return: Словарь {action, confidence, source="sac"}.
        """
        default: Dict[str, Any] = {
            "action": "hold",
            "confidence": 0.0,
            "source": "sac",
        }
        if not self.loaded or self._model is None:
            return default

        try:
            obs = _snap_to_obs(snap, balance, self._norm_stats)
            action, _ = self._model.predict(
                obs, deterministic=True
            )
            a = float(action[0])

            if a > HOLD_ZONE:
                return {
                    "symbol": snap.get("symbol", ""),
                    "action": "buy",
                    "confidence": round(a, 3),
                    "source": "sac",
                }
            if a < -HOLD_ZONE:
                return {
                    "symbol": snap.get("symbol", ""),
                    "action": "sell",
                    "confidence": round(abs(a), 3),
                    "source": "sac",
                }
            return default

        except Exception as e:
            self.logger.error(
                f"Ошибка инференса SAC: {e}", exc_info=True
            )
            return default
