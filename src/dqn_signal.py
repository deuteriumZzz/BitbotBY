"""
Инференс SAC-сигналов (SB3) для торгового бота.

Конвертирует снэпшот рынка в вектор наблюдения (OBS_DIM=21),
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
from reinforcement_learning.rl_env import HOLD_ZONE

logger = logging.getLogger(__name__)

# Колонки obs[0..10] — рыночные фичи (порядок совпадает с TradingEnv)
_MARKET_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "rsi",
    "macd",
    "macd_signal",
    "bb_upper",
    "bb_middle",
    "bb_lower",
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
         bb_upper, bb_middle, bb_lower, balance, position, current_value,
         funding_rate_norm, ob_imbalance, pcr_norm, fear_greed_norm,
         iv_skew_norm, basis_norm, google_trends_norm]

    Использует точные OHLCV-значения из snap["ohlcv"] (добавлены
    в MarketScanner.build_snapshot()). Если поле отсутствует —
    fallback к snap["price"] для цен и к training-mean для volume.

    :param snap: Снэпшот из MarketScanner.build_snapshot().
    :param balance: Свободный баланс USDT.
    :param norm_stats: Словарь {col: [mean, std]} из train_sac или None.
    :return: float32-массив shape=(OBS_DIM,).
    """
    ind = snap.get("indicators", {})
    ohlcv = snap.get("ohlcv", {})
    price = float(snap.get("price", 0.0))

    open_ = float(ohlcv.get("open", price))
    high = float(ohlcv.get("high", price))
    low = float(ohlcv.get("low", price))
    close = float(ohlcv.get("close", price))

    # Для volume: если нет ohlcv — используем training-mean, чтобы
    # нормализация давала 0 (нейтральный сигнал), а не выброс.
    vol_mean = norm_stats["volume"][0] if norm_stats and "volume" in norm_stats else 0.0
    volume = float(ohlcv.get("volume", vol_mean))

    # MACD: предпочитаем числовое значение из ohlcv, иначе
    # декодируем строку "bullish"/"bearish" из indicators.
    raw_macd = ohlcv.get("macd", ind.get("macd", 0))
    if isinstance(raw_macd, str):
        macd_val = 1.0 if raw_macd == "bullish" else -1.0
    else:
        macd_val = float(raw_macd)

    macd_signal_val = float(ohlcv.get("macd_signal", 0.0))

    bb_w = float(ind.get("bb_width", 0.04))
    bb_upper = price * (1.0 + bb_w / 2.0)
    bb_lower = price * (1.0 - bb_w / 2.0)

    # Новые фичи (индексы 14-20) из market_context снэпшота
    ctx = snap.get("market_context", {})
    funding_raw = float(ctx.get("funding_rate", 0.0))
    ob_imb = float(ctx.get("ob_imbalance", 0.0))
    pcr_raw = float(ctx.get("pcr", 1.0))
    fg_raw = float(ctx.get("fear_greed", 50.0))
    iv_raw = float(ctx.get("iv_skew", 0.0))
    basis_raw = float(ctx.get("basis_pct", 0.0))
    gt_raw = float(ctx.get("google_trends", 50.0))

    raw = np.array(
        [
            open_,  # [0]  open
            high,  # [1]  high
            low,  # [2]  low
            close,  # [3]  close
            volume,  # [4]  volume
            float(ind.get("rsi", 50.0)),  # [5]  rsi
            macd_val,  # [6]  macd
            macd_signal_val,  # [7]  macd_signal
            bb_upper,  # [8]  bb_upper
            price,  # [9]  bb_middle
            bb_lower,  # [10] bb_lower
            balance,  # [11] portfolio: balance
            0.0,  # [12] portfolio: position
            balance,  # [13] portfolio: current_value
            funding_raw * 1000.0,  # [14] funding_rate_norm
            float(np.clip(ob_imb, -1.0, 1.0)),  # [15] ob_imbalance
            float(np.clip(pcr_raw / 3.0, 0.0, 1.0)),  # [16] pcr_norm
            fg_raw / 100.0,  # [17] fear_greed_norm
            float(np.clip(iv_raw / 20.0, -1.0, 1.0)),  # [18] iv_skew_norm
            float(np.clip(basis_raw / 5.0, -1.0, 1.0)),  # [19] basis_norm
            gt_raw / 100.0,  # [20] google_trends_norm
        ],
        dtype=np.float32,
    )

    # Нормализация только для первых 11 элементов (OHLCV + индикаторы)
    # Индексы 11-20 не нормализуются: портфельные и контекстные уже в нужном масштабе
    if norm_stats:
        for i, col in enumerate(_MARKET_COLS):
            if col in norm_stats:
                mu, sd = norm_stats[col]
                if sd > 0:
                    raw[i] = (raw[i] - mu) / sd

    return raw


_MAX_OUTLIER_FRAC: float = 0.4  # fraction of market features >3σ before skipping
_DISABLE_AFTER_DRIFTS: int = 10  # consecutive drifty calls before model is disabled


class SACSignal:
    """
    Инференс SAC-модели (Stable-Baselines3) для одного снэпшота рынка.

    Загружает модель из Config.SAC_MODEL_PATH.
    При отсутствии файла модели возвращает hold с conf=0.
    """

    def __init__(self) -> None:
        """Инициализирует и пытается загрузить SAC-модель."""
        self.logger = logging.getLogger(__name__)
        self._model: Any = None
        self._norm_stats: Optional[Dict[str, list]] = None
        self.loaded = False
        self._mtime: float = 0.0
        self._consecutive_drifts: int = 0
        self._try_load()

    def _try_load(self) -> None:
        """
        Загружает SAC-модель и norm_stats из файловой системы.

        Проверяет SAC_MODEL_PATH из конфига. Если файл не найден —
        логирует предупреждение и оставляет loaded=False.
        """
        try:
            from stable_baselines3 import SAC  # noqa: PLC0415
        except ImportError:
            self.logger.error(
                "stable-baselines3 не установлен. " "pip install stable-baselines3"
            )
            return

        path = Config.SAC_MODEL_PATH
        if not path or not os.path.exists(path):
            self.logger.warning(
                "SAC-модель не найдена. "
                "Запустите: python reinforcement_learning/train_sac.py"
            )
            return

        try:
            self._model = SAC.load(path)
            self.loaded = True
            self._mtime = os.path.getmtime(path)
            self.logger.info("SAC загружен из %s", path)
        except Exception as e:
            self.logger.warning("Ошибка загрузки SAC: %s", e)
            return

        norm_path = path.replace(".zip", "_norm_stats.json")
        if os.path.exists(norm_path):
            try:
                with open(norm_path, encoding="utf-8") as f:
                    self._norm_stats = json.load(f)
                self.logger.info("Norm stats загружены из %s", norm_path)
            except (OSError, json.JSONDecodeError) as e:
                self._norm_stats = None
                self.logger.warning("Ошибка загрузки norm stats: %s", e)
        else:
            self.logger.warning("Norm stats не найдены — инференс без нормализации")

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
            # Detect normstats drift: features far outside training distribution
            if self._norm_stats and np.any(np.abs(obs[:11]) > 3.0):
                outliers = [_MARKET_COLS[i] for i in range(11) if abs(obs[i]) > 3.0]
                outlier_frac = len(outliers) / len(_MARKET_COLS)
                if outlier_frac >= _MAX_OUTLIER_FRAC:
                    self._consecutive_drifts += 1
                    self.logger.warning(
                        "SAC normstats drift for %s: %d/%d features out of range "
                        "(consecutive=%d/%d): %s",
                        snap.get("symbol", "?"),
                        len(outliers),
                        len(_MARKET_COLS),
                        self._consecutive_drifts,
                        _DISABLE_AFTER_DRIFTS,
                        outliers,
                    )
                    if self._consecutive_drifts >= _DISABLE_AFTER_DRIFTS:
                        self.logger.error(
                            "SAC disabled: normstats drifted for %d consecutive calls. "
                            "Retrain the model to restore SAC signals.",
                            self._consecutive_drifts,
                        )
                        self.loaded = False
                    return default
                self.logger.warning(
                    "SAC obs outliers (normstats drift?) for %s: %s",
                    snap.get("symbol", "?"),
                    outliers,
                )
            self._consecutive_drifts = 0
            action, _ = self._model.predict(obs, deterministic=True)
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
            self.logger.error("Ошибка инференса SAC: %s", e, exc_info=True)
            return default

    # ── Вариант 1: горячая перезагрузка модели ───────────────────────────────

    def reload_if_updated(self) -> bool:
        """Перезагружает модель если файл изменился (тренер сохранил новую версию)."""
        path = Config.SAC_MODEL_PATH
        if not path or not os.path.exists(path):
            return False
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return False
        if mtime <= self._mtime:
            return False
        self.logger.info("SAC model updated on disk — reloading...")
        self._try_load()
        return True
