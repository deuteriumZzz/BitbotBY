"""
Amazon Chronos — time-series forecasting для предсказания направления цены.

Модель обучена только на числах (временные ряды), не читает текст/новости.
Используется как третья независимая точка зрения в hybrid+enhanced режиме.

Загрузка модели происходит один раз при первом вызове (ленивая инициализация).
Если пакет не установлен — graceful fallback: predict_direction() вернёт "neutral".
"""

from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)

_pipeline = None
_load_attempted: bool = False
_CONTEXT_LEN = 64  # ~16 часов 15m-свечей


def _load_pipeline() -> None:
    global _pipeline, _load_attempted
    if _load_attempted:
        return
    _load_attempted = True
    try:
        import torch
        from chronos import ChronosPipeline  # type: ignore[import]

        _pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map="cpu",
            torch_dtype=torch.bfloat16,
        )
        logger.info("Chronos model loaded (amazon/chronos-t5-small, CPU)")
    except Exception as exc:
        logger.warning("Chronos unavailable — will return 'neutral': %s", exc)


def predict_direction(
    close_prices: Sequence[float], threshold_pct: float = 0.002
) -> str:
    """
    Предсказывает направление движения цены на 1 шаг вперёд.

    :param close_prices: Последние N значений цены закрытия (минимум 10).
    :param threshold_pct: Минимальное отклонение прогноза для классификации (0.002 = 0.2%).
    :return: "up" / "down" / "neutral"
    """
    _load_pipeline()
    if _pipeline is None or len(close_prices) < 10:
        return "neutral"

    try:
        import torch

        prices = list(close_prices[-_CONTEXT_LEN:])
        context = torch.tensor(prices, dtype=torch.float32).unsqueeze(0)
        forecast = _pipeline.predict(context, prediction_length=1)
        # forecast: Tensor[batch=1, num_samples, horizon=1]
        median = float(forecast[0, :, 0].median().item())
        last = prices[-1]

        if last == 0:
            return "neutral"

        diff_pct = (median - last) / last
        if diff_pct > threshold_pct:
            return "up"
        if diff_pct < -threshold_pct:
            return "down"
        return "neutral"
    except Exception as exc:
        logger.warning("Chronos predict error: %s", exc)
        return "neutral"


def is_available() -> bool:
    """True если модель загружена и готова к работе."""
    _load_pipeline()
    return _pipeline is not None
