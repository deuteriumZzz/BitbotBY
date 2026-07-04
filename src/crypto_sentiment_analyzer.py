"""
CryptoBERT — sentiment-классификатор постов про крипту (Bullish/Bearish/Neutral).

Модель дообучена на ~3.2M постов Twitter/Reddit/StockTwits специально под
крипто-текст. Заменяет грубую эвристику upvote_ratio в
MarketContext._get_reddit_sentiment — та отражала согласие голосующих
с постом, а не его содержание (медвежий пост может иметь высокий
upvote_ratio, если все согласны, что новость плохая).

Загрузка модели происходит один раз при первом вызове (ленивая инициализация).
Если пакет не установлен — graceful fallback: predict_sentiment() вернёт "neutral".
"""

from __future__ import annotations

import logging
from typing import Sequence, Tuple

logger = logging.getLogger(__name__)

_pipeline = None
_load_attempted: bool = False
_MODEL_NAME = "ElKulako/cryptobert"
_MAX_CHARS = 200  # текст обрезается — инференс остаётся быстрым на CPU
_BULLISH_THRESHOLD = 0.15
_BEARISH_THRESHOLD = -0.15


def _load_pipeline() -> None:
    global _pipeline, _load_attempted
    if _load_attempted:
        return
    _load_attempted = True
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore[import]

        _pipeline = hf_pipeline(
            "text-classification",
            model=_MODEL_NAME,
            top_k=None,
            device=-1,  # CPU
        )
        logger.info("CryptoBERT model loaded (%s, CPU)", _MODEL_NAME)
    except Exception as exc:
        logger.warning("CryptoBERT unavailable — will return 'neutral': %s", exc)


def predict_sentiment(texts: Sequence[str]) -> Tuple[float, str]:
    """
    Классифицирует тексты постов и возвращает агрегированный sentiment.

    :param texts: Тексты постов (title [+ selftext]).
    :return: (net_score от -1 до 1, "reddit_bullish"/"reddit_bearish"/"neutral").
    """
    _load_pipeline()
    clean_texts = [t[:_MAX_CHARS] for t in texts if t and t.strip()]
    if _pipeline is None or not clean_texts:
        return 0.0, "neutral"

    try:
        results = _pipeline(clean_texts)
        bullish = 0.0
        bearish = 0.0
        for scores in results:
            for item in scores:
                label = str(item.get("label", "")).lower()
                score = float(item.get("score", 0.0))
                if label == "bullish":
                    bullish += score
                elif label == "bearish":
                    bearish += score

        n = len(clean_texts)
        net_score = round((bullish - bearish) / n, 4) if n else 0.0

        if net_score > _BULLISH_THRESHOLD:
            signal = "reddit_bullish"
        elif net_score < _BEARISH_THRESHOLD:
            signal = "reddit_bearish"
        else:
            signal = "neutral"
        return net_score, signal
    except Exception as exc:
        logger.warning("CryptoBERT predict error: %s", exc)
        return 0.0, "neutral"


def is_available() -> bool:
    """True если модель загружена и готова к работе."""
    _load_pipeline()
    return _pipeline is not None
