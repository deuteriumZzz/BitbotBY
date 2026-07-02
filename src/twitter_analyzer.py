"""
Twitter/X sentiment analyzer for crypto tickers.
Skips silently when TWITTER_BEARER_TOKEN is not set.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Tuple

from config import Config

logger = logging.getLogger(__name__)

_TWITTER_SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"


class TwitterAnalyzer:
    """
    Анализатор Twitter/X сентимента для крипто-тикеров.

    Использует Twitter API v2 и VADER для быстрого sentiment-scoring.
    Если TWITTER_BEARER_TOKEN не задан — молча возвращает 0.0 без ошибок.
    Результаты кэшируются на SENTIMENT_CACHE_TTL секунд.

    Лимиты Basic плана ($100/мес): ~10 000 твитов/мес.
    Экономия: опрашиваем только топ-N монет раз в 5 мин через SentimentPoller.
    """

    def __init__(self) -> None:
        self._token = Config.TWITTER_BEARER_TOKEN
        self._max_results = Config.TWITTER_MAX_RESULTS
        self._enabled = bool(self._token)
        self._cache: Dict[str, Tuple[float, float]] = {}
        self._cache_ttl = float(Config.SENTIMENT_CACHE_TTL)

        if not self._enabled:
            logger.debug(
                "TwitterAnalyzer: TWITTER_BEARER_TOKEN not set — sentiment disabled"
            )

    def _ticker_from_symbol(self, symbol: str) -> str:
        """Извлекает тикер: 'BTC/USDT' → 'BTC'."""
        return symbol.split("/")[0].upper()

    async def get_sentiment(self, symbol: str) -> float:
        """
        Возвращает sentiment score -1.0..1.0 для символа.
        0.0 если TWITTER_BEARER_TOKEN не задан или ошибка.

        :param symbol: Символ в формате ccxt ('PEPE/USDT').
        :return: Sentiment score -1.0..1.0, 0.0 при ошибке или без токена.
        """
        if not self._enabled:
            return 0.0

        now = time.monotonic()
        cached = self._cache.get(symbol)
        if cached and now - cached[1] < self._cache_ttl:
            return cached[0]

        try:
            score = await self._fetch_sentiment(symbol)
            self._cache[symbol] = (score, now)
            return score
        except Exception as exc:
            logger.debug("TwitterAnalyzer: error for %s: %s", symbol, exc)
            return 0.0

    async def _fetch_sentiment(self, symbol: str) -> float:
        """
        Выполняет запрос к Twitter API v2 и считает VADER sentiment.

        :param symbol: Символ в формате ccxt.
        :return: Средний compound score VADER по твитам, 0.0 если нет твитов.
        """
        import aiohttp
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        ticker = self._ticker_from_symbol(symbol)
        query = f"(#{ticker} OR ${ticker}) lang:en -is:retweet"

        headers = {"Authorization": f"Bearer {self._token}"}
        params = {
            "query": query,
            "max_results": str(self._max_results),
            "tweet.fields": "created_at,public_metrics",
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(
                _TWITTER_SEARCH_URL,
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status == 429:
                    logger.debug(
                        "TwitterAnalyzer: rate limited for %s — returning 0.0", symbol
                    )
                    return 0.0
                if resp.status != 200:
                    logger.debug("TwitterAnalyzer: HTTP %d for %s", resp.status, symbol)
                    return 0.0
                data = await resp.json()

        tweets = data.get("data", [])
        if not tweets:
            return 0.0

        vader = SentimentIntensityAnalyzer()
        scores = [
            vader.polarity_scores(t.get("text", ""))["compound"]
            for t in tweets
            if t.get("text")
        ]
        if not scores:
            return 0.0

        avg_score = sum(scores) / len(scores)
        logger.debug(
            "TwitterAnalyzer: %s — %d tweets, avg_score=%.3f",
            ticker,
            len(scores),
            avg_score,
        )
        return round(avg_score, 4)
