from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import List, Tuple

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import Config
from src.redis_client import RedisClient

logger = logging.getLogger(__name__)

# Запросы NewsAPI для известных монет
_COIN_QUERIES = {
    "BTC": "Bitcoin OR BTC",
    "ETH": "Ethereum OR ETH",
    "SOL": "Solana OR SOL",
    "ADA": "Cardano OR ADA",
    "DOT": "Polkadot OR DOT",
    "AVAX": "Avalanche OR AVAX",
    "MATIC": "Polygon OR MATIC",
    "LINK": "Chainlink OR LINK",
    "UNI": "Uniswap OR UNI",
    "ATOM": "Cosmos OR ATOM",
    "XRP": "Ripple OR XRP",
    "BNB": "Binance OR BNB",
    "DOGE": "Dogecoin OR DOGE",
    "LTC": "Litecoin OR LTC",
    "TRX": "TRON OR TRX",
    "XLM": "Stellar OR XLM",
    "ALGO": "Algorand OR ALGO",
    "FTM": "Fantom OR FTM",
    "NEAR": "NEAR Protocol OR NEAR",
    "OP": "Optimism OR OP token",
    "ARB": "Arbitrum OR ARB",
}


class NewsAnalyzer:
    """
    Анализирует новостной сентимент для каждой монеты.

    Поддерживает per-symbol запросы, кэширует результаты в Redis
    (TTL=NEWS_UPDATE_INTERVAL, по умолчанию 900 сек = 15 мин).
    При отсутствии NEWS_API_KEY возвращает нейтральный сентимент
    вместо исключения.
    """

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.redis = RedisClient()
        self.logger = logging.getLogger(__name__)
        api_key = os.getenv("NEWS_API_KEY", "")
        self._enabled = bool(api_key)
        if self._enabled:
            from newsapi import NewsApiClient
            self.newsapi = NewsApiClient(api_key=api_key)

    def _cache_key(self, symbol: str) -> str:
        """
        Формирует ключ Redis-кэша для символа.

        :param symbol: Символ ccxt ('BTC/USDT').
        :return: Строка вида 'news:BTC'.
        """
        base = symbol.split("/")[0]
        return f"news:{base}"

    def _load_cached(
        self, symbol: str
    ) -> tuple[float, list[str]] | None:
        """
        Загружает сентимент из Redis-кэша.

        :param symbol: Символ ccxt.
        :return: Кортеж (sentiment, headlines) или None если кэш пуст.
        """
        try:
            raw = self.redis.redis_client.get(
                self._cache_key(symbol)
            )
            if raw:
                data = json.loads(raw.decode("utf-8"))
                return data["sentiment"], data["headlines"]
        except Exception:
            pass
        return None

    def _save_cache(
        self,
        symbol: str,
        sentiment: float,
        headlines: List[str],
    ) -> None:
        """
        Сохраняет сентимент и заголовки в Redis-кэш.

        TTL определяется Config.NEWS_UPDATE_INTERVAL (по умолчанию 900 с).

        :param symbol: Символ ccxt.
        :param sentiment: Compound-оценка VADER от -1 до 1.
        :param headlines: Список заголовков новостей.
        """
        try:
            key = self._cache_key(symbol)
            payload = json.dumps(
                {"sentiment": sentiment, "headlines": headlines}
            )
            self.redis.redis_client.setex(
                key,
                Config.NEWS_UPDATE_INTERVAL,
                payload.encode("utf-8"),
            )
        except Exception as e:
            self.logger.debug(
                f"Cache save failed for {symbol}: {e}"
            )

    async def get_sentiment(
        self, symbol: str
    ) -> Tuple[float, List[str]]:
        """
        Возвращает (compound_score, headlines) для символа.

        Результат кэшируется в Redis на NEWS_UPDATE_INTERVAL секунд.
        При недоступности API возвращает (0.0, []).

        :param symbol: Символ ccxt ('BTC/USDT', 'ETH/USDT', ...).
        :return: (sentiment -1..1, список заголовков)
        """
        cached = self._load_cached(symbol)
        if cached is not None:
            return cached

        if not self._enabled:
            return 0.0, []

        base = symbol.split("/")[0]
        query = _COIN_QUERIES.get(base, f"{base} cryptocurrency")

        articles = await self._fetch_for_symbol(query)
        if not articles:
            return 0.0, []

        sentiments = []
        headlines = []
        for art in articles:
            title = art.get("title", "") or ""
            desc = art.get("description", "") or ""
            text = f"{title} {desc}".strip()
            if text:
                scores = self.analyzer.polarity_scores(text)
                sentiments.append(scores["compound"])
                if title:
                    headlines.append(title[:100])

        sentiment = (
            sum(sentiments) / len(sentiments)
            if sentiments else 0.0
        )
        self._save_cache(symbol, sentiment, headlines[:5])
        self.logger.info(
            f"News [{base}]: sentiment={sentiment:.3f}, "
            f"articles={len(articles)}"
        )
        return sentiment, headlines[:5]

    async def _fetch_for_symbol(self, query: str) -> list:
        """
        Выполняет запрос к NewsAPI в thread executor.

        :param query: Поисковый запрос (например, "Bitcoin OR BTC").
        :return: Список статей или пустой список при ошибке.
        """
        if not self._enabled:
            return []
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.newsapi.get_everything(
                    q=query,
                    language="en",
                    sort_by="publishedAt",
                    page_size=10,
                ),
            )
            return response.get("articles", [])
        except Exception as e:
            self.logger.warning(
                f"NewsAPI fetch failed ({query}): {e}"
            )
            return []

    async def analyze_news_async(self) -> float:
        """
        Возвращает общий сентимент по крипторынку через BTC/USDT.

        Метод сохранён для обратной совместимости.

        :return: Compound-оценка VADER от -1 до 1.
        """
        sentiment, _ = await self.get_sentiment("BTC/USDT")
        return sentiment
