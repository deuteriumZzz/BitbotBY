from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import List, Tuple

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
        self.redis = RedisClient()
        self.logger = logging.getLogger(__name__)
        news_key = os.getenv("NEWS_API_KEY", "")
        self._enabled = bool(news_key)
        if self._enabled:
            from newsapi import NewsApiClient

            self.newsapi = NewsApiClient(api_key=news_key)
        # Claude API sentiment (preferred); VADER is fallback
        self._anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._use_claude = bool(self._anthropic_key)

    def _cache_key(self, symbol: str) -> str:
        """
        Формирует ключ Redis-кэша для символа.

        :param symbol: Символ ccxt ('BTC/USDT').
        :return: Строка вида 'news:BTC'.
        """
        base = symbol.split("/")[0]
        return f"news:{base}"

    def _load_cached(self, symbol: str) -> tuple[float, list[str]] | None:
        """
        Загружает сентимент из Redis-кэша.

        :param symbol: Символ ccxt.
        :return: Кортеж (sentiment, headlines) или None если кэш пуст.
        """
        try:
            raw = self.redis.redis_client.get(self._cache_key(symbol))
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
            payload = json.dumps({"sentiment": sentiment, "headlines": headlines})
            self.redis.redis_client.setex(
                key,
                Config.NEWS_UPDATE_INTERVAL,
                payload.encode("utf-8"),
            )
        except Exception as e:
            self.logger.debug(f"Cache save failed for {symbol}: {e}")

    async def get_sentiment(self, symbol: str) -> Tuple[float, List[str]]:
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

        headlines = [
            (art.get("title", "") or "")[:100] for art in articles if art.get("title")
        ]

        if self._use_claude:
            sentiment = await self._score_with_claude(base, headlines)
        else:
            sentiment = self._score_with_vader(articles)

        self._save_cache(symbol, sentiment, headlines[:5])
        self.logger.info(
            "News [%s]: sentiment=%.3f, articles=%d (scorer=%s)",
            base,
            sentiment,
            len(articles),
            "claude" if self._use_claude else "vader",
        )
        return sentiment, headlines[:5]

    async def _score_with_claude(self, coin: str, headlines: List[str]) -> float:
        """
        Score sentiment with Claude API (haiku-tier for cost efficiency).

        Batches all headlines in one prompt, returns float -1..1.
        Falls back to VADER if the API call fails.

        :param coin: Coin ticker, e.g. "BTC".
        :param headlines: List of news headline strings.
        :return: Sentiment score -1.0 (very bearish) to +1.0 (very bullish).
        """
        if not headlines:
            return 0.0
        try:
            import anthropic

            bullet_list = "\n".join(f"- {h}" for h in headlines[:10])
            prompt = (
                f"Rate the overall market sentiment for {coin} cryptocurrency "
                f"based on these recent news headlines:\n\n{bullet_list}\n\n"
                "Reply with ONLY a single decimal number between -1.0 (extremely "
                "bearish) and 1.0 (extremely bullish). No explanation."
            )
            client = anthropic.AsyncAnthropic(api_key=self._anthropic_key)
            message = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=16,
                messages=[{"role": "user", "content": prompt}],
            )
            block = message.content[0]
            text = block.text.strip() if hasattr(block, "text") else ""
            score = max(-1.0, min(1.0, float(text)))
            return score
        except Exception as exc:
            self.logger.warning(
                "Claude sentiment failed, falling back to VADER: %s", exc
            )
            return self._score_with_vader_headlines(headlines)

    def _score_with_vader(self, articles: list) -> float:
        """VADER-фолбек: вычисляет sentiment по заголовку и описанию каждой статьи."""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            analyzer = SentimentIntensityAnalyzer()
            scores = []
            for art in articles:
                title = art.get("title", "") or ""
                desc = art.get("description", "") or ""
                text = f"{title} {desc}".strip()
                if text:
                    scores.append(analyzer.polarity_scores(text)["compound"])
            return sum(scores) / len(scores) if scores else 0.0
        except Exception:
            return 0.0

    def _score_with_vader_headlines(self, headlines: List[str]) -> float:
        """VADER-фолбек для списка только заголовков (без описаний)."""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            analyzer = SentimentIntensityAnalyzer()
            scores = [analyzer.polarity_scores(h)["compound"] for h in headlines if h]
            return sum(scores) / len(scores) if scores else 0.0
        except Exception:
            return 0.0

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
            self.logger.warning(f"NewsAPI fetch failed ({query}): {e}")
            return []

    async def analyze_news_async(self) -> float:
        """
        Возвращает общий сентимент по крипторынку через BTC/USDT.

        Метод сохранён для обратной совместимости.

        :return: Compound-оценка VADER от -1 до 1.
        """
        sentiment, _ = await self.get_sentiment("BTC/USDT")
        return sentiment
