"""
Twitter/X sentiment analyzer for crypto tickers.

Backend приоритет:
  1. Официальный Twitter API v2 (TWITTER_BEARER_TOKEN) — если задан.
  2. twscrape (TWSCRAPE_ACCOUNTS) — ротация нескольких X-аккаунтов вместо
     платного официального API, если bearer token не задан.
  3. Ни то ни другое — молча возвращает 0.0.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Tuple
from urllib.parse import unquote

from config import Config

logger = logging.getLogger(__name__)

_TWITTER_SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"


def _parse_twscrape_accounts(raw: str) -> List[Tuple[str, str, str, str]]:
    """Разбирает TWSCRAPE_ACCOUNTS в список (username, password, email, email_password).

    :param raw: "user:pass:email:email_pass,user2:pass2:email2:email_pass2".
    :return: Список валидных 4-элементных кортежей; записи с неверным
        числом полей пропускаются с предупреждением в лог.
    """
    accounts = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        fields = entry.split(":")
        if len(fields) != 4:
            logger.warning(
                "TWSCRAPE_ACCOUNTS: пропускаю запись с неверным числом полей "
                "(ожидается user:pass:email:email_pass)"
            )
            continue
        # Приложение кодирует «:», «,» и «%» внутри значений как %3A, %2C, %25.
        accounts.append(
            (
                unquote(fields[0]),
                unquote(fields[1]),
                unquote(fields[2]),
                unquote(fields[3]),
            )
        )
    return accounts


class TwitterAnalyzer:
    """
    Анализатор Twitter/X сентимента для крипто-тикеров.

    Официальный backend — Twitter API v2 + VADER. Если TWITTER_BEARER_TOKEN
    не задан, но задан TWSCRAPE_ACCOUNTS — переключается на twscrape
    (github.com/vladkens/twscrape): скрапинг через пул реальных X-аккаунтов
    с автоматической ротацией при рейт-лимите, без официального API.
    Без обоих — молча возвращает 0.0 без ошибок.
    Результаты кэшируются на SENTIMENT_CACHE_TTL секунд.

    Лимиты Basic плана официального API ($100/мес): ~10 000 твитов/мес.
    """

    def __init__(self) -> None:
        self._token = Config.TWITTER_BEARER_TOKEN
        self._max_results = Config.TWITTER_MAX_RESULTS
        self._twscrape_raw_accounts = Config.TWSCRAPE_ACCOUNTS
        self._twscrape_db_path = Config.TWSCRAPE_DB_PATH

        if self._token:
            self._backend = "api"
        elif self._twscrape_raw_accounts:
            self._backend = "twscrape"
        else:
            self._backend = None
        self._enabled = self._backend is not None

        self._cache: Dict[str, Tuple[float, float]] = {}
        self._cache_ttl = float(Config.SENTIMENT_CACHE_TTL)

        # Ленивая инициализация пула аккаунтов twscrape (первый запрос за
        # процесс логинит все аккаунты; дальше переиспользуется).
        self._twscrape_api = None
        self._twscrape_ready = False
        self._twscrape_lock = asyncio.Lock()

        if not self._enabled:
            logger.debug(
                "TwitterAnalyzer: ни TWITTER_BEARER_TOKEN, ни TWSCRAPE_ACCOUNTS "
                "не заданы — sentiment отключён"
            )
        elif self._backend == "twscrape":
            logger.info(
                "TwitterAnalyzer: используется twscrape fallback "
                "(официальный TWITTER_BEARER_TOKEN не задан)"
            )

    def _ticker_from_symbol(self, symbol: str) -> str:
        """Извлекает тикер: 'BTC/USDT' → 'BTC'."""
        return symbol.split("/")[0].upper()

    async def get_sentiment(self, symbol: str) -> float:
        """
        Возвращает sentiment score -1.0..1.0 для символа.
        0.0 если ни один backend не настроен или произошла ошибка.

        :param symbol: Символ в формате ccxt ('PEPE/USDT').
        :return: Sentiment score -1.0..1.0, 0.0 при ошибке или без backend'а.
        """
        if not self._enabled:
            return 0.0

        now = time.monotonic()
        cached = self._cache.get(symbol)
        if cached and now - cached[1] < self._cache_ttl:
            return cached[0]

        try:
            if self._backend == "api":
                score = await self._fetch_sentiment_api(symbol)
            else:
                score = await self._fetch_sentiment_twscrape(symbol)
            self._cache[symbol] = (score, now)
            return score
        except Exception as exc:
            logger.debug("TwitterAnalyzer: error for %s: %s", symbol, exc)
            return 0.0

    async def _fetch_sentiment_api(self, symbol: str) -> float:
        """
        Выполняет запрос к официальному Twitter API v2 и считает VADER sentiment.

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
            "TwitterAnalyzer(api): %s — %d tweets, avg_score=%.3f",
            ticker,
            len(scores),
            avg_score,
        )
        return round(avg_score, 4)

    async def _ensure_twscrape_pool(self):
        """
        Лениво создаёт twscrape.API и логинит все аккаунты из
        TWSCRAPE_ACCOUNTS (один раз за время жизни процесса — twscrape
        сам кэширует сессионные куки в TWSCRAPE_DB_PATH между запусками).

        :return: Готовый к поиску twscrape.API.
        """
        if self._twscrape_ready:
            return self._twscrape_api

        async with self._twscrape_lock:
            if self._twscrape_ready:
                return self._twscrape_api

            from twscrape import API

            api = API(self._twscrape_db_path)
            accounts = _parse_twscrape_accounts(self._twscrape_raw_accounts)
            for username, password, email, email_password in accounts:
                await api.pool.add_account(username, password, email, email_password)
            await api.pool.login_all()

            self._twscrape_api = api
            self._twscrape_ready = True
            return api

    async def _fetch_sentiment_twscrape(self, symbol: str) -> float:
        """
        Скрапит твиты через twscrape (без официального API) и считает
        VADER sentiment — тот же метод оценки, что и в официальном backend'е,
        чтобы значения из обоих источников были сопоставимы.

        :param symbol: Символ в формате ccxt.
        :return: Средний compound score VADER по твитам, 0.0 если нет твитов.
        """
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        ticker = self._ticker_from_symbol(symbol)
        query = f"(#{ticker} OR ${ticker}) lang:en -is:retweet"

        api = await self._ensure_twscrape_pool()

        texts: List[str] = []
        async for tweet in api.search(query, limit=self._max_results):
            content = getattr(tweet, "rawContent", "") or ""
            if content:
                texts.append(content)

        if not texts:
            return 0.0

        vader = SentimentIntensityAnalyzer()
        scores = [vader.polarity_scores(t)["compound"] for t in texts]
        avg_score = sum(scores) / len(scores)
        logger.debug(
            "TwitterAnalyzer(twscrape): %s — %d tweets, avg_score=%.3f",
            ticker,
            len(scores),
            avg_score,
        )
        return round(avg_score, 4)
