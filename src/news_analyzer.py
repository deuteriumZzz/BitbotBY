from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from config import Config
from src.constants import REDIS_TTL_TRADING_STATE
from src.redis_client import RedisClient

# AI_DAILY_BUDGET применяется ко всем провайдерам (Claude / DeepSeek / OpenAI).
# CLAUDE_DAILY_BUDGET оставлен как алиас для обратной совместимости.
_AI_DAILY_BUDGET: int = int(
    os.getenv("AI_DAILY_BUDGET", os.getenv("CLAUDE_DAILY_BUDGET", "200"))
)

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


_RSS_FEEDS: Dict[str, List[str]] = {
    "BTC": [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
    ],
    "ETH": [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
    ],
    "_default": [
        "https://cointelegraph.com/rss",
    ],
}

_RSS_CACHE_TTL: int = 900  # 15 минут


class NewsAnalyzer:
    """
    Анализирует новостной сентимент для каждой монеты.

    Поддерживает per-symbol запросы, кэширует результаты в Redis
    (TTL=NEWS_UPDATE_INTERVAL, по умолчанию 900 сек = 15 мин).
    При отсутствии NEWS_API_KEY возвращает нейтральный сентимент
    вместо исключения.

    AI-провайдер для сентимента выбирается через AI_PROVIDER (.env):
      auto      → Claude → DeepSeek → OpenAI (первый найденный ключ)
      anthropic → только Claude
      deepseek  → только DeepSeek
      openai    → только ChatGPT
    """

    def __init__(self) -> None:
        self.redis = RedisClient()
        self.logger = logging.getLogger(__name__)
        news_key = os.getenv("NEWS_API_KEY", "")
        self._enabled = bool(news_key)
        if self._enabled:
            from newsapi import NewsApiClient

            self.newsapi = NewsApiClient(api_key=news_key)

        # Выбор AI-провайдера для сентимента
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
        openai_key = os.getenv("OPENAI_API_KEY", "")
        provider = os.getenv("AI_PROVIDER", "auto")

        if provider == "anthropic":
            self._ai_scorer = "claude" if anthropic_key else "vader"
        elif provider == "deepseek":
            self._ai_scorer = "deepseek" if deepseek_key else "vader"
        elif provider == "openai":
            self._ai_scorer = "openai" if openai_key else "vader"
        else:  # auto
            if anthropic_key:
                self._ai_scorer = "claude"
            elif deepseek_key:
                self._ai_scorer = "deepseek"
            elif openai_key:
                self._ai_scorer = "openai"
            else:
                self._ai_scorer = "vader"

        # Создаём клиент один раз, а не при каждом вызове
        self._claude_client: Optional[object] = None
        self._compat_client: Optional[object] = None  # DeepSeek / OpenAI

        if self._ai_scorer == "claude":
            import anthropic

            self._claude_client = anthropic.AsyncAnthropic(api_key=anthropic_key)
        elif self._ai_scorer == "deepseek":
            from openai import AsyncOpenAI

            self._compat_client = AsyncOpenAI(
                api_key=deepseek_key,
                base_url="https://api.deepseek.com",
            )
        elif self._ai_scorer == "openai":
            from openai import AsyncOpenAI

            self._compat_client = AsyncOpenAI(api_key=openai_key)

        # Единый бюджет-гард: применяется ко всем провайдерам
        self._ai_calls_today: int = 0
        self._ai_date: str = ""

        # RSS-кэш: symbol_base → (headlines, timestamp)
        self._rss_cache: Dict[str, Tuple[List[str], float]] = {}

        self.logger.info("NewsAnalyzer: AI scorer = %s", self._ai_scorer)

    # ── Budget guard ──────────────────────────────────────────────────────────

    def _check_budget(self) -> bool:
        """
        Проверяет дневной лимит вызовов AI (сбрасывается в UTC полночь).
        Счётчик хранится в Redis чтобы пережить рестарты бота.

        :return: True если лимит не исчерпан (вызов разрешён).
        """
        today = datetime.utcnow().date().isoformat()
        redis_key = f"ai_budget:{today}"
        try:
            count = self.redis.redis_client.incr(redis_key)
            if count == 1:
                # Истекает в конце дня — TTL ставится только при первом инкременте
                self.redis.redis_client.expire(redis_key, REDIS_TTL_TRADING_STATE)
            if count > _AI_DAILY_BUDGET:
                self.logger.warning(
                    "%s daily budget (%d calls) exhausted — VADER fallback",
                    self._ai_scorer.upper(),
                    _AI_DAILY_BUDGET,
                )
                return False
            # Синхронизируем с in-memory счётчиком, чтобы логи
            # всегда отражали точное значение.
            self._ai_calls_today = int(count)
            return True
        except Exception:
            # Redis недоступен — переходим на in-memory счётчик
            if self._ai_date != today:
                self._ai_date = today
                self._ai_calls_today = 0
            if self._ai_calls_today >= _AI_DAILY_BUDGET:
                self.logger.warning(
                    "%s daily budget (%d calls) exhausted — VADER fallback",
                    self._ai_scorer.upper(),
                    _AI_DAILY_BUDGET,
                )
                return False
            self._ai_calls_today += 1
            return True

    # ── Cache ─────────────────────────────────────────────────────────────────

    def _cache_key(self, symbol: str) -> str:
        base = symbol.split("/")[0]
        return f"news:{base}"

    def _load_cached(self, symbol: str) -> tuple[float, list[str]] | None:
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
        try:
            key = self._cache_key(symbol)
            payload = json.dumps({"sentiment": sentiment, "headlines": headlines})
            self.redis.redis_client.setex(
                key,
                Config.NEWS_UPDATE_INTERVAL,
                payload.encode("utf-8"),
            )
        except Exception as e:
            self.logger.debug("Cache save failed for %s: %s", symbol, e)

    # ── RSS fallback ──────────────────────────────────────────────────────────

    async def _fetch_rss_headlines(self, symbol: str, limit: int = 10) -> List[str]:
        base = symbol.split("/")[0].upper()
        now = time.monotonic()
        cached = self._rss_cache.get(base)
        if cached is not None and (now - cached[1]) < _RSS_CACHE_TTL:
            return cached[0]

        feeds = _RSS_FEEDS.get(base, _RSS_FEEDS["_default"])
        ticker_lower = base.lower()
        cutoff = datetime.now(tz=timezone.utc).timestamp() - 86_400

        def _parse_feeds() -> List[str]:
            try:
                import feedparser
            except ImportError:
                return []
            results: List[str] = []
            for url in feeds:
                try:
                    feed = feedparser.parse(url)
                    for entry in feed.entries:
                        title: str = entry.get("title", "") or ""
                        if not title:
                            continue
                        pub = entry.get("published_parsed")
                        if pub is not None:
                            import calendar
                            ts = float(calendar.timegm(pub))
                            if ts < cutoff:
                                continue
                        if ticker_lower in title.lower() or base in title:
                            results.append(title[:120])
                        if len(results) >= limit:
                            break
                except Exception:
                    continue
            return results

        try:
            loop = asyncio.get_running_loop()
            headlines = await loop.run_in_executor(None, _parse_feeds)
        except Exception as exc:
            self.logger.warning("RSS fetch failed for %s: %s", base, exc)
            headlines = []

        self._rss_cache[base] = (headlines, now)
        return headlines

    # ── Public API ────────────────────────────────────────────────────────────

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

        base = symbol.split("/")[0]
        query = _COIN_QUERIES.get(base, f"{base} cryptocurrency")

        articles: list = []
        if self._enabled:
            articles = await self._fetch_for_symbol(query)

        if articles:
            headlines = [
                (art.get("title", "") or "")[:100] for art in articles if art.get("title")
            ]
        else:
            headlines = await self._fetch_rss_headlines(symbol)

        if not headlines:
            return 0.0, []

        if self._ai_scorer == "claude":
            sentiment = await self._score_with_claude(base, headlines)
        elif self._ai_scorer == "deepseek":
            sentiment = await self._score_with_deepseek(base, headlines)
        elif self._ai_scorer == "openai":
            sentiment = await self._score_with_openai(base, headlines)
        else:
            sentiment = self._score_with_vader(articles)

        self._save_cache(symbol, sentiment, headlines[:5])
        self.logger.info(
            "News [%s]: sentiment=%.3f, articles=%d (scorer=%s, calls_today=%d/%d)",
            base,
            sentiment,
            len(articles),
            self._ai_scorer,
            self._ai_calls_today,
            _AI_DAILY_BUDGET,
        )
        return sentiment, headlines[:5]

    # ── AI scorers ────────────────────────────────────────────────────────────

    def _build_sentiment_prompt(self, coin: str, headlines: List[str]) -> str:
        bullet_list = "\n".join(f"- {h}" for h in headlines[:10])
        return (
            f"Rate the overall market sentiment for {coin} cryptocurrency "
            f"based on these recent news headlines:\n\n{bullet_list}\n\n"
            "Reply with ONLY a single decimal number between -1.0 (extremely "
            "bearish) and 1.0 (extremely bullish). No explanation."
        )

    async def _score_with_claude(self, coin: str, headlines: List[str]) -> float:
        """
        Оценивает сентимент через Claude API (haiku-tier).

        Применяет дневной бюджет-гард (AI_DAILY_BUDGET).
        Фолбек на VADER при ошибке или исчерпании лимита.
        """
        if not headlines or not self._check_budget():
            return self._score_with_vader_headlines(headlines)
        try:
            client = self._claude_client
            prompt = self._build_sentiment_prompt(coin, headlines)
            message = await client.messages.create(  # type: ignore[union-attr]
                model="claude-haiku-4-5-20251001",
                max_tokens=16,
                messages=[{"role": "user", "content": prompt}],
            )
            block = message.content[0]
            text = block.text.strip() if hasattr(block, "text") else ""
            return max(-1.0, min(1.0, float(text)))
        except Exception as exc:
            self.logger.warning("Claude sentiment failed → VADER: %s", exc)
            return self._score_with_vader_headlines(headlines)

    async def _score_with_deepseek(self, coin: str, headlines: List[str]) -> float:
        """
        Оценивает сентимент через DeepSeek API (OpenAI-совместимый).

        Применяет тот же дневной бюджет-гард (AI_DAILY_BUDGET).
        Фолбек на VADER при ошибке или исчерпании лимита.
        """
        if not headlines or not self._check_budget():
            return self._score_with_vader_headlines(headlines)
        try:
            prompt = self._build_sentiment_prompt(coin, headlines)
            client = self._compat_client
            response = await client.chat.completions.create(  # type: ignore[union-attr]
                model=Config.DEEPSEEK_MODEL,
                max_tokens=16,
                messages=[{"role": "user", "content": prompt}],
            )
            text = (response.choices[0].message.content or "").strip()
            return max(-1.0, min(1.0, float(text)))
        except Exception as exc:
            self.logger.warning("DeepSeek sentiment failed → VADER: %s", exc)
            return self._score_with_vader_headlines(headlines)

    async def _score_with_openai(self, coin: str, headlines: List[str]) -> float:
        """
        Оценивает сентимент через OpenAI API (ChatGPT).

        Применяет тот же дневной бюджет-гард (AI_DAILY_BUDGET).
        Фолбек на VADER при ошибке или исчерпании лимита.
        """
        if not headlines or not self._check_budget():
            return self._score_with_vader_headlines(headlines)
        try:
            prompt = self._build_sentiment_prompt(coin, headlines)
            client = self._compat_client
            response = await client.chat.completions.create(  # type: ignore[union-attr]
                model=Config.OPENAI_MODEL,
                max_tokens=16,
                messages=[{"role": "user", "content": prompt}],
            )
            text = (response.choices[0].message.content or "").strip()
            return max(-1.0, min(1.0, float(text)))
        except Exception as exc:
            self.logger.warning("OpenAI sentiment failed → VADER: %s", exc)
            return self._score_with_vader_headlines(headlines)

    # ── VADER fallbacks ───────────────────────────────────────────────────────

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

    # ── NewsAPI fetch ─────────────────────────────────────────────────────────

    async def _fetch_for_symbol(self, query: str) -> list:
        """
        Выполняет запрос к NewsAPI в thread executor.

        :param query: Поисковый запрос (например, "Bitcoin OR BTC").
        :return: Список статей или пустой список при ошибке.
        """
        if not self._enabled:
            return []
        try:
            loop = asyncio.get_running_loop()
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
            self.logger.warning("NewsAPI fetch failed (%s): %s", query, e)
            return []
