"""
Сентимент соцсетей для торгового цикла: Telegram + Twitter.

Telegram — мгновенно из окна свежих постов (TelegramSentiment), в любом
профиле, по монетам текущего скана. Twitter — опросом по расписанию в
фоновой asyncio-задаче и только в профиле meme (как было).
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Tuple

from config import Config

logger = logging.getLogger(__name__)

_REDIS_KEY_PREFIX = "bot:sentiment:"


def combine(tw: float, tg: float) -> float:
    """Telegram весит 2×: памп-группы сигнализируют раньше Twitter."""
    if tw == 0.0 or tg == 0.0:
        return tw or tg
    return round((tw + tg * 2) / 3, 4)


class SentimentPoller:
    """
    Twitter опрашивается по расписанию (оценка — в Redis
    bot:sentiment:{symbol} с TTL), Telegram считается из окна при запросе.
    Оба источника опциональны — без ключей оценка 0.0 и 0 упоминаний.

    Использование:
        poller = SentimentPoller(twitter, telegram, redis_client)
        await poller.start(symbols)
        await poller.sync(symbols, twitter=..., channels=..., window_min=...)
        score, mentions = await poller.get_score("PEPE/USDT")
        await poller.stop()
    """

    def __init__(self, twitter, telegram, redis_client) -> None:
        self._twitter = twitter
        self._telegram = telegram
        self._redis = redis_client
        self._task: asyncio.Task | None = None
        self._symbols: List[str] = []
        self._twitter_on = False
        self._poll_interval = Config.SENTIMENT_POLL_INTERVAL
        self._cache_ttl = Config.SENTIMENT_CACHE_TTL

    async def start(self, symbols: List[str]) -> None:
        """Подключает Telegram и запускает фоновый опрос Twitter."""
        self._symbols = list(symbols)
        await self._telegram.start()
        self._task = asyncio.create_task(self._loop(), name="sentiment_poller")
        logger.info("SentimentPoller: started for %d symbols", len(symbols))

    async def stop(self) -> None:
        """Останавливает опрос и отключает Telegram-клиент."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._telegram.stop()
        logger.info("SentimentPoller: stopped")

    async def sync(
        self,
        symbols: List[str],
        *,
        twitter: bool,
        channels: List[str],
        window_min: float,
    ) -> None:
        """Каждый цикл: монеты активного профиля, каналы и окно из настроек.

        Скан и профиль (в т.ч. сменённый детектором сезона) меняются — список
        монет для каналов меняется вместе с ними.
        """
        self._symbols = list(symbols)
        self._twitter_on = twitter
        self._telegram.configure(channels, window_min)
        await self._telegram.start()  # подхватывает вход и новые каналы
        self._telegram.report(self._symbols)

    async def get_score(self, symbol: str) -> Tuple[float, int]:
        """(оценка -1..1, упоминаний в свежих постах Telegram)."""
        tg, n = self._telegram.snapshot(symbol)
        tw = self._twitter_score(symbol) if self._twitter_on else 0.0
        return combine(tw, tg), n

    def _twitter_score(self, symbol: str) -> float:
        try:
            raw = self._redis.redis_client.get(f"{_REDIS_KEY_PREFIX}{symbol}")
            if raw is None:
                return 0.0
            return float(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception:
            return 0.0

    async def _loop(self) -> None:
        """Опрос Twitter."""
        while True:
            try:
                if self._twitter_on:
                    await self._poll_twitter()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("SentimentPoller: poll error — %s", exc)
            await asyncio.sleep(self._poll_interval)

    async def _poll_twitter(self) -> None:
        symbols = list(self._symbols)
        if not symbols:
            return
        results = await asyncio.gather(
            *[self._twitter.get_sentiment(sym) for sym in symbols],
            return_exceptions=True,
        )
        stored = 0
        for sym, result in zip(symbols, results):
            if isinstance(result, BaseException):
                logger.debug("SentimentPoller: %s error — %s", sym, result)
                continue
            try:
                self._redis.redis_client.setex(
                    f"{_REDIS_KEY_PREFIX}{sym}",
                    self._cache_ttl,
                    str(round(float(result), 4)),
                )
                stored += 1
            except Exception as exc:
                logger.debug("SentimentPoller: redis write %s — %s", sym, exc)
        logger.debug("SentimentPoller: twitter %d/%d symbols", stored, len(symbols))
