"""
Фоновый опросчик сентимента: агрегирует оценки Twitter + Telegram в Redis.
Запускается как asyncio-задача параллельно с основным торговым циклом.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List

from config import Config

logger = logging.getLogger(__name__)

_REDIS_KEY_PREFIX = "bot:sentiment:"


class SentimentPoller:
    """
    Опрашивает Twitter и Telegram по расписанию для списка символов.

    Сохраняет агрегированную оценку в Redis: bot:sentiment:{symbol} с TTL.
    Оба источника опциональны — отсутствие ключей → 0.0 без ошибок.

    Использование:
        poller = SentimentPoller(twitter, telegram, redis_client)
        await poller.start(symbols)
        score = await poller.get_score("PEPE/USDT")
        await poller.stop()
    """

    def __init__(self, twitter, telegram, redis_client) -> None:
        self._twitter = twitter
        self._telegram = telegram
        self._redis = redis_client
        self._task: asyncio.Task | None = None
        self._symbols: List[str] = []
        self._poll_interval = Config.SENTIMENT_POLL_INTERVAL
        self._cache_ttl = Config.SENTIMENT_CACHE_TTL

    async def start(self, symbols: List[str]) -> None:
        """Запускает фоновый цикл опроса для переданного списка символов."""
        self._symbols = list(symbols)
        await self._telegram.start()
        self._task = asyncio.create_task(self._loop(), name="sentiment_poller")
        logger.info(
            "SentimentPoller: started for %d symbols, interval=%ds",
            len(symbols),
            self._poll_interval,
        )

    async def stop(self) -> None:
        """Останавливает цикл опроса и отключает Telegram-клиент."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._telegram.stop()
        logger.info("SentimentPoller: stopped")

    def update_symbols(self, symbols: List[str]) -> None:
        """Обновляет список символов на лету при обновлении топ-N сканером."""
        self._symbols = list(symbols)

    async def get_score(self, symbol: str) -> float:
        """Читает оценку сентимента из Redis. Возвращает 0.0 если недоступно."""
        try:
            raw = self._redis.redis_client.get(f"{_REDIS_KEY_PREFIX}{symbol}")
            if raw is None:
                return 0.0
            val = raw.decode() if isinstance(raw, bytes) else raw
            return float(val)
        except Exception:
            return 0.0

    async def _loop(self) -> None:
        """Основной цикл опроса."""
        while True:
            try:
                await self._poll_all()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("SentimentPoller: poll error — %s", exc)
            await asyncio.sleep(self._poll_interval)

    async def _poll_all(self) -> None:
        symbols = list(self._symbols)
        if not symbols:
            return

        results = await asyncio.gather(
            *[self._fetch_combined(sym) for sym in symbols],
            return_exceptions=True,
        )

        stored = 0
        for sym, result in zip(symbols, results):
            if isinstance(result, BaseException):
                logger.debug("SentimentPoller: %s error — %s", sym, result)
                continue
            score: float = result
            try:
                self._redis.redis_client.setex(
                    f"{_REDIS_KEY_PREFIX}{sym}",
                    self._cache_ttl,
                    str(round(score, 4)),
                )
                stored += 1
            except Exception as exc:
                logger.debug("SentimentPoller: redis write %s — %s", sym, exc)

        logger.debug("SentimentPoller: updated %d/%d symbols", stored, len(symbols))

    async def _fetch_combined(self, symbol: str) -> float:
        """Запрашивает Twitter + Telegram, возвращает взвешенное среднее.

        Telegram весит 2×: памп-группы сигнализируют раньше Twitter.
        """
        results = await asyncio.gather(
            self._twitter.get_sentiment(symbol),
            self._telegram.get_sentiment(symbol),
            return_exceptions=True,
        )
        tw_raw, tg_raw = results[0], results[1]
        tw = float(tw_raw) if not isinstance(tw_raw, BaseException) else 0.0
        tg = float(tg_raw) if not isinstance(tg_raw, BaseException) else 0.0

        if tw == 0.0 and tg == 0.0:
            return 0.0
        if tw == 0.0:
            return tg
        if tg == 0.0:
            return tw
        # Telegram weighted 2x — pump groups signal earlier than Twitter
        return round((tw + tg * 2) / 3, 4)
