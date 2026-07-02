"""
Анализатор Telegram-сентимента для обнаружения памп-активности мемкоинов.
Мониторит публичные крипто-каналы через Telethon MTProto.
Молча пропускается если TELEGRAM_SENTIMENT_API_ID не задан.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from config import Config

logger = logging.getLogger(__name__)

_BULLISH_KEYWORDS = frozenset(
    {
        "pump",
        "moon",
        "buy",
        "bullish",
        "launch",
        "gem",
        "100x",
        "1000x",
        "listing",
        "breakout",
        "surge",
        "rally",
        "ath",
    }
)

_BEARISH_KEYWORDS = frozenset(
    {
        "dump",
        "sell",
        "bearish",
        "crash",
        "scam",
        "rug",
        "rugpull",
        "exit",
        "warning",
        "avoid",
    }
)


class TelegramSentiment:
    """
    Анализатор сентимента через публичные Telegram крипто-каналы.

    Использует Telethon MTProto для чтения последних сообщений из настроенных
    каналов и оценки упоминаний тикеров конкретных монет.
    Молча возвращает 0.0 если ключи не заданы.

    Session-файл: data/tg_sentiment.session (создаётся при первой авторизации).
    """

    def __init__(self) -> None:
        self._api_id = Config.TELEGRAM_SENTIMENT_API_ID
        self._api_hash = Config.TELEGRAM_SENTIMENT_API_HASH
        self._phone = Config.TELEGRAM_SENTIMENT_PHONE
        self._channels: List[str] = [
            ch.strip()
            for ch in Config.TELEGRAM_SENTIMENT_CHANNELS.split(",")
            if ch.strip()
        ]
        self._enabled = bool(self._api_id and self._api_hash)
        self._client: Optional[Any] = None
        self._cache: Dict[str, Tuple[float, float]] = {}
        self._cache_ttl = float(Config.SENTIMENT_CACHE_TTL)

        if not self._enabled:
            logger.debug("TelegramSentiment: credentials not set — sentiment disabled")

    async def start(self) -> None:
        """Подключает и аутентифицирует Telethon-клиент.

        Ничего не делает если credentials не заданы (отключён).
        """
        if not self._enabled:
            return
        try:
            from telethon import TelegramClient

            self._client = TelegramClient(
                "data/tg_sentiment",
                int(self._api_id),
                self._api_hash,
            )
            await self._client.start(phone=self._phone)
            logger.info("TelegramSentiment: connected to Telegram MTProto")
        except Exception as exc:
            logger.warning("TelegramSentiment: connection failed — %s", exc)
            self._client = None

    async def stop(self) -> None:
        """Отключает Telethon-клиент."""
        if self._client:
            try:
                await self._client.disconnect()
            except Exception:
                pass
            self._client = None

    async def get_sentiment(self, symbol: str) -> float:
        """
        Возвращает оценку сентимента -1.0..1.0 на основе недавних упоминаний в Telegram.
        0.0 если отключён или при ошибке.

        :param symbol: Символ в формате ccxt ('PEPE/USDT').
        :return: Оценка сентимента -1.0..1.0.
        """
        if not self._enabled or not self._client:
            return 0.0

        now = time.monotonic()
        cached = self._cache.get(symbol)
        if cached and now - cached[1] < self._cache_ttl:
            return cached[0]

        try:
            score = await self._scan_channels(symbol)
            self._cache[symbol] = (score, now)
            return score
        except Exception as exc:
            logger.debug("TelegramSentiment: error for %s: %s", symbol, exc)
            return 0.0

    async def _scan_channels(self, symbol: str) -> float:
        """
        Сканирует настроенные каналы на упоминания тикера в последних 100 сообщениях.

        Оценка за упоминание:
        +0.1 базовых, +0.2 за каждое бычье ключевое слово рядом, -0.3 за медвежье.
        Результат ограничен [-1.0, 1.0].
        """
        ticker = symbol.split("/")[0].upper()
        score = 0.0
        mention_count = 0

        for channel in self._channels:
            try:
                messages = await self._client.get_messages(channel, limit=100)
                for msg in messages:
                    text = (msg.text or "").lower()
                    if ticker.lower() not in text:
                        continue
                    mention_count += 1
                    score += 0.1
                    words = set(text.split())
                    score += 0.2 * len(words & _BULLISH_KEYWORDS)
                    score -= 0.3 * len(words & _BEARISH_KEYWORDS)
            except Exception as exc:
                logger.debug("TelegramSentiment: channel %s error: %s", channel, exc)

        result = max(-1.0, min(1.0, score))
        logger.debug(
            "TelegramSentiment: %s — %d mentions, score=%.3f",
            ticker,
            mention_count,
            result,
        )
        return round(result, 4)
