"""
Анализатор Telegram-сентимента для обнаружения памп-активности мемкоинов.
Мониторит публичные крипто-каналы через Telethon MTProto.

Вход выполняется в приложении (src/sentiment/telegram_login.py); здесь бот
только подключается с сохранённой StringSession и никогда не просит код.
Что происходит с подключением — в Redis `STATUS_KEY` для приложения.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from config import Config
from src.settings import secrets_store

logger = logging.getLogger(__name__)

STATUS_KEY = "bot:telegram:status"
# Не чаще раза в минуту: без сессии или при сбое сети не долбим Telegram.
_RETRY_S = 60.0

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
    Возвращает 0.0, пока нет ключей или входа.
    """

    def __init__(self, redis: Any = None) -> None:
        self._redis = redis
        self._channels: List[str] = [
            ch.strip()
            for ch in Config.TELEGRAM_SENTIMENT_CHANNELS.split(",")
            if ch.strip()
        ]
        self._client: Optional[Any] = None
        self._session = ""  # с какой сессией подключён клиент
        self._retry_at = 0.0
        self._revoked = ""  # сессия, которую Telegram отклонил: ждём новый вход
        self._cache: Dict[str, Tuple[float, float]] = {}
        self._cache_ttl = float(Config.SENTIMENT_CACHE_TTL)
        self.state = "off"

    def _status(self, state: str, message: str) -> None:
        self.state = state
        if self._redis is None:
            return
        payload = {
            "state": state,
            "message": message,
            "at": time.time(),
            "channels": self._channels,
        }
        try:
            self._redis.redis_client.set(
                STATUS_KEY, json.dumps(payload, ensure_ascii=False)
            )
        except Exception as e:
            logger.debug("TelegramSentiment: status not saved: %s", e)

    async def start(self) -> None:
        """Подключается с сессией из хранилища; без неё — только честный статус.

        Безопасно звать каждый цикл: хранилище перечитывается не чаще раза
        в _RETRY_S, так что новый вход, удаление ключей и обрыв связи
        подхватываются без перезапуска бота.
        """
        now = time.monotonic()
        if now < self._retry_at:
            return
        self._retry_at = now + _RETRY_S
        api_id, api_hash, _ = (
            secrets_store.get_value(k) or "" for k in secrets_store.TG_KEYS
        )
        session = secrets_store.get_value(secrets_store.TG_SESSION_ENV) or ""
        if self._client is not None:
            if self._client.is_connected() and session == self._session:
                return
            await self._disconnect()  # вход удалён, заменён или связь потеряна
        if not (api_id and api_hash):
            self._status("off", "Ключи не заданы — каналы не читаются.")
            return
        if not session:
            self._status(
                "no_session",
                "Вход не выполнен: нажмите «Войти» на карточке «Памп-каналы "
                "Telegram».",
            )
            return
        if session == self._revoked:
            return
        from telethon import TelegramClient  # noqa: PLC0415
        from telethon.sessions import StringSession  # noqa: PLC0415

        client: Optional[Any] = None
        try:
            string_session = StringSession(session)
        except Exception:
            authorized = False  # строка повреждена — как отозванный вход
        else:
            try:
                client = TelegramClient(string_session, int(api_id), api_hash)
                await client.connect()
                authorized = await client.is_user_authorized()
            except Exception as exc:
                logger.warning("TelegramSentiment: connection failed — %s", exc)
                self._status("error", "Нет связи с Telegram — повторю через минуту.")
                return
        if not authorized:
            if client is not None:
                await client.disconnect()
            self._revoked = session
            self._status(
                "revoked",
                "Telegram больше не принимает этот вход (сеанс завершён в "
                "настройках Telegram или сменился пароль). Нажмите «Войти» заново.",
            )
            return
        self._client, self._session = client, session
        logger.info("TelegramSentiment: connected to Telegram MTProto")
        self._status("ok", f"Подключено, каналов: {len(self._channels)}.")

    async def _disconnect(self) -> None:
        if self._client:
            try:
                await self._client.disconnect()
            except Exception:
                pass
            self._client = None

    async def stop(self) -> None:
        """Отключает Telethon-клиент."""
        await self._disconnect()
        self._status("stopped", "Чтение каналов остановлено.")

    async def get_sentiment(self, symbol: str) -> float:
        """
        Возвращает оценку сентимента -1.0..1.0 на основе недавних упоминаний в Telegram.
        0.0 если отключён или при ошибке.

        :param symbol: Символ в формате ccxt ('PEPE/USDT').
        :return: Оценка сентимента -1.0..1.0.
        """
        if not self._client:
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
