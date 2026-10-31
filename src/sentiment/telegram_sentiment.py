"""
Telegram-каналы: свежие посты в реальном времени и точное упоминание монет.

Вход выполняется в приложении (src/sentiment/telegram_login.py); здесь бот
только подключается с сохранённой StringSession и никогда не просит код.

Посты приходят событием NewMessage — бот подписан на каналы от аккаунта
пользователя, поэтому Telegram присылает их сразу, без опроса. Они лежат в
скользящем окне (TELEGRAM_WINDOW_MIN), и оценка монеты считается из окна
мгновенно. Что происходит с подключением — в Redis `STATUS_KEY`.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter, deque
from functools import lru_cache
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    List,
    NamedTuple,
    Optional,
    Pattern,
    Set,
    Tuple,
)

from config import Config
from src.settings import secrets_store

logger = logging.getLogger(__name__)

STATUS_KEY = "bot:telegram:status"
# Не чаще раза в минуту: без сессии или при сбое сети не долбим Telegram.
_RETRY_S = 60.0
# ponytail: потолок окна в памяти — ~80 постов в минуту за час; больше не держим.
_MAX_POSTS = 5000

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

# Тикеры, которые совпадают с обычными словами: считаем их только с $, # или
# парой к доллару, иначе «BUY NOW» в каждом посте стало бы упоминанием монет.
_COMMON_WORDS = frozenset(
    {w.upper() for w in _BULLISH_KEYWORDS | _BEARISH_KEYWORDS}
    | {
        "THE",
        "AND",
        "FOR",
        "NOW",
        "NEW",
        "ALL",
        "TOP",
        "ONE",
        "BIG",
        "HOT",
        "WIN",
        "FUN",
        "SUN",
        "GAS",
        "CAT",
        "DOG",
        "HOLD",
        "LONG",
        "SHORT",
        "USD",
        "USDT",
        "USDC",
        "CEO",
        "NFT",
        "APP",
        "AI",
        "OK",
        "GO",
        "UP",
        "PEOPLE",
        "TRUMP",
    }
)


class Post(NamedTuple):
    at: float  # время поста (unix)
    channel: str
    text: str


@lru_cache(maxsize=4096)
def _patterns(base: str) -> Tuple[Pattern[str], Optional[Pattern[str]]]:
    """(с пометкой: $WIF, #WIF, WIF/USDT, WIFUSDT — регистр любой;
    просто слово: только заглавными, от 3 знаков и не обычное слово).

    1000PEPE на бирже — это PEPE в постах.
    """
    names = [base]
    short = re.fullmatch(r"10+([A-Z].*)", base)
    if short:
        names.append(short.group(1))
    alt = "|".join(re.escape(n) for n in names)
    marked = re.compile(
        rf"(?<![\w$#])(?:[$#](?:{alt})|(?:{alt})\s?[/-]?\s?USD[TC]?)(?!\w)",
        re.IGNORECASE,
    )
    plain = [re.escape(n) for n in names if len(n) >= 3 and n not in _COMMON_WORDS]
    word = re.compile(rf"(?<![\w$#])(?:{'|'.join(plain)})(?!\w)") if plain else None
    return marked, word


def mentions(text: str, base: str) -> bool:
    """Упомянута ли монета точно: никаких подстрок вроде «wifi» для WIF."""
    marked, word = _patterns(base.upper())
    return bool(marked.search(text) or (word and word.search(text)))


def post_score(text: str) -> float:
    """+0.1 за упоминание, +0.2 за бычье слово, -0.3 за медвежье."""
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    return (
        0.1
        + 0.2 * len(words & _BULLISH_KEYWORDS)
        - 0.3 * len(words & _BEARISH_KEYWORDS)
    )


def channel_names(raw: Any) -> List[str]:
    """«@Chan, https://t.me/chan2 -100123» → ['chan', 'chan2', '-100123']."""
    out: List[str] = []
    for part in re.split(r"[,\s]+", str(raw or "")):
        name = re.sub(r"^(?:https?://)?(?:t\.me/|telegram\.me/)?@?", "", part)
        name = name.strip("/").lower()
        if name and name not in out:
            out.append(name)
    return out


class TelegramSentiment:
    """
    Сентимент по Telegram-каналам из окна свежих постов.

    Возвращает (0.0, 0 упоминаний), пока нет ключей, входа или постов.
    """

    def __init__(self, redis: Any = None) -> None:
        self._redis = redis
        self._channels: List[str] = channel_names(Config.TELEGRAM_SENTIMENT_CHANNELS)
        self.window_s = float(Config.TELEGRAM_WINDOW_MIN) * 60
        self._posts: Deque[Post] = deque(maxlen=_MAX_POSTS)
        self._client: Optional[Any] = None
        self._session = ""  # с какой сессией подключён клиент
        self._retry_at = 0.0
        self._revoked = ""  # сессия, которую Telegram отклонил: ждём новый вход
        self._watched: Set[str] = set()  # каналы, на которые уже подписаны
        self._ids: Dict[int, str] = {}  # id чата Telegram → имя канала
        self._failed: Dict[str, str] = {}  # канал → почему не читается
        self.state = "off"
        self._message = ""
        # Каждый живой пост (не догрузка истории) — для сделок по доверенным
        # каналам; вызывается сразу из обработчика события.
        self.on_post: Optional[Callable[[Post], Awaitable[None]]] = None

    # ── Настройки и статус ────────────────────────────────────────────────────

    def configure(self, channels: List[str], window_min: float) -> None:
        """Каналы и окно из настроек — каждый цикл, без перезапуска."""
        self._channels = list(channels)
        self.window_s = max(60.0, float(window_min) * 60)

    def _status(self, state: str, message: str) -> None:
        self.state, self._message = state, message
        self.report()

    def report(self, symbols: Optional[List[str]] = None) -> None:
        """Статус для приложения: подключение, посты в окне, упоминания монет."""
        if self._redis is None:
            return
        payload: Dict[str, Any] = {
            "state": self.state,
            "message": self._message,
            "at": time.time(),
            "channels": self._channels,
            "failed": {c: r for c, r in self._failed.items() if c in self._channels},
            "window_min": round(self.window_s / 60),
        }
        if self.state == "ok":
            fresh = self.fresh()
            payload["posts"] = len(fresh)
            n = Counter(
                s.split("/")[0]
                for s in symbols or []
                for p in fresh
                if mentions(p.text, s.split("/")[0])
            )
            payload["mentions"] = dict(n.most_common(10))
        try:
            self._redis.redis_client.set(
                STATUS_KEY, json.dumps(payload, ensure_ascii=False)
            )
        except Exception as e:
            logger.debug("TelegramSentiment: status not saved: %s", e)

    # ── Подключение ───────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Подключается с сессией из хранилища и подписывается на каналы.

        Безопасно звать каждый цикл: хранилище перечитывается не чаще раза
        в _RETRY_S, так что новый вход, удаление ключей и обрыв связи
        подхватываются без перезапуска бота; новые каналы — сразу.
        """
        now = time.monotonic()
        if now >= self._retry_at:
            self._retry_at = now + _RETRY_S
            await self._connect()
        if self._client is not None and set(self._channels) - self._watched:
            await self._watch()

    async def _connect(self) -> None:
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
        from telethon import TelegramClient, events  # noqa: PLC0415
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
        client.add_event_handler(self._on_message, events.NewMessage())
        self._client, self._session = client, session
        self._watched, self._ids, self._failed = set(), {}, {}
        logger.info("TelegramSentiment: connected to Telegram MTProto")
        self._status("ok", "Подключено, посты приходят сразу.")

    async def _watch(self) -> None:
        """Подписка на новые каналы и догрузка их постов за окно."""
        from telethon import errors, utils  # noqa: PLC0415
        from telethon.tl.functions.channels import JoinChannelRequest  # noqa

        client = self._client
        todo = [c for c in self._channels if c not in self._watched]
        if any(c.lstrip("-").isdigit() for c in todo):
            await client.get_dialogs()  # приватные каналы по id — только из диалогов
        for name in todo:
            self._watched.add(name)
            try:
                entity = await client.get_entity(
                    int(name) if name.lstrip("-").isdigit() else name
                )
                if getattr(entity, "left", False):
                    await client(JoinChannelRequest(entity))
                self._ids[utils.get_peer_id(entity)] = name
                await self._backfill(entity, name)
                self._failed.pop(name, None)
            except errors.FloodWaitError as e:
                self._watched.discard(name)
                self._failed[name] = f"Telegram просит подождать {e.seconds} с"
                break
            except Exception as e:
                logger.warning("TelegramSentiment: channel %s — %s", name, e)
                self._failed[name] = "канал не найден или закрыт"
        self.report()

    async def _backfill(self, entity: Any, name: str) -> None:
        since = time.time() - self.window_s
        async for msg in self._client.iter_messages(entity, limit=100):
            at = msg.date.timestamp()
            if at < since:
                break
            if msg.raw_text:
                self._posts.append(Post(at, name, msg.raw_text))

    async def _on_message(self, event: Any) -> None:
        name = self._ids.get(event.chat_id)
        if name is None or name not in self._channels or not event.raw_text:
            return  # личные чаты и чужие каналы не храним
        post = Post(event.message.date.timestamp(), name, event.raw_text)
        self._posts.append(post)
        if self.on_post is not None:
            await self.on_post(post)

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

    # ── Оценка ────────────────────────────────────────────────────────────────

    def fresh(self) -> List[Post]:
        cutoff = time.time() - self.window_s
        while self._posts and self._posts[0].at < cutoff:
            self._posts.popleft()
        return [p for p in self._posts if p.at >= cutoff]

    def snapshot(self, symbol: str) -> Tuple[float, int]:
        """(оценка -1..1, сколько свежих постов упоминают монету).

        0 упоминаний — «нет данных», а не нейтральное настроение.
        """
        base = symbol.split("/")[0]
        score, n = 0.0, 0
        for p in self.fresh():
            if mentions(p.text, base):
                n += 1
                score += post_score(p.text)
        return round(max(-1.0, min(1.0, score)), 4), n
