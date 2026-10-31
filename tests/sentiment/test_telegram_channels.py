"""Каналы Telegram: точное упоминание монет, окно свежих постов, живые посты."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.sentiment import telegram_sentiment as ts
from src.sentiment.sentiment_poller import SentimentPoller, combine
from src.sentiment.telegram_sentiment import (
    Post,
    TelegramSentiment,
    channel_names,
    mentions,
)

# ── точное упоминание ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "🚀 $WIF breaking out, next stop $5",
        "#WIF holders, are you ready?",
        "LONG WIF/USDT entry 2.31 SL 2.15 TP 2.80",
        "wifusdt perp: funding flipped",
        "Buy WIF now before the listing",
        "Заходим в WIF, цель +30%",
        "$wif",
    ],
)
def test_wif_is_mentioned(text):
    assert mentions(text, "WIF")


@pytest.mark.parametrize(
    "text",
    [
        "my wifi is down again lol",
        "WIFI module on sale",
        "Wif is a dog in a hat",  # просто слово не заглавными — не тикер
        "$WIFE token launches today",
        "SWIF airdrop",
    ],
)
def test_wif_is_not_a_substring_match(text):
    assert not mentions(text, "WIF")


def test_common_words_need_a_marker():
    assert not mentions("someone said ONE more pump", "ONE")
    assert not mentions("PUMP IT NOW!!!", "PUMP")
    assert mentions("$ONE looks strong", "ONE")
    assert mentions("PUMP/USDT long", "PUMP")


def test_short_tickers_and_multiplied_contracts():
    assert not mentions("A big day for crypto", "A")
    assert mentions("$A is moving", "A")
    assert mentions("PEPE about to send", "1000PEPE")  # контракт 1000PEPE
    assert mentions("$PEPE", "1000PEPE")
    assert not mentions("PEPE about to send", "1000BONK")


def test_channel_names_normalised():
    assert channel_names("@Chan, https://t.me/chan2/ -100123\nchan") == [
        "chan",
        "chan2",
        "-100123",
    ]


# ── окно и оценка ─────────────────────────────────────────────────────────────


def _tg(*posts):
    tg = TelegramSentiment()
    tg.configure(["alpha"], 60)
    for age_min, text in posts:
        tg._posts.append(Post(time.time() - age_min * 60, "alpha", text))
    return tg


def test_snapshot_counts_only_fresh_exact_mentions():
    tg = _tg(
        (90, "$WIF moon"),  # старше окна
        (10, "$WIF pump incoming, bullish"),
        (5, "my wifi broke"),
        (1, "WIF rug warning, avoid"),
    )
    score, n = tg.snapshot("WIF/USDT:USDT")
    assert n == 2
    # (0.1 + 0.2*2) + (0.1 - 0.3*3)
    assert score == pytest.approx(-0.3)
    assert tg.snapshot("PEPE/USDT") == (0.0, 0)  # нет данных ≠ нейтрально


def test_window_follows_settings():
    tg = _tg((30, "$WIF"))
    assert tg.snapshot("WIF/USDT")[1] == 1
    tg.configure(["alpha"], 15)
    assert tg.snapshot("WIF/USDT")[1] == 0


def test_poller_combines_sources():
    assert combine(0.0, 0.6) == 0.6 and combine(0.3, 0.0) == 0.3
    assert combine(0.3, 0.6) == 0.5


async def test_poller_twitter_only_in_meme():
    tg = _tg((1, "$WIF moon"))
    redis = SimpleNamespace(redis_client=SimpleNamespace(get=lambda k: "0.9"))
    p = SentimentPoller(None, tg, redis)
    p._twitter_on = False
    assert await p.get_score("WIF/USDT") == (0.3, 1)
    p._twitter_on = True
    assert await p.get_score("WIF/USDT") == (0.5, 1)


# ── живые посты ───────────────────────────────────────────────────────────────


def _event(chat_id, text):
    return SimpleNamespace(
        chat_id=chat_id,
        raw_text=text,
        message=SimpleNamespace(date=datetime.now(timezone.utc)),
    )


async def test_live_posts_only_from_configured_channels():
    tg = TelegramSentiment()
    tg.configure(["alpha", "beta"], 60)
    tg._ids = {-1001: "alpha", -1002: "beta", -1003: "gone"}
    seen = []

    async def on_post(post):
        seen.append(post)

    tg.on_post = on_post
    await tg._on_message(_event(-1001, "$WIF long"))
    await tg._on_message(_event(42, "private chat about $WIF"))  # личный чат
    await tg._on_message(_event(-1003, "$WIF"))  # канал убрали из настроек
    await tg._on_message(_event(-1002, ""))  # картинка без подписи
    assert [p.channel for p in seen] == ["alpha"]
    assert tg.snapshot("WIF/USDT")[1] == 1


class FakeClient:
    """Telethon-клиент для подписки: alpha уже читаем, beta — нет, ghost нет."""

    def __init__(self):
        self.joined = []

    async def get_dialogs(self):
        return []

    async def get_entity(self, name):
        if name == "ghost":
            raise ValueError("No user has 'ghost' as username")
        return SimpleNamespace(
            name=name, left=name == "beta", title=f"{name.title()} Calls"
        )

    async def __call__(self, request):
        self.joined.append(request.channel.name)

    def iter_messages(self, entity, limit):
        now = datetime.now(timezone.utc).timestamp()

        async def gen():
            for age, text in ((5, f"$WIF from {entity.name}"), (120, "$WIF old")):
                yield SimpleNamespace(
                    date=datetime.fromtimestamp(now - age * 60, timezone.utc),
                    raw_text=text,
                )

        return gen()


async def test_watch_joins_new_channels_and_backfills_the_window(monkeypatch):
    from telethon import utils

    ids = {"alpha": -1001, "beta": -1002}
    monkeypatch.setattr(utils, "get_peer_id", lambda e: ids[e.name])
    tg = TelegramSentiment()
    tg.configure(["alpha", "beta", "ghost"], 60)
    tg._client = client = FakeClient()
    await tg._watch()
    assert client.joined == ["beta"]  # на alpha уже подписаны
    assert tg._ids == {-1001: "alpha", -1002: "beta"}
    assert "ghost" in tg._failed
    assert tg._titles == {"alpha": "Alpha Calls", "beta": "Beta Calls"}
    assert tg.snapshot("WIF/USDT")[1] == 2  # по посту из окна с каждого канала
    await tg._watch()  # повторно не подписываемся
    assert client.joined == ["beta"]


def test_status_shows_posts_and_mentions():
    saved = {}
    tg = _tg((1, "$WIF moon"), (2, "nothing here"))
    tg._redis = SimpleNamespace(
        redis_client=SimpleNamespace(set=lambda k, v: saved.update({k: v}))
    )
    tg.state = "ok"
    tg.report(["WIF/USDT", "PEPE/USDT"])
    st = json.loads(saved[ts.STATUS_KEY])
    assert st["posts"] == 2 and st["mentions"] == {"WIF": 1}
    assert st["window_min"] == 60


def test_status_names_each_channel_and_what_happens_to_it():
    saved = {}
    tg = _tg((3, "$WIF moon"), (90, "old post"))
    tg.configure(["alpha", "beta", "gamma"], 60)
    tg._redis = SimpleNamespace(
        redis_client=SimpleNamespace(set=lambda k, v: saved.update({k: v}))
    )
    tg.state, tg._ids = "ok", {-1001: "alpha"}
    tg._titles = {"alpha": "Alpha Calls"}
    tg._failed = {"gamma": "канал не найден или закрыт"}
    tg.report([])
    per = json.loads(saved[ts.STATUS_KEY])["per_channel"]
    assert per["alpha"]["title"] == "Alpha Calls"
    assert per["alpha"]["state"] == "reading" and per["alpha"]["posts"] == 1
    assert per["alpha"]["last_at"] == pytest.approx(time.time() - 180, abs=5)
    assert per["beta"] == {
        "title": "",
        "state": "connecting",
        "why": "",
        "posts": 0,
        "last_at": None,
    }
    assert per["gamma"]["state"] == "failed" and per["gamma"]["why"]
