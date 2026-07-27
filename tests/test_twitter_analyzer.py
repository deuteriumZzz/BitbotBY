"""Тесты для TwitterAnalyzer — официальный API + twscrape fallback."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.twitter_analyzer import TwitterAnalyzer, _parse_twscrape_accounts


# ---------------------------------------------------------------------------
# _parse_twscrape_accounts()
# ---------------------------------------------------------------------------


def test_parse_single_account():
    accounts = _parse_twscrape_accounts("user1:pass1:mail1@x.com:mailpass1")
    assert accounts == [("user1", "pass1", "mail1@x.com", "mailpass1")]


def test_parse_multiple_accounts():
    accounts = _parse_twscrape_accounts(
        "user1:pass1:mail1@x.com:mailpass1,user2:pass2:mail2@x.com:mailpass2"
    )
    assert accounts == [
        ("user1", "pass1", "mail1@x.com", "mailpass1"),
        ("user2", "pass2", "mail2@x.com", "mailpass2"),
    ]


def test_parse_empty_string_gives_empty_list():
    assert _parse_twscrape_accounts("") == []


def test_parse_skips_malformed_entry():
    accounts = _parse_twscrape_accounts(
        "user1:pass1:mail1@x.com:mailpass1,badentry:onlytwofields"
    )
    assert accounts == [("user1", "pass1", "mail1@x.com", "mailpass1")]


# ---------------------------------------------------------------------------
# Backend selection (__init__)
# ---------------------------------------------------------------------------


def _make_analyzer(token: str, twscrape_accounts: str) -> TwitterAnalyzer:
    with patch("src.twitter_analyzer.Config") as mock_config:
        mock_config.TWITTER_BEARER_TOKEN = token
        mock_config.TWITTER_MAX_RESULTS = 10
        mock_config.TWSCRAPE_ACCOUNTS = twscrape_accounts
        mock_config.TWSCRAPE_DB_PATH = "data/twscrape_accounts.db"
        mock_config.SENTIMENT_CACHE_TTL = 300
        return TwitterAnalyzer()


def test_official_token_takes_priority_over_twscrape():
    analyzer = _make_analyzer(token="tok123", twscrape_accounts="u:p:e:ep")
    assert analyzer._backend == "api"
    assert analyzer._enabled is True


def test_falls_back_to_twscrape_without_token():
    analyzer = _make_analyzer(token="", twscrape_accounts="u:p:e:ep")
    assert analyzer._backend == "twscrape"
    assert analyzer._enabled is True


def test_disabled_without_token_or_accounts():
    analyzer = _make_analyzer(token="", twscrape_accounts="")
    assert analyzer._backend is None
    assert analyzer._enabled is False


@pytest.mark.asyncio
async def test_disabled_analyzer_returns_zero():
    analyzer = _make_analyzer(token="", twscrape_accounts="")
    assert await analyzer.get_sentiment("BTC/USDT") == 0.0


# ---------------------------------------------------------------------------
# twscrape fetch path (mocked sys.modules["twscrape"], no real dependency needed)
# ---------------------------------------------------------------------------


class _FakeTweet:
    def __init__(self, text: str) -> None:
        self.rawContent = text


def _fake_twscrape_module(tweet_texts):
    async def _search_gen(query, limit=10):
        for text in tweet_texts:
            yield _FakeTweet(text)

    fake_api_instance = MagicMock()
    fake_api_instance.pool.add_account = AsyncMock()
    fake_api_instance.pool.login_all = AsyncMock()
    fake_api_instance.search = MagicMock(side_effect=_search_gen)

    fake_module = MagicMock()
    fake_module.API = MagicMock(return_value=fake_api_instance)
    return fake_module, fake_api_instance


@pytest.mark.asyncio
async def test_twscrape_fetch_positive_sentiment():
    analyzer = _make_analyzer(token="", twscrape_accounts="u1:p1:e1:ep1")
    fake_module, fake_api = _fake_twscrape_module(
        ["I love this coin, amazing gains!", "great project, bullish!"]
    )
    with patch.dict("sys.modules", {"twscrape": fake_module}):
        score = await analyzer.get_sentiment("BTC/USDT")
    assert score > 0
    fake_api.pool.add_account.assert_awaited_once_with("u1", "p1", "e1", "ep1")
    fake_api.pool.login_all.assert_awaited_once()


@pytest.mark.asyncio
async def test_twscrape_fetch_no_tweets_returns_zero():
    analyzer = _make_analyzer(token="", twscrape_accounts="u1:p1:e1:ep1")
    fake_module, _ = _fake_twscrape_module([])
    with patch.dict("sys.modules", {"twscrape": fake_module}):
        score = await analyzer.get_sentiment("BTC/USDT")
    assert score == 0.0


@pytest.mark.asyncio
async def test_twscrape_pool_logged_in_only_once_across_calls():
    analyzer = _make_analyzer(token="", twscrape_accounts="u1:p1:e1:ep1")
    fake_module, fake_api = _fake_twscrape_module(["neutral text"])
    with patch.dict("sys.modules", {"twscrape": fake_module}):
        await analyzer.get_sentiment("BTC/USDT")
        # разные символы — кэш sentiment не должен мешать повторному login
        fake_api.search = MagicMock(
            side_effect=lambda query, limit=10: _fake_twscrape_module(["neutral"])[0]
        )
        await analyzer.get_sentiment("ETH/USDT")
    fake_api.pool.login_all.assert_awaited_once()


@pytest.mark.asyncio
async def test_twscrape_error_is_swallowed_and_returns_zero():
    analyzer = _make_analyzer(token="", twscrape_accounts="u1:p1:e1:ep1")
    fake_module = MagicMock()
    fake_module.API = MagicMock(side_effect=RuntimeError("boom"))
    with patch.dict("sys.modules", {"twscrape": fake_module}):
        score = await analyzer.get_sentiment("BTC/USDT")
    assert score == 0.0


@pytest.mark.asyncio
async def test_cache_hit_skips_backend_call():
    analyzer = _make_analyzer(token="", twscrape_accounts="u1:p1:e1:ep1")
    analyzer._cache["BTC/USDT"] = (0.55, __import__("time").monotonic())
    fake_module, fake_api = _fake_twscrape_module(["should not be read"])
    with patch.dict("sys.modules", {"twscrape": fake_module}):
        score = await analyzer.get_sentiment("BTC/USDT")
    assert score == 0.55
    fake_api.pool.add_account.assert_not_awaited()
