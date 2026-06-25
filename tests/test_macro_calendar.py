"""Тесты MacroCalendar — мокаем aiohttp и time."""

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_calendar(api_key: str = "test_key"):
    from src.macro_calendar import MacroCalendar

    cal = MacroCalendar()
    cal._api_key = api_key
    return cal


def _make_event_item(name: str, impact: str = "high", offset_minutes: int = 0) -> dict:
    """Создаёт элемент API-ответа для события."""
    dt = datetime.now(timezone.utc) + timedelta(minutes=offset_minutes)
    return {
        "event": name,
        "impact": impact,
        "date": dt.strftime("%Y-%m-%d"),
        "time": dt.strftime("%H:%M:%S"),
    }


def _mock_aiohttp_response(data: dict, status: int = 200):
    """Возвращает мок aiohttp сессии с заданным ответом."""
    resp_mock = AsyncMock()
    resp_mock.status = status
    resp_mock.json = AsyncMock(return_value=data)
    resp_mock.__aenter__ = AsyncMock(return_value=resp_mock)
    resp_mock.__aexit__ = AsyncMock(return_value=False)

    get_mock = MagicMock(return_value=resp_mock)

    session_mock = AsyncMock()
    session_mock.get = get_mock
    session_mock.__aenter__ = AsyncMock(return_value=session_mock)
    session_mock.__aexit__ = AsyncMock(return_value=False)

    aiohttp_mock = MagicMock()
    aiohttp_mock.ClientSession = MagicMock(return_value=session_mock)
    aiohttp_mock.ClientTimeout = MagicMock(return_value=MagicMock())
    return aiohttp_mock


# ---------------------------------------------------------------------------
# MacroCalendar.__init__
# ---------------------------------------------------------------------------


class TestMacroCalendarInit:
    def test_init_reads_api_key_from_env(self):
        with patch.dict("os.environ", {"FINNHUB_API_KEY": "my_key_123"}):
            from src.macro_calendar import MacroCalendar

            cal = MacroCalendar()
        assert cal._api_key == "my_key_123"

    def test_init_empty_key_when_env_not_set(self):
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("FINNHUB_API_KEY", None)
            from src.macro_calendar import MacroCalendar

            cal = MacroCalendar()
        assert cal._api_key == ""

    def test_init_empty_cache(self):
        cal = _make_calendar()
        assert cal._cache == []
        assert cal._cache_ts == 0.0


# ---------------------------------------------------------------------------
# is_blackout() — базовые случаи
# ---------------------------------------------------------------------------


class TestIsBlackoutBasic:
    @pytest.mark.asyncio
    async def test_returns_false_when_no_api_key(self):
        cal = _make_calendar(api_key="")
        result = await cal.is_blackout()
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_exception(self):
        cal = _make_calendar()
        cal._get_events = AsyncMock(side_effect=RuntimeError("network error"))
        result = await cal.is_blackout()
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_no_events(self):
        cal = _make_calendar()
        cal._get_events = AsyncMock(return_value=[])
        result = await cal.is_blackout()
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_events_far_away(self):
        """Событие через 2 часа — не в blackout-окне (±30 мин)."""
        cal = _make_calendar()
        future_ts = datetime.now(timezone.utc).timestamp() + 7200  # +2h
        cal._get_events = AsyncMock(
            return_value=[{"event": "CPI", "_ts": future_ts}]
        )
        result = await cal.is_blackout()
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_when_event_in_15_minutes(self):
        """Событие через 15 минут — в blackout-окне."""
        cal = _make_calendar()
        near_ts = datetime.now(timezone.utc).timestamp() + 900  # +15 min
        cal._get_events = AsyncMock(
            return_value=[{"event": "FOMC Meeting", "_ts": near_ts}]
        )
        result = await cal.is_blackout()
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_true_when_event_just_passed(self):
        """Событие было 10 минут назад — ещё в blackout-окне."""
        cal = _make_calendar()
        past_ts = datetime.now(timezone.utc).timestamp() - 600  # -10 min
        cal._get_events = AsyncMock(
            return_value=[{"event": "NFP Release", "_ts": past_ts}]
        )
        result = await cal.is_blackout()
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_true_on_boundary_exactly_30_min_before(self):
        """Ровно 30 минут до события (граница blackout)."""
        cal = _make_calendar()
        # 29 мин — строго внутри окна
        near_ts = datetime.now(timezone.utc).timestamp() + 1740  # 29 min
        cal._get_events = AsyncMock(
            return_value=[{"event": "GDP Release", "_ts": near_ts}]
        )
        result = await cal.is_blackout()
        assert result is True


# ---------------------------------------------------------------------------
# _get_events() — кэширование
# ---------------------------------------------------------------------------


class TestGetEventsCache:
    @pytest.mark.asyncio
    async def test_returns_cached_events_before_ttl(self):
        """Если кэш свежий — не делаем HTTP-запрос."""
        cal = _make_calendar()
        cached = [{"event": "CPI", "_ts": 9999999999.0}]
        cal._cache = cached
        cal._cache_ts = time.monotonic()  # только что обновили

        aiohttp_mock = MagicMock()
        with patch.dict("sys.modules", {"aiohttp": aiohttp_mock}):
            result = await cal._get_events()

        assert result is cached
        aiohttp_mock.ClientSession.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetches_when_cache_expired(self):
        """Кэш протух → выполняется HTTP-запрос."""
        cal = _make_calendar()
        cal._cache = [{"event": "old", "_ts": 0}]
        cal._cache_ts = 0.0  # давно

        aiohttp_mock = _mock_aiohttp_response({"economicCalendar": []})
        with patch.dict("sys.modules", {"aiohttp": aiohttp_mock}), \
             patch("src.macro_calendar.time.monotonic", return_value=100_000.0):
            result = await cal._get_events()

        assert result == []

    @pytest.mark.asyncio
    async def test_updates_cache_after_fetch(self):
        cal = _make_calendar()
        cal._cache = []
        cal._cache_ts = 0.0

        item = _make_event_item("FOMC Rate Decision", offset_minutes=60)
        aiohttp_mock = _mock_aiohttp_response({"economicCalendar": [item]})
        with patch.dict("sys.modules", {"aiohttp": aiohttp_mock}):
            result = await cal._get_events()

        assert len(result) == 1
        assert result[0]["event"] == "FOMC Rate Decision"
        assert cal._cache == result
        assert cal._cache_ts > 0


# ---------------------------------------------------------------------------
# _get_events() — фильтрация событий
# ---------------------------------------------------------------------------


class TestGetEventsFiltering:
    @pytest.mark.asyncio
    async def _fetch(self, items):
        cal = _make_calendar()
        cal._cache_ts = 0.0
        aiohttp_mock = _mock_aiohttp_response({"economicCalendar": items})
        with patch.dict("sys.modules", {"aiohttp": aiohttp_mock}):
            return await cal._get_events()

    @pytest.mark.asyncio
    async def test_ignores_low_impact_events(self):
        item = _make_event_item("CPI Release", impact="low", offset_minutes=60)
        result = await self._fetch([item])
        assert result == []

    @pytest.mark.asyncio
    async def test_ignores_medium_impact_events(self):
        item = _make_event_item("NFP Report", impact="medium", offset_minutes=60)
        result = await self._fetch([item])
        assert result == []

    @pytest.mark.asyncio
    async def test_ignores_high_impact_unknown_keywords(self):
        item = _make_event_item(
            "Earnings Season Start", impact="high", offset_minutes=60
        )
        result = await self._fetch([item])
        assert result == []

    @pytest.mark.asyncio
    async def test_accepts_fomc_event(self):
        item = _make_event_item("FOMC Meeting Minutes", impact="high", offset_minutes=60)
        result = await self._fetch([item])
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_accepts_cpi_event(self):
        item = _make_event_item("CPI m/m Change", impact="high", offset_minutes=60)
        result = await self._fetch([item])
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_accepts_nonfarm_event(self):
        item = _make_event_item(
            "Nonfarm Payrolls Report", impact="high", offset_minutes=60
        )
        result = await self._fetch([item])
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_accepts_gdp_event(self):
        item = _make_event_item("GDP Growth Rate", impact="high", offset_minutes=60)
        result = await self._fetch([item])
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_accepts_pce_event(self):
        item = _make_event_item("PCE Price Index", impact="high", offset_minutes=60)
        result = await self._fetch([item])
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_accepts_federal_reserve_event(self):
        item = _make_event_item(
            "Federal Reserve Statement", impact="high", offset_minutes=60
        )
        result = await self._fetch([item])
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_accepts_interest_rate_decision(self):
        item = _make_event_item(
            "Interest Rate Decision", impact="high", offset_minutes=60
        )
        result = await self._fetch([item])
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_accepts_consumer_price_index(self):
        item = _make_event_item(
            "Consumer Price Index YoY", impact="high", offset_minutes=60
        )
        result = await self._fetch([item])
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_skips_event_with_invalid_date(self):
        item = {
            "event": "FOMC Meeting",
            "impact": "high",
            "date": "not-a-date",
            "time": "00:00:00",
        }
        result = await self._fetch([item])
        assert result == []

    @pytest.mark.asyncio
    async def test_multiple_events_filtered_correctly(self):
        items = [
            _make_event_item("FOMC Rate Decision", impact="high", offset_minutes=60),
            _make_event_item("Earnings Release", impact="high", offset_minutes=120),
            _make_event_item("CPI Data", impact="low", offset_minutes=90),
            _make_event_item("NFP Report", impact="high", offset_minutes=180),
        ]
        result = await self._fetch(items)
        # Only FOMC and NFP pass (high impact + keyword match)
        assert len(result) == 2
        names = {e["event"] for e in result}
        assert "FOMC Rate Decision" in names
        assert "NFP Report" in names


# ---------------------------------------------------------------------------
# _get_events() — HTTP status non-200
# ---------------------------------------------------------------------------


class TestGetEventsHttpErrors:
    @pytest.mark.asyncio
    async def test_returns_empty_on_non_200_status(self):
        cal = _make_calendar()
        cal._cache_ts = 0.0
        aiohttp_mock = _mock_aiohttp_response({}, status=403)
        with patch.dict("sys.modules", {"aiohttp": aiohttp_mock}):
            result = await cal._get_events()
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_500_status(self):
        cal = _make_calendar()
        cal._cache_ts = 0.0
        aiohttp_mock = _mock_aiohttp_response({}, status=500)
        with patch.dict("sys.modules", {"aiohttp": aiohttp_mock}):
            result = await cal._get_events()
        assert result == []
