"""
Тесты для src/season_detector.py

Покрытие:
  - SeasonDetector.__init__
  - fetch_data  (CoinGecko OK / fallback to Paprika)
  - _fetch_coingecko  (200, non-200, exception)
  - _fetch_paprika    (200, non-200, exception, stablecoin filter, BTC normalization)
  - compute_index     (happy path, no BTC row, no alts, exception)
  - classify          (altcoin season, bluechip season, neutral)
  - should_alert      (streak, cooldown, signal==profile, signal=None, resets)
  - format_message    (altcoin target, bluechip target, bar rendering)
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — aiohttp mock factories
# ---------------------------------------------------------------------------


def _make_response(status: int, json_data) -> MagicMock:
    """Creates a mock aiohttp response context manager."""
    resp = MagicMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data)
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


def _make_session(*responses) -> MagicMock:
    """
    Creates a mock aiohttp.ClientSession context manager whose
    session.get() returns responses in order (cycling if needed).
    """
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    response_iter = iter(responses)

    def _get(_url, **_kwargs):
        try:
            return next(response_iter)
        except StopIteration:
            # Return last response again if exhausted
            return responses[-1]

    session.get = MagicMock(side_effect=_get)
    return session


# ---------------------------------------------------------------------------
# Minimal CoinGecko payloads
# ---------------------------------------------------------------------------

_CG_GLOBAL = {
    "data": {
        "market_cap_percentage": {"btc": 48.0}
    }
}

_CG_MARKETS = [
    {
        "id": "bitcoin",
        "symbol": "btc",
        "price_change_percentage_30d_in_currency": 10.0,
    },
    {
        "id": "ethereum",
        "symbol": "eth",
        "price_change_percentage_30d_in_currency": 15.0,
    },
    {
        "id": "solana",
        "symbol": "sol",
        "price_change_percentage_30d_in_currency": 5.0,
    },
]

# ---------------------------------------------------------------------------
# Fixture: fresh SeasonDetector
# ---------------------------------------------------------------------------


@pytest.fixture
def detector():
    from src.season_detector import SeasonDetector

    return SeasonDetector()


# ===========================================================================
# SeasonDetector.__init__
# ===========================================================================


class TestInit:
    def test_streak_initialised_to_zero(self, detector):
        assert detector._streak == {"altcoin": 0, "bluechip": 0}

    def test_last_alert_initialised_to_zero(self, detector):
        assert detector._last_alert == {"altcoin": 0.0, "bluechip": 0.0}


# ===========================================================================
# SeasonDetector.fetch_data
# ===========================================================================


class TestFetchData:
    @pytest.mark.asyncio
    async def test_returns_coingecko_when_available(self, detector):
        """fetch_data returns CoinGecko result when _fetch_coingecko succeeds."""
        cg_result = {"global": _CG_GLOBAL, "markets": _CG_MARKETS}
        detector._fetch_coingecko = AsyncMock(return_value=cg_result)
        detector._fetch_paprika = AsyncMock(return_value=None)

        result = await detector.fetch_data()

        assert result == cg_result
        detector._fetch_paprika.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_to_paprika_when_coingecko_fails(self, detector):
        """fetch_data falls back to CoinPaprika when CoinGecko returns None."""
        paprika_result = {"global": _CG_GLOBAL, "markets": _CG_MARKETS}
        detector._fetch_coingecko = AsyncMock(return_value=None)
        detector._fetch_paprika = AsyncMock(return_value=paprika_result)

        result = await detector.fetch_data()

        assert result == paprika_result
        detector._fetch_paprika.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_both_fail(self, detector):
        detector._fetch_coingecko = AsyncMock(return_value=None)
        detector._fetch_paprika = AsyncMock(return_value=None)

        result = await detector.fetch_data()

        assert result is None


# ===========================================================================
# SeasonDetector._fetch_coingecko
# ===========================================================================


class TestFetchCoingecko:
    @pytest.mark.asyncio
    async def test_returns_data_on_200(self, detector):
        global_resp = _make_response(200, _CG_GLOBAL)
        markets_resp = _make_response(200, _CG_MARKETS)
        session = _make_session(global_resp, markets_resp)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientTimeout = MagicMock(return_value=MagicMock())
        mock_aiohttp.ClientSession = MagicMock(return_value=session)

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await detector._fetch_coingecko()

        assert result is not None
        assert "global" in result
        assert "markets" in result

    @pytest.mark.asyncio
    async def test_returns_none_on_global_non_200(self, detector):
        global_resp = _make_response(429, {})
        session = _make_session(global_resp)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientTimeout = MagicMock(return_value=MagicMock())
        mock_aiohttp.ClientSession = MagicMock(return_value=session)

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await detector._fetch_coingecko()

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_markets_non_200(self, detector):
        global_resp = _make_response(200, _CG_GLOBAL)
        markets_resp = _make_response(503, {})
        session = _make_session(global_resp, markets_resp)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientTimeout = MagicMock(return_value=MagicMock())
        mock_aiohttp.ClientSession = MagicMock(return_value=session)

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await detector._fetch_coingecko()

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self, detector):
        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientTimeout = MagicMock(return_value=MagicMock())
        mock_aiohttp.ClientSession = MagicMock(side_effect=Exception("network error"))

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            result = await detector._fetch_coingecko()

        assert result is None


# ===========================================================================
# SeasonDetector._fetch_paprika
# ===========================================================================

_PAPRIKA_GLOBAL_DATA = {"bitcoin_dominance_percentage": 47.5}

_PAPRIKA_TICKERS_DATA = [
    {
        "id": "btc-bitcoin",
        "symbol": "BTC",
        "quotes": {"USD": {"percent_change_30d": 10.0}},
    },
    {
        "id": "eth-ethereum",
        "symbol": "ETH",
        "quotes": {"USD": {"percent_change_30d": 20.0}},
    },
    {
        "id": "sol-solana",
        "symbol": "SOL",
        "quotes": {"USD": {"percent_change_30d": 5.0}},
    },
    # stablecoin — must be filtered
    {
        "id": "usdt-tether",
        "symbol": "USDT",
        "quotes": {"USD": {"percent_change_30d": 0.1}},
    },
]


class TestFetchPaprika:
    @pytest.mark.asyncio
    async def test_returns_normalised_data_on_200(self, detector):
        global_resp = _make_response(200, _PAPRIKA_GLOBAL_DATA)
        tickers_resp = _make_response(200, _PAPRIKA_TICKERS_DATA)
        session = _make_session(global_resp, tickers_resp)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientTimeout = MagicMock(return_value=MagicMock())
        mock_aiohttp.ClientSession = MagicMock(return_value=session)

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            result = await detector._fetch_paprika()

        assert result is not None
        assert result["global"]["data"]["market_cap_percentage"]["btc"] == 47.5
        # USDT stablecoin should be filtered out
        symbols = [m["symbol"] for m in result["markets"]]
        assert "usdt" not in symbols

    @pytest.mark.asyncio
    async def test_btc_ticker_normalised_to_bitcoin_id(self, detector):
        global_resp = _make_response(200, _PAPRIKA_GLOBAL_DATA)
        tickers_resp = _make_response(200, _PAPRIKA_TICKERS_DATA)
        session = _make_session(global_resp, tickers_resp)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientTimeout = MagicMock(return_value=MagicMock())
        mock_aiohttp.ClientSession = MagicMock(return_value=session)

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            result = await detector._fetch_paprika()

        ids = [m["id"] for m in result["markets"]]
        assert "bitcoin" in ids

    @pytest.mark.asyncio
    async def test_returns_none_on_global_non_200(self, detector):
        global_resp = _make_response(500, {})
        session = _make_session(global_resp)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientTimeout = MagicMock(return_value=MagicMock())
        mock_aiohttp.ClientSession = MagicMock(return_value=session)

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            result = await detector._fetch_paprika()

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_tickers_non_200(self, detector):
        global_resp = _make_response(200, _PAPRIKA_GLOBAL_DATA)
        tickers_resp = _make_response(429, {})
        session = _make_session(global_resp, tickers_resp)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientTimeout = MagicMock(return_value=MagicMock())
        mock_aiohttp.ClientSession = MagicMock(return_value=session)

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            result = await detector._fetch_paprika()

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self, detector):
        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientTimeout = MagicMock(return_value=MagicMock())
        mock_aiohttp.ClientSession = MagicMock(side_effect=RuntimeError("boom"))

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            result = await detector._fetch_paprika()

        assert result is None

    @pytest.mark.asyncio
    async def test_filters_all_known_stablecoins(self, detector):
        stablecoin_tickers = [
            {"id": f"{sym}-x", "symbol": sym.upper(), "quotes": {"USD": {"percent_change_30d": 0.0}}}
            for sym in ("usdt", "usdc", "dai", "tusd", "fdusd", "busd")
        ]
        btc_ticker = {
            "id": "btc-bitcoin",
            "symbol": "BTC",
            "quotes": {"USD": {"percent_change_30d": 5.0}},
        }
        global_resp = _make_response(200, _PAPRIKA_GLOBAL_DATA)
        tickers_resp = _make_response(200, [btc_ticker] + stablecoin_tickers)
        session = _make_session(global_resp, tickers_resp)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientTimeout = MagicMock(return_value=MagicMock())
        mock_aiohttp.ClientSession = MagicMock(return_value=session)

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            result = await detector._fetch_paprika()

        assert result is not None
        assert len(result["markets"]) == 1  # only BTC remains

    @pytest.mark.asyncio
    async def test_missing_quotes_defaults_to_zero(self, detector):
        tickers = [
            {"id": "btc-bitcoin", "symbol": "BTC", "quotes": {}},
            {"id": "eth-ethereum", "symbol": "ETH"},  # no quotes key at all
        ]
        global_resp = _make_response(200, _PAPRIKA_GLOBAL_DATA)
        tickers_resp = _make_response(200, tickers)
        session = _make_session(global_resp, tickers_resp)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientTimeout = MagicMock(return_value=MagicMock())
        mock_aiohttp.ClientSession = MagicMock(return_value=session)

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            result = await detector._fetch_paprika()

        assert result is not None
        for m in result["markets"]:
            assert m["price_change_percentage_30d_in_currency"] == 0.0


# ===========================================================================
# SeasonDetector.compute_index
# ===========================================================================


def _make_data(btc_dom: float, btc_30d: float, alts: list[dict]) -> dict:
    """Build a minimal data dict compatible with compute_index."""
    markets = [
        {
            "id": "bitcoin",
            "symbol": "btc",
            "price_change_percentage_30d_in_currency": btc_30d,
        }
    ] + alts
    return {
        "global": {"data": {"market_cap_percentage": {"btc": btc_dom}}},
        "markets": markets,
    }


def _alt(coin_id: str, change_30d: float) -> dict:
    return {
        "id": coin_id,
        "symbol": coin_id[:3],
        "price_change_percentage_30d_in_currency": change_30d,
    }


class TestComputeIndex:
    def test_basic_altcoin_outperformance(self, detector):
        data = _make_data(
            btc_dom=47.0,
            btc_30d=10.0,
            alts=[
                _alt("ethereum", 20.0),   # outperforms
                _alt("solana", 5.0),      # does not
                _alt("cardano", 15.0),    # outperforms
                _alt("polkadot", 8.0),    # does not
            ],
        )
        result = detector.compute_index(data)
        assert result is not None
        assert result["altcoin_index"] == 50.0  # 2/4 = 50%
        assert result["btc_dominance"] == 47.0
        assert result["btc_30d"] == 10.0
        assert result["alts_count"] == 4
        assert result["alts_outperformed"] == 2

    def test_all_alts_outperform_btc(self, detector):
        data = _make_data(
            btc_dom=40.0,
            btc_30d=5.0,
            alts=[_alt(f"coin{i}", 20.0) for i in range(10)],
        )
        result = detector.compute_index(data)
        assert result["altcoin_index"] == 100.0
        assert result["alts_outperformed"] == 10

    def test_no_alts_outperform_btc(self, detector):
        data = _make_data(
            btc_dom=55.0,
            btc_30d=30.0,
            alts=[_alt(f"coin{i}", 5.0) for i in range(5)],
        )
        result = detector.compute_index(data)
        assert result["altcoin_index"] == 0.0
        assert result["alts_outperformed"] == 0

    def test_returns_none_when_no_btc_row(self, detector):
        data = {
            "global": {"data": {"market_cap_percentage": {"btc": 48.0}}},
            "markets": [_alt("ethereum", 10.0)],
        }
        result = detector.compute_index(data)
        assert result is None

    def test_returns_none_when_no_alts(self, detector):
        data = {
            "global": {"data": {"market_cap_percentage": {"btc": 48.0}}},
            "markets": [
                {
                    "id": "bitcoin",
                    "symbol": "btc",
                    "price_change_percentage_30d_in_currency": 10.0,
                }
            ],
        }
        result = detector.compute_index(data)
        assert result is None

    def test_stablecoins_excluded_from_alts(self, detector):
        stablecoins = [
            _alt("tether", 0.1),
            _alt("usd-coin", 0.1),
            _alt("dai", 0.1),
            _alt("true-usd", 0.1),
            _alt("first-digital-usd", 0.1),
        ]
        data = _make_data(
            btc_dom=48.0,
            btc_30d=10.0,
            alts=stablecoins + [_alt("ethereum", 20.0)],
        )
        result = detector.compute_index(data)
        assert result is not None
        # Only "ethereum" is a real alt
        assert result["alts_count"] == 1
        assert result["alts_outperformed"] == 1

    def test_none_price_change_treated_as_zero(self, detector):
        alts = [
            {"id": "ethereum", "symbol": "eth", "price_change_percentage_30d_in_currency": None},
        ]
        data = _make_data(btc_dom=48.0, btc_30d=10.0, alts=alts)
        result = detector.compute_index(data)
        # None → 0.0, BTC is +10% → eth does NOT outperform
        assert result["alts_outperformed"] == 0

    def test_returns_none_on_exception(self, detector):
        # Pass a broken data structure to trigger the except branch
        result = detector.compute_index({"global": None, "markets": []})
        assert result is None

    def test_btc_30d_none_treated_as_zero(self, detector):
        """BTC row with None price_change should be treated as 0.0."""
        markets = [
            {"id": "bitcoin", "symbol": "btc", "price_change_percentage_30d_in_currency": None},
            _alt("ethereum", 5.0),
        ]
        data = {
            "global": {"data": {"market_cap_percentage": {"btc": 48.0}}},
            "markets": markets,
        }
        result = detector.compute_index(data)
        assert result is not None
        assert result["btc_30d"] == 0.0
        # ETH at +5% > BTC at 0% → outperforms
        assert result["alts_outperformed"] == 1


# ===========================================================================
# SeasonDetector.classify
# ===========================================================================


class TestClassify:
    def _index(self, ai: float, dom: float) -> dict:
        return {"altcoin_index": ai, "btc_dominance": dom}

    def test_altcoin_season_both_conditions_met(self, detector):
        # ai=80 >= 75 AND dom=43 < 45
        result = detector.classify(self._index(80.0, 43.0))
        assert result == "altcoin"

    def test_altcoin_season_high_ai_but_high_dom(self, detector):
        # ai=80 >= 75 BUT dom=46 >= 45 → not alt season
        result = detector.classify(self._index(80.0, 46.0))
        assert result != "altcoin"

    def test_altcoin_season_boundary_ai_exactly_75(self, detector):
        result = detector.classify(self._index(75.0, 40.0))
        assert result == "altcoin"

    def test_altcoin_season_boundary_dom_exactly_45(self, detector):
        # dom == 45 is NOT < 45, so NOT alt season
        result = detector.classify(self._index(80.0, 45.0))
        assert result != "altcoin"

    def test_bluechip_season_low_ai(self, detector):
        # ai=20 <= 25
        result = detector.classify(self._index(20.0, 50.0))
        assert result == "bluechip"

    def test_bluechip_season_high_dominance(self, detector):
        # dom=55 > 52
        result = detector.classify(self._index(50.0, 55.0))
        assert result == "bluechip"

    def test_bluechip_season_boundary_ai_exactly_25(self, detector):
        result = detector.classify(self._index(25.0, 48.0))
        assert result == "bluechip"

    def test_bluechip_season_boundary_dom_exactly_52(self, detector):
        result = detector.classify(self._index(50.0, 52.0))
        assert result != "bluechip"

    def test_neutral_zone_returns_none(self, detector):
        # ai=50, dom=48 — neither condition met
        result = detector.classify(self._index(50.0, 48.0))
        assert result is None

    def test_neutral_near_alt_boundary(self, detector):
        # ai=74.9 < 75
        result = detector.classify(self._index(74.9, 43.0))
        assert result is None


# ===========================================================================
# SeasonDetector.should_alert
# ===========================================================================


class TestShouldAlert:
    def _now(self) -> float:
        return time.time()

    def test_returns_false_when_signal_is_none(self, detector):
        result = detector.should_alert(None, "neutral", self._now())
        assert result is False

    def test_returns_false_when_signal_matches_profile(self, detector):
        result = detector.should_alert("altcoin", "altcoin", self._now())
        assert result is False

    def test_returns_false_on_first_signal_streak_not_reached(self, detector):
        # _CONFIRM_STREAK = 2, first call gives streak=1
        result = detector.should_alert("altcoin", "bluechip", self._now())
        assert result is False
        assert detector._streak["altcoin"] == 1

    def test_returns_true_on_second_signal_after_cooldown(self, detector):
        # First call: streak → 1
        detector.should_alert("altcoin", "bluechip", self._now())
        # Second call: streak → 2 >= CONFIRM_STREAK, last_alert=0 → elapsed huge
        result = detector.should_alert("altcoin", "bluechip", self._now())
        assert result is True

    def test_returns_false_within_cooldown(self, detector):
        # Warm up streak to 2 and fire alert
        detector.should_alert("altcoin", "bluechip", self._now())
        now = self._now()
        detector.should_alert("altcoin", "bluechip", now)
        # Immediately call again — cooldown not expired
        result = detector.should_alert("altcoin", "bluechip", now + 60)
        assert result is False

    def test_returns_true_after_cooldown_expires(self, detector):
        # Fire first alert
        detector.should_alert("altcoin", "bluechip", self._now())
        t = self._now()
        detector.should_alert("altcoin", "bluechip", t)
        # Advance time by more than COOLDOWN_H (24h) = 86400s
        far_future = t + 86401
        # Streak was reset after last alert to 2, need to rebuild
        # streak is still 2 from before — call once more in the future
        result = detector.should_alert("altcoin", "bluechip", far_future)
        assert result is True

    def test_opposite_signal_resets_streak(self, detector):
        # Build altcoin streak to 1
        detector.should_alert("altcoin", "bluechip", self._now())
        assert detector._streak["altcoin"] == 1
        # Now a bluechip signal arrives — altcoin streak should reset
        detector.should_alert("bluechip", "altcoin", self._now())
        assert detector._streak["altcoin"] == 0

    def test_none_signal_resets_both_streaks(self, detector):
        detector._streak["altcoin"] = 1
        detector._streak["bluechip"] = 1
        detector.should_alert(None, "neutral", self._now())
        assert detector._streak["altcoin"] == 0
        assert detector._streak["bluechip"] == 0

    def test_last_alert_updated_on_fired_alert(self, detector):
        detector.should_alert("bluechip", "altcoin", self._now())
        now = self._now()
        detector.should_alert("bluechip", "altcoin", now)
        assert detector._last_alert["bluechip"] == now

    def test_streak_increments_correctly(self, detector):
        t = self._now()
        detector.should_alert("altcoin", "bluechip", t)
        assert detector._streak["altcoin"] == 1
        detector.should_alert("altcoin", "bluechip", t)
        assert detector._streak["altcoin"] == 2

    def test_bluechip_signal_returns_true_after_confirm(self, detector):
        detector.should_alert("bluechip", "altcoin", self._now())
        result = detector.should_alert("bluechip", "altcoin", self._now())
        assert result is True


# ===========================================================================
# SeasonDetector.format_message
# ===========================================================================


def _sample_index(ai: float = 80.0, dom: float = 42.0, btc_30d: float = 5.0,
                  count: int = 70, total: int = 90) -> dict:
    return {
        "altcoin_index": ai,
        "btc_dominance": dom,
        "btc_30d": btc_30d,
        "alts_outperformed": count,
        "alts_count": total,
    }


class TestFormatMessage:
    def test_altcoin_message_contains_header(self, detector):
        msg = detector.format_message("altcoin", _sample_index())
        assert "АЛЬТСЕЗОН" in msg

    def test_altcoin_message_contains_dominance(self, detector):
        idx = _sample_index(dom=42.5)
        msg = detector.format_message("altcoin", idx)
        assert "42.5%" in msg

    def test_altcoin_message_contains_counts(self, detector):
        idx = _sample_index(count=70, total=90)
        msg = detector.format_message("altcoin", idx)
        assert "70" in msg
        assert "90" in msg

    def test_altcoin_message_switch_recommendation(self, detector):
        msg = detector.format_message("altcoin", _sample_index())
        assert "Альткоины" in msg

    def test_bluechip_message_contains_header(self, detector):
        msg = detector.format_message("bluechip", _sample_index(ai=20.0, dom=54.0))
        assert "BTC SEASON" in msg

    def test_bluechip_message_contains_dominance(self, detector):
        idx = _sample_index(ai=20.0, dom=54.0)
        msg = detector.format_message("bluechip", idx)
        assert "54.0%" in msg

    def test_bluechip_message_switch_recommendation(self, detector):
        msg = detector.format_message("bluechip", _sample_index())
        assert "Блючипы" in msg

    def test_message_contains_btc_30d(self, detector):
        idx = _sample_index(btc_30d=12.3)
        msg = detector.format_message("altcoin", idx)
        assert "+12.3%" in msg

    def test_message_contains_altcoin_index_value(self, detector):
        idx = _sample_index(ai=80.0)
        msg = detector.format_message("altcoin", idx)
        assert "80/100" in msg

    def test_bar_fully_filled_at_100(self, detector):
        idx = _sample_index(ai=100.0)
        msg = detector.format_message("altcoin", idx)
        # 10 filled blocks, 0 empty
        assert "██████████" in msg
        assert "░" not in msg

    def test_bar_empty_at_zero(self, detector):
        idx = _sample_index(ai=0.0)
        msg = detector.format_message("bluechip", idx)
        # 0 filled blocks, 10 empty
        assert "░░░░░░░░░░" in msg
        assert "█" not in msg

    def test_bar_mixed_at_50(self, detector):
        idx = _sample_index(ai=50.0)
        msg = detector.format_message("altcoin", idx)
        # 5 filled, 5 empty
        assert "█████░░░░░" in msg

    def test_negative_btc_30d_formatted_with_sign(self, detector):
        idx = _sample_index(btc_30d=-8.7)
        msg = detector.format_message("bluechip", idx)
        assert "-8.7%" in msg

    def test_returns_string(self, detector):
        result = detector.format_message("altcoin", _sample_index())
        assert isinstance(result, str)
        assert len(result) > 0
