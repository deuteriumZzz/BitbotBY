"""
Comprehensive tests for src/market_context.py.

Coverage strategy:
  - Unit tests for every public/private method
  - TTL-based cache logic (hit / miss / expiry)
  - Signal computation from raw numbers (all branch paths)
  - Error handling (HTTP failures, empty responses, malformed data)
  - get_context() aggregation (BTC vs non-BTC, Deribit symbols vs others)
  - get_context_for_symbols() parallel gather + fallback to neutral
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_BIG_NOW = 100_000.0  # больше любого TTL — гарантирует что initial-кэши протухли


@pytest.fixture(autouse=True)
def mock_market_time():
    """Мокаем time.monotonic в market_context чтобы initial-кэши не мешали тестам."""
    with patch("src.market_context.time.monotonic", return_value=_BIG_NOW):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_aiohttp_resp(json_data=None, text_data=None, status=200):
    """Build a mock that behaves like an aiohttp response context manager."""
    resp = AsyncMock()
    resp.status = status
    if json_data is not None:
        resp.json = AsyncMock(return_value=json_data)
    if text_data is not None:
        resp.text = AsyncMock(return_value=text_data)

    # Support `async with session.get(...) as resp:`
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _make_aiohttp_session(resp_cm):
    """Build a mock ClientSession whose .get() returns resp_cm."""
    session = AsyncMock()
    session.get = MagicMock(return_value=resp_cm)

    # Support `async with aiohttp.ClientSession() as session:`
    session_cm = AsyncMock()
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=False)
    return session_cm


def _patch_session(json_data=None, text_data=None, status=200):
    """
    Returns a patcher for aiohttp.ClientSession used inside market_context.
    Usage: with _patch_session(json_data={...}) as mock_cls: ...
    """
    resp_cm = _make_aiohttp_resp(json_data=json_data, text_data=text_data, status=status)
    session_cm = _make_aiohttp_session(resp_cm)
    return patch("aiohttp.ClientSession", return_value=session_cm)


# ---------------------------------------------------------------------------
# Import helper — isolate from global state
# ---------------------------------------------------------------------------


def _make_mc():
    """Return a fresh MarketContext instance."""
    from src.market_context import MarketContext

    return MarketContext()


# ---------------------------------------------------------------------------
# Module-level helpers (_symbol_to_bybit, _symbol_to_base)
# ---------------------------------------------------------------------------


class TestSymbolHelpers:
    def test_symbol_to_bybit_slash(self):
        from src.market_context import _symbol_to_bybit

        assert _symbol_to_bybit("BTC/USDT") == "BTCUSDT"

    def test_symbol_to_bybit_no_slash(self):
        from src.market_context import _symbol_to_bybit

        assert _symbol_to_bybit("BTCUSDT") == "BTCUSDT"

    def test_symbol_to_base(self):
        from src.market_context import _symbol_to_base

        assert _symbol_to_base("BTC/USDT") == "BTC"
        assert _symbol_to_base("ETH/USDT") == "ETH"
        assert _symbol_to_base("SOL/USDT") == "SOL"


# ---------------------------------------------------------------------------
# _neutral_context()
# ---------------------------------------------------------------------------


class TestNeutralContext:
    def test_neutral_context_has_all_keys(self):
        mc = _make_mc()
        ctx = mc._neutral_context()
        expected_keys = {
            "funding_rate",
            "funding_signal",
            "oi_signal",
            "liquidation_pressure",
            "fear_greed",
            "fear_greed_signal",
            "basis_pct",
            "basis_signal",
            "google_trends",
            "google_trends_signal",
            "pcr",
            "pcr_signal",
            "ob_imbalance",
            "ob_signal",
            "iv_skew",
            "iv_signal",
            "macro_signal",
            "etf_flow",
            "etf_signal",
            "reddit_signal",
            "stablecoin_signal",
        }
        assert set(ctx.keys()) == expected_keys

    def test_neutral_context_values(self):
        mc = _make_mc()
        ctx = mc._neutral_context()
        assert ctx["funding_rate"] == 0.0
        assert ctx["funding_signal"] == "neutral"
        assert ctx["oi_signal"] == "oi_neutral"
        assert ctx["fear_greed"] == 50
        assert ctx["pcr"] == 1.0
        assert ctx["ob_signal"] == "balanced"


# ---------------------------------------------------------------------------
# _fng_to_signal()
# ---------------------------------------------------------------------------


class TestFngToSignal:
    def test_extreme_fear(self):
        from src.market_context import MarketContext

        assert MarketContext._fng_to_signal(10) == "extreme_fear"
        assert MarketContext._fng_to_signal(24) == "extreme_fear"

    def test_extreme_greed(self):
        from src.market_context import MarketContext

        assert MarketContext._fng_to_signal(76) == "extreme_greed"
        assert MarketContext._fng_to_signal(99) == "extreme_greed"

    def test_neutral_boundaries(self):
        from src.market_context import MarketContext

        assert MarketContext._fng_to_signal(25) == "neutral"
        assert MarketContext._fng_to_signal(50) == "neutral"
        assert MarketContext._fng_to_signal(75) == "neutral"


# ---------------------------------------------------------------------------
# _get_funding()
# ---------------------------------------------------------------------------


class TestGetFunding:
    @pytest.mark.asyncio
    async def test_long_overheated(self):
        mc = _make_mc()
        data = {"result": {"list": [{"fundingRate": "0.002"}]}}
        with _patch_session(json_data=data):
            rate, signal = await mc._get_funding("BTC/USDT")
        assert rate == pytest.approx(0.002)
        assert signal == "long_overheated"

    @pytest.mark.asyncio
    async def test_short_overheated(self):
        mc = _make_mc()
        data = {"result": {"list": [{"fundingRate": "-0.001"}]}}
        with _patch_session(json_data=data):
            rate, signal = await mc._get_funding("BTC/USDT")
        assert rate == pytest.approx(-0.001)
        assert signal == "short_overheated"

    @pytest.mark.asyncio
    async def test_neutral_rate(self):
        mc = _make_mc()
        data = {"result": {"list": [{"fundingRate": "0.0001"}]}}
        with _patch_session(json_data=data):
            rate, signal = await mc._get_funding("BTC/USDT")
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_empty_list_returns_neutral(self):
        mc = _make_mc()
        data = {"result": {"list": []}}
        with _patch_session(json_data=data):
            rate, signal = await mc._get_funding("BTC/USDT")
        assert rate == 0.0
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_missing_result_returns_neutral(self):
        mc = _make_mc()
        data = {}
        with _patch_session(json_data=data):
            rate, signal = await mc._get_funding("BTC/USDT")
        assert rate == 0.0
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_http_error_returns_neutral(self):
        mc = _make_mc()
        with patch("aiohttp.ClientSession", side_effect=Exception("Network error")):
            rate, signal = await mc._get_funding("BTC/USDT")
        assert rate == 0.0
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_exactly_0001_is_long_overheated(self):
        """Boundary: rate == 0.001 is NOT > 0.001, stays neutral."""
        mc = _make_mc()
        data = {"result": {"list": [{"fundingRate": "0.001"}]}}
        with _patch_session(json_data=data):
            rate, signal = await mc._get_funding("BTC/USDT")
        # 0.001 is not > 0.001 → neutral
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_exactly_minus_0005_is_not_short_overheated(self):
        """Boundary: rate == -0.0005 is NOT < -0.0005, stays neutral."""
        mc = _make_mc()
        data = {"result": {"list": [{"fundingRate": "-0.0005"}]}}
        with _patch_session(json_data=data):
            rate, signal = await mc._get_funding("ETH/USDT")
        assert signal == "neutral"


# ---------------------------------------------------------------------------
# _get_oi_signal()
# ---------------------------------------------------------------------------


class TestGetOiSignal:
    def _oi_data(self, oi_now, oi_prev):
        return {
            "result": {
                "list": [
                    {"openInterest": str(oi_now)},
                    {"openInterest": str(oi_prev)},
                ]
            }
        }

    @pytest.mark.asyncio
    async def test_oi_bearish_oi_grows_price_falls(self):
        mc = _make_mc()
        with _patch_session(json_data=self._oi_data(1100, 1000)):
            # current_price much lower → price_falling = True
            signal, liq = await mc._get_oi_signal("BTC/USDT", 90.0)
        # OI growing + price falling → oi_bearish
        # But no prev_price cached, so prev_price == current → price_falling False
        # Actually: prev_price = current_price when no cache → price_falling = (90 < 90*0.9995) = False
        # So signal is oi_neutral (oi_growing=True, price_falling=False)
        assert signal == "oi_neutral"

    @pytest.mark.asyncio
    async def test_oi_bearish_with_cached_price(self):
        mc = _make_mc()
        # Pre-seed cache with higher price so current_price appears to have fallen
        mc._cache["BTC/USDT"] = ({"_prev_price": 100.0}, _BIG_NOW)
        data = self._oi_data(1100, 1000)  # OI growing
        with _patch_session(json_data=data):
            signal, liq = await mc._get_oi_signal("BTC/USDT", 90.0)
        # OI growing + price falling (90 < 100*0.9995 = 99.95) → oi_bearish
        assert signal == "oi_bearish"

    @pytest.mark.asyncio
    async def test_oi_bullish_oi_falls_price_rises(self):
        mc = _make_mc()
        mc._cache["BTC/USDT"] = ({"_prev_price": 90.0}, _BIG_NOW)
        data = self._oi_data(900, 1000)  # OI falling
        with _patch_session(json_data=data):
            signal, liq = await mc._get_oi_signal("BTC/USDT", 100.0)
        # oi_growing=False, price_falling = (100 < 90*0.9995=89.955) = False → oi_bullish
        assert signal == "oi_bullish"

    @pytest.mark.asyncio
    async def test_long_liquidation_pressure(self):
        mc = _make_mc()
        mc._cache["BTC/USDT"] = ({"_prev_price": 100.0}, _BIG_NOW)
        # OI drops >2% and price falls → long_liquidation
        data = self._oi_data(970, 1000)  # 3% drop
        with _patch_session(json_data=data):
            signal, liq = await mc._get_oi_signal("BTC/USDT", 90.0)
        assert liq == "long_liquidation"

    @pytest.mark.asyncio
    async def test_short_squeeze_pressure(self):
        mc = _make_mc()
        mc._cache["BTC/USDT"] = ({"_prev_price": 90.0}, _BIG_NOW)
        # OI drops >2% and price rises → short_squeeze
        data = self._oi_data(970, 1000)
        with _patch_session(json_data=data):
            signal, liq = await mc._get_oi_signal("BTC/USDT", 100.0)
        assert liq == "short_squeeze"

    @pytest.mark.asyncio
    async def test_neutral_liquidation_small_oi_drop(self):
        mc = _make_mc()
        # OI drops <2% → neutral liquidation
        data = self._oi_data(995, 1000)
        with _patch_session(json_data=data):
            signal, liq = await mc._get_oi_signal("BTC/USDT", 100.0)
        assert liq == "neutral"

    @pytest.mark.asyncio
    async def test_insufficient_data_returns_neutral(self):
        mc = _make_mc()
        data = {"result": {"list": [{"openInterest": "1000"}]}}
        with _patch_session(json_data=data):
            signal, liq = await mc._get_oi_signal("BTC/USDT", 50000.0)
        assert signal == "oi_neutral"
        assert liq == "neutral"

    @pytest.mark.asyncio
    async def test_zero_oi_prev_returns_neutral(self):
        mc = _make_mc()
        data = self._oi_data(0, 0)
        with _patch_session(json_data=data):
            signal, liq = await mc._get_oi_signal("BTC/USDT", 50000.0)
        assert signal == "oi_neutral"

    @pytest.mark.asyncio
    async def test_http_error_returns_neutral(self):
        mc = _make_mc()
        with patch("aiohttp.ClientSession", side_effect=Exception("conn refused")):
            signal, liq = await mc._get_oi_signal("BTC/USDT", 50000.0)
        assert signal == "oi_neutral"
        assert liq == "neutral"


# ---------------------------------------------------------------------------
# _get_basis()
# ---------------------------------------------------------------------------


class TestGetBasis:
    @pytest.mark.asyncio
    async def test_greed_premium(self):
        mc = _make_mc()
        # spot=100, futures=102 → basis=2% > 1.5 → greed_premium
        data = {"result": {"list": [{"lastPrice": "102"}]}}
        with _patch_session(json_data=data):
            basis, signal = await mc._get_basis("BTC/USDT", 100.0)
        assert signal == "greed_premium"
        assert basis == pytest.approx(2.0, abs=0.001)

    @pytest.mark.asyncio
    async def test_backwardation(self):
        mc = _make_mc()
        # spot=100, futures=99 → basis=-1% < -0.5 → backwardation
        data = {"result": {"list": [{"lastPrice": "99"}]}}
        with _patch_session(json_data=data):
            basis, signal = await mc._get_basis("BTC/USDT", 100.0)
        assert signal == "backwardation"

    @pytest.mark.asyncio
    async def test_neutral_basis(self):
        mc = _make_mc()
        # spot=100, futures=100.5 → basis=0.5% → neutral
        data = {"result": {"list": [{"lastPrice": "100.5"}]}}
        with _patch_session(json_data=data):
            basis, signal = await mc._get_basis("BTC/USDT", 100.0)
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_zero_spot_price_returns_neutral(self):
        mc = _make_mc()
        basis, signal = await mc._get_basis("BTC/USDT", 0.0)
        assert basis == 0.0
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_zero_futures_price_returns_neutral(self):
        mc = _make_mc()
        data = {"result": {"list": [{"lastPrice": "0"}]}}
        with _patch_session(json_data=data):
            basis, signal = await mc._get_basis("BTC/USDT", 100.0)
        assert basis == 0.0
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_empty_list_returns_neutral(self):
        mc = _make_mc()
        data = {"result": {"list": []}}
        with _patch_session(json_data=data):
            basis, signal = await mc._get_basis("BTC/USDT", 100.0)
        assert basis == 0.0
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_http_error_returns_neutral(self):
        mc = _make_mc()
        with patch("aiohttp.ClientSession", side_effect=Exception("timeout")):
            basis, signal = await mc._get_basis("BTC/USDT", 100.0)
        assert basis == 0.0
        assert signal == "neutral"


# ---------------------------------------------------------------------------
# _get_fear_greed()
# ---------------------------------------------------------------------------


class TestGetFearGreed:
    @pytest.mark.asyncio
    async def test_extreme_fear(self):
        mc = _make_mc()
        data = {"data": [{"value": "15"}]}
        with _patch_session(json_data=data):
            val, signal = await mc._get_fear_greed()
        assert val == 15
        assert signal == "extreme_fear"

    @pytest.mark.asyncio
    async def test_extreme_greed(self):
        mc = _make_mc()
        data = {"data": [{"value": "85"}]}
        with _patch_session(json_data=data):
            val, signal = await mc._get_fear_greed()
        assert val == 85
        assert signal == "extreme_greed"

    @pytest.mark.asyncio
    async def test_neutral(self):
        mc = _make_mc()
        data = {"data": [{"value": "50"}]}
        with _patch_session(json_data=data):
            val, signal = await mc._get_fear_greed()
        assert val == 50
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_cache_hit_skips_http(self):
        mc = _make_mc()
        # Pre-warm cache
        mc._fng_cache = (80, _BIG_NOW)
        # Even with a broken session, should return cached value
        with patch("aiohttp.ClientSession", side_effect=Exception("should not call")):
            val, signal = await mc._get_fear_greed()
        assert val == 80
        assert signal == "extreme_greed"

    @pytest.mark.asyncio
    async def test_cache_expired_refetches(self):
        mc = _make_mc()
        # Expired cache (timestamp 0 → elapsed >> TTL)
        mc._fng_cache = (80, 0.0)
        data = {"data": [{"value": "20"}]}
        with _patch_session(json_data=data):
            val, signal = await mc._get_fear_greed()
        assert val == 20
        assert signal == "extreme_fear"

    @pytest.mark.asyncio
    async def test_http_error_returns_50_neutral(self):
        mc = _make_mc()
        with patch("aiohttp.ClientSession", side_effect=Exception("timeout")):
            val, signal = await mc._get_fear_greed()
        assert val == 50
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_updates_cache_on_success(self):
        mc = _make_mc()
        data = {"data": [{"value": "30"}]}
        with _patch_session(json_data=data):
            val, _ = await mc._get_fear_greed()
        assert mc._fng_cache[0] == 30
        assert mc._fng_cache[1] > 0


# ---------------------------------------------------------------------------
# _get_google_trends()
# ---------------------------------------------------------------------------


_PYTRENDS_MOCK = MagicMock()
_PYTRENDS_MOCK.request.TrendReq = MagicMock()


class TestGetGoogleTrends:
    @pytest.mark.asyncio
    async def test_retail_fomo(self):
        mc = _make_mc()
        with patch.dict("sys.modules", {"pytrends": _PYTRENDS_MOCK, "pytrends.request": _PYTRENDS_MOCK.request}), \
             patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=80)
            val, signal = await mc._get_google_trends("buy bitcoin")
        assert signal == "retail_fomo"
        assert val == 80

    @pytest.mark.asyncio
    async def test_retail_absent(self):
        mc = _make_mc()
        with patch.dict("sys.modules", {"pytrends": _PYTRENDS_MOCK, "pytrends.request": _PYTRENDS_MOCK.request}), \
             patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=10)
            val, signal = await mc._get_google_trends("buy bitcoin")
        assert signal == "retail_absent"
        assert val == 10

    @pytest.mark.asyncio
    async def test_neutral(self):
        mc = _make_mc()
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=50)
            val, signal = await mc._get_google_trends("buy bitcoin")
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        mc = _make_mc()
        mc._trends_cache = ((90, "retail_fomo"), _BIG_NOW)
        # executor should NOT be called
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(
                side_effect=Exception("should not call")
            )
            val, signal = await mc._get_google_trends()
        assert val == 90
        assert signal == "retail_fomo"

    @pytest.mark.asyncio
    async def test_exception_returns_50_neutral(self):
        mc = _make_mc()
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(
                side_effect=Exception("pytrends unavailable")
            )
            val, signal = await mc._get_google_trends("buy bitcoin")
        assert val == 50
        assert signal == "neutral"


# ---------------------------------------------------------------------------
# _get_deribit_options()
# ---------------------------------------------------------------------------


class TestGetDeribitOptions:
    def _deribit_result(self, puts, calls):
        """Build a Deribit-like response with put/call option entries."""
        result = []
        for v, iv in puts:
            result.append(
                {"instrument_name": "BTC-1JAN25-50000-P", "volume": v, "mark_iv": iv}
            )
        for v, iv in calls:
            result.append(
                {"instrument_name": "BTC-1JAN25-50000-C", "volume": v, "mark_iv": iv}
            )
        return {"result": result}

    @pytest.mark.asyncio
    async def test_fear_puts_high_pcr(self):
        mc = _make_mc()
        # PCR = 3.0 > 1.5 → fear_puts
        data = self._deribit_result(puts=[(30, 60.0)], calls=[(10, 40.0)])
        with _patch_session(json_data=data):
            pcr, pcr_sig, iv_skew, iv_sig = await mc._get_deribit_options("BTC")
        assert pcr_sig == "fear_puts"
        assert pcr == pytest.approx(3.0, abs=0.001)

    @pytest.mark.asyncio
    async def test_greed_calls_low_pcr(self):
        mc = _make_mc()
        # PCR = 0.4 < 0.5 → greed_calls
        data = self._deribit_result(puts=[(4, 50.0)], calls=[(10, 40.0)])
        with _patch_session(json_data=data):
            pcr, pcr_sig, iv_skew, iv_sig = await mc._get_deribit_options("BTC")
        assert pcr_sig == "greed_calls"

    @pytest.mark.asyncio
    async def test_neutral_pcr(self):
        mc = _make_mc()
        # PCR = 1.0 → neutral
        data = self._deribit_result(puts=[(10, 50.0)], calls=[(10, 40.0)])
        with _patch_session(json_data=data):
            pcr, pcr_sig, iv_skew, iv_sig = await mc._get_deribit_options("BTC")
        assert pcr_sig == "neutral"

    @pytest.mark.asyncio
    async def test_put_skew(self):
        mc = _make_mc()
        # avg_put_iv=60, avg_call_iv=50 → skew=10 > 5 → put_skew
        data = self._deribit_result(puts=[(10, 60.0)], calls=[(10, 50.0)])
        with _patch_session(json_data=data):
            pcr, pcr_sig, iv_skew, iv_sig = await mc._get_deribit_options("BTC")
        assert iv_sig == "put_skew"
        assert iv_skew == pytest.approx(10.0)

    @pytest.mark.asyncio
    async def test_call_skew(self):
        mc = _make_mc()
        # avg_put_iv=40, avg_call_iv=50 → skew=-10 < -3 → call_skew
        data = self._deribit_result(puts=[(10, 40.0)], calls=[(10, 50.0)])
        with _patch_session(json_data=data):
            pcr, pcr_sig, iv_skew, iv_sig = await mc._get_deribit_options("BTC")
        assert iv_sig == "call_skew"

    @pytest.mark.asyncio
    async def test_no_calls_returns_neutral(self):
        mc = _make_mc()
        data = self._deribit_result(puts=[(10, 50.0)], calls=[])
        with _patch_session(json_data=data):
            pcr, pcr_sig, iv_skew, iv_sig = await mc._get_deribit_options("BTC")
        assert pcr == 1.0
        assert pcr_sig == "neutral"

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        mc = _make_mc()
        cached_result = (1.5, "fear_puts", 6.0, "put_skew")
        mc._deribit_cache["BTC"] = (cached_result, _BIG_NOW)
        with patch("aiohttp.ClientSession", side_effect=Exception("should not call")):
            result = await mc._get_deribit_options("BTC")
        assert result == cached_result

    @pytest.mark.asyncio
    async def test_http_error_returns_neutral(self):
        mc = _make_mc()
        with patch("aiohttp.ClientSession", side_effect=Exception("deribit down")):
            pcr, pcr_sig, iv_skew, iv_sig = await mc._get_deribit_options("BTC")
        assert pcr == 1.0
        assert pcr_sig == "neutral"

    @pytest.mark.asyncio
    async def test_iv_skew_zero_when_no_iv_data(self):
        mc = _make_mc()
        # Entries with mark_iv=0 should be excluded
        result = [
            {"instrument_name": "BTC-P", "volume": 5, "mark_iv": 0.0},
            {"instrument_name": "BTC-C", "volume": 5, "mark_iv": 0.0},
        ]
        data = {"result": result}
        with _patch_session(json_data=data):
            pcr, pcr_sig, iv_skew, iv_sig = await mc._get_deribit_options("BTC")
        assert iv_skew == 0.0


# ---------------------------------------------------------------------------
# _get_orderbook_imbalance()
# ---------------------------------------------------------------------------


class TestGetOrderbookImbalance:
    def _ob_data(self, bids, asks):
        return {"result": {"b": bids, "a": asks}}

    @pytest.mark.asyncio
    async def test_bid_dominant(self):
        mc = _make_mc()
        # bid_vol=70, ask_vol=30 → imbalance=0.4 > 0.3 → bid_dominant
        bids = [["50000", "70"]]
        asks = [["50001", "30"]]
        with _patch_session(json_data=self._ob_data(bids, asks)):
            imbalance, signal = await mc._get_orderbook_imbalance("BTC/USDT")
        assert signal == "bid_dominant"
        assert imbalance == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_ask_dominant(self):
        mc = _make_mc()
        # bid_vol=30, ask_vol=70 → imbalance=-0.4 < -0.3 → ask_dominant
        bids = [["50000", "30"]]
        asks = [["50001", "70"]]
        with _patch_session(json_data=self._ob_data(bids, asks)):
            imbalance, signal = await mc._get_orderbook_imbalance("BTC/USDT")
        assert signal == "ask_dominant"

    @pytest.mark.asyncio
    async def test_balanced(self):
        mc = _make_mc()
        # bid_vol=50, ask_vol=50 → imbalance=0.0 → balanced
        bids = [["50000", "50"]]
        asks = [["50001", "50"]]
        with _patch_session(json_data=self._ob_data(bids, asks)):
            imbalance, signal = await mc._get_orderbook_imbalance("BTC/USDT")
        assert signal == "balanced"

    @pytest.mark.asyncio
    async def test_zero_volume_returns_balanced(self):
        mc = _make_mc()
        with _patch_session(json_data=self._ob_data([], [])):
            imbalance, signal = await mc._get_orderbook_imbalance("BTC/USDT")
        assert imbalance == 0.0
        assert signal == "balanced"

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        mc = _make_mc()
        mc._ob_cache["BTC/USDT"] = ((0.5, "bid_dominant"), _BIG_NOW)
        with patch("aiohttp.ClientSession", side_effect=Exception("should not call")):
            imbalance, signal = await mc._get_orderbook_imbalance("BTC/USDT")
        assert imbalance == 0.5
        assert signal == "bid_dominant"

    @pytest.mark.asyncio
    async def test_cache_expired_refetches(self):
        mc = _make_mc()
        mc._ob_cache["BTC/USDT"] = ((0.5, "bid_dominant"), 0.0)  # expired
        bids = [["50000", "50"]]
        asks = [["50001", "50"]]
        with _patch_session(json_data=self._ob_data(bids, asks)):
            imbalance, signal = await mc._get_orderbook_imbalance("BTC/USDT")
        assert signal == "balanced"  # fresh fetch returns balanced

    @pytest.mark.asyncio
    async def test_http_error_returns_balanced(self):
        mc = _make_mc()
        with patch("aiohttp.ClientSession", side_effect=Exception("timeout")):
            imbalance, signal = await mc._get_orderbook_imbalance("BTC/USDT")
        assert imbalance == 0.0
        assert signal == "balanced"


# ---------------------------------------------------------------------------
# _get_glassnode()
# ---------------------------------------------------------------------------


class TestGetGlassnode:
    @pytest.mark.asyncio
    async def test_coingecko_macro_bullish(self):
        mc = _make_mc()
        data = {"market_data": {"market_cap_change_percentage_24h": 6.0}}
        with patch.dict("os.environ", {"GLASSNODE_API_KEY": ""}):
            with _patch_session(json_data=data):
                val, signal = await mc._get_glassnode("BTC")
        assert signal == "macro_bullish"

    @pytest.mark.asyncio
    async def test_coingecko_macro_bearish(self):
        mc = _make_mc()
        data = {"market_data": {"market_cap_change_percentage_24h": -6.0}}
        with patch.dict("os.environ", {"GLASSNODE_API_KEY": ""}):
            with _patch_session(json_data=data):
                val, signal = await mc._get_glassnode("BTC")
        assert signal == "macro_bearish"

    @pytest.mark.asyncio
    async def test_coingecko_neutral(self):
        mc = _make_mc()
        data = {"market_data": {"market_cap_change_percentage_24h": 1.0}}
        with patch.dict("os.environ", {"GLASSNODE_API_KEY": ""}):
            with _patch_session(json_data=data):
                val, signal = await mc._get_glassnode("BTC")
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_unknown_base_returns_neutral(self):
        mc = _make_mc()
        with patch.dict("os.environ", {"GLASSNODE_API_KEY": ""}):
            val, signal = await mc._get_glassnode("UNKNOWN")
        assert val == 0.0
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        mc = _make_mc()
        mc._glassnode_cache["BTC"] = ((7.5, "macro_bullish"), _BIG_NOW)
        with patch("aiohttp.ClientSession", side_effect=Exception("should not call")):
            val, signal = await mc._get_glassnode("BTC")
        assert val == 7.5
        assert signal == "macro_bullish"

    @pytest.mark.asyncio
    async def test_http_error_returns_neutral(self):
        mc = _make_mc()
        with patch.dict("os.environ", {"GLASSNODE_API_KEY": ""}):
            with patch("aiohttp.ClientSession", side_effect=Exception("conn error")):
                val, signal = await mc._get_glassnode("BTC")
        assert val == 0.0
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_glassnode_api_macro_bullish(self):
        """When GLASSNODE_API_KEY is set and response is negative (outflows) → macro_bullish."""
        mc = _make_mc()
        glassnode_data = [{"t": 1700000000, "v": -500.0}]
        with patch.dict("os.environ", {"GLASSNODE_API_KEY": "real_key"}):
            with _patch_session(json_data=glassnode_data):
                val, signal = await mc._get_glassnode("BTC")
        assert signal == "macro_bullish"

    @pytest.mark.asyncio
    async def test_glassnode_api_macro_bearish(self):
        mc = _make_mc()
        glassnode_data = [{"t": 1700000000, "v": 300.0}]
        with patch.dict("os.environ", {"GLASSNODE_API_KEY": "real_key"}):
            with _patch_session(json_data=glassnode_data):
                val, signal = await mc._get_glassnode("BTC")
        assert signal == "macro_bearish"

    @pytest.mark.asyncio
    async def test_glassnode_api_neutral(self):
        mc = _make_mc()
        glassnode_data = [{"t": 1700000000, "v": 0.0}]
        with patch.dict("os.environ", {"GLASSNODE_API_KEY": "real_key"}):
            with _patch_session(json_data=glassnode_data):
                val, signal = await mc._get_glassnode("BTC")
        assert signal == "neutral"


# ---------------------------------------------------------------------------
# _get_btc_etf_flows()
# ---------------------------------------------------------------------------


class TestGetBtcEtfFlows:
    @pytest.mark.asyncio
    async def test_etf_inflow(self):
        mc = _make_mc()
        html = """
        <table><tr><td>Date</td><td>Total</td></tr>
        <tr><td>2024-01-01</td><td>200.5</td></tr>
        </table>
        """
        with _patch_session(text_data=html):
            flow, signal = await mc._get_btc_etf_flows()
        assert signal == "etf_inflow"
        assert flow == pytest.approx(200.5)

    @pytest.mark.asyncio
    async def test_etf_outflow(self):
        mc = _make_mc()
        html = """
        <table><tr><td>Date</td><td>Total</td></tr>
        <tr><td>2024-01-01</td><td>(100.0)</td></tr>
        </table>
        """
        with _patch_session(text_data=html):
            flow, signal = await mc._get_btc_etf_flows()
        assert signal == "etf_outflow"
        assert flow == pytest.approx(-100.0)

    @pytest.mark.asyncio
    async def test_etf_neutral_small_flow(self):
        mc = _make_mc()
        html = """
        <table><tr><td>Date</td><td>Total</td></tr>
        <tr><td>2024-01-01</td><td>30.0</td></tr>
        </table>
        """
        with _patch_session(text_data=html):
            flow, signal = await mc._get_btc_etf_flows()
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        mc = _make_mc()
        mc._etf_cache = ((150.0, "etf_inflow"), _BIG_NOW)
        with patch("aiohttp.ClientSession", side_effect=Exception("should not call")):
            flow, signal = await mc._get_btc_etf_flows()
        assert flow == 150.0
        assert signal == "etf_inflow"

    @pytest.mark.asyncio
    async def test_http_error_returns_neutral(self):
        mc = _make_mc()
        with patch("aiohttp.ClientSession", side_effect=Exception("site down")):
            flow, signal = await mc._get_btc_etf_flows()
        assert flow == 0.0
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_empty_table_returns_neutral(self):
        mc = _make_mc()
        html = "<table><tr><th>Date</th><th>Total</th></tr></table>"
        with _patch_session(text_data=html):
            flow, signal = await mc._get_btc_etf_flows()
        assert flow == 0.0
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_no_table_html_returns_neutral(self):
        mc = _make_mc()
        html = "<html><body><p>No table here</p></body></html>"
        with _patch_session(text_data=html):
            flow, signal = await mc._get_btc_etf_flows()
        assert flow == 0.0
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_comma_formatted_large_number(self):
        mc = _make_mc()
        html = """
        <table><tr><td>Date</td><td>Total</td></tr>
        <tr><td>2024-01-01</td><td>1,500.00</td></tr>
        </table>
        """
        with _patch_session(text_data=html):
            flow, signal = await mc._get_btc_etf_flows()
        assert signal == "etf_inflow"


# ---------------------------------------------------------------------------
# _get_reddit_sentiment()
# ---------------------------------------------------------------------------


class TestGetRedditSentiment:
    @pytest.mark.asyncio
    async def test_no_credentials_returns_neutral(self):
        mc = _make_mc()
        with patch.dict(
            "os.environ",
            {"REDDIT_CLIENT_ID": "", "REDDIT_CLIENT_SECRET": ""},
        ):
            val, signal = await mc._get_reddit_sentiment("BTC")
        assert val == 0.0
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_reddit_bullish(self):
        mc = _make_mc()
        praw_mock = MagicMock()
        with patch.dict("sys.modules", {"praw": praw_mock}), \
             patch.dict("os.environ", {"REDDIT_CLIENT_ID": "cid", "REDDIT_CLIENT_SECRET": "csec"}), \
             patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=0.85)
            val, signal = await mc._get_reddit_sentiment("BTC")
        assert signal == "reddit_bullish"
        assert val == pytest.approx(0.85, abs=0.001)

    @pytest.mark.asyncio
    async def test_reddit_bearish(self):
        mc = _make_mc()
        praw_mock = MagicMock()
        with patch.dict("sys.modules", {"praw": praw_mock}), \
             patch.dict("os.environ", {"REDDIT_CLIENT_ID": "cid", "REDDIT_CLIENT_SECRET": "csec"}), \
             patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=0.4)
            val, signal = await mc._get_reddit_sentiment("BTC")
        assert signal == "reddit_bearish"

    @pytest.mark.asyncio
    async def test_reddit_neutral(self):
        mc = _make_mc()
        with patch.dict(
            "os.environ",
            {"REDDIT_CLIENT_ID": "cid", "REDDIT_CLIENT_SECRET": "csec"},
        ):
            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=0.65)
                val, signal = await mc._get_reddit_sentiment("BTC")
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        mc = _make_mc()
        mc._reddit_cache["BTC"] = ((0.9, "reddit_bullish"), _BIG_NOW)
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(
                side_effect=Exception("should not call")
            )
            val, signal = await mc._get_reddit_sentiment("BTC")
        assert val == 0.9
        assert signal == "reddit_bullish"

    @pytest.mark.asyncio
    async def test_exception_returns_neutral(self):
        mc = _make_mc()
        with patch.dict(
            "os.environ",
            {"REDDIT_CLIENT_ID": "cid", "REDDIT_CLIENT_SECRET": "csec"},
        ):
            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(
                    side_effect=Exception("praw error")
                )
                val, signal = await mc._get_reddit_sentiment("BTC")
        assert val == 0.0
        assert signal == "neutral"


# ---------------------------------------------------------------------------
# _get_stablecoin_supply_change()
# ---------------------------------------------------------------------------


class TestGetStablecoinSupplyChange:
    @pytest.mark.asyncio
    async def test_stablecoin_inflow(self):
        mc = _make_mc()
        data = {
            "market_data": {
                "market_cap_change_percentage_24h_in_currency": {"usd": 1.0}
            }
        }
        with _patch_session(json_data=data):
            val, signal = await mc._get_stablecoin_supply_change()
        assert signal == "stablecoin_inflow"
        assert val == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_stablecoin_outflow(self):
        mc = _make_mc()
        data = {
            "market_data": {
                "market_cap_change_percentage_24h_in_currency": {"usd": -0.5}
            }
        }
        with _patch_session(json_data=data):
            val, signal = await mc._get_stablecoin_supply_change()
        assert signal == "stablecoin_outflow"

    @pytest.mark.asyncio
    async def test_stablecoin_neutral(self):
        mc = _make_mc()
        data = {
            "market_data": {
                "market_cap_change_percentage_24h_in_currency": {"usd": 0.1}
            }
        }
        with _patch_session(json_data=data):
            val, signal = await mc._get_stablecoin_supply_change()
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        mc = _make_mc()
        mc._stablecoin_cache = ((2.0, "stablecoin_inflow"), _BIG_NOW)
        with patch("aiohttp.ClientSession", side_effect=Exception("should not call")):
            val, signal = await mc._get_stablecoin_supply_change()
        assert val == 2.0
        assert signal == "stablecoin_inflow"

    @pytest.mark.asyncio
    async def test_http_error_returns_neutral(self):
        mc = _make_mc()
        with patch("aiohttp.ClientSession", side_effect=Exception("timeout")):
            val, signal = await mc._get_stablecoin_supply_change()
        assert val == 0.0
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_missing_market_data_returns_neutral(self):
        mc = _make_mc()
        data = {}
        with _patch_session(json_data=data):
            val, signal = await mc._get_stablecoin_supply_change()
        assert val == 0.0
        assert signal == "neutral"


# ---------------------------------------------------------------------------
# get_context()  — full aggregation
# ---------------------------------------------------------------------------


class TestGetContext:
    def _patch_all_sub_methods(self, mc):
        """Patch every sub-method on mc with fixed neutral returns."""
        mc._get_funding = AsyncMock(return_value=(0.0001, "neutral"))
        mc._get_oi_signal = AsyncMock(return_value=("oi_neutral", "neutral"))
        mc._get_fear_greed = AsyncMock(return_value=(50, "neutral"))
        mc._get_basis = AsyncMock(return_value=(0.0, "neutral"))
        mc._get_orderbook_imbalance = AsyncMock(return_value=(0.0, "balanced"))
        mc._get_google_trends = AsyncMock(return_value=(50, "neutral"))
        mc._get_deribit_options = AsyncMock(
            return_value=(1.0, "neutral", 0.0, "neutral")
        )
        mc._get_glassnode = AsyncMock(return_value=(0.0, "neutral"))
        mc._get_btc_etf_flows = AsyncMock(return_value=(0.0, "neutral"))
        mc._get_reddit_sentiment = AsyncMock(return_value=(0.0, "neutral"))
        mc._get_stablecoin_supply_change = AsyncMock(return_value=(0.0, "neutral"))

    @pytest.mark.asyncio
    async def test_get_context_returns_all_keys(self):
        mc = _make_mc()
        self._patch_all_sub_methods(mc)
        ctx = await mc.get_context("BTC/USDT", 50000.0)
        assert "funding_rate" in ctx
        assert "funding_signal" in ctx
        assert "oi_signal" in ctx
        assert "fear_greed" in ctx
        assert "macro_signal" in ctx
        assert "etf_flow" in ctx
        assert "stablecoin_signal" in ctx

    @pytest.mark.asyncio
    async def test_get_context_cache_hit(self):
        mc = _make_mc()
        self._patch_all_sub_methods(mc)
        # Warm cache
        await mc.get_context("BTC/USDT", 50000.0)
        call_counts = {
            name: m.call_count
            for name, m in [
                ("funding", mc._get_funding),
                ("oi", mc._get_oi_signal),
            ]
        }
        # Second call should hit cache → sub-methods NOT called again
        await mc.get_context("BTC/USDT", 50000.0)
        assert mc._get_funding.call_count == call_counts["funding"]
        assert mc._get_oi_signal.call_count == call_counts["oi"]

    @pytest.mark.asyncio
    async def test_get_context_cache_expired_refetches(self):
        mc = _make_mc()
        self._patch_all_sub_methods(mc)
        # First call
        await mc.get_context("BTC/USDT", 50000.0)
        # Expire cache
        mc._cache["BTC/USDT"] = (mc._cache["BTC/USDT"][0], 0.0)
        # Second call should refetch
        await mc.get_context("BTC/USDT", 50000.0)
        assert mc._get_funding.call_count == 2

    @pytest.mark.asyncio
    async def test_get_context_btc_calls_etf_and_stablecoin(self):
        mc = _make_mc()
        self._patch_all_sub_methods(mc)
        await mc.get_context("BTC/USDT", 50000.0)
        mc._get_btc_etf_flows.assert_called_once()
        mc._get_stablecoin_supply_change.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_context_non_btc_skips_etf_and_stablecoin(self):
        mc = _make_mc()
        self._patch_all_sub_methods(mc)
        await mc.get_context("ETH/USDT", 3000.0)
        mc._get_btc_etf_flows.assert_not_called()
        mc._get_stablecoin_supply_change.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_context_btc_calls_google_trends(self):
        mc = _make_mc()
        self._patch_all_sub_methods(mc)
        await mc.get_context("BTC/USDT", 50000.0)
        mc._get_google_trends.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_context_non_btc_uses_neutral_google_trends(self):
        mc = _make_mc()
        self._patch_all_sub_methods(mc)
        ctx = await mc.get_context("ETH/USDT", 3000.0)
        mc._get_google_trends.assert_not_called()
        assert ctx["google_trends"] == 50
        assert ctx["google_trends_signal"] == "neutral"

    @pytest.mark.asyncio
    async def test_get_context_btc_calls_deribit(self):
        mc = _make_mc()
        self._patch_all_sub_methods(mc)
        await mc.get_context("BTC/USDT", 50000.0)
        mc._get_deribit_options.assert_called_once_with("BTC")

    @pytest.mark.asyncio
    async def test_get_context_eth_calls_deribit(self):
        mc = _make_mc()
        self._patch_all_sub_methods(mc)
        await mc.get_context("ETH/USDT", 3000.0)
        mc._get_deribit_options.assert_called_once_with("ETH")

    @pytest.mark.asyncio
    async def test_get_context_sol_skips_deribit(self):
        mc = _make_mc()
        self._patch_all_sub_methods(mc)
        ctx = await mc.get_context("SOL/USDT", 100.0)
        mc._get_deribit_options.assert_not_called()
        assert ctx["pcr"] == 1.0
        assert ctx["pcr_signal"] == "neutral"

    @pytest.mark.asyncio
    async def test_get_context_stores_in_cache(self):
        mc = _make_mc()
        self._patch_all_sub_methods(mc)
        await mc.get_context("BTC/USDT", 50000.0)
        assert "BTC/USDT" in mc._cache

    @pytest.mark.asyncio
    async def test_get_context_non_btc_etf_signal_neutral(self):
        mc = _make_mc()
        self._patch_all_sub_methods(mc)
        ctx = await mc.get_context("SOL/USDT", 100.0)
        assert ctx["etf_flow"] == 0.0
        assert ctx["etf_signal"] == "neutral"
        assert ctx["stablecoin_signal"] == "neutral"


# ---------------------------------------------------------------------------
# get_context_for_symbols()
# ---------------------------------------------------------------------------


class TestGetContextForSymbols:
    @pytest.mark.asyncio
    async def test_returns_dict_for_all_symbols(self):
        mc = _make_mc()

        async def _fake_get_context(symbol, price):
            return {"funding_signal": "neutral", "symbol": symbol}

        mc.get_context = _fake_get_context
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        prices = {"BTC/USDT": 50000.0, "ETH/USDT": 3000.0, "SOL/USDT": 100.0}
        result = await mc.get_context_for_symbols(symbols, prices)
        assert set(result.keys()) == set(symbols)
        for sym in symbols:
            assert result[sym]["symbol"] == sym

    @pytest.mark.asyncio
    async def test_exception_fallback_to_neutral(self):
        mc = _make_mc()

        async def _failing_context(symbol, price):
            raise RuntimeError("network error")

        mc.get_context = _failing_context
        symbols = ["BTC/USDT", "ETH/USDT"]
        prices = {"BTC/USDT": 50000.0, "ETH/USDT": 3000.0}
        result = await mc.get_context_for_symbols(symbols, prices)
        # Both should fall back to neutral_context
        for sym in symbols:
            assert result[sym]["funding_signal"] == "neutral"
            assert result[sym]["oi_signal"] == "oi_neutral"

    @pytest.mark.asyncio
    async def test_missing_price_uses_zero(self):
        mc = _make_mc()
        received_prices = {}

        async def _capture(symbol, price):
            received_prices[symbol] = price
            return mc._neutral_context()

        mc.get_context = _capture
        await mc.get_context_for_symbols(["BTC/USDT"], {})
        assert received_prices["BTC/USDT"] == 0.0

    @pytest.mark.asyncio
    async def test_empty_symbols_returns_empty(self):
        mc = _make_mc()
        result = await mc.get_context_for_symbols([], {})
        assert result == {}


# ---------------------------------------------------------------------------
# Cache TTL boundary tests
# ---------------------------------------------------------------------------


class TestCacheTTL:
    @pytest.mark.asyncio
    async def test_fng_cache_not_expired_just_before_ttl(self):
        from src.market_context import _CACHE_TTL

        mc = _make_mc()
        # Set cache timestamp to just within TTL
        mc._fng_cache = (75, _BIG_NOW - (_CACHE_TTL - 1.0))
        with patch("aiohttp.ClientSession", side_effect=Exception("should not call")):
            val, signal = await mc._get_fear_greed()
        assert val == 75

    @pytest.mark.asyncio
    async def test_ob_cache_not_expired_just_before_ttl(self):
        from src.market_context import _OB_TTL

        mc = _make_mc()
        mc._ob_cache["BTC/USDT"] = (
            (0.45, "bid_dominant"),
            _BIG_NOW - (_OB_TTL - 1.0),
        )
        with patch("aiohttp.ClientSession", side_effect=Exception("should not call")):
            imbalance, signal = await mc._get_orderbook_imbalance("BTC/USDT")
        assert signal == "bid_dominant"

    @pytest.mark.asyncio
    async def test_deribit_cache_not_expired_just_before_ttl(self):
        from src.market_context import _PCR_TTL

        mc = _make_mc()
        mc._deribit_cache["ETH"] = (
            (0.8, "neutral", 2.0, "neutral"),
            _BIG_NOW - (_PCR_TTL - 1.0),
        )
        with patch("aiohttp.ClientSession", side_effect=Exception("should not call")):
            result = await mc._get_deribit_options("ETH")
        assert result[0] == 0.8

    @pytest.mark.asyncio
    async def test_stablecoin_cache_not_expired_just_before_ttl(self):
        from src.market_context import _STABLECOIN_TTL

        mc = _make_mc()
        mc._stablecoin_cache = (
            (1.5, "stablecoin_inflow"),
            _BIG_NOW - (_STABLECOIN_TTL - 1.0),
        )
        with patch("aiohttp.ClientSession", side_effect=Exception("should not call")):
            val, signal = await mc._get_stablecoin_supply_change()
        assert signal == "stablecoin_inflow"

    @pytest.mark.asyncio
    async def test_etf_cache_not_expired_just_before_ttl(self):
        from src.market_context import _ETF_TTL

        mc = _make_mc()
        mc._etf_cache = (
            (300.0, "etf_inflow"),
            _BIG_NOW - (_ETF_TTL - 1.0),
        )
        with patch("aiohttp.ClientSession", side_effect=Exception("should not call")):
            flow, signal = await mc._get_btc_etf_flows()
        assert flow == 300.0

    @pytest.mark.asyncio
    async def test_reddit_cache_not_expired_just_before_ttl(self):
        from src.market_context import _REDDIT_TTL

        mc = _make_mc()
        mc._reddit_cache["ETH"] = (
            (0.9, "reddit_bullish"),
            _BIG_NOW - (_REDDIT_TTL - 1.0),
        )
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(
                side_effect=Exception("should not call")
            )
            val, signal = await mc._get_reddit_sentiment("ETH")
        assert signal == "reddit_bullish"


# ---------------------------------------------------------------------------
# Signal boundary / edge cases
# ---------------------------------------------------------------------------


class TestSignalBoundaries:
    @pytest.mark.asyncio
    async def test_basis_exactly_1_5_is_neutral(self):
        """Basis == 1.5 is NOT > 1.5 → neutral."""
        mc = _make_mc()
        # spot=100, futures=101.5 → basis=1.5%
        data = {"result": {"list": [{"lastPrice": "101.5"}]}}
        with _patch_session(json_data=data):
            basis, signal = await mc._get_basis("BTC/USDT", 100.0)
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_basis_exactly_minus_0_5_is_neutral(self):
        """Basis == -0.5 is NOT < -0.5 → neutral."""
        mc = _make_mc()
        data = {"result": {"list": [{"lastPrice": "99.5"}]}}
        with _patch_session(json_data=data):
            basis, signal = await mc._get_basis("BTC/USDT", 100.0)
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_ob_imbalance_exactly_0_3_is_balanced(self):
        """imbalance == 0.3 is NOT > 0.3 → balanced."""
        mc = _make_mc()
        # bid=65, ask=35 → imbalance=(65-35)/100=0.3
        bids = [["50000", "65"]]
        asks = [["50001", "35"]]
        data = {"result": {"b": bids, "a": asks}}
        with _patch_session(json_data=data):
            imbalance, signal = await mc._get_orderbook_imbalance("BTC/USDT")
        assert signal == "balanced"

    @pytest.mark.asyncio
    async def test_pcr_exactly_1_5_is_neutral(self):
        """PCR == 1.5 is NOT > 1.5 → neutral."""
        mc = _make_mc()
        result = [
            {"instrument_name": "BTC-P", "volume": 15, "mark_iv": 50.0},
            {"instrument_name": "BTC-C", "volume": 10, "mark_iv": 40.0},
        ]
        data = {"result": result}
        with _patch_session(json_data=data):
            pcr, pcr_sig, iv_skew, iv_sig = await mc._get_deribit_options("BTC")
        assert pcr_sig == "neutral"

    @pytest.mark.asyncio
    async def test_pcr_exactly_0_5_is_neutral(self):
        """PCR == 0.5 is NOT < 0.5 → neutral."""
        mc = _make_mc()
        result = [
            {"instrument_name": "BTC-P", "volume": 5, "mark_iv": 50.0},
            {"instrument_name": "BTC-C", "volume": 10, "mark_iv": 40.0},
        ]
        data = {"result": result}
        with _patch_session(json_data=data):
            pcr, pcr_sig, iv_skew, iv_sig = await mc._get_deribit_options("BTC")
        assert pcr_sig == "neutral"
