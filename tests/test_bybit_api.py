"""Тесты BybitAPI клиента."""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import ccxt.async_support as ccxt
import pandas as pd
import pytest

from src.bybit_api import BybitAPI, _fetch_ohlcv_with_retry, _is_retryable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_api_with_mock_exchange() -> BybitAPI:
    """Создаёт BybitAPI с замоканным exchange и отключённым Redis."""
    api = BybitAPI.__new__(BybitAPI)
    api.logger = logging.getLogger("test")
    api.redis = MagicMock()
    api.redis.load_market_data.return_value = None
    api.redis._available = False
    mock_exchange = AsyncMock()
    mock_exchange.markets = {}
    # amount_to_precision — sync метод ccxt; AsyncMock сделал бы его корутиной,
    # что ломает float(exchange.amount_to_precision(...)) в round_quantity.
    mock_exchange.amount_to_precision = MagicMock()
    api.exchange = mock_exchange
    return api


_SAMPLE_OHLCV = [
    [1_700_000_000_000, 30000.0, 30500.0, 29800.0, 30200.0, 123.4],
    [1_700_000_060_000, 30200.0, 30600.0, 30100.0, 30400.0, 234.5],
    [1_700_000_120_000, 30400.0, 30700.0, 30300.0, 30600.0, 345.6],
]


# ---------------------------------------------------------------------------
# _is_retryable
# ---------------------------------------------------------------------------

class TestIsRetryable:
    def test_retryable_network_error(self):
        assert _is_retryable(ccxt.NetworkError("net err")) is True

    def test_retryable_request_timeout(self):
        assert _is_retryable(ccxt.RequestTimeout("timeout")) is True

    def test_retryable_exchange_not_available(self):
        assert _is_retryable(ccxt.ExchangeNotAvailable("down")) is True

    def test_not_retryable_auth_error(self):
        assert _is_retryable(ccxt.AuthenticationError("bad key")) is False

    def test_not_retryable_value_error(self):
        assert _is_retryable(ValueError("bad value")) is False

    def test_not_retryable_generic_exception(self):
        assert _is_retryable(Exception("generic")) is False


# ---------------------------------------------------------------------------
# BybitAPI._process_ohlcv
# ---------------------------------------------------------------------------

class TestProcessOhlcv:
    def setup_method(self):
        self.api = make_api_with_mock_exchange()

    def test_process_ohlcv_returns_dataframe(self):
        df = self.api._process_ohlcv(_SAMPLE_OHLCV)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 3

    def test_process_ohlcv_sets_timestamp_index(self):
        df = self.api._process_ohlcv(_SAMPLE_OHLCV)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "timestamp"

    def test_process_ohlcv_correct_values(self):
        df = self.api._process_ohlcv(_SAMPLE_OHLCV)
        assert df["close"].iloc[0] == pytest.approx(30200.0)
        assert df["volume"].iloc[2] == pytest.approx(345.6)


# ---------------------------------------------------------------------------
# BybitAPI.get_ohlcv
# ---------------------------------------------------------------------------

class TestGetOhlcv:
    def setup_method(self):
        self.api = make_api_with_mock_exchange()

    @pytest.mark.asyncio
    async def test_get_ohlcv_cache_hit(self):
        """Cache hit: Redis returns a DataFrame → no exchange call."""
        cached_df = pd.DataFrame({"close": [1.0, 2.0]})
        self.api.redis.load_market_data.return_value = cached_df

        result = await self.api.get_ohlcv("BTC/USDT", "1h", 100)

        assert result is cached_df
        self.api.exchange.fetch_ohlcv.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_ohlcv_cache_miss_fetches_exchange(self):
        """Cache miss: exchange.fetch_ohlcv is called and result is cached."""
        self.api.redis.load_market_data.return_value = None
        self.api.exchange.fetch_ohlcv.return_value = _SAMPLE_OHLCV

        result = await self.api.get_ohlcv("BTC/USDT", "1h", 3)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        self.api.exchange.fetch_ohlcv.assert_called_once()
        self.api.redis.save_market_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_ohlcv_network_error_raises(self):
        """NetworkError from exchange propagates out."""
        self.api.redis.load_market_data.return_value = None
        self.api.exchange.fetch_ohlcv.side_effect = ccxt.NetworkError("net")

        with pytest.raises(ccxt.NetworkError):
            await self.api.get_ohlcv("BTC/USDT", "1h", 100)

    @pytest.mark.asyncio
    async def test_get_ohlcv_auth_error_raises(self):
        """AuthenticationError from exchange propagates out."""
        self.api.redis.load_market_data.return_value = None
        self.api.exchange.fetch_ohlcv.side_effect = ccxt.AuthenticationError("bad key")

        with pytest.raises(ccxt.AuthenticationError):
            await self.api.get_ohlcv("BTC/USDT", "1h", 100)

    @pytest.mark.asyncio
    async def test_get_ohlcv_cache_key_format(self):
        """Cache key is symbol:timeframe:limit."""
        self.api.redis.load_market_data.return_value = None
        self.api.exchange.fetch_ohlcv.return_value = _SAMPLE_OHLCV

        await self.api.get_ohlcv("ETH/USDT", "4h", 50)

        self.api.redis.load_market_data.assert_called_with("ETH/USDT:4h:50")


# ---------------------------------------------------------------------------
# BybitAPI.get_balance
# ---------------------------------------------------------------------------

class TestGetBalance:
    def setup_method(self):
        self.api = make_api_with_mock_exchange()

    @pytest.mark.asyncio
    async def test_get_balance_success(self):
        balance = {"USDT": {"free": 5000.0}, "total": {}}
        self.api.exchange.fetch_balance.return_value = balance

        result = await self.api.get_balance()

        assert result is balance

    @pytest.mark.asyncio
    async def test_get_balance_network_error_returns_none(self):
        self.api.exchange.fetch_balance.side_effect = ccxt.NetworkError("net")

        result = await self.api.get_balance()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_balance_auth_error_raises(self):
        self.api.exchange.fetch_balance.side_effect = ccxt.AuthenticationError("bad key")

        with pytest.raises(ccxt.AuthenticationError):
            await self.api.get_balance()

    @pytest.mark.asyncio
    async def test_get_balance_generic_error_returns_none(self):
        self.api.exchange.fetch_balance.side_effect = RuntimeError("unexpected")

        result = await self.api.get_balance()

        assert result is None


# ---------------------------------------------------------------------------
# BybitAPI.fetch_positions
# ---------------------------------------------------------------------------

class TestFetchPositions:
    def setup_method(self):
        self.api = make_api_with_mock_exchange()

    @pytest.mark.asyncio
    async def test_fetch_positions_filters_zero_contracts(self):
        self.api.exchange.fetch_positions.return_value = [
            {"symbol": "BTC/USDT", "contracts": 0},
            {"symbol": "ETH/USDT", "contracts": 0.0},
            {"symbol": "ADA/USDT", "contracts": None},
        ]

        result = await self.api.fetch_positions()

        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_positions_keeps_nonzero(self):
        self.api.exchange.fetch_positions.return_value = [
            {"symbol": "BTC/USDT", "contracts": 1.5},
            {"symbol": "ETH/USDT", "contracts": 0},
        ]

        result = await self.api.fetch_positions()

        assert len(result) == 1
        assert result[0]["symbol"] == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_fetch_positions_network_error_returns_empty(self):
        self.api.exchange.fetch_positions.side_effect = ccxt.NetworkError("net")

        result = await self.api.fetch_positions()

        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_positions_auth_error_raises(self):
        self.api.exchange.fetch_positions.side_effect = ccxt.AuthenticationError("bad")

        with pytest.raises(ccxt.AuthenticationError):
            await self.api.fetch_positions()

    @pytest.mark.asyncio
    async def test_fetch_positions_returns_none_fallback_to_empty(self):
        """If exchange returns None, result should be []."""
        self.api.exchange.fetch_positions.return_value = None

        result = await self.api.fetch_positions()

        assert result == []


# ---------------------------------------------------------------------------
# BybitAPI.get_current_price
# ---------------------------------------------------------------------------

class TestGetCurrentPrice:
    def setup_method(self):
        self.api = make_api_with_mock_exchange()

    @pytest.mark.asyncio
    async def test_get_current_price_success(self):
        self.api.exchange.fetch_ticker.return_value = {"last": 50000.0, "bid": 49999.0}

        result = await self.api.get_current_price("BTC/USDT")

        assert result == pytest.approx(50000.0)
        self.api.exchange.fetch_ticker.assert_called_once_with("BTC/USDT")

    @pytest.mark.asyncio
    async def test_get_current_price_network_error_returns_none(self):
        self.api.exchange.fetch_ticker.side_effect = ccxt.NetworkError("net")

        result = await self.api.get_current_price("BTC/USDT")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_current_price_generic_error_returns_none(self):
        self.api.exchange.fetch_ticker.side_effect = RuntimeError("unexpected")

        result = await self.api.get_current_price("BTC/USDT")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_current_price_auth_error_raises(self):
        self.api.exchange.fetch_ticker.side_effect = ccxt.AuthenticationError("bad")

        with pytest.raises(ccxt.AuthenticationError):
            await self.api.get_current_price("BTC/USDT")


# ---------------------------------------------------------------------------
# BybitAPI.cancel_order
# ---------------------------------------------------------------------------

class TestCancelOrder:
    def setup_method(self):
        self.api = make_api_with_mock_exchange()

    @pytest.mark.asyncio
    async def test_cancel_order_success(self):
        self.api.exchange.cancel_order.return_value = {"id": "123"}

        result = await self.api.cancel_order("123", "BTC/USDT")

        assert result is True
        self.api.exchange.cancel_order.assert_called_once_with("123", "BTC/USDT")

    @pytest.mark.asyncio
    async def test_cancel_order_network_error(self):
        self.api.exchange.cancel_order.side_effect = ccxt.NetworkError("net")

        result = await self.api.cancel_order("123", "BTC/USDT")

        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_order_auth_error_raises(self):
        self.api.exchange.cancel_order.side_effect = ccxt.AuthenticationError("bad")

        with pytest.raises(ccxt.AuthenticationError):
            await self.api.cancel_order("123", "BTC/USDT")

    @pytest.mark.asyncio
    async def test_cancel_order_generic_error_returns_false(self):
        self.api.exchange.cancel_order.side_effect = RuntimeError("unexpected")

        result = await self.api.cancel_order("123", "BTC/USDT")

        assert result is False


# ---------------------------------------------------------------------------
# BybitAPI.round_quantity
# ---------------------------------------------------------------------------

class TestRoundQuantity:
    def setup_method(self):
        self.api = make_api_with_mock_exchange()

    def test_round_quantity_no_exchange(self):
        """exchange=None → quantity returned unchanged."""
        self.api.exchange = None
        result = self.api.round_quantity("BTC/USDT", 1.23456)
        assert result == pytest.approx(1.23456)

    def test_round_quantity_symbol_not_in_markets(self):
        """Symbol absent from markets → quantity unchanged."""
        self.api.exchange.markets = {}
        result = self.api.round_quantity("XYZ/USDT", 2.5)
        assert result == pytest.approx(2.5)

    def test_round_quantity_rounds_to_precision(self):
        """amount_to_precision used and result returned."""
        self.api.exchange.markets = {
            "BTC/USDT": {"limits": {"amount": {"min": 0.001}}}
        }
        self.api.exchange.amount_to_precision.return_value = "0.12300"

        result = self.api.round_quantity("BTC/USDT", 0.123456)

        assert result == pytest.approx(0.123)
        self.api.exchange.amount_to_precision.assert_called_once_with("BTC/USDT", 0.123456)

    def test_round_quantity_below_minimum(self):
        """Rounded value below min_qty → 0.0 returned."""
        self.api.exchange.markets = {
            "BTC/USDT": {"limits": {"amount": {"min": 0.01}}}
        }
        self.api.exchange.amount_to_precision.return_value = "0.001"

        result = self.api.round_quantity("BTC/USDT", 0.001)

        assert result == pytest.approx(0.0)

    def test_round_quantity_no_min_limit(self):
        """No min limit set → positive rounded value returned normally."""
        self.api.exchange.markets = {
            "BTC/USDT": {"limits": {"amount": {"min": None}}}
        }
        self.api.exchange.amount_to_precision.return_value = "0.500"

        result = self.api.round_quantity("BTC/USDT", 0.5)

        assert result == pytest.approx(0.5)

    def test_round_quantity_exception_returns_original(self):
        """Exception in amount_to_precision → original quantity returned."""
        self.api.exchange.markets = {
            "BTC/USDT": {"limits": {"amount": {"min": 0.001}}}
        }
        self.api.exchange.amount_to_precision.side_effect = RuntimeError("bad")

        result = self.api.round_quantity("BTC/USDT", 1.5)

        assert result == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# BybitAPI.create_order
# ---------------------------------------------------------------------------

class TestCreateOrder:
    def setup_method(self):
        self.api = make_api_with_mock_exchange()

    @pytest.mark.asyncio
    async def test_create_order_lock_fails_returns_none(self):
        """acquire_lock returns False → create_order returns None immediately."""
        self.api.redis.acquire_lock.return_value = False

        result = await self.api.create_order("BTC/USDT", "market", "buy", 0.01)

        assert result is None
        self.api.exchange.create_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_order_success(self):
        """Successful order creation returns the exchange order dict."""
        self.api.redis.acquire_lock.return_value = True
        order = {"id": "order-1", "filled": 0.01, "status": "open"}
        self.api.exchange.create_order.return_value = order

        result = await self.api.create_order("BTC/USDT", "market", "buy", 0.01)

        assert result is order
        self.api.redis.release_lock.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_order_insufficient_funds_returns_none(self):
        """InsufficientFunds → None returned, lock released."""
        self.api.redis.acquire_lock.return_value = True
        self.api.exchange.create_order.side_effect = ccxt.InsufficientFunds("no funds")

        result = await self.api.create_order("BTC/USDT", "market", "buy", 0.01)

        assert result is None
        self.api.redis.release_lock.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_order_auth_error_raises(self):
        """AuthenticationError propagates out, lock still released."""
        self.api.redis.acquire_lock.return_value = True
        self.api.exchange.create_order.side_effect = ccxt.AuthenticationError("bad")

        with pytest.raises(ccxt.AuthenticationError):
            await self.api.create_order("BTC/USDT", "market", "buy", 0.01)

        self.api.redis.release_lock.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_order_saves_state_on_success(self):
        """Successful order → redis.save_trading_state called."""
        self.api.redis.acquire_lock.return_value = True
        order = {"id": "order-99", "filled": 0.0, "status": "open"}
        self.api.exchange.create_order.return_value = order

        await self.api.create_order("BTC/USDT", "limit", "sell", 0.1, price=30000.0)

        self.api.redis.save_trading_state.assert_called_once()


# ---------------------------------------------------------------------------
# BybitAPI.place_exchange_sl_tp
# ---------------------------------------------------------------------------

class TestPlaceExchangeSlTp:
    def setup_method(self):
        self.api = make_api_with_mock_exchange()

    @pytest.mark.asyncio
    async def test_place_sl_tp_both_placed(self):
        """sl_price > 0 and tp_price > 0 → both orders created, IDs returned."""
        sl_order = {"id": "sl-1"}
        tp_order = {"id": "tp-1"}
        self.api.exchange.create_order.side_effect = [sl_order, tp_order]

        sl_id, tp_id = await self.api.place_exchange_sl_tp(
            "BTC/USDT", "sell", 0.01, sl_price=29000.0, tp_price=32000.0
        )

        assert sl_id == "sl-1"
        assert tp_id == "tp-1"
        assert self.api.exchange.create_order.call_count == 2

    @pytest.mark.asyncio
    async def test_place_sl_tp_zero_prices_skip(self):
        """sl_price=0, tp_price=0 → no orders placed, (None, None) returned."""
        sl_id, tp_id = await self.api.place_exchange_sl_tp(
            "BTC/USDT", "sell", 0.01, sl_price=0, tp_price=0
        )

        assert sl_id is None
        assert tp_id is None
        self.api.exchange.create_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_place_sl_tp_only_sl(self):
        """Only sl_price set → only one order, tp_id is None."""
        sl_order = {"id": "sl-only"}
        self.api.exchange.create_order.return_value = sl_order

        sl_id, tp_id = await self.api.place_exchange_sl_tp(
            "BTC/USDT", "sell", 0.01, sl_price=29000.0, tp_price=0
        )

        assert sl_id == "sl-only"
        assert tp_id is None
        assert self.api.exchange.create_order.call_count == 1

    @pytest.mark.asyncio
    async def test_place_sl_tp_exchange_error_returns_none_ids(self):
        """Exchange error on SL/TP → logged, returns (None, None)."""
        self.api.exchange.create_order.side_effect = ccxt.NetworkError("net")

        sl_id, tp_id = await self.api.place_exchange_sl_tp(
            "BTC/USDT", "sell", 0.01, sl_price=29000.0, tp_price=32000.0
        )

        assert sl_id is None
        assert tp_id is None


# ---------------------------------------------------------------------------
# BybitAPI.fetch_order_status
# ---------------------------------------------------------------------------

class TestFetchOrderStatus:
    def setup_method(self):
        self.api = make_api_with_mock_exchange()

    @pytest.mark.asyncio
    async def test_fetch_order_status_success(self):
        order = {"id": "ord-1", "status": "closed"}
        self.api.exchange.fetch_order.return_value = order

        result = await self.api.fetch_order_status("ord-1", "BTC/USDT")

        assert result is order

    @pytest.mark.asyncio
    async def test_fetch_order_status_network_error_returns_none(self):
        self.api.exchange.fetch_order.side_effect = ccxt.NetworkError("net")

        result = await self.api.fetch_order_status("ord-1", "BTC/USDT")

        assert result is None
