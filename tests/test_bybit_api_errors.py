"""
Тесты error paths src/bybit_api.py.

Покрывают непокрытые ветки (строки 95-131, 179, 184-185, 189-193,
241, 248-269, 282, 308, 315-317, 330, 356, 358-361, 367-371, 441, 489-490):
- initialize(): testnet, AuthenticationError, NetworkError, общее исключение
- get_ohlcv(): CancelledError, InsufficientFunds, NetworkError, общее исключение
- create_order(): CancelledError, AuthenticationError, InsufficientFunds,
                  NetworkError-retry-loop, общее исключение-retry-loop,
                  блокировка занята
- get_balance(): CancelledError, NetworkError, общее исключение
- fetch_positions(): CancelledError, общее исключение
- get_current_price(): CancelledError
- fetch_order_status(): CancelledError, AuthenticationError, общее исключение
- cancel_order(): CancelledError
- close(): exchange is not None
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import ccxt.async_support as ccxt
import pytest

from src.bybit_api import BybitAPI

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_api() -> BybitAPI:
    api = BybitAPI.__new__(BybitAPI)
    api.logger = logging.getLogger("test")
    api.redis = MagicMock()
    api.redis._available = False
    api.redis.acquire_lock.return_value = True
    api.redis.release_lock.return_value = None
    api.redis.save_trading_state.return_value = None
    mock_exchange = AsyncMock()
    mock_exchange.markets = {}
    mock_exchange.amount_to_precision = MagicMock()
    api.exchange = mock_exchange
    return api


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


class TestInitialize:
    @pytest.mark.asyncio
    async def test_initialize_mainnet_calls_load_markets(self):
        api = BybitAPI.__new__(BybitAPI)
        api.logger = logging.getLogger("test")
        api.redis = MagicMock()

        mock_exchange = AsyncMock()
        mock_exchange_cls = MagicMock(return_value=mock_exchange)

        with patch("src.bybit_api.ccxt.bybit", mock_exchange_cls):
            await api.initialize("key", "secret", testnet=False)

        mock_exchange.load_markets.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initialize_testnet_uses_testnet_urls(self):
        api = BybitAPI.__new__(BybitAPI)
        api.logger = logging.getLogger("test")
        api.redis = MagicMock()

        mock_exchange = AsyncMock()
        mock_exchange_cls = MagicMock(return_value=mock_exchange)

        with patch("src.bybit_api.ccxt.bybit", mock_exchange_cls):
            await api.initialize("key", "secret", testnet=True)

        call_kwargs = mock_exchange_cls.call_args[0][0]
        assert "urls" in call_kwargs
        assert "testnet" in call_kwargs["urls"]["api"]["public"]

    @pytest.mark.asyncio
    async def test_initialize_auth_error_raises(self):
        api = BybitAPI.__new__(BybitAPI)
        api.logger = logging.getLogger("test")
        api.redis = MagicMock()

        mock_exchange = AsyncMock()
        mock_exchange.load_markets.side_effect = ccxt.AuthenticationError("bad key")
        mock_exchange_cls = MagicMock(return_value=mock_exchange)

        with patch("src.bybit_api.ccxt.bybit", mock_exchange_cls):
            with pytest.raises(ccxt.AuthenticationError):
                await api.initialize("bad", "key")

    @pytest.mark.asyncio
    async def test_initialize_network_error_raises(self):
        api = BybitAPI.__new__(BybitAPI)
        api.logger = logging.getLogger("test")
        api.redis = MagicMock()

        mock_exchange = AsyncMock()
        mock_exchange.load_markets.side_effect = ccxt.NetworkError("down")
        mock_exchange_cls = MagicMock(return_value=mock_exchange)

        with patch("src.bybit_api.ccxt.bybit", mock_exchange_cls):
            with pytest.raises(ccxt.NetworkError):
                await api.initialize("k", "s")

    @pytest.mark.asyncio
    async def test_initialize_generic_error_raises(self):
        api = BybitAPI.__new__(BybitAPI)
        api.logger = logging.getLogger("test")
        api.redis = MagicMock()

        mock_exchange = AsyncMock()
        mock_exchange.load_markets.side_effect = RuntimeError("unexpected")
        mock_exchange_cls = MagicMock(return_value=mock_exchange)

        with patch("src.bybit_api.ccxt.bybit", mock_exchange_cls):
            with pytest.raises(RuntimeError):
                await api.initialize("k", "s")


# ---------------------------------------------------------------------------
# get_ohlcv error paths
# ---------------------------------------------------------------------------


class TestGetOhlcvErrors:
    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        api = _make_api()
        api.redis.load_market_data.return_value = None
        with patch(
            "src.bybit_api._fetch_ohlcv_with_retry",
            side_effect=asyncio.CancelledError(),
        ):
            with pytest.raises(asyncio.CancelledError):
                await api.get_ohlcv("BTC/USDT")

    @pytest.mark.asyncio
    async def test_insufficient_funds_raises(self):
        api = _make_api()
        api.redis.load_market_data.return_value = None
        with patch(
            "src.bybit_api._fetch_ohlcv_with_retry",
            side_effect=ccxt.InsufficientFunds("no funds"),
        ):
            with pytest.raises(ccxt.InsufficientFunds):
                await api.get_ohlcv("BTC/USDT")

    @pytest.mark.asyncio
    async def test_network_error_raises(self):
        api = _make_api()
        api.redis.load_market_data.return_value = None
        with patch(
            "src.bybit_api._fetch_ohlcv_with_retry",
            side_effect=ccxt.NetworkError("net"),
        ):
            with pytest.raises(ccxt.NetworkError):
                await api.get_ohlcv("BTC/USDT")

    @pytest.mark.asyncio
    async def test_generic_exception_raises(self):
        api = _make_api()
        api.redis.load_market_data.return_value = None
        with patch(
            "src.bybit_api._fetch_ohlcv_with_retry", side_effect=ValueError("bad data")
        ):
            with pytest.raises(ValueError):
                await api.get_ohlcv("BTC/USDT")


# ---------------------------------------------------------------------------
# create_order error paths
# ---------------------------------------------------------------------------


class TestCreateOrderErrors:
    @pytest.mark.asyncio
    async def test_lock_not_acquired_returns_none(self):
        api = _make_api()
        api.redis.acquire_lock.return_value = False
        result = await api.create_order("BTC/USDT", "market", "buy", 0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        api = _make_api()
        api.exchange.create_order.side_effect = asyncio.CancelledError()
        with pytest.raises(asyncio.CancelledError):
            await api.create_order("BTC/USDT", "market", "buy", 0.1)

    @pytest.mark.asyncio
    async def test_auth_error_raises(self):
        api = _make_api()
        api.exchange.create_order.side_effect = ccxt.AuthenticationError("bad")
        with pytest.raises(ccxt.AuthenticationError):
            await api.create_order("BTC/USDT", "market", "buy", 0.1)

    @pytest.mark.asyncio
    async def test_insufficient_funds_returns_none(self):
        api = _make_api()
        api.exchange.create_order.side_effect = ccxt.InsufficientFunds("broke")
        result = await api.create_order("BTC/USDT", "market", "buy", 0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_network_error_retries_and_returns_none(self):
        api = _make_api()
        api.exchange.create_order.side_effect = ccxt.NetworkError("net")
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await api.create_order("BTC/USDT", "market", "buy", 0.1)
        assert result is None
        assert api.exchange.create_order.call_count == 3

    @pytest.mark.asyncio
    async def test_generic_error_retries_and_returns_none(self):
        api = _make_api()
        api.exchange.create_order.side_effect = RuntimeError("crash")
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await api.create_order("BTC/USDT", "market", "buy", 0.1)
        assert result is None
        assert api.exchange.create_order.call_count == 3

    @pytest.mark.asyncio
    async def test_success_on_second_attempt(self):
        api = _make_api()
        order = {"id": "abc", "filled": 0.1, "status": "closed"}
        api.exchange.create_order.side_effect = [ccxt.NetworkError("net"), order]
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await api.create_order("BTC/USDT", "market", "buy", 0.1)
        assert result == order


# ---------------------------------------------------------------------------
# get_balance error paths
# ---------------------------------------------------------------------------


class TestGetBalanceErrors:
    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        api = _make_api()
        api.exchange.fetch_balance.side_effect = asyncio.CancelledError()
        with pytest.raises(asyncio.CancelledError):
            await api.get_balance()

    @pytest.mark.asyncio
    async def test_network_error_returns_none(self):
        api = _make_api()
        api.exchange.fetch_balance.side_effect = ccxt.NetworkError("net")
        result = await api.get_balance()
        assert result is None

    @pytest.mark.asyncio
    async def test_generic_error_returns_none(self):
        api = _make_api()
        api.exchange.fetch_balance.side_effect = RuntimeError("crash")
        result = await api.get_balance()
        assert result is None


# ---------------------------------------------------------------------------
# fetch_positions error paths
# ---------------------------------------------------------------------------


class TestFetchPositionsErrors:
    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        api = _make_api()
        api.exchange.fetch_positions.side_effect = asyncio.CancelledError()
        with pytest.raises(asyncio.CancelledError):
            await api.fetch_positions()

    @pytest.mark.asyncio
    async def test_generic_error_returns_empty(self):
        api = _make_api()
        api.exchange.fetch_positions.side_effect = RuntimeError("crash")
        result = await api.fetch_positions()
        assert result == []

    @pytest.mark.asyncio
    async def test_filters_zero_contracts(self):
        api = _make_api()
        api.exchange.fetch_positions.return_value = [
            {"contracts": 0.0, "symbol": "BTC/USDT"},
            {"contracts": 1.0, "symbol": "ETH/USDT"},
        ]
        result = await api.fetch_positions()
        assert len(result) == 1
        assert result[0]["symbol"] == "ETH/USDT"


# ---------------------------------------------------------------------------
# get_current_price error paths
# ---------------------------------------------------------------------------


class TestGetCurrentPriceErrors:
    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        api = _make_api()
        api.exchange.fetch_ticker.side_effect = asyncio.CancelledError()
        with pytest.raises(asyncio.CancelledError):
            await api.get_current_price("BTC/USDT")


# ---------------------------------------------------------------------------
# fetch_order_status error paths
# ---------------------------------------------------------------------------


class TestFetchOrderStatusErrors:
    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        api = _make_api()
        api.exchange.fetch_order.side_effect = asyncio.CancelledError()
        with pytest.raises(asyncio.CancelledError):
            await api.fetch_order_status("123", "BTC/USDT")

    @pytest.mark.asyncio
    async def test_auth_error_raises(self):
        api = _make_api()
        api.exchange.fetch_order.side_effect = ccxt.AuthenticationError("bad")
        with pytest.raises(ccxt.AuthenticationError):
            await api.fetch_order_status("123", "BTC/USDT")

    @pytest.mark.asyncio
    async def test_generic_error_returns_none(self):
        api = _make_api()
        api.exchange.fetch_order.side_effect = RuntimeError("crash")
        result = await api.fetch_order_status("123", "BTC/USDT")
        assert result is None


# ---------------------------------------------------------------------------
# cancel_order: CancelledError
# ---------------------------------------------------------------------------


class TestCancelOrderCancelled:
    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        api = _make_api()
        api.exchange.cancel_order.side_effect = asyncio.CancelledError()
        with pytest.raises(asyncio.CancelledError):
            await api.cancel_order("123", "BTC/USDT")


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


class TestClose:
    @pytest.mark.asyncio
    async def test_close_calls_exchange_close(self):
        api = _make_api()
        await api.close()
        api.exchange.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_with_no_exchange_is_noop(self):
        api = _make_api()
        api.exchange = None
        await api.close()  # не должно бросать исключение
