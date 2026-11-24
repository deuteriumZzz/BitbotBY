"""
Integration tests for BybitAPI against Bybit testnet.

Run with:
    BYBIT_TESTNET_API_KEY=xxx BYBIT_TESTNET_API_SECRET=yyy \
    pytest tests/integration/ -m integration -v
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

SYMBOL = "BTC/USDT"


class TestConnection:
    async def test_exchange_initialized(self, bybit_api):
        assert bybit_api.exchange is not None

    async def test_markets_loaded(self, bybit_api):
        assert bybit_api.exchange.markets
        assert SYMBOL in bybit_api.exchange.markets

    async def test_symbol_has_limits(self, bybit_api):
        market = bybit_api.exchange.markets[SYMBOL]
        assert market.get("limits") is not None


class TestPublicData:
    async def test_ohlcv_returns_dataframe(self, bybit_api):
        import pandas as pd

        df = await bybit_api.get_ohlcv(SYMBOL, timeframe="1h", limit=50)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)

    async def test_ohlcv_values_are_positive(self, bybit_api):
        df = await bybit_api.get_ohlcv(SYMBOL, timeframe="1h", limit=10)
        assert (df["close"] > 0).all()
        assert (df["volume"] >= 0).all()
        assert (df["high"] >= df["low"]).all()

    async def test_current_price_is_positive(self, bybit_api):
        price = await bybit_api.get_current_price(SYMBOL)
        assert price is not None
        assert price > 0


class TestPrivateData:
    async def test_balance_has_expected_keys(self, bybit_api):
        balance = await bybit_api.get_balance()
        assert balance is not None
        assert "total" in balance
        assert "free" in balance
        assert "used" in balance

    async def test_fetch_positions_returns_list(self, bybit_api):
        positions = await bybit_api.fetch_positions()
        assert isinstance(positions, list)
        # Testnet may have no open positions -- that is fine


class TestOrderLifecycle:
    async def test_create_limit_order_and_cancel(self, bybit_api):
        price = await bybit_api.get_current_price(SYMBOL)
        assert price is not None

        # Limit buy at 50% below market -- will never fill
        limit_price = round(price * 0.5, 2)

        market = bybit_api.exchange.markets[SYMBOL]
        min_qty = float(
            (market.get("limits") or {}).get("amount", {}).get("min") or 0.001
        )

        order = await bybit_api.create_order(
            symbol=SYMBOL,
            order_type="limit",
            side="buy",
            amount=min_qty,
            price=limit_price,
        )
        assert order is not None, "Order creation returned None"
        order_id = order.get("id")
        assert order_id, "Order has no id"

        cancelled = await bybit_api.cancel_order(order_id, SYMBOL)
        assert cancelled is True

    async def test_fetch_order_status_after_place(self, bybit_api):
        price = await bybit_api.get_current_price(SYMBOL)
        assert price is not None
        limit_price = round(price * 0.5, 2)

        market = bybit_api.exchange.markets[SYMBOL]
        min_qty = float(
            (market.get("limits") or {}).get("amount", {}).get("min") or 0.001
        )

        order = await bybit_api.create_order(
            symbol=SYMBOL,
            order_type="limit",
            side="buy",
            amount=min_qty,
            price=limit_price,
        )
        assert order is not None
        order_id = order["id"]

        status = await bybit_api.fetch_order_status(order_id, SYMBOL)
        assert status is not None
        assert status.get("status") in ("open", "new", "untriggered", "canceled")

        await bybit_api.cancel_order(order_id, SYMBOL)


class TestUtils:
    async def test_round_quantity_returns_float(self, bybit_api):
        result = bybit_api.round_quantity(SYMBOL, 0.123456789)
        assert isinstance(result, float)
        assert result >= 0

    async def test_round_quantity_below_minimum_returns_zero(self, bybit_api):
        result = bybit_api.round_quantity(SYMBOL, 1e-12)
        assert result == 0.0

    async def test_round_quantity_unknown_symbol_passthrough(self, bybit_api):
        result = bybit_api.round_quantity("FAKE/USDT", 1.5)
        assert result == 1.5
