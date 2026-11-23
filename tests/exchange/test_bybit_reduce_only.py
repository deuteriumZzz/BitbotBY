"""Регрессия C-9: закрывающие ордера на фьючерсах должны идти с reduceOnly.

Без него рыночное закрытие, пришедшее после срабатывания биржевого SL/TP,
открыло бы встречную позицию.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.exchange.bybit_api import BybitAPI


def _api(market_type: str) -> BybitAPI:
    api = BybitAPI.__new__(BybitAPI)
    api.logger = logging.getLogger("test")
    api.redis = MagicMock()
    api.redis.acquire_lock.return_value = True
    api._runtime_config = MagicMock()
    api._runtime_config.get_market_type.return_value = market_type
    api.exchange = AsyncMock()
    api.exchange.create_order.return_value = {"id": "1", "filled": 0.1}
    return api


async def _params(api: BybitAPI, **kw) -> dict:
    await api.create_order("BTC/USDT", "market", "sell", 0.1, **kw)
    return api.exchange.create_order.await_args.kwargs["params"]


@pytest.mark.asyncio
async def test_reduce_only_set_on_futures() -> None:
    params = await _params(_api("linear"), reduce_only=True)
    assert params["reduceOnly"] is True
    assert params["clientOrderId"].startswith("bb-")


@pytest.mark.asyncio
async def test_reduce_only_not_sent_on_spot() -> None:
    params = await _params(_api("spot"), reduce_only=True)
    assert "reduceOnly" not in params


@pytest.mark.asyncio
async def test_opening_order_has_no_reduce_only() -> None:
    params = await _params(_api("linear"))
    assert "reduceOnly" not in params
