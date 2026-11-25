"""Paper-биржа: те же торговые вызовы, что у Bybit; исполнение — симуляция по реальной цене."""

from unittest.mock import AsyncMock, patch

import pytest

from bitbot.adapters.exchange.paper import PaperExchange, Realism
from bitbot.domain.order_states import Placed, Rejected
from tests.exchange.test_bybit_api_errors import _make_api


def _paper(price=100.0, **realism):
    return PaperExchange(AsyncMock(return_value=price), Realism(**realism))


async def test_market_order_fills_at_current_price_slippage_against_us():
    buy = await _paper(slippage_pct=0.001).create_order(
        "BTC/USDT", "market", "buy", 0.5
    )
    sell = await _paper(slippage_pct=0.001).create_order(
        "BTC/USDT", "market", "sell", 0.5
    )
    assert isinstance(buy, Placed) and isinstance(sell, Placed)
    assert buy.order["average"] == pytest.approx(100.1)  # купить — дороже
    assert sell.order["average"] == pytest.approx(99.9)  # продать — дешевле
    assert buy.order["filled"] == 0.5 and buy.order["fee"]["cost"] == 0.0
    flat = await _paper().create_order("BTC/USDT", "market", "buy", 1.0)
    assert flat.order["average"] == 100.0  # ручки по умолчанию: как раньше


async def test_no_price_means_rejected_not_a_fake_fill():
    out = await _paper(price=None).create_order("BTC/USDT", "market", "buy", 1.0)
    assert isinstance(out, Rejected)


async def test_stops_live_in_the_book_until_cancelled():
    ex = _paper()
    sl, tp = await ex.place_exchange_sl_tp("BTC/USDT", "sell", 0.5, 95.0, 110.0)
    assert sl and tp and set(ex.stops) == {sl, tp}
    assert ex.stops[sl]["kind"] == "sl" and ex.stops[tp]["price"] == 110.0
    assert await ex.cancel_order(sl, "BTC/USDT") is True
    assert await ex.cancel_order(sl, "BTC/USDT") is False  # уже снят
    assert set(ex.stops) == {tp}
    no_tp = await ex.place_exchange_sl_tp("BTC/USDT", "sell", 0.5, 95.0, 0.0)
    assert no_tp[1] is None


async def test_bybit_api_in_paper_never_touches_the_real_exchange():
    api = _make_api()
    api.paper = _paper()
    api._leverage_cache = {}
    api._current_market_type = lambda: "linear"
    with patch("bitbot.adapters.exchange.bybit.Config.PAPER_TRADING", True):
        assert isinstance(
            await api.create_order("BTC/USDT", "market", "buy", 1.0), Placed
        )
        sl, tp = await api.place_exchange_sl_tp("BTC/USDT", "sell", 1.0, 95.0, 110.0)
        assert await api.cancel_order(sl, "BTC/USDT") is True
        assert await api.set_leverage("BTC/USDT", 5) is True
    api.exchange.create_order.assert_not_called()
    api.exchange.cancel_order.assert_not_called()
    api.exchange.set_leverage.assert_not_called()
    api.redis.acquire_lock.assert_not_called()
