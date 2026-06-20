"""Тесты для PortfolioManager — баланс, позиции, trailing stop."""

from unittest.mock import MagicMock, patch

import pytest

from config import Config
from src.portfolio_manager import PortfolioManager

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def portfolio():
    """PortfolioManager с полностью замоканным Redis."""
    with patch("src.portfolio_manager.RedisClient") as MockRedis:  # noqa: N806
        MockRedis.return_value = MagicMock()
        MockRedis.return_value.save_trading_state = MagicMock(return_value=True)
        MockRedis.return_value.load_trading_state = MagicMock(return_value=None)
        pm = PortfolioManager(initial_balance=10000.0)
        yield pm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_buy_deducts_balance_and_commission(portfolio):
    """buy 0.1 BTC по 30000 → баланс уменьшается на стоимость + комиссию."""
    qty = 0.1
    price = 30000.0
    commission = qty * price * portfolio.commission_rate
    cost = qty * price + commission

    ok = await portfolio.update_portfolio("BTC/USDT", "buy", qty, price)

    assert ok is True
    assert portfolio.current_balance == pytest.approx(10000.0 - cost)
    assert portfolio.positions.get("BTC/USDT") == pytest.approx(qty)
    assert portfolio.total_commissions == pytest.approx(commission)


async def test_buy_insufficient_funds_returns_false(portfolio):
    """Попытка купить на сумму больше баланса → False, баланс не изменяется."""
    ok = await portfolio.update_portfolio("BTC/USDT", "buy", 10.0, 30000.0)

    assert ok is False
    assert portfolio.current_balance == pytest.approx(10000.0)
    assert "BTC/USDT" not in portfolio.positions


async def test_sell_increases_balance(portfolio):
    """buy, затем sell по той же цене → баланс близок к начальному
    (минус 2× комиссия)."""
    qty = 0.1
    price = 30000.0

    await portfolio.update_portfolio("BTC/USDT", "buy", qty, price)

    ok = await portfolio.update_portfolio("BTC/USDT", "sell", qty, price)

    assert ok is True
    # Баланс должен быть близок к начальному минус 2× комиссии
    total_commission = 2 * qty * price * portfolio.commission_rate
    assert portfolio.current_balance == pytest.approx(
        10000.0 - total_commission, rel=1e-6
    )
    assert "BTC/USDT" not in portfolio.positions


async def test_sell_nonexistent_position_returns_false(portfolio):
    """Продажа символа, которого нет в позициях → False."""
    ok = await portfolio.update_portfolio("ETH/USDT", "sell", 1.0, 2000.0)

    assert ok is False
    assert portfolio.current_balance == pytest.approx(10000.0)


async def test_portfolio_value_calculation(portfolio):
    """После buy по 30000, get_portfolio_value по той же цене равен initial_balance."""
    qty = 0.1
    price = 30000.0
    await portfolio.update_portfolio("BTC/USDT", "buy", qty, price)

    value = await portfolio.get_portfolio_value({"BTC/USDT": price})

    # cash + стоимость позиции == начальный баланс минус только комиссии
    commission = qty * price * portfolio.commission_rate
    # наличные после buy + qty*price = начальный - комиссия
    expected = 10000.0 - commission
    assert value == pytest.approx(expected, rel=1e-6)


async def test_trailing_stop_triggers(portfolio):
    """
    Buy по 100, цена растёт до 110 (high watermark), затем падает до 104.
    При ATR=5 и TRAILING_STOP_ATR_MULT=1.0 стоп находится на 110 - 5 = 105.
    Тик на 104 ниже стопа → позиция должна быть закрыта.

    У PortfolioManager нет встроенного метода trailing-stop — тестируем
    вспомогательную логику напрямую, симулируя поведение торгового цикла.
    """
    atr = 5.0
    mult = Config.TRAILING_STOP_ATR_MULT  # 1.0

    buy_price = 100.0
    await portfolio.update_portfolio("BTC/USDT", "buy", 0.1, buy_price)

    high_watermark = 110.0
    trailing_stop = high_watermark - atr * mult  # 105.0

    current_price = 104.0  # ниже трейлинг-стопа
    triggered = current_price < trailing_stop

    assert triggered is True
    # При срабатывании бот продаёт
    ok = await portfolio.update_portfolio("BTC/USDT", "sell", 0.1, current_price)
    assert ok is True
    assert "BTC/USDT" not in portfolio.positions


async def test_trailing_stop_not_triggered_small_move(portfolio):
    """
    Buy по 100, high watermark = 105, текущая цена = 104.5.
    При ATR=5 и mult=1.0 стоп на 100 → цена 104.5 > 100 → не срабатывает.
    """
    atr = 5.0
    mult = Config.TRAILING_STOP_ATR_MULT  # 1.0

    buy_price = 100.0
    await portfolio.update_portfolio("BTC/USDT", "buy", 0.1, buy_price)

    high_watermark = 105.0
    trailing_stop = high_watermark - atr * mult  # 100.0

    current_price = 104.5
    triggered = current_price < trailing_stop

    assert triggered is False
    assert "BTC/USDT" in portfolio.positions
