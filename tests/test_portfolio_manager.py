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
    """PortfolioManager with Redis fully mocked."""
    with patch("src.portfolio_manager.RedisClient") as MockRedis:
        MockRedis.return_value = MagicMock()
        MockRedis.return_value.save_trading_state = MagicMock(return_value=True)
        MockRedis.return_value.load_trading_state = MagicMock(return_value=None)
        pm = PortfolioManager(initial_balance=10000.0)
        yield pm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_buy_deducts_balance_and_commission(portfolio):
    """buy 0.1 BTC at 30000 → balance decreases by cost + commission."""
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
    """Trying to buy more than balance covers → False, balance unchanged."""
    ok = await portfolio.update_portfolio("BTC/USDT", "buy", 10.0, 30000.0)

    assert ok is False
    assert portfolio.current_balance == pytest.approx(10000.0)
    assert "BTC/USDT" not in portfolio.positions


async def test_sell_increases_balance(portfolio):
    """buy then sell at same price → balance close to initial (minus 2× commission)."""
    qty = 0.1
    price = 30000.0

    await portfolio.update_portfolio("BTC/USDT", "buy", qty, price)

    ok = await portfolio.update_portfolio("BTC/USDT", "sell", qty, price)

    assert ok is True
    # Balance should be near initial minus 2× commissions
    total_commission = 2 * qty * price * portfolio.commission_rate
    assert portfolio.current_balance == pytest.approx(
        10000.0 - total_commission, rel=1e-6
    )
    assert "BTC/USDT" not in portfolio.positions


async def test_sell_nonexistent_position_returns_false(portfolio):
    """Selling a symbol that is not in positions → False."""
    ok = await portfolio.update_portfolio("ETH/USDT", "sell", 1.0, 2000.0)

    assert ok is False
    assert portfolio.current_balance == pytest.approx(10000.0)


async def test_portfolio_value_calculation(portfolio):
    """After buy at 30000, get_portfolio_value at same price equals initial_balance."""
    qty = 0.1
    price = 30000.0
    await portfolio.update_portfolio("BTC/USDT", "buy", qty, price)

    value = await portfolio.get_portfolio_value({"BTC/USDT": price})

    # cash + position value == initial minus commissions only
    commission = qty * price * portfolio.commission_rate
    expected = 10000.0 - commission  # cash after buy + qty*price = initial - commission
    assert value == pytest.approx(expected, rel=1e-6)


async def test_trailing_stop_triggers(portfolio):
    """
    Buy at 100, price rises to 110 (high watermark), then drops to 104.
    With ATR=5 and TRAILING_STOP_ATR_MULT=1.0 the stop sits at 110 - 5 = 105.
    A tick at 104 is below the stop → position should be closed.

    PortfolioManager doesn't have a built-in trailing-stop method — we test
    the helper logic directly by simulating what the trading loop would do.
    """
    atr = 5.0
    mult = Config.TRAILING_STOP_ATR_MULT  # 1.0

    buy_price = 100.0
    await portfolio.update_portfolio("BTC/USDT", "buy", 0.1, buy_price)

    high_watermark = 110.0
    trailing_stop = high_watermark - atr * mult  # 105.0

    current_price = 104.0  # below trailing stop
    triggered = current_price < trailing_stop

    assert triggered is True
    # When triggered, bot would sell
    ok = await portfolio.update_portfolio("BTC/USDT", "sell", 0.1, current_price)
    assert ok is True
    assert "BTC/USDT" not in portfolio.positions


async def test_trailing_stop_not_triggered_small_move(portfolio):
    """
    Buy at 100, high watermark = 105, current = 104.5.
    With ATR=5 and mult=1.0 stop is at 100 → price 104.5 > 100 → no trigger.
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
