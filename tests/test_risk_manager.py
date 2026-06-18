import pytest
from unittest.mock import MagicMock


# Patch Redis so RiskManager doesn't need a live server
@pytest.fixture(autouse=True)
def mock_redis(monkeypatch):
    monkeypatch.setattr(
        "src.redis_client.RedisClient.__init__",
        lambda self: None,
    )
    monkeypatch.setattr(
        "src.redis_client.RedisClient.save_trading_state",
        lambda *a, **kw: None,
    )


from src.risk_management import RiskManager


@pytest.mark.asyncio
async def test_position_size_basic():
    rm = RiskManager(
        initial_balance=10000.0, risk_per_trade=0.02
    )
    # risk = 200 USDT, price diff = 100
    size = await rm.calculate_position_size(
        current_balance=10000.0,
        entry_price=1100.0,
        stop_loss=1000.0,
    )
    assert abs(size - 2.0) < 0.01  # 200/100 = 2.0


@pytest.mark.asyncio
async def test_position_size_zero_diff():
    rm = RiskManager(10000.0)
    size = await rm.calculate_position_size(
        10000.0, entry_price=100.0, stop_loss=100.0
    )
    assert size == 0


@pytest.mark.asyncio
async def test_stop_loss_buy_atr():
    rm = RiskManager(10000.0)
    sl = await rm.calculate_stop_loss(
        entry_price=1000.0,
        signal={"action": "buy", "atr": 20.0},
    )
    assert sl == pytest.approx(1000.0 - 1.5 * 20.0)


@pytest.mark.asyncio
async def test_stop_loss_sell_atr():
    rm = RiskManager(10000.0)
    sl = await rm.calculate_stop_loss(
        entry_price=1000.0,
        signal={"action": "sell", "atr": 20.0},
    )
    assert sl == pytest.approx(1000.0 + 1.5 * 20.0)


@pytest.mark.asyncio
async def test_stop_loss_fallback_no_atr():
    rm = RiskManager(10000.0)
    sl = await rm.calculate_stop_loss(
        entry_price=1000.0,
        signal={"action": "buy", "atr": 0.0},
    )
    # fallback: entry * (1 - STOP_LOSS_PERCENT)
    assert sl < 1000.0


def test_daily_loss_limit_ok():
    rm = RiskManager(10000.0)
    # Lost 3% — limit is 5% — should be OK
    assert rm.check_daily_loss_limit(9700.0) is True


def test_daily_loss_limit_breached():
    rm = RiskManager(10000.0)
    # Lost 6% — over limit
    assert rm.check_daily_loss_limit(9400.0) is False


def test_daily_loss_limit_exact():
    rm = RiskManager(10000.0)
    # Lost exactly 5% — breached (>= limit)
    assert rm.check_daily_loss_limit(9500.0) is False
