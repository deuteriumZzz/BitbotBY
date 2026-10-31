import pytest


# Patch Redis so RiskManager doesn't need a live server
@pytest.fixture(autouse=True)
def mock_redis(monkeypatch):
    monkeypatch.setattr(
        "src.infra.redis_client.RedisClient.__init__",
        lambda self: None,
    )
    monkeypatch.setattr(
        "src.infra.redis_client.RedisClient.save_trading_state",
        lambda *a, **kw: None,
    )


from src.core.risk_management import RiskManager


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
