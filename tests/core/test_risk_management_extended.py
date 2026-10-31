"""
Расширенные тесты src/core/risk_management.py.

Покрывают строки, не охваченные test_risk_manager.py:
- calculate_kelly_size (граничные случаи)
- check_daily_loss_limit (граничные случаи)
- apply_risk_cap через kelly
"""

from __future__ import annotations

import pytest

from src.core.risk_management import RiskManager

# mock_redis автоматически применяется через conftest autouse


@pytest.fixture(autouse=True)
def mock_redis(monkeypatch):
    """Патч RedisClient чтобы не требовался живой Redis."""
    monkeypatch.setattr(
        "src.infra.redis_client.RedisClient.__init__", lambda self: None
    )
    monkeypatch.setattr(
        "src.infra.redis_client.RedisClient.save_trading_state", lambda *a, **kw: None
    )


# ---------------------------------------------------------------------------
# calculate_kelly_size
# ---------------------------------------------------------------------------


class TestCalculateKellySize:
    """Тесты критерия Келли (Half-Kelly)."""

    def test_kelly_typical_values(self):
        """Обычный случай: win_rate=0.6, reward/risk=2 → положительный размер."""
        rm = RiskManager(10000.0, 0.02)
        size = rm.calculate_kelly_size(
            entry_price=100.0,
            stop_loss=98.0,
            take_profit=104.0,
            win_rate=0.6,
            current_balance=10000.0,
        )
        assert size > 0

    def test_kelly_zero_when_win_rate_too_low(self):
        """Отрицательный Kelly → размер = 0."""
        rm = RiskManager(10000.0, 0.02)
        # win_rate=0.1, reward/risk=1 → Kelly=(0.1*1-0.9)/1 = -0.8 < 0
        size = rm.calculate_kelly_size(
            entry_price=100.0,
            stop_loss=99.0,
            take_profit=101.0,
            win_rate=0.1,
            current_balance=10000.0,
        )
        assert size == 0.0

    def test_kelly_zero_when_entry_equals_stop_loss(self):
        """entry==stop_loss → risk=0 → размер = 0."""
        rm = RiskManager(10000.0)
        size = rm.calculate_kelly_size(
            entry_price=100.0,
            stop_loss=100.0,
            take_profit=105.0,
            win_rate=0.6,
            current_balance=10000.0,
        )
        assert size == 0.0

    def test_kelly_zero_when_entry_price_zero(self):
        """entry_price=0 → размер = 0."""
        rm = RiskManager(10000.0)
        size = rm.calculate_kelly_size(
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            win_rate=0.6,
            current_balance=10000.0,
        )
        assert size == 0.0

    def test_kelly_capped_at_20_percent_of_balance(self):
        """Half-Kelly не может превысить 20% баланса."""
        rm = RiskManager(10000.0)
        # win_rate=0.99, reward/risk=10 → очень большой Kelly
        size = rm.calculate_kelly_size(
            entry_price=100.0,
            stop_loss=90.0,
            take_profit=200.0,
            win_rate=0.99,
            current_balance=10000.0,
        )
        max_position_value = 10000.0 * 0.20
        assert size * 100.0 <= max_position_value + 1e-9

    def test_kelly_non_negative_always(self):
        """Размер позиции никогда не отрицательный."""
        rm = RiskManager(10000.0)
        for win_rate in [0.0, 0.01, 0.5, 0.99, 1.0]:
            size = rm.calculate_kelly_size(
                entry_price=100.0,
                stop_loss=95.0,
                take_profit=115.0,
                win_rate=win_rate,
                current_balance=10000.0,
            )
            assert size >= 0.0, f"negative size for win_rate={win_rate}"

    def test_kelly_with_win_rate_1(self):
        """win_rate=1.0 даёт позитивный размер, ограниченный кэпом."""
        rm = RiskManager(10000.0)
        size = rm.calculate_kelly_size(
            entry_price=100.0,
            stop_loss=90.0,
            take_profit=120.0,
            win_rate=1.0,
            current_balance=10000.0,
        )
        assert size > 0
        assert size * 100.0 <= 10000.0 * 0.20 + 1e-9

    def test_kelly_larger_balance_gives_larger_size(self):
        """Больший баланс → пропорционально больший размер позиции."""
        rm = RiskManager(10000.0)
        size_small = rm.calculate_kelly_size(100.0, 95.0, 115.0, 0.6, 1000.0)
        size_large = rm.calculate_kelly_size(100.0, 95.0, 115.0, 0.6, 10000.0)
        assert size_large > size_small


# ---------------------------------------------------------------------------
# check_daily_loss_limit (дополнительные случаи)
# ---------------------------------------------------------------------------


class TestCheckDailyLossLimitExtended:
    """Дополнительные граничные случаи для check_daily_loss_limit."""

    def test_no_loss_allows_trading(self):
        """Баланс равен начальному → торговля разрешена."""
        rm = RiskManager(10000.0)
        assert rm.check_daily_loss_limit(10000.0) is True

    def test_profit_allows_trading(self):
        """Баланс выше начального (прибыль) → торговля разрешена."""
        rm = RiskManager(10000.0)
        assert rm.check_daily_loss_limit(11000.0) is True

    def test_just_below_limit_allows_trading(self):
        """Убыток чуть ниже лимита → торговля разрешена."""
        rm = RiskManager(10000.0)
        # DAILY_LOSS_LIMIT = 0.05 → limit = 500
        # loss = 499 → ok
        assert rm.check_daily_loss_limit(9501.0) is True

    def test_custom_risk_per_trade(self):
        """Нестандартный risk_per_trade не влияет на дневной лимит."""
        rm = RiskManager(initial_balance=5000.0, risk_per_trade=0.01)
        # 5% от 5000 = 250; потеряли 300 → должно быть False
        assert rm.check_daily_loss_limit(4700.0) is False
