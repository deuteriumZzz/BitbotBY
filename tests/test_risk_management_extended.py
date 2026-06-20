"""
Расширенные тесты src/risk_management.py.

Покрывают строки, не охваченные test_risk_manager.py:
- calculate_kelly_size (граничные случаи)
- calculate_stop_loss (hold-action, fallback sell)
- calculate_take_profit (все ветки)
- validate_signal (все ветки)
- apply_risk_cap через kelly
"""

from __future__ import annotations

import pytest

# mock_redis автоматически применяется через conftest autouse


@pytest.fixture(autouse=True)
def mock_redis(monkeypatch):
    """Патч RedisClient чтобы не требовался живой Redis."""
    monkeypatch.setattr("src.redis_client.RedisClient.__init__", lambda self: None)
    monkeypatch.setattr(
        "src.redis_client.RedisClient.save_trading_state", lambda *a, **kw: None
    )


from src.risk_management import RiskManager


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
# calculate_stop_loss
# ---------------------------------------------------------------------------

class TestCalculateStopLossExtended:
    """Дополнительные тесты для calculate_stop_loss (непокрытые ветки)."""

    @pytest.mark.asyncio
    async def test_stop_loss_hold_with_atr(self):
        """action=hold с ATR → возвращается entry_price (без изменений)."""
        rm = RiskManager(10000.0)
        sl = await rm.calculate_stop_loss(100.0, {"action": "hold", "atr": 5.0})
        assert sl == 100.0

    @pytest.mark.asyncio
    async def test_stop_loss_sell_fallback_no_atr(self):
        """action=sell без ATR → entry * (1 + STOP_LOSS_PERCENT)."""
        rm = RiskManager(10000.0)
        sl = await rm.calculate_stop_loss(100.0, {"action": "sell", "atr": 0.0})
        assert sl > 100.0  # для sell стоп выше цены входа

    @pytest.mark.asyncio
    async def test_stop_loss_hold_no_atr(self):
        """action=hold без ATR → возвращается entry_price."""
        rm = RiskManager(10000.0)
        sl = await rm.calculate_stop_loss(100.0, {"action": "hold"})
        assert sl == 100.0


# ---------------------------------------------------------------------------
# calculate_take_profit
# ---------------------------------------------------------------------------

class TestCalculateTakeProfitExtended:
    """Тесты calculate_take_profit для всех веток."""

    @pytest.mark.asyncio
    async def test_take_profit_buy(self):
        """Buy TP = entry + 2 * risk."""
        rm = RiskManager(10000.0)
        tp = await rm.calculate_take_profit(100.0, {"action": "buy", "atr": 2.0})
        sl = 100.0 - 1.5 * 2.0  # = 97.0
        risk = 100.0 - sl         # = 3.0
        expected = 100.0 + risk * 2  # = 106.0
        assert tp == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_take_profit_sell(self):
        """Sell TP = entry - 2 * risk."""
        rm = RiskManager(10000.0)
        tp = await rm.calculate_take_profit(100.0, {"action": "sell", "atr": 2.0})
        sl = 100.0 + 1.5 * 2.0  # = 103.0
        risk = sl - 100.0         # = 3.0
        expected = 100.0 - risk * 2  # = 94.0
        assert tp == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_take_profit_hold(self):
        """Hold → TP = entry (без изменений)."""
        rm = RiskManager(10000.0)
        tp = await rm.calculate_take_profit(100.0, {"action": "hold"})
        assert tp == 100.0


# ---------------------------------------------------------------------------
# validate_signal
# ---------------------------------------------------------------------------

class TestValidateSignalExtended:
    """Расширенные тесты validate_signal."""

    @pytest.mark.asyncio
    async def test_validate_signal_buy_high_confidence(self):
        """Buy с высокой уверенностью — проходит валидацию."""
        rm = RiskManager(10000.0)
        result = await rm.validate_signal({"action": "buy", "confidence": 0.9}, None)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_signal_sell_high_confidence(self):
        """Sell с высокой уверенностью — проходит валидацию."""
        rm = RiskManager(10000.0)
        result = await rm.validate_signal({"action": "sell", "confidence": 0.9}, None)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_signal_hold_always_false(self):
        """Hold никогда не проходит валидацию."""
        rm = RiskManager(10000.0)
        result = await rm.validate_signal({"action": "hold", "confidence": 0.95}, None)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_signal_low_confidence(self):
        """Сигнал с уверенностью ниже MIN_SIGNAL_CONFIDENCE отклоняется."""
        rm = RiskManager(10000.0)
        result = await rm.validate_signal({"action": "buy", "confidence": 0.01}, None)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_signal_zero_confidence(self):
        """Нулевая уверенность — отклоняется."""
        rm = RiskManager(10000.0)
        result = await rm.validate_signal({"action": "buy", "confidence": 0.0}, None)
        assert result is False


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


# ---------------------------------------------------------------------------
# calculate_position_size (дополнительные случаи)
# ---------------------------------------------------------------------------

class TestCalculatePositionSizeExtended:
    """Дополнительные тесты calculate_position_size."""

    @pytest.mark.asyncio
    async def test_position_size_scales_with_risk_per_trade(self):
        """Удвоение risk_per_trade удваивает размер позиции."""
        rm1 = RiskManager(10000.0, risk_per_trade=0.01)
        rm2 = RiskManager(10000.0, risk_per_trade=0.02)
        size1 = await rm1.calculate_position_size(10000.0, 100.0, 90.0)
        size2 = await rm2.calculate_position_size(10000.0, 100.0, 90.0)
        assert size2 == pytest.approx(size1 * 2)

    @pytest.mark.asyncio
    async def test_position_size_larger_when_tight_stop(self):
        """Более узкий стоп → больший размер позиции."""
        rm = RiskManager(10000.0, risk_per_trade=0.02)
        size_tight = await rm.calculate_position_size(10000.0, 100.0, 99.0)  # 1 diff
        size_wide = await rm.calculate_position_size(10000.0, 100.0, 90.0)   # 10 diff
        assert size_tight > size_wide
