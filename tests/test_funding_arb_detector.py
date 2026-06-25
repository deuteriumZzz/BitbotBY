"""Тесты FundingArbDetector."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.funding_arb_detector import (
    FundingArbDetector,
    _FUNDING_ALERT_COOLDOWN,
    _FUNDING_EXTREME,
    _FUNDING_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_BIG_NOW = 100_000.0  # больше 8 часов в секундах — cooldown точно прошёл


def _make_detector() -> FundingArbDetector:
    return FundingArbDetector()


def _make_telegram() -> MagicMock:
    tg = MagicMock()
    tg.notify = AsyncMock()
    return tg


@pytest.fixture(autouse=True)
def mock_monotonic():
    """Мокаем time.monotonic чтобы cooldown не срабатывал на чистой системе."""
    with patch("src.funding_arb_detector.time.monotonic", return_value=_BIG_NOW):
        yield


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_last_alert_at_empty_on_init(self):
        det = _make_detector()
        assert det._last_alert_at == {}


# ---------------------------------------------------------------------------
# check_and_alert() — below threshold
# ---------------------------------------------------------------------------


class TestBelowThreshold:
    @pytest.mark.asyncio
    async def test_no_alert_when_rate_below_threshold(self):
        det = _make_detector()
        tg = _make_telegram()
        await det.check_and_alert("BTC/USDT", _FUNDING_THRESHOLD - 0.0001, tg)
        tg.notify.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_alert_when_rate_is_zero(self):
        det = _make_detector()
        tg = _make_telegram()
        await det.check_and_alert("ETH/USDT", 0.0, tg)
        tg.notify.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_alert_when_negative_rate_below_threshold(self):
        """Отрицательный rate, но abs() ниже порога."""
        det = _make_detector()
        tg = _make_telegram()
        await det.check_and_alert("SOL/USDT", -(_FUNDING_THRESHOLD - 0.0001), tg)
        tg.notify.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_alert_at_exactly_threshold_boundary(self):
        """abs(rate) == threshold — строго меньше не проходит."""
        det = _make_detector()
        tg = _make_telegram()
        # Граничное значение: < threshold → не алертим
        await det.check_and_alert("BTC/USDT", _FUNDING_THRESHOLD - 1e-10, tg)
        tg.notify.assert_not_called()


# ---------------------------------------------------------------------------
# check_and_alert() — above threshold, first alert
# ---------------------------------------------------------------------------


class TestAboveThreshold:
    @pytest.mark.asyncio
    async def test_sends_alert_for_positive_rate_above_threshold(self):
        det = _make_detector()
        tg = _make_telegram()
        await det.check_and_alert("BTC/USDT", _FUNDING_THRESHOLD + 0.0001, tg)
        tg.notify.assert_called_once()

    @pytest.mark.asyncio
    async def test_sends_alert_for_negative_rate_above_threshold(self):
        det = _make_detector()
        tg = _make_telegram()
        await det.check_and_alert("ETH/USDT", -(_FUNDING_THRESHOLD + 0.0001), tg)
        tg.notify.assert_called_once()

    @pytest.mark.asyncio
    async def test_message_contains_symbol(self):
        det = _make_detector()
        tg = _make_telegram()
        await det.check_and_alert("SOL/USDT", 0.001, tg)
        msg = tg.notify.call_args[0][0]
        assert "SOL/USDT" in msg

    @pytest.mark.asyncio
    async def test_message_contains_funding_arb_header(self):
        det = _make_detector()
        tg = _make_telegram()
        await det.check_and_alert("BTC/USDT", 0.001, tg)
        msg = tg.notify.call_args[0][0]
        assert "FUNDING ARB" in msg

    @pytest.mark.asyncio
    async def test_message_contains_apy(self):
        det = _make_detector()
        tg = _make_telegram()
        await det.check_and_alert("BTC/USDT", 0.001, tg)
        msg = tg.notify.call_args[0][0]
        assert "APY" in msg

    @pytest.mark.asyncio
    async def test_message_level_high_for_threshold_rate(self):
        """Rate выше threshold, но ниже extreme → уровень HIGH."""
        det = _make_detector()
        tg = _make_telegram()
        rate = (_FUNDING_THRESHOLD + _FUNDING_EXTREME) / 2  # между ними
        await det.check_and_alert("BTC/USDT", rate, tg)
        msg = tg.notify.call_args[0][0]
        assert "HIGH" in msg

    @pytest.mark.asyncio
    async def test_message_level_extreme_for_extreme_rate(self):
        """Rate >= extreme → уровень EXTREME."""
        det = _make_detector()
        tg = _make_telegram()
        await det.check_and_alert("BTC/USDT", _FUNDING_EXTREME, tg)
        msg = tg.notify.call_args[0][0]
        assert "EXTREME" in msg

    @pytest.mark.asyncio
    async def test_positive_rate_direction_long_spot_short_perp(self):
        """Положительный rate → longs pay shorts → Long SPOT + Short PERP."""
        det = _make_detector()
        tg = _make_telegram()
        await det.check_and_alert("BTC/USDT", 0.001, tg)
        msg = tg.notify.call_args[0][0]
        assert "Short PERP" in msg
        assert "longs pay shorts" in msg

    @pytest.mark.asyncio
    async def test_negative_rate_direction_short_spot_long_perp(self):
        """Отрицательный rate → shorts pay longs → Short SPOT + Long PERP."""
        det = _make_detector()
        tg = _make_telegram()
        await det.check_and_alert("ETH/USDT", -0.001, tg)
        msg = tg.notify.call_args[0][0]
        assert "Long PERP" in msg
        assert "shorts pay longs" in msg

    @pytest.mark.asyncio
    async def test_base_asset_extracted_correctly(self):
        """Базовый актив извлекается из symbol до '/'."""
        det = _make_detector()
        tg = _make_telegram()
        await det.check_and_alert("SOL/USDT", 0.001, tg)
        msg = tg.notify.call_args[0][0]
        assert "SOL" in msg
        # Убедимся что /USDT не попал в стратегию
        strategy_line = [ln for ln in msg.split("\n") if "Strategy" in ln][0]
        assert "USDT" not in strategy_line

    @pytest.mark.asyncio
    async def test_message_contains_delta_neutral(self):
        det = _make_detector()
        tg = _make_telegram()
        await det.check_and_alert("BTC/USDT", 0.001, tg)
        msg = tg.notify.call_args[0][0]
        assert "delta neutral" in msg.lower()

    @pytest.mark.asyncio
    async def test_stores_last_alert_timestamp(self):
        det = _make_detector()
        tg = _make_telegram()
        await det.check_and_alert("BTC/USDT", 0.001, tg)
        assert "BTC/USDT" in det._last_alert_at
        assert det._last_alert_at["BTC/USDT"] == _BIG_NOW


# ---------------------------------------------------------------------------
# check_and_alert() — cooldown
# ---------------------------------------------------------------------------


class TestCooldown:
    @pytest.mark.asyncio
    async def test_no_second_alert_within_cooldown(self):
        det = _make_detector()
        tg = _make_telegram()
        # Первый алерт
        await det.check_and_alert("BTC/USDT", 0.001, tg)
        assert tg.notify.call_count == 1

        # Второй вызов сразу после — cooldown ещё не прошёл
        await det.check_and_alert("BTC/USDT", 0.001, tg)
        assert tg.notify.call_count == 1  # не увеличился

    @pytest.mark.asyncio
    async def test_different_symbols_get_independent_cooldowns(self):
        det = _make_detector()
        tg = _make_telegram()
        await det.check_and_alert("BTC/USDT", 0.001, tg)
        await det.check_and_alert("ETH/USDT", 0.001, tg)
        # Оба должны сработать
        assert tg.notify.call_count == 2

    @pytest.mark.asyncio
    async def test_alert_fires_again_after_cooldown_expires(self):
        det = _make_detector()
        tg = _make_telegram()

        # Устанавливаем время последнего алерта в прошлое — за пределами cooldown
        past_time = _BIG_NOW - _FUNDING_ALERT_COOLDOWN - 1
        det._last_alert_at["BTC/USDT"] = past_time

        await det.check_and_alert("BTC/USDT", 0.001, tg)
        tg.notify.assert_called_once()

    @pytest.mark.asyncio
    async def test_alert_suppressed_just_before_cooldown_ends(self):
        det = _make_detector()
        tg = _make_telegram()

        # 1 секунда до окончания cooldown
        recent_time = _BIG_NOW - _FUNDING_ALERT_COOLDOWN + 1
        det._last_alert_at["ETH/USDT"] = recent_time

        await det.check_and_alert("ETH/USDT", 0.001, tg)
        tg.notify.assert_not_called()


# ---------------------------------------------------------------------------
# check_and_alert() — telegram failure
# ---------------------------------------------------------------------------


class TestTelegramFailure:
    @pytest.mark.asyncio
    async def test_no_crash_when_telegram_raises(self):
        """Ошибка telegram.notify() не должна пробрасываться."""
        det = _make_detector()
        tg = MagicMock()
        tg.notify = AsyncMock(side_effect=Exception("Telegram timeout"))

        # Не должно выбросить исключение
        await det.check_and_alert("BTC/USDT", 0.001, tg)

    @pytest.mark.asyncio
    async def test_last_alert_at_still_updated_on_telegram_failure(self):
        """Даже если notify упал — timestamp обновлён (cooldown соблюдён)."""
        det = _make_detector()
        tg = MagicMock()
        tg.notify = AsyncMock(side_effect=Exception("Telegram timeout"))

        await det.check_and_alert("BTC/USDT", 0.001, tg)
        assert "BTC/USDT" in det._last_alert_at


# ---------------------------------------------------------------------------
# APY calculation
# ---------------------------------------------------------------------------


class TestApyCalculation:
    @pytest.mark.asyncio
    async def test_apy_calculation_correct(self):
        """APY = abs(rate) * 3 * 365 * 100."""
        det = _make_detector()
        tg = _make_telegram()
        rate = 0.001  # 0.1% per 8h
        expected_apy = rate * 3 * 365 * 100  # = 109.5%

        await det.check_and_alert("BTC/USDT", rate, tg)
        msg = tg.notify.call_args[0][0]
        assert f"{expected_apy:.0f}%" in msg

    @pytest.mark.asyncio
    async def test_funding_rate_formatted_in_message(self):
        """Funding rate отображается как процент с 4 знаками."""
        det = _make_detector()
        tg = _make_telegram()
        rate = 0.001  # → "0.1000%"
        await det.check_and_alert("BTC/USDT", rate, tg)
        msg = tg.notify.call_args[0][0]
        assert "0.1000%" in msg
