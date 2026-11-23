"""Регрессии блока 2: дневной лимит при рестарте, SAC без модели, сверка (H-7)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.risk_management import RiskManager
from src.core.signal_combiner import SignalCombiner
from tests.core.test_reconciliation import make_bot
from tests.core.test_signal_combiner import make_snapshot

# ── Дневной лимит и рестарт ───────────────────────────────────────────────────


class _FakeRaw:
    def __init__(self) -> None:
        self.data: dict = {}

    def get(self, key):
        return self.data.get(key)

    def setex(self, key, ttl, value) -> None:
        self.data[key] = value


class _FakeRedis:
    def __init__(self) -> None:
        self._available = True
        self.redis_client = _FakeRaw()


def _cfg(cfg: MagicMock) -> MagicMock:
    cfg.PAPER_TRADING = False
    cfg.MAX_POSITIONS = 3
    cfg.DRAWDOWN_SCALE_ENABLED = False
    cfg.DAILY_LOSS_LIMIT = 0.05
    return cfg


def test_day_reference_survives_restart_same_day() -> None:
    redis = _FakeRedis()
    with patch("src.core.risk_management.Config") as cfg:
        _cfg(cfg)
        first = RiskManager(10_000.0, redis_client=redis)
        first.set_day_reference(10_000.0)
        # «рестарт»: новый менеджер, а начальный баланс уже другой
        second = RiskManager(8_000.0, redis_client=redis)
        assert second.day_reference_restored is True
        # потеряно 6% от утреннего баланса 10 000 → лимит 5% превышен
        assert second.check_daily_loss_limit(9_400.0) is False


def test_stale_day_reference_is_ignored() -> None:
    redis = _FakeRedis()
    yesterday = (datetime.utcnow() - timedelta(days=1)).date().isoformat()
    redis.redis_client.data["bot:risk:daily_ref:live"] = json.dumps(
        {"date": yesterday, "balance": 1.0}
    )
    with patch("src.core.risk_management.Config") as cfg:
        _cfg(cfg)
        rm = RiskManager(10_000.0, redis_client=redis)
    assert rm.day_reference_restored is False
    assert rm._day_start_balance == 10_000.0


def test_works_without_redis() -> None:
    with patch("src.core.risk_management.Config") as cfg:
        _cfg(cfg)
        rm = RiskManager(10_000.0)
        rm.set_day_reference(9_000.0)
    assert rm._day_start_balance == 9_000.0
    assert rm.day_reference_restored is False


def test_redis_failure_does_not_break_risk_manager() -> None:
    redis = MagicMock()
    redis._available = True
    redis.redis_client.get.side_effect = RuntimeError("down")
    redis.redis_client.setex.side_effect = RuntimeError("down")
    with patch("src.core.risk_management.Config") as cfg:
        _cfg(cfg)
        rm = RiskManager(10_000.0, redis_client=redis)
        rm.set_day_reference(9_500.0)
    assert rm._day_start_balance == 9_500.0


# ── SAC-режимы без модели ─────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["hybrid", "dqn"])
async def test_sac_modes_do_not_trade_without_model(mode: str) -> None:
    ai = MagicMock()
    ai.analyze = AsyncMock(return_value=[])
    rc = MagicMock()
    rc.get_mode.return_value = mode
    rc.get_signal_confidence.return_value = 0.5
    rc.get_chronos_enabled.return_value = False
    with patch("src.core.signal_combiner.SACSignal") as sac_cls:
        sac_cls.return_value.loaded = False
        combiner = SignalCombiner(ai=ai, rc=rc)
        ctx = {"BTC/USDT": {"funding_signal": "long_overheated", "funding_rate": 0.01}}
        result = await combiner.combine(
            [make_snapshot()], balance=10_000, market_context=ctx
        )
    assert result == []
    ai.analyze.assert_not_called()  # без SAC AI тоже не тратим впустую


@pytest.mark.asyncio
async def test_ai_mode_unaffected_by_missing_sac() -> None:
    ai = MagicMock()
    ai.analyze = AsyncMock(return_value=[])
    rc = MagicMock()
    rc.get_mode.return_value = "ai"
    rc.get_signal_confidence.return_value = 0.5
    with patch("src.core.signal_combiner.SACSignal") as sac_cls:
        sac_cls.return_value.loaded = False
        combiner = SignalCombiner(ai=ai, rc=rc)
        await combiner.combine([make_snapshot()], balance=10_000)
    ai.analyze.assert_awaited_once()


# ── H-7: устаревшая позиция при сверке ────────────────────────────────────────


@pytest.mark.asyncio
async def test_reconcile_marks_trade_orphaned_for_stale_position() -> None:
    bot = make_bot()
    bot._monitored["BTC/USDT"] = {
        "trade_id": 7,
        "qty": 0.01,
        "entry": 50000.0,
        "side": "buy",
    }
    bot.api.fetch_positions = AsyncMock(return_value=[])
    bot.trade_history.mark_orphaned = AsyncMock()

    await bot._reconcile_positions()

    assert "BTC/USDT" not in bot._monitored
    bot.trade_history.mark_orphaned.assert_awaited_once_with(7)


@pytest.mark.asyncio
async def test_reconcile_survives_mark_orphaned_failure() -> None:
    bot = make_bot()
    bot._monitored["ETH/USDT"] = {"trade_id": 9, "qty": 1.0, "side": "buy"}
    bot.api.fetch_positions = AsyncMock(return_value=[])
    bot.trade_history.mark_orphaned = AsyncMock(side_effect=RuntimeError("db"))

    await bot._reconcile_positions()  # не должно падать

    assert "ETH/USDT" not in bot._monitored
