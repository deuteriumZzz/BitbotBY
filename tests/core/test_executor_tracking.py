"""Регрессии: позиция не теряется при сбое записи сделки; предупреждение об SL."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tests.core.test_order_executor_extra import (
    _BT_STATS,
    _TOP_BUY,
    _make_cfg,
    _make_executor,
)


async def _run(executor, cfg) -> None:
    with patch("src.core.order_executor.Config", cfg):
        with patch("src.core.order_executor._ac_impact", return_value=0.0):
            with patch(
                "src.core.trade_history.get_backtest_stats", return_value=_BT_STATS
            ):
                await executor.execute(_TOP_BUY, {}, 10_000.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("paper", [True, False])
async def test_position_tracked_when_record_open_fails(paper: bool) -> None:
    executor, monitored, _ = _make_executor()
    executor._trade_history.record_open = AsyncMock(side_effect=RuntimeError("db"))
    await _run(executor, _make_cfg(paper=paper))
    pos = monitored.get("BTC/USDT")
    assert pos is not None, "позиция потеряна из мониторинга"
    assert pos["trade_id"] is None


@pytest.mark.asyncio
async def test_live_warns_when_exchange_sl_missing() -> None:
    executor, _, _ = _make_executor()
    executor._api.place_exchange_sl_tp = AsyncMock(return_value=(None, "TP_ID"))
    await _run(executor, _make_cfg(paper=False))
    text = executor._telegram.notify.await_args_list[-1].args[0]
    assert "Биржевой SL не выставлен" in text


@pytest.mark.asyncio
async def test_live_no_warning_when_exchange_sl_present() -> None:
    executor, _, _ = _make_executor()
    await _run(executor, _make_cfg(paper=False))
    text = executor._telegram.notify.await_args_list[-1].args[0]
    assert "Биржевой SL не выставлен" not in text
