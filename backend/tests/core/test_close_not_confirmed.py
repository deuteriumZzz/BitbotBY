"""Закрытие не подтверждено биржей (отказ или исход неизвестен) — позиция не бросается.

Раньше аварийное закрытие, закрытие убыточных при просадке, частичное и
принудительное закрытие отменяли биржевые SL/TP, а результат ордера не смотрели:
позиция пропадала из монитора и считалась закрытой, оставаясь открытой на бирже
без защиты. Теперь любой путь закрытия: позиция остаётся в мониторе, SL/TP
ставятся заново на весь объём, сделка не записывается закрытой.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from bitbot.domain.order_states import Rejected, Unknown
from tests.core.test_position_monitor_new import _make_cfg, _make_monitor, _open_pos

SYM = "BTC/USDT"


async def _close(path, monitor, pos, monitored, lock):
    if path == "emergency":
        await monitor.close_all_positions(monitored, lock)
    elif path == "drawdown_loser":
        await monitor.close_losers_tighten_winners(monitored, lock, {SYM: 48000.0})
    elif path == "partial":
        assert await monitor.close_fraction(SYM, pos, 0.5, monitored, lock) == (
            None,
            False,
        )
    else:
        await monitor._force_close_at_market(SYM, pos, 48000.0, monitored, lock)


@pytest.mark.parametrize("outcome", [Rejected("отказ"), Unknown("bb-1")])
@pytest.mark.parametrize("path", ["emergency", "drawdown_loser", "partial", "force"])
async def test_unconfirmed_close_keeps_position_protected(path, outcome):
    cfg = _make_cfg(paper=False)
    monitor = _make_monitor(cfg=cfg)
    monitor._api.create_order = AsyncMock(return_value=outcome)
    pos = _open_pos(side="buy", entry=50000, sl=49000, tp=52000, qty=0.01)
    pos.update(exchange_sl_id="sl-old", exchange_tp_id="tp-old")
    monitored = {SYM: dict(pos)}
    with patch("bitbot.execution.position_monitor.Config", cfg):
        await _close(path, monitor, pos, monitored, asyncio.Lock())

    assert SYM in monitored  # наблюдение не потеряно
    monitor._api.place_exchange_sl_tp.assert_awaited_once_with(
        SYM, "sell", 0.01, 49000.0, 52000.0
    )  # защита на ВЕСЬ объём
    assert monitored[SYM]["exchange_sl_id"] == "sl_id_new"
    monitor._trade_history.record_close.assert_not_awaited()
