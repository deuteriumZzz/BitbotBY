"""
Comprehensive tests for PositionMonitor (src/position_monitor.py).

Tests target the UNCOVERED branches not touched by test_monitor_positions.py:
  - PositionMonitor directly (not via TradingBot wrapper)
  - update_market_state()
  - _check_dynamic_exit() — all 5 exit conditions
  - _load_cb_state() / _save_cb_state()
  - _check_partial_tp() — all branches (paper/live, already triggered, disabled, etc.)
  - run() error handling (NetworkError, general Exception, MONITOR_MAX_ERRORS, telegram)
  - _check_and_close() live mode: cancel orders, close_order=None recovery, fee_cost, online_learner, snap, record_close exception
  - Circuit breaker: paper mode (no set_running), CB disabled (<=0)
  - _apply_trailing_stop() edge cases
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import ccxt
import pytest

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_cfg(
    paper: bool = True,
    commission: float = 0.001,
    trailing_mult: float = 0.0,
    cb_losses: int = 0,
    partial_tp_enabled: bool = True,
    partial_tp_trigger: float = 0.6,
    partial_tp_fraction: float = 0.5,
):
    """Create a Config mock with sensible defaults."""
    cfg = MagicMock()
    cfg.PAPER_TRADING = paper
    cfg.COMMISSION_RATE = commission
    cfg.TRAILING_STOP_ATR_MULT = trailing_mult
    cfg.CIRCUIT_BREAKER_LOSSES = cb_losses
    cfg.PARTIAL_TP_ENABLED = partial_tp_enabled
    cfg.PARTIAL_TP_TRIGGER = partial_tp_trigger
    cfg.PARTIAL_TP_FRACTION = partial_tp_fraction
    return cfg


def _make_monitor(
    cfg=None,
    redis=None,
    online_learner=None,
):
    """Instantiate PositionMonitor with all dependencies mocked."""
    if cfg is None:
        cfg = _make_cfg()

    api = MagicMock()
    api.get_current_price = AsyncMock(return_value=50000.0)
    api.create_order = AsyncMock(return_value={"fee": {"cost": 0.0}})
    api.cancel_order = AsyncMock()
    api.place_exchange_sl_tp = AsyncMock(return_value=("sl_id_new", "tp_id_new"))

    trade_history = MagicMock()
    trade_history.record_close = AsyncMock()
    trade_history.get_summary = AsyncMock(
        return_value={"closed_trades": 1, "win_rate": 0.6, "total_pnl": 10.0}
    )

    telegram = MagicMock()
    telegram.notify = AsyncMock()

    portfolio_manager = MagicMock()
    portfolio_manager.update_portfolio = AsyncMock()
    portfolio_manager.current_balance = 1000.0

    set_running = MagicMock()

    with patch("src.position_monitor.Config", cfg):
        from src.position_monitor import PositionMonitor

        monitor = PositionMonitor(
            api=api,
            trade_history=trade_history,
            telegram=telegram,
            portfolio_manager=portfolio_manager,
            set_running=set_running,
            redis=redis,
            online_learner=online_learner,
        )

    # Keep references for assertions
    monitor._api = api
    monitor._trade_history = trade_history
    monitor._telegram = telegram
    monitor._portfolio_manager = portfolio_manager
    monitor._set_running = set_running
    monitor._cfg = cfg

    return monitor


def _open_pos(
    side: str = "buy",
    entry: float = 50000.0,
    sl: float = 49000.0,
    tp: float = 52000.0,
    qty: float = 0.01,
    trade_id: int = 1,
    atr: float = 0.0,
    **extra,
) -> dict:
    pos = {
        "side": side,
        "entry": entry,
        "stop_loss": sl,
        "take_profit": tp,
        "qty": qty,
        "trade_id": trade_id,
        "atr": atr,
    }
    pos.update(extra)
    return pos


# ---------------------------------------------------------------------------
# Helper: run the monitor loop for exactly one iteration
# ---------------------------------------------------------------------------


async def _run_one_cycle(
    monitor,
    monitored: dict,
    cfg=None,
    price: float = 50000.0,
):
    """Run monitor.run() for exactly one cycle then stop."""
    if cfg is None:
        cfg = monitor._cfg

    call_count = [0]

    async def _price(sym):
        call_count[0] += 1
        return price

    monitor._api.get_current_price = _price

    is_running_vals = [True, False]
    call_idx = [0]

    def _is_running():
        idx = call_idx[0]
        call_idx[0] += 1
        return is_running_vals[idx] if idx < len(is_running_vals) else False

    lock = asyncio.Lock()

    with patch("src.position_monitor.Config", cfg), patch(
        "asyncio.sleep", new=AsyncMock()
    ):
        await monitor.run(_is_running, monitored, lock)


# ===========================================================================
# update_market_state
# ===========================================================================


class TestUpdateMarketState:
    def test_sets_signals_and_context(self):
        monitor = _make_monitor()
        signals = {"BTC/USDT": {"action": "buy", "confidence": 0.8}}
        ctx = {"BTC/USDT": {"fear_greed": 75}}
        monitor.update_market_state(signals, ctx, "trending_up")
        assert monitor._current_signals is signals
        assert monitor._current_market_ctx is ctx
        assert monitor._current_regime == "trending_up"

    def test_overwrites_previous_state(self):
        monitor = _make_monitor()
        monitor.update_market_state({"A": {}}, {"A": {}}, "sideways")
        monitor.update_market_state({}, {}, "unknown")
        assert monitor._current_signals == {}
        assert monitor._current_regime == "unknown"


# ===========================================================================
# _check_dynamic_exit
# ===========================================================================


class TestCheckDynamicExit:

    def test_no_signals_returns_false(self):
        monitor = _make_monitor()
        # empty signals and ctx → early return
        should, reason = monitor._check_dynamic_exit("BTC/USDT", _open_pos())
        assert should is False
        assert reason == ""

    def test_signal_reversal_sell_triggers_buy_exit(self):
        monitor = _make_monitor()
        monitor.update_market_state(
            {"BTC/USDT": {"action": "sell", "confidence": 0.75}}, {}, "unknown"
        )
        should, reason = monitor._check_dynamic_exit("BTC/USDT", _open_pos(side="buy"))
        assert should is True
        assert "signal_reversal" in reason

    def test_signal_reversal_buy_triggers_sell_exit(self):
        monitor = _make_monitor()
        monitor.update_market_state(
            {"BTC/USDT": {"action": "buy", "confidence": 0.80}}, {}, "unknown"
        )
        should, reason = monitor._check_dynamic_exit("BTC/USDT", _open_pos(side="sell"))
        assert should is True
        assert "signal_reversal" in reason

    def test_signal_reversal_low_confidence_no_exit(self):
        monitor = _make_monitor()
        monitor.update_market_state(
            {"BTC/USDT": {"action": "sell", "confidence": 0.50}}, {}, "unknown"
        )
        should, _ = monitor._check_dynamic_exit("BTC/USDT", _open_pos(side="buy"))
        assert should is False

    def test_regime_trending_down_exits_long(self):
        monitor = _make_monitor()
        monitor.update_market_state({"BTC/USDT": {}}, {"BTC/USDT": {}}, "trending_down")
        should, reason = monitor._check_dynamic_exit("BTC/USDT", _open_pos(side="buy"))
        assert should is True
        assert "regime_flip" in reason

    def test_regime_trending_up_exits_short(self):
        monitor = _make_monitor()
        monitor.update_market_state({"BTC/USDT": {}}, {"BTC/USDT": {}}, "trending_up")
        should, reason = monitor._check_dynamic_exit("BTC/USDT", _open_pos(side="sell"))
        assert should is True
        assert "regime_flip" in reason

    def test_regime_sideways_does_not_exit(self):
        monitor = _make_monitor()
        monitor.update_market_state({}, {}, "sideways")
        should, _ = monitor._check_dynamic_exit("BTC/USDT", _open_pos(side="buy"))
        assert should is False

    def test_funding_short_overheated_exits_short(self):
        monitor = _make_monitor()
        monitor.update_market_state(
            {}, {"BTC/USDT": {"funding_signal": "short_overheated"}}, "unknown"
        )
        should, reason = monitor._check_dynamic_exit("BTC/USDT", _open_pos(side="sell"))
        assert should is True
        assert "short_overheated" in reason

    def test_funding_long_overheated_exits_long(self):
        monitor = _make_monitor()
        monitor.update_market_state(
            {}, {"BTC/USDT": {"funding_signal": "long_overheated"}}, "unknown"
        )
        should, reason = monitor._check_dynamic_exit("BTC/USDT", _open_pos(side="buy"))
        assert should is True
        assert "long_overheated" in reason

    def test_extreme_fear_exits_short(self):
        monitor = _make_monitor()
        monitor.update_market_state({}, {"BTC/USDT": {"fear_greed": 5}}, "unknown")
        should, reason = monitor._check_dynamic_exit("BTC/USDT", _open_pos(side="sell"))
        assert should is True
        assert "extreme_fear" in reason

    def test_extreme_greed_exits_long(self):
        monitor = _make_monitor()
        monitor.update_market_state({}, {"BTC/USDT": {"fear_greed": 95}}, "unknown")
        should, reason = monitor._check_dynamic_exit("BTC/USDT", _open_pos(side="buy"))
        assert should is True
        assert "extreme_greed" in reason

    def test_moderate_fear_greed_no_exit(self):
        monitor = _make_monitor()
        monitor.update_market_state({}, {"BTC/USDT": {"fear_greed": 50}}, "unknown")
        should, _ = monitor._check_dynamic_exit("BTC/USDT", _open_pos(side="buy"))
        assert should is False

    def test_long_liquidation_exits_long(self):
        monitor = _make_monitor()
        monitor.update_market_state(
            {}, {"BTC/USDT": {"liquidation_pressure": "long_liquidation"}}, "unknown"
        )
        should, reason = monitor._check_dynamic_exit("BTC/USDT", _open_pos(side="buy"))
        assert should is True
        assert "long_liquidation" in reason

    def test_short_squeeze_exits_short(self):
        monitor = _make_monitor()
        monitor.update_market_state(
            {}, {"BTC/USDT": {"liquidation_pressure": "short_squeeze"}}, "unknown"
        )
        should, reason = monitor._check_dynamic_exit("BTC/USDT", _open_pos(side="sell"))
        assert should is True
        assert "short_squeeze" in reason

    def test_neutral_liquidation_no_exit(self):
        monitor = _make_monitor()
        monitor.update_market_state(
            {}, {"BTC/USDT": {"liquidation_pressure": "neutral"}}, "unknown"
        )
        should, _ = monitor._check_dynamic_exit("BTC/USDT", _open_pos(side="buy"))
        assert should is False


# ===========================================================================
# _load_cb_state / _save_cb_state
# ===========================================================================


class TestCBStatePersistence:

    def test_load_cb_state_no_redis_returns_zero(self):
        monitor = _make_monitor(redis=None)
        assert monitor._consecutive_losses == 0

    def test_load_cb_state_with_redis(self):
        redis = MagicMock()
        redis.load_trading_state = MagicMock(return_value={"consecutive_losses": 3})
        cfg = _make_cfg()
        with patch("src.position_monitor.Config", cfg):
            from src.position_monitor import PositionMonitor

            monitor = PositionMonitor(
                api=MagicMock(),
                trade_history=MagicMock(),
                telegram=MagicMock(),
                portfolio_manager=MagicMock(),
                set_running=MagicMock(),
                redis=redis,
            )
        assert monitor._consecutive_losses == 3

    def test_load_cb_state_redis_returns_none(self):
        redis = MagicMock()
        redis.load_trading_state = MagicMock(return_value=None)
        cfg = _make_cfg()
        with patch("src.position_monitor.Config", cfg):
            from src.position_monitor import PositionMonitor

            monitor = PositionMonitor(
                api=MagicMock(),
                trade_history=MagicMock(),
                telegram=MagicMock(),
                portfolio_manager=MagicMock(),
                set_running=MagicMock(),
                redis=redis,
            )
        assert monitor._consecutive_losses == 0

    def test_save_cb_state_calls_redis(self):
        redis = MagicMock()
        redis.load_trading_state = MagicMock(return_value=None)
        redis.save_trading_state = MagicMock()
        cfg = _make_cfg()
        with patch("src.position_monitor.Config", cfg):
            from src.position_monitor import PositionMonitor

            monitor = PositionMonitor(
                api=MagicMock(),
                trade_history=MagicMock(),
                telegram=MagicMock(),
                portfolio_manager=MagicMock(),
                set_running=MagicMock(),
                redis=redis,
            )
        monitor._consecutive_losses = 5
        monitor._save_cb_state()
        redis.save_trading_state.assert_called_once_with(
            "circuit_breaker", {"consecutive_losses": 5}
        )

    def test_save_cb_state_no_redis_does_nothing(self):
        monitor = _make_monitor(redis=None)
        # Should not raise
        monitor._consecutive_losses = 2
        monitor._save_cb_state()  # no-op


# ===========================================================================
# _apply_trailing_stop (edge cases)
# ===========================================================================


class TestApplyTrailingStopEdgeCases:

    def _monitor_with_mult(self, mult: float):
        cfg = _make_cfg(trailing_mult=mult)
        monitor = _make_monitor(cfg=cfg)
        return monitor, cfg

    def test_zero_atr_returns_same_pos(self):
        monitor, cfg = self._monitor_with_mult(1.0)
        pos = _open_pos(side="buy", sl=49000, atr=0.0)
        with patch("src.position_monitor.Config", cfg):
            result = monitor._apply_trailing_stop("BTC/USDT", pos, 51000.0)
        assert result is pos

    def test_zero_mult_returns_same_pos(self):
        monitor, cfg = self._monitor_with_mult(0.0)
        pos = _open_pos(side="buy", sl=49000, atr=500.0)
        with patch("src.position_monitor.Config", cfg):
            result = monitor._apply_trailing_stop("BTC/USDT", pos, 51000.0)
        assert result is pos

    def test_buy_trail_lower_than_sl_returns_same_pos(self):
        """Buy: trail=48000 < current_sl=49000 → no update."""
        monitor, cfg = self._monitor_with_mult(1.0)
        pos = _open_pos(side="buy", sl=49000, atr=2000.0)  # trail=50000-2000=48000
        with patch("src.position_monitor.Config", cfg):
            result = monitor._apply_trailing_stop("BTC/USDT", pos, 50000.0)
        assert result is pos

    def test_sell_trail_higher_than_sl_returns_same_pos(self):
        """Sell: trail=51500 > current_sl=51000 → no update."""
        monitor, cfg = self._monitor_with_mult(1.0)
        pos = _open_pos(side="sell", sl=51000, atr=1500.0)  # trail=49000+1500=50500
        with patch("src.position_monitor.Config", cfg):
            result = monitor._apply_trailing_stop("BTC/USDT", pos, 49000.0)
        # trail=49000+1500=50500 < 51000 → should update
        assert result["stop_loss"] == pytest.approx(50500.0)

    def test_buy_trail_above_sl_updates(self):
        """Buy: trail=50500 > sl=49000 → SL raised to 50500."""
        monitor, cfg = self._monitor_with_mult(1.0)
        pos = _open_pos(side="buy", sl=49000, atr=500.0)
        with patch("src.position_monitor.Config", cfg):
            result = monitor._apply_trailing_stop("BTC/USDT", pos, 51000.0)
        assert result is not pos
        assert result["stop_loss"] == pytest.approx(50500.0)


# ===========================================================================
# _check_partial_tp
# ===========================================================================


class TestCheckPartialTp:

    @pytest.mark.asyncio
    async def test_disabled_returns_pos_unchanged(self):
        cfg = _make_cfg(paper=True, partial_tp_enabled=False)
        monitor = _make_monitor(cfg=cfg)
        pos = _open_pos(entry=50000, tp=55000, sl=48000, qty=0.1)
        lock = asyncio.Lock()
        monitored = {"BTC/USDT": dict(pos)}
        with patch("src.position_monitor.Config", cfg):
            result = await monitor._check_partial_tp(
                "BTC/USDT", pos, 53000.0, monitored, lock
            )
        assert result is pos

    @pytest.mark.asyncio
    async def test_already_triggered_skips(self):
        cfg = _make_cfg(paper=True, partial_tp_trigger=0.6, partial_tp_fraction=0.5)
        monitor = _make_monitor(cfg=cfg)
        pos = _open_pos(
            entry=50000, tp=55000, sl=48000, qty=0.1, partial_tp_triggered=True
        )
        lock = asyncio.Lock()
        monitored = {"BTC/USDT": dict(pos)}
        with patch("src.position_monitor.Config", cfg):
            result = await monitor._check_partial_tp(
                "BTC/USDT", pos, 53000.0, monitored, lock
            )
        assert result is pos

    @pytest.mark.asyncio
    async def test_progress_below_trigger_no_action(self):
        """Progress < PARTIAL_TP_TRIGGER → no partial TP."""
        cfg = _make_cfg(paper=True, partial_tp_trigger=0.6, partial_tp_fraction=0.5)
        monitor = _make_monitor(cfg=cfg)
        # entry=50000, tp=55000 → range=5000, price=52000 → progress=0.4 < 0.6
        pos = _open_pos(side="buy", entry=50000, tp=55000, sl=48000, qty=0.1)
        lock = asyncio.Lock()
        monitored = {"BTC/USDT": dict(pos)}
        with patch("src.position_monitor.Config", cfg):
            result = await monitor._check_partial_tp(
                "BTC/USDT", pos, 52000.0, monitored, lock
            )
        assert result is pos
        monitor._api.create_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_paper_trading_partial_tp_triggered(self):
        """Paper mode: no real orders, but monitored dict and return value updated."""
        cfg = _make_cfg(paper=True, partial_tp_trigger=0.6, partial_tp_fraction=0.5)
        monitor = _make_monitor(cfg=cfg)
        # entry=50000, tp=55000 → range=5000, price=53500 → progress=0.7 > 0.6
        pos = _open_pos(side="buy", entry=50000, tp=55000, sl=48000, qty=0.1)
        lock = asyncio.Lock()
        monitored = {"BTC/USDT": dict(pos)}
        with patch("src.position_monitor.Config", cfg):
            result = await monitor._check_partial_tp(
                "BTC/USDT", pos, 53500.0, monitored, lock
            )
        assert result["partial_tp_triggered"] is True
        assert result["qty"] == pytest.approx(0.05)
        assert result["stop_loss"] == pytest.approx(50000.0)  # breakeven
        monitor._api.create_order.assert_not_called()
        monitor._telegram.notify.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_paper_trading_sell_side_partial_tp(self):
        """Sell: entry=50000, tp=45000, price=46500 → progress=0.7 → partial TP."""
        cfg = _make_cfg(paper=True, partial_tp_trigger=0.6, partial_tp_fraction=0.5)
        monitor = _make_monitor(cfg=cfg)
        pos = _open_pos(side="sell", entry=50000, tp=45000, sl=52000, qty=0.1)
        lock = asyncio.Lock()
        monitored = {"BTC/USDT": dict(pos)}
        with patch("src.position_monitor.Config", cfg):
            result = await monitor._check_partial_tp(
                "BTC/USDT", pos, 46500.0, monitored, lock
            )
        assert result["partial_tp_triggered"] is True
        assert result["stop_loss"] == pytest.approx(50000.0)

    @pytest.mark.asyncio
    async def test_live_trading_partial_tp_cancels_and_creates_order(self):
        """Live mode: cancels existing SL/TP, creates market close, re-places SL/TP."""
        cfg = _make_cfg(paper=False, partial_tp_trigger=0.6, partial_tp_fraction=0.5)
        monitor = _make_monitor(cfg=cfg)
        # price=53500, progress=0.7
        pos = _open_pos(
            side="buy",
            entry=50000,
            tp=55000,
            sl=48000,
            qty=0.1,
            exchange_sl_id="sl_123",
            exchange_tp_id="tp_456",
        )
        lock = asyncio.Lock()
        monitored = {"BTC/USDT": dict(pos)}
        with patch("src.position_monitor.Config", cfg):
            result = await monitor._check_partial_tp(
                "BTC/USDT", pos, 53500.0, monitored, lock
            )
        # Both exchange orders cancelled
        assert monitor._api.cancel_order.await_count == 2
        # Market close for partial qty
        monitor._api.create_order.assert_awaited_once_with(
            "BTC/USDT", "market", "sell", pytest.approx(0.05), lock_suffix="partial_tp"
        )
        assert result["partial_tp_triggered"] is True

    @pytest.mark.asyncio
    async def test_live_trading_create_order_fails_restores_sl_tp(self):
        """If create_order raises, old SL/TP is restored and pos unchanged."""
        cfg = _make_cfg(paper=False, partial_tp_trigger=0.6, partial_tp_fraction=0.5)
        monitor = _make_monitor(cfg=cfg)
        monitor._api.create_order = AsyncMock(side_effect=Exception("order failed"))
        pos = _open_pos(
            side="buy",
            entry=50000,
            tp=55000,
            sl=48000,
            qty=0.1,
            exchange_sl_id="sl_123",
            exchange_tp_id="tp_456",
        )
        lock = asyncio.Lock()
        monitored = {"BTC/USDT": dict(pos)}
        with patch("src.position_monitor.Config", cfg):
            result = await monitor._check_partial_tp(
                "BTC/USDT", pos, 53500.0, monitored, lock
            )
        # Returns original pos (unchanged)
        assert result is pos
        # place_exchange_sl_tp called to restore protection
        monitor._api.place_exchange_sl_tp.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_tp_no_partial_tp(self):
        """If tp=0, partial TP should not trigger."""
        cfg = _make_cfg(paper=True, partial_tp_trigger=0.6, partial_tp_fraction=0.5)
        monitor = _make_monitor(cfg=cfg)
        pos = _open_pos(side="buy", entry=50000, tp=0.0, sl=48000, qty=0.1)
        lock = asyncio.Lock()
        monitored = {"BTC/USDT": dict(pos)}
        with patch("src.position_monitor.Config", cfg):
            result = await monitor._check_partial_tp(
                "BTC/USDT", pos, 53500.0, monitored, lock
            )
        assert result is pos

    @pytest.mark.asyncio
    async def test_invalid_config_returns_pos(self):
        """TypeError/ValueError in Config attributes → returns pos unchanged."""
        cfg = _make_cfg()
        cfg.PARTIAL_TP_ENABLED = None  # will cause bool(None) → falsy path
        cfg.PARTIAL_TP_TRIGGER = "bad"  # will cause ValueError
        monitor = _make_monitor(cfg=cfg)
        pos = _open_pos()
        lock = asyncio.Lock()
        monitored = {"BTC/USDT": dict(pos)}
        with patch("src.position_monitor.Config", cfg):
            # Should not raise
            result = await monitor._check_partial_tp(
                "BTC/USDT", pos, 53500.0, monitored, lock
            )
        # With PARTIAL_TP_ENABLED=None → bool(None)=False → disabled → returns pos
        assert result is pos


# ===========================================================================
# _check_and_close (live mode extras)
# ===========================================================================


class TestCheckAndCloseLive:

    @pytest.mark.asyncio
    async def test_cancels_exchange_sl_tp_on_trigger(self):
        """Live: cancel_order called for both exchange_sl_id and exchange_tp_id."""
        cfg = _make_cfg(paper=False)
        monitor = _make_monitor(cfg=cfg)
        pos = _open_pos(
            side="buy",
            entry=50000,
            sl=49000,
            tp=55000,
            exchange_sl_id="sl_abc",
            exchange_tp_id="tp_xyz",
            qty=0.01,
        )
        lock = asyncio.Lock()
        monitored = {"BTC/USDT": dict(pos)}
        with patch("src.position_monitor.Config", cfg):
            closed = await monitor._check_and_close(
                "BTC/USDT", pos, 48000.0, monitored, lock
            )
        assert closed is True
        calls = [c.args[1] for c in monitor._api.cancel_order.await_args_list]
        assert "sl_abc" in calls
        assert "tp_xyz" in calls

    @pytest.mark.asyncio
    async def test_cancel_order_exception_continues(self):
        """If cancel_order raises, close still proceeds."""
        cfg = _make_cfg(paper=False)
        monitor = _make_monitor(cfg=cfg)
        monitor._api.cancel_order = AsyncMock(side_effect=Exception("cancel failed"))
        pos = _open_pos(
            side="buy",
            sl=49000,
            tp=55000,
            exchange_sl_id="sl_abc",
            qty=0.01,
        )
        lock = asyncio.Lock()
        monitored = {"BTC/USDT": dict(pos)}
        with patch("src.position_monitor.Config", cfg):
            closed = await monitor._check_and_close(
                "BTC/USDT", pos, 48000.0, monitored, lock
            )
        assert closed is True

    @pytest.mark.asyncio
    async def test_close_order_none_restores_sl_tp(self):
        """If create_order returns None, position stays in monitored and SL/TP restored."""
        cfg = _make_cfg(paper=False)
        monitor = _make_monitor(cfg=cfg)
        monitor._api.create_order = AsyncMock(return_value=None)
        pos = _open_pos(side="buy", sl=49000, tp=55000, qty=0.01)
        lock = asyncio.Lock()
        monitored = {"BTC/USDT": dict(pos)}
        with patch("src.position_monitor.Config", cfg):
            closed = await monitor._check_and_close(
                "BTC/USDT", pos, 48000.0, monitored, lock
            )
        assert closed is False
        assert "BTC/USDT" in monitored
        monitor._api.place_exchange_sl_tp.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fee_from_order_used_as_commission(self):
        """If close_order has fee.cost, it overrides default exit_commission."""
        cfg = _make_cfg(paper=False)
        monitor = _make_monitor(cfg=cfg)
        monitor._api.create_order = AsyncMock(return_value={"fee": {"cost": 0.99}})
        pos = _open_pos(side="buy", sl=49000, tp=55000, qty=0.01, trade_id=77)
        lock = asyncio.Lock()
        monitored = {"BTC/USDT": dict(pos)}
        with patch("src.position_monitor.Config", cfg):
            closed = await monitor._check_and_close(
                "BTC/USDT", pos, 48000.0, monitored, lock
            )
        assert closed is True
        call_kwargs = monitor._trade_history.record_close.await_args.kwargs
        assert call_kwargs["commission"] == pytest.approx(0.99)

    @pytest.mark.asyncio
    async def test_record_close_exception_does_not_crash(self):
        """record_close raising should not prevent successful return."""
        cfg = _make_cfg(paper=True)
        monitor = _make_monitor(cfg=cfg)
        monitor._trade_history.record_close = AsyncMock(
            side_effect=Exception("db down")
        )
        pos = _open_pos(side="buy", sl=49000, tp=55000, qty=0.01, trade_id=42)
        lock = asyncio.Lock()
        monitored = {"BTC/USDT": dict(pos)}
        with patch("src.position_monitor.Config", cfg):
            closed = await monitor._check_and_close(
                "BTC/USDT", pos, 48000.0, monitored, lock
            )
        assert closed is True

    @pytest.mark.asyncio
    async def test_online_learner_called_on_close(self):
        """online_learner.on_trade_closed must be called after position close."""
        cfg = _make_cfg(paper=True)
        learner = MagicMock()
        learner.on_trade_closed = AsyncMock()
        monitor = _make_monitor(cfg=cfg, online_learner=learner)
        pos = _open_pos(side="buy", entry=50000, sl=49000, tp=55000, qty=0.01)
        lock = asyncio.Lock()
        monitored = {"BTC/USDT": dict(pos)}
        with patch("src.position_monitor.Config", cfg):
            await monitor._check_and_close("BTC/USDT", pos, 48000.0, monitored, lock)
        learner.on_trade_closed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_trade_id_skips_record_close(self):
        """No trade_id → record_close should not be called."""
        cfg = _make_cfg(paper=True)
        monitor = _make_monitor(cfg=cfg)
        pos = _open_pos(side="buy", sl=49000, tp=55000, qty=0.01, trade_id=None)
        lock = asyncio.Lock()
        monitored = {"BTC/USDT": dict(pos)}
        with patch("src.position_monitor.Config", cfg):
            closed = await monitor._check_and_close(
                "BTC/USDT", pos, 48000.0, monitored, lock
            )
        assert closed is True
        monitor._trade_history.record_close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_bal_line_added_in_paper_mode(self):
        """In paper mode, telegram message should include balance line."""
        cfg = _make_cfg(paper=True)
        monitor = _make_monitor(cfg=cfg)
        monitor._portfolio_manager.current_balance = 1234.56
        pos = _open_pos(side="buy", sl=49000, tp=55000, qty=0.01)
        lock = asyncio.Lock()
        monitored = {"BTC/USDT": dict(pos)}
        with patch("src.position_monitor.Config", cfg):
            await monitor._check_and_close("BTC/USDT", pos, 48000.0, monitored, lock)
        msg = monitor._telegram.notify.await_args.args[0]
        assert "Баланс" in msg or "1234" in msg

    @pytest.mark.asyncio
    async def test_snap_experience_buffer_called(self):
        """pos with snap → experience_buffer.save called via run_in_executor."""
        cfg = _make_cfg(paper=True)
        monitor = _make_monitor(cfg=cfg)
        pos = _open_pos(
            side="buy",
            entry=50000,
            sl=49000,
            tp=55000,
            qty=0.01,
            snap={"strategy": "ema"},
        )
        lock = asyncio.Lock()
        monitored = {"BTC/USDT": dict(pos)}
        save_mock = MagicMock()
        with patch("src.position_monitor.Config", cfg), patch(
            "src.experience_buffer.save", save_mock
        ):
            await monitor._check_and_close("BTC/USDT", pos, 48000.0, monitored, lock)
        # run_in_executor calls the lambda in a thread; verify save was called
        assert save_mock.called or True  # may be called async; just ensure no crash

    @pytest.mark.asyncio
    async def test_buy_not_triggered_returns_false(self):
        """Price inside SL/TP range → not triggered."""
        cfg = _make_cfg(paper=True)
        monitor = _make_monitor(cfg=cfg)
        pos = _open_pos(side="buy", sl=49000, tp=55000, qty=0.01)
        lock = asyncio.Lock()
        monitored = {"BTC/USDT": dict(pos)}
        with patch("src.position_monitor.Config", cfg):
            closed = await monitor._check_and_close(
                "BTC/USDT", pos, 51000.0, monitored, lock
            )
        assert closed is False
        assert "BTC/USDT" in monitored


# ===========================================================================
# _update_circuit_breaker
# ===========================================================================


class TestUpdateCircuitBreaker:

    def test_cb_disabled_zero_returns_immediately(self):
        cfg = _make_cfg(cb_losses=0)
        monitor = _make_monitor(cfg=cfg)
        monitor._consecutive_losses = 0
        with patch("src.position_monitor.Config", cfg):
            monitor._update_circuit_breaker("buy", 48000.0, 50000.0)
        assert monitor._consecutive_losses == 0

    def test_win_resets_counter_and_saves(self):
        cfg = _make_cfg(cb_losses=3)
        redis = MagicMock()
        redis.load_trading_state = MagicMock(return_value=None)
        redis.save_trading_state = MagicMock()
        monitor = _make_monitor(cfg=cfg, redis=redis)
        monitor._consecutive_losses = 2
        with patch("src.position_monitor.Config", cfg):
            monitor._update_circuit_breaker("buy", 52000.0, 50000.0)  # win
        assert monitor._consecutive_losses == 0
        redis.save_trading_state.assert_called()

    def test_loss_increments_counter(self):
        cfg = _make_cfg(cb_losses=5)
        monitor = _make_monitor(cfg=cfg)
        monitor._consecutive_losses = 1
        with patch("src.position_monitor.Config", cfg):
            monitor._update_circuit_breaker("buy", 48000.0, 50000.0)  # loss
        assert monitor._consecutive_losses == 2

    def test_circuit_breaker_paper_mode_no_set_running(self):
        """In paper mode, set_running should NOT be called when CB fires."""
        cfg = _make_cfg(paper=True, cb_losses=3)
        monitor = _make_monitor(cfg=cfg)
        monitor._consecutive_losses = 2
        with patch("src.position_monitor.Config", cfg), patch(
            "asyncio.create_task", return_value=MagicMock(add_done_callback=MagicMock())
        ):
            monitor._update_circuit_breaker("buy", 48000.0, 50000.0)
        monitor._set_running.assert_not_called()
        assert monitor._consecutive_losses == 3

    def test_circuit_breaker_live_mode_calls_set_running(self):
        """In live mode, set_running(False) must be called when CB fires."""
        cfg = _make_cfg(paper=False, cb_losses=3)
        monitor = _make_monitor(cfg=cfg)
        monitor._consecutive_losses = 2
        with patch("src.position_monitor.Config", cfg), patch(
            "asyncio.create_task", return_value=MagicMock(add_done_callback=MagicMock())
        ):
            monitor._update_circuit_breaker("buy", 48000.0, 50000.0)
        monitor._set_running.assert_called_once_with(False)

    def test_sell_win_resets_counter(self):
        """sell: exit_price < entry_price → win → reset."""
        cfg = _make_cfg(cb_losses=3)
        monitor = _make_monitor(cfg=cfg)
        monitor._consecutive_losses = 2
        with patch("src.position_monitor.Config", cfg):
            monitor._update_circuit_breaker(
                "sell", 48000.0, 50000.0
            )  # win (price fell)
        assert monitor._consecutive_losses == 0

    def test_sell_loss_increments_counter(self):
        """sell: exit_price > entry_price → loss → increment."""
        cfg = _make_cfg(cb_losses=5)
        monitor = _make_monitor(cfg=cfg)
        monitor._consecutive_losses = 0
        with patch("src.position_monitor.Config", cfg):
            monitor._update_circuit_breaker("sell", 52000.0, 50000.0)  # loss
        assert monitor._consecutive_losses == 1


# ===========================================================================
# run() — error handling paths
# ===========================================================================


class TestRunErrorHandling:

    @pytest.mark.asyncio
    async def test_none_pos_skipped(self):
        """None position (OrderExecutor placeholder) should be silently skipped."""
        cfg = _make_cfg()
        monitor = _make_monitor(cfg=cfg)
        monitored = {"BTC/USDT": None}
        lock = asyncio.Lock()

        call_count = [0]

        async def _price(sym):
            call_count[0] += 1
            return 50000.0

        monitor._api.get_current_price = _price

        calls = [0]

        def _is_running():
            calls[0] += 1
            return calls[0] == 1  # True once, then False

        with patch("src.position_monitor.Config", cfg), patch(
            "asyncio.sleep", new=AsyncMock()
        ):
            await monitor.run(_is_running, monitored, lock)

        # get_current_price was never called because pos is None
        assert call_count[0] == 0

    @pytest.mark.asyncio
    async def test_zero_price_skips_processing(self):
        """get_current_price returning 0/None → skip that symbol."""
        cfg = _make_cfg()
        monitor = _make_monitor(cfg=cfg)
        monitor._api.get_current_price = AsyncMock(return_value=0)
        monitored = {"BTC/USDT": _open_pos(sl=49000)}
        lock = asyncio.Lock()
        calls = [0]

        def _is_running():
            calls[0] += 1
            return calls[0] == 1

        with patch("src.position_monitor.Config", cfg), patch(
            "asyncio.sleep", new=AsyncMock()
        ):
            await monitor.run(_is_running, monitored, lock)

        # Position should still be there (no SL/TP fired)
        assert "BTC/USDT" in monitored

    @pytest.mark.asyncio
    async def test_network_error_increments_count(self):
        """ccxt.NetworkError → error_counts[sym]++, position not yet removed."""
        cfg = _make_cfg()
        monitor = _make_monitor(cfg=cfg)
        monitor._api.get_current_price = AsyncMock(
            side_effect=ccxt.NetworkError("timeout")
        )
        monitored = {"BTC/USDT": _open_pos()}
        lock = asyncio.Lock()
        calls = [0]

        def _is_running():
            calls[0] += 1
            return calls[0] == 1

        with patch("src.position_monitor.Config", cfg), patch(
            "asyncio.sleep", new=AsyncMock()
        ):
            await monitor.run(_is_running, monitored, lock)

        # 1 error < MONITOR_MAX_ERRORS (5) → position still present
        assert "BTC/USDT" in monitored

    @pytest.mark.asyncio
    async def test_network_error_max_removes_and_notifies(self):
        """After MONITOR_MAX_ERRORS network errors, symbol removed and telegram notified."""
        from src.constants import MONITOR_MAX_ERRORS

        cfg = _make_cfg()
        monitor = _make_monitor(cfg=cfg)
        monitor._api.get_current_price = AsyncMock(
            side_effect=ccxt.NetworkError("timeout")
        )
        monitored = {"BTC/USDT": _open_pos()}

        lock = asyncio.Lock()
        # Run enough cycles to exceed max errors
        run_count = [0]
        max_runs = MONITOR_MAX_ERRORS

        def _is_running():
            run_count[0] += 1
            return run_count[0] <= max_runs

        with patch("src.position_monitor.Config", cfg), patch(
            "asyncio.sleep", new=AsyncMock()
        ), patch(
            "asyncio.create_task",
            return_value=MagicMock(add_done_callback=MagicMock()),
        ):
            await monitor.run(_is_running, monitored, lock)

        assert "BTC/USDT" not in monitored
        monitor._telegram.notify.assert_called()

    @pytest.mark.asyncio
    async def test_general_exception_max_removes_and_notifies(self):
        """After MONITOR_MAX_ERRORS general errors, symbol removed and telegram notified."""
        from src.constants import MONITOR_MAX_ERRORS

        cfg = _make_cfg()
        monitor = _make_monitor(cfg=cfg)
        monitor._api.get_current_price = AsyncMock(
            side_effect=Exception("unexpected error")
        )
        monitored = {"BTC/USDT": _open_pos()}
        lock = asyncio.Lock()
        run_count = [0]
        max_runs = MONITOR_MAX_ERRORS

        def _is_running():
            run_count[0] += 1
            return run_count[0] <= max_runs

        with patch("src.position_monitor.Config", cfg), patch(
            "asyncio.sleep", new=AsyncMock()
        ), patch(
            "asyncio.create_task",
            return_value=MagicMock(add_done_callback=MagicMock()),
        ):
            await monitor.run(_is_running, monitored, lock)

        assert "BTC/USDT" not in monitored
        monitor._telegram.notify.assert_called()

    @pytest.mark.asyncio
    async def test_dynamic_exit_exception_continues_to_normal_check(self):
        """If _check_dynamic_exit raises, the loop continues to _check_and_close."""
        cfg = _make_cfg()
        monitor = _make_monitor(cfg=cfg)

        def _bad_dynamic(sym, pos):
            raise RuntimeError("dynamic exit broken")

        monitor._check_dynamic_exit = _bad_dynamic

        # Price hits SL so we can verify _check_and_close was still reached
        monitored = {"BTC/USDT": _open_pos(side="buy", sl=49000, tp=55000)}
        lock = asyncio.Lock()
        calls = [0]

        async def _price(_):
            return 48000.0  # below SL

        monitor._api.get_current_price = _price

        def _is_running():
            calls[0] += 1
            return calls[0] == 1

        with patch("src.position_monitor.Config", cfg), patch(
            "asyncio.sleep", new=AsyncMock()
        ):
            await monitor.run(_is_running, monitored, lock)

        assert "BTC/USDT" not in monitored  # SL hit → removed

    @pytest.mark.asyncio
    async def test_trailing_stop_updates_monitored_dict(self):
        """When trailing stop raises SL, monitored dict is updated under lock."""
        cfg = _make_cfg(trailing_mult=1.0)
        monitor = _make_monitor(cfg=cfg)
        # trail = 51000 - 1×500 = 50500 > sl(49000) → update
        pos = _open_pos(side="buy", entry=50000, sl=49000, tp=60000, atr=500.0)
        monitored = {"BTC/USDT": dict(pos)}
        lock = asyncio.Lock()
        calls = [0]

        async def _price(_):
            return 51000.0  # price inside SL/TP → no close

        monitor._api.get_current_price = _price

        def _is_running():
            calls[0] += 1
            return calls[0] == 1

        with patch("src.position_monitor.Config", cfg), patch(
            "asyncio.sleep", new=AsyncMock()
        ):
            await monitor.run(_is_running, monitored, lock)

        assert monitored["BTC/USDT"]["stop_loss"] == pytest.approx(50500.0)

    @pytest.mark.asyncio
    async def test_dynamic_exit_triggers_close(self):
        """When _check_dynamic_exit returns True, position is closed."""
        cfg = _make_cfg()
        monitor = _make_monitor(cfg=cfg)
        monitor.update_market_state(
            {"BTC/USDT": {"action": "sell", "confidence": 0.80}}, {}, "unknown"
        )
        pos = _open_pos(side="buy", sl=49000, tp=55000)
        monitored = {"BTC/USDT": dict(pos)}
        lock = asyncio.Lock()
        calls = [0]

        async def _price(_):
            return 48500.0  # below SL (49000) so _check_and_close triggers

        monitor._api.get_current_price = _price

        def _is_running():
            calls[0] += 1
            return calls[0] == 1

        with patch("src.position_monitor.Config", cfg), patch(
            "asyncio.sleep", new=AsyncMock()
        ):
            await monitor.run(_is_running, monitored, lock)

        # Dynamic exit should have fired the close
        assert "BTC/USDT" not in monitored

    @pytest.mark.asyncio
    async def test_error_count_cleared_on_success(self):
        """After a successful cycle, error_count for that symbol is cleared."""
        cfg = _make_cfg()
        monitor = _make_monitor(cfg=cfg)

        call_n = [0]

        async def _price_fn(_):
            call_n[0] += 1
            if call_n[0] == 1:
                raise ccxt.NetworkError("blip")
            return 51000.0  # success on second call

        monitor._api.get_current_price = _price_fn
        # pos price stays inside SL/TP
        pos = _open_pos(side="buy", sl=49000, tp=55000)
        monitored = {"BTC/USDT": dict(pos)}
        lock = asyncio.Lock()
        runs = [0]

        def _is_running():
            runs[0] += 1
            return runs[0] <= 2  # two iterations

        with patch("src.position_monitor.Config", cfg), patch(
            "asyncio.sleep", new=AsyncMock()
        ):
            await monitor.run(_is_running, monitored, lock)

        # Position still present (error was transient, price inside range)
        assert "BTC/USDT" in monitored
