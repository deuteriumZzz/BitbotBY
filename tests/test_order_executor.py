"""Прямые юнит-тесты для OrderExecutor."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.order_executor import OrderExecutor

# ── Вспомогательные функции ───────────────────────────────────────────────────


def _make_df(n: int = 50, price: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    prices = price + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame(
        {
            "close": prices,
            "volume": [1_000_000.0] * n,
            "high": prices * 1.01,
            "low": prices * 0.99,
        }
    )


def _make_cfg(paper: bool = True, max_pos: int = 5, max_corr: float = 0.0) -> MagicMock:
    cfg = MagicMock()
    cfg.PAPER_TRADING = paper
    cfg.MAX_POSITIONS = max_pos
    cfg.SYMBOL = "BTC/USDT"
    cfg.MAX_CORRELATION = max_corr
    cfg.DEFAULT_STRATEGY = "rsi"
    cfg.RISK_PER_TRADE = 0.01
    cfg.TIMEFRAME = "15m"
    cfg.COMMISSION_RATE = 0.001
    cfg.AUTO_EXECUTE = True
    cfg.TELEGRAM_CONFIRM_TIMEOUT = 0
    return cfg


def _make_executor(monitored: dict | None = None) -> tuple[OrderExecutor, dict]:
    """Возвращает (executor, monitored_dict) со всеми замоканными зависимостями."""
    if monitored is None:
        monitored = {}
    lock = asyncio.Lock()

    api = MagicMock()
    api.round_quantity = MagicMock(side_effect=lambda sym, qty: round(qty, 6))
    api.create_order = AsyncMock(return_value={"id": "ORDER123"})
    api.place_exchange_sl_tp = AsyncMock(return_value=("SL_ID", "TP_ID"))
    api.set_leverage = AsyncMock(return_value=True)

    trade_history = MagicMock()
    trade_history.get_win_rate = AsyncMock(return_value=0.6)
    trade_history.get_trade_count = AsyncMock(return_value=15)
    trade_history.get_expected_value = AsyncMock(return_value=0.02)
    trade_history.record_open = AsyncMock(return_value=42)

    telegram = MagicMock()
    telegram.ask_confirm = AsyncMock(return_value=True)
    telegram.notify = AsyncMock()

    risk_manager = MagicMock()
    risk_manager.calculate_kelly_size = MagicMock(return_value=0.005)

    corr_filter = MagicMock()
    corr_filter.is_allowed = MagicMock(return_value=True)
    corr_filter.max_correlation = MagicMock(return_value=0.0)

    paper_balance = [10_000.0]

    executor = OrderExecutor(
        api=api,
        trade_history=trade_history,
        telegram=telegram,
        risk_manager=risk_manager,
        portfolio_optimizer=MagicMock(),
        corr_filter=corr_filter,
        get_monitored=lambda: monitored,
        get_lock=lambda: lock,
        get_paper_balance=lambda: paper_balance[0],
        set_paper_balance=lambda v: paper_balance.__setitem__(0, v),
        set_last_trade_at=lambda v: None,
        get_current_regime=lambda: "trending",
    )
    executor._paper_balance = paper_balance
    return executor, monitored


_TOP_BUY = {
    "symbol": "BTC/USDT",
    "action": "buy",
    "entry": 100.0,
    "stop_loss": 95.0,
    "take_profit": 110.0,
    "confidence": 0.8,
    "strategy": "rsi",
    "atr": 1.5,
    "alloc_fraction": 0.01,
}

_BT_STATS = {"win_rate": 0.6, "total_trades": 20, "ev": 0.01}


# ── Защитные проверки ─────────────────────────────────────────────────────────


class TestGuards:
    @pytest.mark.asyncio
    async def test_skips_when_max_positions_reached(self):
        monitored = {
            f"SYM{i}/USDT": {"qty": 0.1, "side": "buy", "entry": 100.0}
            for i in range(3)
        }
        executor, _ = _make_executor(monitored=monitored)
        with patch("src.order_executor.Config", _make_cfg(max_pos=3)):
            await executor.execute(_TOP_BUY, {}, 10_000.0)
        executor._trade_history.record_open.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_duplicate_symbol(self):
        monitored = {"BTC/USDT": {"qty": 0.1, "side": "buy", "entry": 100.0}}
        executor, _ = _make_executor(monitored=monitored)
        with patch("src.order_executor.Config", _make_cfg(max_pos=5)):
            await executor.execute(_TOP_BUY, {}, 10_000.0)
        executor._trade_history.record_open.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_correlation_blocked(self):
        executor, monitored = _make_executor()
        monitored["ETH/USDT"] = {"qty": 0.5, "side": "buy", "entry": 200.0}
        executor._corr_filter.is_allowed = MagicMock(return_value=False)
        executor._corr_filter.max_correlation = MagicMock(return_value=0.95)
        with patch("src.order_executor.Config", _make_cfg(max_pos=5, max_corr=0.7)):
            await executor.execute(_TOP_BUY, {}, 10_000.0)
        executor._trade_history.record_open.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_hold_action(self):
        executor, monitored = _make_executor()
        with patch("src.order_executor.Config", _make_cfg()):
            await executor.execute({**_TOP_BUY, "action": "hold"}, {}, 10_000.0)
        assert "BTC/USDT" not in monitored

    @pytest.mark.asyncio
    async def test_skips_when_telegram_rejects(self):
        executor, monitored = _make_executor()
        executor._telegram.ask_confirm = AsyncMock(return_value=False)
        # AUTO_EXECUTE=False → диалог показывается, Skip отменяет сделку
        cfg = _make_cfg()
        cfg.AUTO_EXECUTE = False
        with patch("src.order_executor.Config", cfg):
            with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                await executor.execute(_TOP_BUY, {}, 10_000.0)
        assert "BTC/USDT" not in monitored

    @pytest.mark.asyncio
    async def test_skips_zero_entry_price(self):
        executor, monitored = _make_executor()
        with patch("src.order_executor.Config", _make_cfg()):
            with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                await executor.execute({**_TOP_BUY, "entry": 0.0}, {}, 10_000.0)
        assert "BTC/USDT" not in monitored


# ── Бумажная торговля ─────────────────────────────────────────────────────────


class TestPaperTrading:
    @pytest.mark.asyncio
    async def test_records_position_in_monitored(self):
        executor, monitored = _make_executor()
        with patch("src.order_executor.Config", _make_cfg()):
            with patch("src.order_executor._ac_impact", return_value=0.0):
                with patch(
                    "src.trade_history.get_backtest_stats", return_value=_BT_STATS
                ):
                    await executor.execute(_TOP_BUY, {"BTC/USDT": _make_df()}, 10_000.0)

        assert "BTC/USDT" in monitored
        pos = monitored["BTC/USDT"]
        assert pos["side"] == "buy"
        assert pos["trade_id"] == 42

    @pytest.mark.asyncio
    async def test_does_not_call_create_order(self):
        executor, _ = _make_executor()
        with patch("src.order_executor.Config", _make_cfg()):
            with patch("src.order_executor._ac_impact", return_value=0.0):
                with patch(
                    "src.trade_history.get_backtest_stats", return_value=_BT_STATS
                ):
                    await executor.execute(_TOP_BUY, {"BTC/USDT": _make_df()}, 10_000.0)

        executor._api.create_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_paper_balance_decreases_on_buy(self):
        executor, _ = _make_executor()
        initial = executor._paper_balance[0]
        with patch("src.order_executor.Config", _make_cfg()):
            with patch("src.order_executor._ac_impact", return_value=0.0):
                with patch(
                    "src.trade_history.get_backtest_stats", return_value=_BT_STATS
                ):
                    await executor.execute(_TOP_BUY, {"BTC/USDT": _make_df()}, 10_000.0)

        assert executor._paper_balance[0] < initial


# ── Расчёт размера позиции ────────────────────────────────────────────────────


class TestSizing:
    def test_kelly_path_with_enough_trades(self):
        executor, _ = _make_executor()
        executor._risk_manager.calculate_kelly_size = MagicMock(return_value=0.002)

        with patch("src.order_executor.Config", _make_cfg()):
            qty = executor._size_position(
                top={**_TOP_BUY, "alloc_fraction": 0.01},
                balance=10_000.0,
                live_wr=0.6,
                live_n=15,
                entry=100.0,
            )
        # portfolio_qty = 10000 * 0.01 / 100 = 1.0; kelly = 0.002 → итог = 0.002
        assert qty == pytest.approx(0.002)

    def test_conservative_cap_when_few_trades(self):
        executor, _ = _make_executor()
        cfg = _make_cfg()
        cfg.RISK_PER_TRADE = 0.01
        executor.configure_risk(
            max_positions=cfg.MAX_POSITIONS,
            risk_per_trade=0.01,
            drawdown_scale_enabled=True,
        )

        with patch("src.order_executor.Config", cfg):
            qty = executor._size_position(
                top={**_TOP_BUY, "alloc_fraction": 0.05},
                balance=10_000.0,
                live_wr=0.5,
                live_n=3,  # < KELLY_MIN_TRADES
                entry=100.0,
            )
        # conservative = 10000 * 0.01 / 100 = 1.0; portfolio = 5.0 → мин = 1.0
        assert qty == pytest.approx(1.0)

    def test_zero_kelly_falls_back_to_portfolio_qty(self):
        executor, _ = _make_executor()
        executor._risk_manager.calculate_kelly_size = MagicMock(return_value=0.0)

        with patch("src.order_executor.Config", _make_cfg()):
            qty = executor._size_position(
                top={**_TOP_BUY, "alloc_fraction": 0.01},
                balance=10_000.0,
                live_wr=0.6,
                live_n=15,
                entry=100.0,
            )
        # kelly_qty = 0 → откат к portfolio_qty = 10000 * 0.01 / 100 = 1.0
        assert qty == pytest.approx(1.0)
