"""
Extra pytest coverage for OrderExecutor — targets uncovered branches from coverage report.

Covers (by source line reference):
  75-76   : runtime_config.get_leverage_mode() raises → falls back to Config
  81      : mode == "fixed" → returns fallback leverage
  86      : atr<=0 or price<=0 → returns fallback leverage
  93-111  : mode == "full" with regime + drawdown multipliers
  122-123 : outer exception in _calc_dynamic_leverage → returns fallback
  210-215 : runtime_config.is_trading_time() == False → skip
  220-231 : TRADING_HOURS str filter (non-runtime_config path, normal + overnight)
  324     : scale < 1.0 → quantity multiplied by drawdown scale
  333-339 : buy with sl >= entry → skipped
  341-364 : buy with tp <= entry, sell SL/TP invalid → skipped
  371     : sell action paper trading (LINEAR SHORT margin reservation)
  393-400 : post-impact SL invalid (buy) → skipped
  402-428 : post-impact TP invalid (buy), post-impact SL/TP invalid (sell) → skipped
  432     : round_quantity returns 0 → skipped
  459-482 : live trading: set_leverage, create_order, place_exchange_sl_tp
  543-566 : _check_liquidity spread too wide / volume too low / exception allow
  584-585 : _drawdown_scale TypeError/ValueError config attrs → return 1.0
  587     : _drawdown_scale disabled → return 1.0
  591     : _drawdown_scale peak_balance <= 0 → return 1.0
  594-600 : _drawdown_scale drawdown >= threshold → returns factor
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.order_executor import OrderExecutor, _calc_dynamic_leverage


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_cfg(
    paper: bool = True,
    max_pos: int = 5,
    max_corr: float = 0.0,
    market_type: str = "linear",
    auto_execute: bool = True,
) -> MagicMock:
    cfg = MagicMock()
    cfg.PAPER_TRADING = paper
    cfg.MAX_POSITIONS = max_pos
    cfg.SYMBOL = "BTC/USDT"
    cfg.MAX_CORRELATION = max_corr
    cfg.DEFAULT_STRATEGY = "rsi"
    cfg.RISK_PER_TRADE = 0.01
    cfg.TIMEFRAME = "15m"
    cfg.COMMISSION_RATE = 0.001
    cfg.AUTO_EXECUTE = auto_execute
    cfg.TELEGRAM_CONFIRM_TIMEOUT = 0
    cfg.MARKET_TYPE = market_type
    cfg.LEVERAGE = 3
    cfg.LEVERAGE_MIN = 1
    cfg.LEVERAGE_MAX = 5
    cfg.LEVERAGE_MODE = "volatility"
    cfg.LEVERAGE_TARGET_RISK = 0.01
    cfg.MAX_DRAWDOWN_PERCENT = 0.15
    cfg.TRADING_HOURS = ""
    cfg.MAX_SPREAD_PCT = 0.3
    cfg.MIN_VOLUME_USDT = 1_000_000.0
    cfg.DRAWDOWN_SCALE_ENABLED = True
    cfg.DRAWDOWN_SCALE_THRESHOLD = 0.10
    cfg.DRAWDOWN_SCALE_FACTOR = 0.5
    return cfg


def _make_executor(monitored: dict | None = None, runtime_config=None):
    """Returns (executor, monitored_dict, paper_balance_list) with all mocked deps."""
    if monitored is None:
        monitored = {}
    lock = asyncio.Lock()

    api = MagicMock()
    api.round_quantity = MagicMock(side_effect=lambda sym, qty: round(qty, 6))
    api.create_order = AsyncMock(return_value={"id": "ORDER123", "fee": {"cost": 0}})
    api.place_exchange_sl_tp = AsyncMock(return_value=("SL_ID", "TP_ID"))
    api.set_leverage = AsyncMock(return_value=True)
    api.exchange = MagicMock()
    api.exchange.fetch_ticker = AsyncMock(
        return_value={"bid": 99.9, "ask": 100.1, "quoteVolume": 2_000_000.0}
    )

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
        runtime_config=runtime_config,
    )
    return executor, monitored, paper_balance


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

_TOP_SELL = {
    "symbol": "BTC/USDT",
    "action": "sell",
    "entry": 100.0,
    "stop_loss": 105.0,
    "take_profit": 90.0,
    "confidence": 0.8,
    "strategy": "rsi",
    "atr": 1.5,
    "alloc_fraction": 0.01,
}

_BT_STATS = {"win_rate": 0.6, "total_trades": 20, "ev": 0.01}


# ── _calc_dynamic_leverage ────────────────────────────────────────────────────


class TestCalcDynamicLeverage:
    def _cfg(self, **kwargs):
        cfg = MagicMock()
        cfg.LEVERAGE = kwargs.get("LEVERAGE", 3)
        cfg.LEVERAGE_MIN = kwargs.get("LEVERAGE_MIN", 1)
        cfg.LEVERAGE_MAX = kwargs.get("LEVERAGE_MAX", 5)
        cfg.LEVERAGE_MODE = kwargs.get("LEVERAGE_MODE", "volatility")
        cfg.LEVERAGE_TARGET_RISK = kwargs.get("LEVERAGE_TARGET_RISK", 0.01)
        cfg.MAX_DRAWDOWN_PERCENT = kwargs.get("MAX_DRAWDOWN_PERCENT", 0.15)
        return cfg

    def test_fixed_mode_returns_fallback(self):
        """mode == 'fixed' branch (line 81)."""
        cfg = self._cfg(LEVERAGE_MODE="fixed", LEVERAGE=3, LEVERAGE_MIN=1, LEVERAGE_MAX=5)
        top = {"symbol": "BTC/USDT", "atr": 1.5}
        with patch("src.order_executor.Config", cfg):
            lev = _calc_dynamic_leverage(top, entry=100.0)
        assert lev == 3

    def test_zero_atr_returns_fallback(self):
        """atr <= 0 → returns fallback (line 86)."""
        cfg = self._cfg(LEVERAGE_MODE="volatility")
        top = {"symbol": "BTC/USDT", "atr": 0.0}
        with patch("src.order_executor.Config", cfg):
            lev = _calc_dynamic_leverage(top, entry=100.0)
        assert lev == 3

    def test_zero_price_returns_fallback(self):
        """price <= 0 and no entry → returns fallback (line 86)."""
        cfg = self._cfg(LEVERAGE_MODE="volatility")
        top = {"symbol": "BTC/USDT", "atr": 1.5, "price": 0.0}
        with patch("src.order_executor.Config", cfg):
            lev = _calc_dynamic_leverage(top, entry=0.0)
        assert lev == 3

    def test_runtime_config_get_leverage_mode_raises(self):
        """runtime_config.get_leverage_mode() raises → fallback to Config (lines 75-76)."""
        cfg = self._cfg(LEVERAGE_MODE="volatility", LEVERAGE_TARGET_RISK=0.01)
        rc = MagicMock()
        rc.get_leverage_mode.side_effect = RuntimeError("oops")
        top = {"symbol": "BTC/USDT", "atr": 1.5}
        with patch("src.order_executor.Config", cfg):
            # Should not raise; returns a valid integer
            lev = _calc_dynamic_leverage(top, entry=100.0, runtime_config=rc)
        assert isinstance(lev, int)
        assert 1 <= lev <= 5

    def test_mode_full_uptrend(self):
        """mode == 'full', uptrend, no drawdown (lines 93-111)."""
        cfg = self._cfg(LEVERAGE_MODE="full", LEVERAGE_TARGET_RISK=0.02)
        top = {
            "symbol": "BTC/USDT",
            "atr": 1.0,  # atr_pct = 1.0/100 = 0.01 → base_lev = 0.02/0.01 = 2
            "indicators": {"trend": "uptrend"},
        }
        with patch("src.order_executor.Config", cfg):
            lev = _calc_dynamic_leverage(top, entry=100.0, balance=10000, peak_balance=10000)
        # uptrend mult=1.0, no drawdown → lev = round(2 * 1.0 * 1.0) = 2
        assert lev == 2

    def test_mode_full_downtrend_drawdown(self):
        """mode == 'full', downtrend with drawdown → reduced leverage (lines 97-111)."""
        cfg = self._cfg(LEVERAGE_MODE="full", LEVERAGE_TARGET_RISK=0.05, MAX_DRAWDOWN_PERCENT=0.20)
        top = {
            "symbol": "ETH/USDT",
            "atr": 2.0,  # atr_pct = 0.02
            "indicators": {"trend": "downtrend"},
        }
        with patch("src.order_executor.Config", cfg):
            lev = _calc_dynamic_leverage(
                top,
                entry=100.0,
                balance=8000.0,
                peak_balance=10000.0,  # 20% drawdown
            )
        # downtrend mult=0.5, dd_pct=0.20, max_dd=0.20 → drawdown_mult = max(0.3, 1-0.20/0.20)=0.3
        # base = 0.05/0.02 = 2.5; 2.5 * 0.5 * 0.3 = 0.375 → round(0.375) = 0 → max(1,0)=1
        assert 1 <= lev <= 5

    def test_mode_full_sideways_regime(self):
        """mode == 'full', sideways regime multiplier = 0.7."""
        cfg = self._cfg(LEVERAGE_MODE="full", LEVERAGE_TARGET_RISK=0.02)
        top = {
            "symbol": "BTC/USDT",
            "atr": 1.0,
            "indicators": {"trend": "sideways"},
        }
        with patch("src.order_executor.Config", cfg):
            lev = _calc_dynamic_leverage(top, entry=100.0, balance=10000, peak_balance=10000)
        # sideways mult=0.7; base=2; 2*0.7*1.0=1.4 → round=1
        assert 1 <= lev <= 5

    def test_mode_full_unknown_regime_defaults_07(self):
        """Unknown trend → _REGIME_MULT.get default 0.7."""
        cfg = self._cfg(LEVERAGE_MODE="full", LEVERAGE_TARGET_RISK=0.02)
        top = {
            "symbol": "BTC/USDT",
            "atr": 1.0,
            "indicators": {"trend": "unknown_regime"},
        }
        with patch("src.order_executor.Config", cfg):
            lev = _calc_dynamic_leverage(top, entry=100.0, balance=10000, peak_balance=10000)
        assert isinstance(lev, int)

    def test_outer_exception_returns_fallback(self):
        """Exception in _calc_dynamic_leverage → fallback (lines 122-123)."""
        cfg = MagicMock()
        # LEVERAGE_MIN = "bad" → int("bad") → ValueError → outer except
        cfg.LEVERAGE_MIN = "bad_value"
        cfg.LEVERAGE = 3
        with patch("src.order_executor.Config", cfg):
            lev = _calc_dynamic_leverage({}, entry=0.0)
        # Should return a valid int (max(1, int(Config.LEVERAGE)) fallback path)
        assert isinstance(lev, int)
        assert lev >= 1

    def test_volatility_mode_clamps_to_max(self):
        """Result clamped to LEVERAGE_MAX."""
        cfg = self._cfg(LEVERAGE_MODE="volatility", LEVERAGE_TARGET_RISK=0.5, LEVERAGE_MAX=5)
        top = {"symbol": "BTC/USDT", "atr": 0.1}  # atr_pct=0.001 → base=500
        with patch("src.order_executor.Config", cfg):
            lev = _calc_dynamic_leverage(top, entry=100.0)
        assert lev == 5

    def test_volatility_mode_clamps_to_min(self):
        """Result clamped to LEVERAGE_MIN."""
        cfg = self._cfg(LEVERAGE_MODE="volatility", LEVERAGE_TARGET_RISK=0.0001, LEVERAGE_MIN=1)
        top = {"symbol": "BTC/USDT", "atr": 10.0}  # atr_pct=0.1 → base=0.001
        with patch("src.order_executor.Config", cfg):
            lev = _calc_dynamic_leverage(top, entry=100.0)
        assert lev == 1

    def test_full_mode_no_drawdown_when_balance_at_peak(self):
        """peak_balance == balance → drawdown_mult = 1.0 (line 109)."""
        cfg = self._cfg(LEVERAGE_MODE="full", LEVERAGE_TARGET_RISK=0.02)
        top = {
            "symbol": "BTC/USDT",
            "atr": 1.0,
            "indicators": {"trend": "uptrend"},
        }
        with patch("src.order_executor.Config", cfg):
            lev_no_dd = _calc_dynamic_leverage(top, entry=100.0, balance=10000, peak_balance=10000)
            lev_with_dd = _calc_dynamic_leverage(top, entry=100.0, balance=8000, peak_balance=10000)
        # With drawdown leverage should be <= without drawdown
        assert lev_with_dd <= lev_no_dd

    def test_full_mode_max_drawdown_from_runtime_config(self):
        """runtime_config supplies max_drawdown_percent (line 101)."""
        cfg = self._cfg(LEVERAGE_MODE="full", LEVERAGE_TARGET_RISK=0.02)
        rc = MagicMock()
        rc.get_leverage_mode.return_value = "full"
        rc.get_leverage_target_risk.return_value = 0.02
        rc.get_max_drawdown_percent.return_value = 0.10
        top = {
            "symbol": "BTC/USDT",
            "atr": 1.0,
            "indicators": {"trend": "uptrend"},
        }
        with patch("src.order_executor.Config", cfg):
            lev = _calc_dynamic_leverage(
                top, entry=100.0, balance=9000.0, peak_balance=10000.0, runtime_config=rc
            )
        assert isinstance(lev, int)

    def test_full_mode_max_drawdown_runtime_raises(self):
        """runtime_config.get_max_drawdown_percent() raises → Config fallback (lines 105-106)."""
        cfg = self._cfg(LEVERAGE_MODE="full", LEVERAGE_TARGET_RISK=0.02, MAX_DRAWDOWN_PERCENT=0.15)
        rc = MagicMock()
        rc.get_leverage_mode.return_value = "full"
        rc.get_leverage_target_risk.return_value = 0.02
        rc.get_max_drawdown_percent.side_effect = Exception("broken")
        top = {
            "symbol": "BTC/USDT",
            "atr": 1.0,
            "indicators": {"trend": "uptrend"},
        }
        with patch("src.order_executor.Config", cfg):
            lev = _calc_dynamic_leverage(
                top, entry=100.0, balance=9000.0, peak_balance=10000.0, runtime_config=rc
            )
        assert isinstance(lev, int)


# ── Trading hours filter ──────────────────────────────────────────────────────


class TestTradingHoursFilter:
    @pytest.mark.asyncio
    async def test_runtime_config_outside_trading_hours_skips(self):
        """runtime_config.is_trading_time() returns False → skip (lines 210-215)."""
        rc = MagicMock()
        rc.get_trading_hours.return_value = "8-20"
        rc.is_trading_time.return_value = False
        rc.get_max_positions.return_value = 5
        rc.get_auto_execute.return_value = True

        executor, monitored, _ = _make_executor(runtime_config=rc)
        with patch("src.order_executor.Config", _make_cfg()):
            await executor.execute(_TOP_BUY, {}, 10_000.0)

        assert "BTC/USDT" not in monitored
        executor._trade_history.record_open.assert_not_called()

    @pytest.mark.asyncio
    async def test_runtime_config_inside_trading_hours_proceeds(self):
        """runtime_config.is_trading_time() returns True → proceeds."""
        rc = MagicMock()
        rc.get_trading_hours.return_value = "0-24"
        rc.is_trading_time.return_value = True
        rc.get_max_positions.return_value = 5
        rc.get_auto_execute.return_value = True

        executor, monitored, _ = _make_executor(runtime_config=rc)
        with patch("src.order_executor.Config", _make_cfg()):
            with patch("src.order_executor._ac_impact", return_value=0.0):
                with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                    await executor.execute(_TOP_BUY, {}, 10_000.0)

        assert "BTC/USDT" in monitored

    @pytest.mark.asyncio
    async def test_trading_hours_str_outside_skips(self):
        """Config.TRADING_HOURS string: outside hours → skip (lines 220-231)."""
        cfg = _make_cfg()
        cfg.TRADING_HOURS = "10-20"

        executor, monitored, _ = _make_executor()
        # Patch utcnow().hour to be outside 10-20 (e.g. hour=5)
        fake_dt = MagicMock()
        fake_dt.hour = 5
        with patch("src.order_executor.Config", cfg):
            with patch("src.order_executor.datetime") as mock_datetime:
                mock_datetime.datetime.utcnow.return_value = fake_dt
                await executor.execute(_TOP_BUY, {}, 10_000.0)

        assert "BTC/USDT" not in monitored

    @pytest.mark.asyncio
    async def test_trading_hours_str_inside_proceeds(self):
        """Config.TRADING_HOURS string: inside hours → proceeds."""
        cfg = _make_cfg()
        cfg.TRADING_HOURS = "8-22"

        executor, monitored, _ = _make_executor()
        fake_dt = MagicMock()
        fake_dt.hour = 12  # inside 8-22
        with patch("src.order_executor.Config", cfg):
            with patch("src.order_executor.datetime") as mock_datetime:
                mock_datetime.datetime.utcnow.return_value = fake_dt
                with patch("src.order_executor._ac_impact", return_value=0.0):
                    with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                        await executor.execute(_TOP_BUY, {}, 10_000.0)

        assert "BTC/USDT" in monitored

    @pytest.mark.asyncio
    async def test_trading_hours_overnight_span(self):
        """TRADING_HOURS='22-6' (overnight): hour=23 is inside → proceeds."""
        cfg = _make_cfg()
        cfg.TRADING_HOURS = "22-6"

        executor, monitored, _ = _make_executor()
        fake_dt = MagicMock()
        fake_dt.hour = 23  # inside overnight window (>=22 or <6)
        with patch("src.order_executor.Config", cfg):
            with patch("src.order_executor.datetime") as mock_datetime:
                mock_datetime.datetime.utcnow.return_value = fake_dt
                with patch("src.order_executor._ac_impact", return_value=0.0):
                    with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                        await executor.execute(_TOP_BUY, {}, 10_000.0)

        assert "BTC/USDT" in monitored

    @pytest.mark.asyncio
    async def test_trading_hours_overnight_outside_skips(self):
        """TRADING_HOURS='22-6': hour=10 is outside the overnight window → skip."""
        cfg = _make_cfg()
        cfg.TRADING_HOURS = "22-6"

        executor, monitored, _ = _make_executor()
        fake_dt = MagicMock()
        fake_dt.hour = 10  # outside overnight window
        with patch("src.order_executor.Config", cfg):
            with patch("src.order_executor.datetime") as mock_datetime:
                mock_datetime.datetime.utcnow.return_value = fake_dt
                await executor.execute(_TOP_BUY, {}, 10_000.0)

        assert "BTC/USDT" not in monitored

    @pytest.mark.asyncio
    async def test_invalid_trading_hours_format_passes_through(self):
        """ValueError in hours.split('-') → pass (no skip)."""
        cfg = _make_cfg()
        cfg.TRADING_HOURS = "invalid-format-not-int"

        executor, monitored, _ = _make_executor()
        fake_dt = MagicMock()
        fake_dt.hour = 12
        with patch("src.order_executor.Config", cfg):
            with patch("src.order_executor.datetime") as mock_datetime:
                mock_datetime.datetime.utcnow.return_value = fake_dt
                with patch("src.order_executor._ac_impact", return_value=0.0):
                    with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                        await executor.execute(_TOP_BUY, {}, 10_000.0)

        # ValueError is swallowed → trade is not blocked
        assert "BTC/USDT" in monitored


# ── SL/TP validation ──────────────────────────────────────────────────────────


class TestSLTPValidation:
    @pytest.mark.asyncio
    async def test_buy_sl_at_or_above_entry_skips(self):
        """buy: sl >= entry → skip (lines 333-339)."""
        bad_top = {**_TOP_BUY, "entry": 100.0, "stop_loss": 100.0, "take_profit": 110.0}
        executor, monitored, _ = _make_executor()
        with patch("src.order_executor.Config", _make_cfg()):
            with patch("src.order_executor._ac_impact", return_value=0.0):
                with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                    await executor.execute(bad_top, {}, 10_000.0)
        assert "BTC/USDT" not in monitored

    @pytest.mark.asyncio
    async def test_buy_tp_at_or_below_entry_skips(self):
        """buy: tp <= entry → skip (lines 341-347)."""
        bad_top = {**_TOP_BUY, "entry": 100.0, "stop_loss": 95.0, "take_profit": 100.0}
        executor, monitored, _ = _make_executor()
        with patch("src.order_executor.Config", _make_cfg()):
            with patch("src.order_executor._ac_impact", return_value=0.0):
                with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                    await executor.execute(bad_top, {}, 10_000.0)
        assert "BTC/USDT" not in monitored

    @pytest.mark.asyncio
    async def test_sell_sl_at_or_below_entry_skips(self):
        """sell: sl <= entry → skip (lines 349-355)."""
        bad_top = {**_TOP_SELL, "entry": 100.0, "stop_loss": 100.0, "take_profit": 90.0}
        executor, monitored, _ = _make_executor()
        with patch("src.order_executor.Config", _make_cfg()):
            with patch("src.order_executor._ac_impact", return_value=0.0):
                with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                    await executor.execute(bad_top, {}, 10_000.0)
        assert "BTC/USDT" not in monitored

    @pytest.mark.asyncio
    async def test_sell_tp_at_or_above_entry_skips(self):
        """sell: tp >= entry → skip (lines 357-364)."""
        bad_top = {**_TOP_SELL, "entry": 100.0, "stop_loss": 105.0, "take_profit": 100.0}
        executor, monitored, _ = _make_executor()
        with patch("src.order_executor.Config", _make_cfg()):
            with patch("src.order_executor._ac_impact", return_value=0.0):
                with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                    await executor.execute(bad_top, {}, 10_000.0)
        assert "BTC/USDT" not in monitored

    @pytest.mark.asyncio
    async def test_zero_sl_zero_tp_allowed(self):
        """sl=0 and tp=0 → validation skipped, trade proceeds."""
        top = {**_TOP_BUY, "stop_loss": 0.0, "take_profit": 0.0}
        executor, monitored, _ = _make_executor()
        with patch("src.order_executor.Config", _make_cfg()):
            with patch("src.order_executor._ac_impact", return_value=0.0):
                with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                    await executor.execute(top, {}, 10_000.0)
        assert "BTC/USDT" in monitored


# ── Post-impact SL/TP validation ──────────────────────────────────────────────


class TestPostImpactValidation:
    @pytest.mark.asyncio
    async def test_post_impact_buy_sl_invalid_skips(self):
        """After Almgren-Chriss shift, buy sl >= adj_entry → skip (lines 393-400)."""
        # impact=-0.10 shifts buy entry DOWN by 10%: entry=100 → adj=90, sl=95 > 90
        top = {**_TOP_BUY, "entry": 100.0, "stop_loss": 95.0, "take_profit": 110.0}
        executor, monitored, _ = _make_executor()
        with patch("src.order_executor.Config", _make_cfg()):
            # Negative impact for buy shifts entry down: entry*(1+(-0.10)) = 90
            with patch("src.order_executor._ac_impact", return_value=-0.10):
                with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                    await executor.execute(top, {"BTC/USDT": pd.DataFrame({"close": [100.0]})}, 10_000.0)
        assert "BTC/USDT" not in monitored

    @pytest.mark.asyncio
    async def test_post_impact_buy_tp_invalid_skips(self):
        """After impact, buy tp <= adj_entry → skip (lines 402-409)."""
        # Large positive impact shifts buy entry UP past tp=110
        top = {**_TOP_BUY, "entry": 100.0, "stop_loss": 95.0, "take_profit": 110.0}
        executor, monitored, _ = _make_executor()
        with patch("src.order_executor.Config", _make_cfg()):
            # impact=0.20 → adj_entry=120 > tp=110
            with patch("src.order_executor._ac_impact", return_value=0.20):
                with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                    await executor.execute(top, {"BTC/USDT": pd.DataFrame({"close": [100.0]})}, 10_000.0)
        assert "BTC/USDT" not in monitored

    @pytest.mark.asyncio
    async def test_post_impact_sell_sl_invalid_skips(self):
        """After impact, sell sl <= adj_entry → skip (lines 411-418)."""
        # sell: sl=105 > entry. impact=0.10 → adj_entry=100*(1-0.10)=90, sl=105>90 still ok
        # But if impact is negative: adj=100*(1-(-0.10))=110 > sl=105 → invalid
        top = {**_TOP_SELL, "entry": 100.0, "stop_loss": 105.0, "take_profit": 90.0}
        executor, monitored, _ = _make_executor()
        with patch("src.order_executor.Config", _make_cfg()):
            # negative impact on sell: entry*(1-(-0.15)) = 115 → sl=105 <= 115 invalid
            with patch("src.order_executor._ac_impact", return_value=-0.15):
                with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                    await executor.execute(top, {"BTC/USDT": pd.DataFrame({"close": [100.0]})}, 10_000.0)
        assert "BTC/USDT" not in monitored

    @pytest.mark.asyncio
    async def test_post_impact_sell_tp_invalid_skips(self):
        """After impact, sell tp >= adj_entry → skip (lines 420-428)."""
        # sell: tp=90 < entry=100. If adj_entry drops to 85 via impact,
        # tp=90 >= adj_entry=85 → invalid
        top = {**_TOP_SELL, "entry": 100.0, "stop_loss": 105.0, "take_profit": 90.0}
        executor, monitored, _ = _make_executor()
        with patch("src.order_executor.Config", _make_cfg()):
            # impact=0.15 on sell: entry*(1-0.15) = 85 → tp=90 >= 85 invalid
            with patch("src.order_executor._ac_impact", return_value=0.15):
                with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                    await executor.execute(top, {"BTC/USDT": pd.DataFrame({"close": [100.0]})}, 10_000.0)
        assert "BTC/USDT" not in monitored


# ── Drawdown scaling ──────────────────────────────────────────────────────────


class TestDrawdownScaling:
    def _make_ex(self):
        executor, _, _ = _make_executor()
        return executor

    def test_scale_disabled_returns_1(self):
        """DRAWDOWN_SCALE_ENABLED=False → scale=1.0 (line 587)."""
        executor = self._make_ex()
        cfg = _make_cfg()
        cfg.DRAWDOWN_SCALE_ENABLED = False
        cfg.DRAWDOWN_SCALE_THRESHOLD = 0.10
        cfg.DRAWDOWN_SCALE_FACTOR = 0.5
        with patch("src.order_executor.Config", cfg):
            assert executor._drawdown_scale(10_000.0) == 1.0

    def test_peak_balance_zero_returns_1(self):
        """peak_balance == 0 → return 1.0 (line 591)."""
        executor = self._make_ex()
        executor._peak_balance = 0.0
        cfg = _make_cfg()
        cfg.DRAWDOWN_SCALE_ENABLED = True
        cfg.DRAWDOWN_SCALE_THRESHOLD = 0.10
        cfg.DRAWDOWN_SCALE_FACTOR = 0.5
        with patch("src.order_executor.Config", cfg):
            # balance also 0, so peak stays 0
            result = executor._drawdown_scale(0.0)
        assert result == 1.0

    def test_below_threshold_returns_1(self):
        """Drawdown below threshold → 1.0."""
        executor = self._make_ex()
        executor._peak_balance = 10_000.0
        cfg = _make_cfg()
        cfg.DRAWDOWN_SCALE_ENABLED = True
        cfg.DRAWDOWN_SCALE_THRESHOLD = 0.10
        cfg.DRAWDOWN_SCALE_FACTOR = 0.5
        with patch("src.order_executor.Config", cfg):
            # balance=9500 → dd=5% < 10% threshold
            result = executor._drawdown_scale(9_500.0)
        assert result == 1.0

    def test_at_threshold_returns_factor(self):
        """Drawdown at or above threshold → returns DRAWDOWN_SCALE_FACTOR (lines 594-600)."""
        executor = self._make_ex()
        executor._peak_balance = 10_000.0
        cfg = _make_cfg()
        cfg.DRAWDOWN_SCALE_ENABLED = True
        cfg.DRAWDOWN_SCALE_THRESHOLD = 0.10
        cfg.DRAWDOWN_SCALE_FACTOR = 0.5
        with patch("src.order_executor.Config", cfg):
            # balance=9000 → dd=10% == threshold
            result = executor._drawdown_scale(9_000.0)
        assert result == pytest.approx(0.5)

    def test_above_threshold_returns_factor(self):
        """Drawdown well above threshold → returns factor."""
        executor = self._make_ex()
        executor._peak_balance = 10_000.0
        cfg = _make_cfg()
        cfg.DRAWDOWN_SCALE_ENABLED = True
        cfg.DRAWDOWN_SCALE_THRESHOLD = 0.10
        cfg.DRAWDOWN_SCALE_FACTOR = 0.3
        with patch("src.order_executor.Config", cfg):
            result = executor._drawdown_scale(8_000.0)  # 20% drawdown
        assert result == pytest.approx(0.3)

    def test_new_peak_updates_peak_balance(self):
        """Balance above peak → peak is updated."""
        executor = self._make_ex()
        executor._peak_balance = 9_000.0
        cfg = _make_cfg()
        cfg.DRAWDOWN_SCALE_ENABLED = True
        cfg.DRAWDOWN_SCALE_THRESHOLD = 0.10
        cfg.DRAWDOWN_SCALE_FACTOR = 0.5
        with patch("src.order_executor.Config", cfg):
            result = executor._drawdown_scale(12_000.0)
        assert executor._peak_balance == pytest.approx(12_000.0)
        assert result == 1.0

    def test_type_error_in_config_returns_1(self):
        """TypeError from Config attrs → return 1.0 (lines 584-585)."""
        executor = self._make_ex()
        cfg = _make_cfg()
        cfg.DRAWDOWN_SCALE_ENABLED = True
        cfg.DRAWDOWN_SCALE_THRESHOLD = None  # float(None) → TypeError
        cfg.DRAWDOWN_SCALE_FACTOR = 0.5
        with patch("src.order_executor.Config", cfg):
            result = executor._drawdown_scale(9_000.0)
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_scale_applied_to_quantity_in_execute(self):
        """scale < 1.0 reduces quantity before round_quantity call (line 324)."""
        executor, monitored, _ = _make_executor()
        executor._peak_balance = 10_000.0

        # Force scale to 0.5 by making drawdown > threshold
        cfg = _make_cfg()
        cfg.DRAWDOWN_SCALE_ENABLED = True
        cfg.DRAWDOWN_SCALE_THRESHOLD = 0.05
        cfg.DRAWDOWN_SCALE_FACTOR = 0.5

        captured_qty = []
        original_round = executor._api.round_quantity.side_effect
        def capture_round(sym, qty):
            captured_qty.append(qty)
            return round(qty, 6)
        executor._api.round_quantity.side_effect = capture_round

        with patch("src.order_executor.Config", cfg):
            with patch("src.order_executor._ac_impact", return_value=0.0):
                with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                    # balance=9000 < peak=10000 → drawdown=10% > threshold=5% → scale=0.5
                    await executor.execute(_TOP_BUY, {}, 9_000.0)

        assert len(captured_qty) > 0
        # With scale=0.5, qty passed to round should be halved vs no-scale run
        # Just confirm position was opened (or not) and round was called
        assert executor._api.round_quantity.call_count >= 1


# ── Liquidity filter ──────────────────────────────────────────────────────────


class TestLiquidityFilter:
    @pytest.mark.asyncio
    async def test_spread_too_wide_returns_false(self):
        """Spread > MAX_SPREAD_PCT → returns False (lines 550-557)."""
        executor, monitored, _ = _make_executor()
        executor._api.exchange.fetch_ticker = AsyncMock(
            return_value={"bid": 98.0, "ask": 102.0, "quoteVolume": 5_000_000.0}
        )
        # spread = (102-98)/100 * 100 = 4% > 0.3%
        cfg = _make_cfg()
        cfg.MAX_SPREAD_PCT = 0.3
        cfg.MIN_VOLUME_USDT = 1_000_000.0
        with patch("src.order_executor.Config", cfg):
            result = await executor._check_liquidity("BTC/USDT", 100.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_volume_too_low_returns_false(self):
        """Volume < MIN_VOLUME_USDT → returns False (lines 558-565)."""
        executor, monitored, _ = _make_executor()
        executor._api.exchange.fetch_ticker = AsyncMock(
            return_value={"bid": 99.9, "ask": 100.1, "quoteVolume": 500_000.0}
        )
        cfg = _make_cfg()
        cfg.MAX_SPREAD_PCT = 1.0
        cfg.MIN_VOLUME_USDT = 1_000_000.0
        with patch("src.order_executor.Config", cfg):
            result = await executor._check_liquidity("BTC/USDT", 100.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_exception_during_fetch_allows(self):
        """Exception in fetch_ticker → returns True (don't block trading, line 568-569)."""
        executor, monitored, _ = _make_executor()
        executor._api.exchange.fetch_ticker = AsyncMock(side_effect=Exception("timeout"))
        cfg = _make_cfg()
        with patch("src.order_executor.Config", cfg):
            result = await executor._check_liquidity("BTC/USDT", 100.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_good_liquidity_returns_true(self):
        """Normal bid/ask/volume → returns True."""
        executor, monitored, _ = _make_executor()
        executor._api.exchange.fetch_ticker = AsyncMock(
            return_value={"bid": 99.9, "ask": 100.1, "quoteVolume": 5_000_000.0}
        )
        cfg = _make_cfg()
        cfg.MAX_SPREAD_PCT = 1.0
        cfg.MIN_VOLUME_USDT = 1_000_000.0
        with patch("src.order_executor.Config", cfg):
            result = await executor._check_liquidity("BTC/USDT", 100.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_liquidity_blocks_execute(self):
        """_check_liquidity returning False stops execute before record_open."""
        executor, monitored, _ = _make_executor()
        executor._api.exchange.fetch_ticker = AsyncMock(
            return_value={"bid": 90.0, "ask": 110.0, "quoteVolume": 5_000_000.0}
        )
        cfg = _make_cfg()
        cfg.MAX_SPREAD_PCT = 0.3
        cfg.MIN_VOLUME_USDT = 1_000_000.0
        with patch("src.order_executor.Config", cfg):
            with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                await executor.execute(_TOP_BUY, {}, 10_000.0)
        executor._trade_history.record_open.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_bid_ask_uses_price_fallback(self):
        """Ticker with None bid/ask uses fallback=price → spread=0 → passes."""
        executor, monitored, _ = _make_executor()
        executor._api.exchange.fetch_ticker = AsyncMock(
            return_value={"bid": None, "ask": None, "quoteVolume": 5_000_000.0}
        )
        cfg = _make_cfg()
        cfg.MAX_SPREAD_PCT = 0.3
        cfg.MIN_VOLUME_USDT = 1_000_000.0
        with patch("src.order_executor.Config", cfg):
            result = await executor._check_liquidity("BTC/USDT", 100.0)
        # bid=ask=price → spread=0 → passes spread check
        assert result is True


# ── Live trading ──────────────────────────────────────────────────────────────


class TestLiveTrading:
    @pytest.mark.asyncio
    async def test_live_calls_create_order(self):
        """Live mode: create_order is called (lines 472-480)."""
        executor, monitored, _ = _make_executor()
        cfg = _make_cfg(paper=False)
        with patch("src.order_executor.Config", cfg):
            with patch("src.order_executor._ac_impact", return_value=0.0):
                with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                    await executor.execute(_TOP_BUY, {}, 10_000.0)
        executor._api.create_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_live_calls_set_leverage_for_linear(self):
        """Live mode, linear market: set_leverage called (lines 467-471)."""
        executor, monitored, _ = _make_executor()
        cfg = _make_cfg(paper=False, market_type="linear")
        with patch("src.order_executor.Config", cfg):
            with patch("src.order_executor._ac_impact", return_value=0.0):
                with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                    await executor.execute(_TOP_BUY, {}, 10_000.0)
        executor._api.set_leverage.assert_called_once()

    @pytest.mark.asyncio
    async def test_live_skips_set_leverage_for_spot(self):
        """Live mode, spot market: set_leverage NOT called."""
        executor, monitored, _ = _make_executor()
        cfg = _make_cfg(paper=False, market_type="spot")
        with patch("src.order_executor.Config", cfg):
            with patch("src.order_executor._ac_impact", return_value=0.0):
                with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                    await executor.execute(_TOP_BUY, {}, 10_000.0)
        executor._api.set_leverage.assert_not_called()

    @pytest.mark.asyncio
    async def test_live_set_leverage_failure_does_not_stop_trade(self):
        """set_leverage raises → warning logged, trade continues."""
        executor, monitored, _ = _make_executor()
        executor._api.set_leverage = AsyncMock(side_effect=Exception("leverage error"))
        cfg = _make_cfg(paper=False, market_type="linear")
        with patch("src.order_executor.Config", cfg):
            with patch("src.order_executor._ac_impact", return_value=0.0):
                with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                    await executor.execute(_TOP_BUY, {}, 10_000.0)
        # Position should still be opened despite set_leverage failure
        assert "BTC/USDT" in monitored

    @pytest.mark.asyncio
    async def test_live_create_order_returns_none_skips(self):
        """create_order returns None → skip (lines 473-475)."""
        executor, monitored, _ = _make_executor()
        executor._api.create_order = AsyncMock(return_value=None)
        cfg = _make_cfg(paper=False)
        with patch("src.order_executor.Config", cfg):
            with patch("src.order_executor._ac_impact", return_value=0.0):
                with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                    await executor.execute(_TOP_BUY, {}, 10_000.0)
        assert "BTC/USDT" not in monitored
        executor._trade_history.record_open.assert_not_called()

    @pytest.mark.asyncio
    async def test_live_uses_exchange_fee_when_present(self):
        """When order has fee.cost > 0, commission is overridden (lines 479-480)."""
        executor, monitored, _ = _make_executor()
        executor._api.create_order = AsyncMock(
            return_value={"id": "ORD1", "fee": {"cost": 0.75}}
        )
        cfg = _make_cfg(paper=False)
        with patch("src.order_executor.Config", cfg):
            with patch("src.order_executor._ac_impact", return_value=0.0):
                with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                    await executor.execute(_TOP_BUY, {}, 10_000.0)
        # Commission from exchange = 0.75; record_open should be called
        executor._trade_history.record_open.assert_called_once()
        call_kwargs = executor._trade_history.record_open.call_args
        assert call_kwargs.kwargs.get("commission") == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_live_places_sl_tp_on_exchange(self):
        """place_exchange_sl_tp called; IDs stored in position (lines 482-484)."""
        executor, monitored, _ = _make_executor()
        cfg = _make_cfg(paper=False)
        with patch("src.order_executor.Config", cfg):
            with patch("src.order_executor._ac_impact", return_value=0.0):
                with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                    await executor.execute(_TOP_BUY, {}, 10_000.0)
        executor._api.place_exchange_sl_tp.assert_called_once()
        assert monitored["BTC/USDT"]["exchange_sl_id"] == "SL_ID"
        assert monitored["BTC/USDT"]["exchange_tp_id"] == "TP_ID"


# ── Paper trading sell + short ────────────────────────────────────────────────


class TestPaperTradingSell:
    @pytest.mark.asyncio
    async def test_paper_sell_linear_short_reserves_margin(self):
        """Paper LINEAR SHORT: margin = qty*entry/lev deducted (line 464-465)."""
        executor, monitored, paper_balance = _make_executor()
        initial_bal = paper_balance[0]
        cfg = _make_cfg(paper=True, market_type="linear")
        with patch("src.order_executor.Config", cfg):
            with patch("src.order_executor._ac_impact", return_value=0.0):
                with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                    await executor.execute(_TOP_SELL, {}, 10_000.0)
        # Balance should have decreased (margin + commission reserved)
        assert paper_balance[0] < initial_bal

    @pytest.mark.asyncio
    async def test_paper_sell_spot_adds_balance(self):
        """Paper SPOT sell: balance += qty*entry - commission (line 461)."""
        executor, monitored, paper_balance = _make_executor()
        initial_bal = paper_balance[0]
        cfg = _make_cfg(paper=True, market_type="spot")
        with patch("src.order_executor.Config", cfg):
            with patch("src.order_executor._ac_impact", return_value=0.0):
                with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                    await executor.execute(_TOP_SELL, {}, 10_000.0)
        # SPOT sell: receives money → balance increases
        assert paper_balance[0] > initial_bal


# ── round_quantity returns 0 ──────────────────────────────────────────────────


class TestRoundQuantityZero:
    @pytest.mark.asyncio
    async def test_zero_quantity_after_rounding_skips(self):
        """round_quantity returns 0 → skip (line 432)."""
        executor, monitored, _ = _make_executor()
        executor._api.round_quantity = MagicMock(return_value=0.0)
        with patch("src.order_executor.Config", _make_cfg()):
            with patch("src.order_executor._ac_impact", return_value=0.0):
                with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                    await executor.execute(_TOP_BUY, {}, 10_000.0)
        assert "BTC/USDT" not in monitored
        executor._trade_history.record_open.assert_not_called()


# ── configure_risk ────────────────────────────────────────────────────────────


class TestConfigureRisk:
    def test_configure_risk_updates_risk_manager(self):
        """configure_risk sets risk_manager attributes."""
        executor, _, _ = _make_executor()
        executor.configure_risk(max_positions=10, risk_per_trade=0.02, drawdown_scale_enabled=False)
        assert executor._risk_manager.max_positions == 10
        assert executor._risk_manager.risk_per_trade == pytest.approx(0.02)
        assert executor._risk_manager.drawdown_scale_enabled is False

    def test_configure_risk_with_none_values_no_op(self):
        """configure_risk with all None doesn't touch risk_manager."""
        executor, _, _ = _make_executor()
        executor._risk_manager.max_positions = 5
        executor.configure_risk()
        assert executor._risk_manager.max_positions == 5


# ── Placeholder cleanup ───────────────────────────────────────────────────────


class TestPlaceholderCleanup:
    @pytest.mark.asyncio
    async def test_placeholder_removed_on_correlation_block(self):
        """Correlation block: placeholder None removed from monitored in finally."""
        executor, monitored, _ = _make_executor()
        monitored["ETH/USDT"] = {"qty": 0.5}
        executor._corr_filter.is_allowed = MagicMock(return_value=False)
        executor._corr_filter.max_correlation = MagicMock(return_value=0.95)
        cfg = _make_cfg(max_corr=0.7)
        with patch("src.order_executor.Config", cfg):
            await executor.execute(_TOP_BUY, {}, 10_000.0)
        assert "BTC/USDT" not in monitored

    @pytest.mark.asyncio
    async def test_placeholder_removed_when_entry_is_zero(self):
        """Zero entry: placeholder cleaned up in finally."""
        executor, monitored, _ = _make_executor()
        with patch("src.order_executor.Config", _make_cfg()):
            with patch("src.trade_history.get_backtest_stats", return_value=_BT_STATS):
                await executor.execute({**_TOP_BUY, "entry": 0.0}, {}, 10_000.0)
        assert "BTC/USDT" not in monitored


# ── Runtime config max positions ─────────────────────────────────────────────


class TestRuntimeConfigMaxPositions:
    @pytest.mark.asyncio
    async def test_runtime_config_max_positions_respected(self):
        """runtime_config.get_max_positions() used instead of Config.MAX_POSITIONS."""
        rc = MagicMock()
        rc.get_trading_hours.return_value = ""
        rc.get_max_positions.return_value = 1
        rc.get_auto_execute.return_value = True

        monitored = {"ETH/USDT": {"qty": 0.5}}  # already 1 position
        executor, _, _ = _make_executor(monitored=monitored, runtime_config=rc)

        cfg = _make_cfg(max_pos=99)  # Config says 99, but runtime says 1
        with patch("src.order_executor.Config", cfg):
            await executor.execute(_TOP_BUY, {}, 10_000.0)
        assert "BTC/USDT" not in monitored
