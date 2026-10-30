"""Тесты для ai_signal_isolation_backtest.py — изоляция AI(LLM)-сигнала
от голых TA-индикаторов (роадмап п.7, follow-up)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.market.market_scanner import MarketScanner
from src.research.ai_signal_isolation_backtest import (
    _ATR_MULT,
    _RR,
    _WARMUP,
    _advance_symbol,
    _build_checkpoints,
    _entry_sl_tp,
    _force_close_tail,
    _Portfolio,
    _print_report,
    _run_ai_round,
    _save_json,
    _SymbolState,
    run_isolation_backtest,
)


def _make_df(
    n: int,
    start_ts: int = 1_700_000_000_000,
    step_ms: int = 900_000,
    price: float = 100.0,
):
    """Плоская серия цен (без волатильности) + постоянный atr — синтетика без
    необходимости гонять add_indicators()."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "timestamp": start_ts + i * step_ms,
                "open": price,
                "high": price * 1.001,
                "low": price * 0.999,
                "close": price,
                "volume": 1000.0,
                "atr": price * 0.02,
            }
        )
    return pd.DataFrame(rows)


def _trend_df(
    n: int,
    start_ts: int = 1_700_000_000_000,
    step_ms: int = 900_000,
    start_price: float = 100.0,
    pct_per_bar: float = 0.0,
):
    """Серия с управляемым трендом — для детерминированных SL/TP срабатываний."""
    rows = []
    price = start_price
    for i in range(n):
        rows.append(
            {
                "timestamp": start_ts + i * step_ms,
                "open": price,
                "high": price * 1.0005,
                "low": price * 0.9995,
                "close": price,
                "volume": 1000.0,
                "atr": start_price * 0.02,
            }
        )
        price *= 1.0 + pct_per_bar
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _entry_sl_tp
# ---------------------------------------------------------------------------


class TestEntrySlTp:
    def test_valid_buy_rec_used_as_is(self):
        sl, tp = _entry_sl_tp("buy", 100.0, atr_val=2.0, rec_sl=95.0, rec_tp=110.0)
        assert sl == 95.0
        assert tp == 110.0

    def test_valid_sell_rec_used_as_is(self):
        sl, tp = _entry_sl_tp("sell", 100.0, atr_val=2.0, rec_sl=105.0, rec_tp=90.0)
        assert sl == 105.0
        assert tp == 90.0

    def test_missing_rec_falls_back_to_atr(self):
        sl, tp = _entry_sl_tp("buy", 100.0, atr_val=2.0, rec_sl=0.0, rec_tp=0.0)
        assert sl == pytest.approx(100.0 - _ATR_MULT * 2.0)
        assert tp == pytest.approx(100.0 + _RR * _ATR_MULT * 2.0)

    def test_wrong_side_rec_falls_back_to_atr(self):
        # AI перепутал стороны: для buy stop_loss выше entry — невалидно
        sl, tp = _entry_sl_tp("buy", 100.0, atr_val=2.0, rec_sl=110.0, rec_tp=95.0)
        assert sl == pytest.approx(100.0 - _ATR_MULT * 2.0)
        assert tp == pytest.approx(100.0 + _RR * _ATR_MULT * 2.0)

    def test_zero_atr_falls_back_to_pct_of_entry(self):
        sl, tp = _entry_sl_tp("sell", 100.0, atr_val=0.0, rec_sl=0.0, rec_tp=0.0)
        assert sl == pytest.approx(102.0)
        assert tp == pytest.approx(96.0)


# ---------------------------------------------------------------------------
# _advance_symbol / _resolve_exit — SL/TP walk-forward
# ---------------------------------------------------------------------------


class TestAdvanceSymbol:
    def _open_long(self, state, entry_i, entry_price=100.0, sl=95.0, tp=110.0):
        state.open_trade = {
            "action": "buy",
            "entry_price": entry_price,
            "stop_loss": sl,
            "take_profit": tp,
            "direction": 1,
            "quantity": 1.0,
            "position_usdt": entry_price,
            "entry_comm": 0.1,
            "entry_i": entry_i,
        }
        state.next_idx = entry_i + 1

    def test_sl_hit_closes_trade_with_negative_pnl(self):
        df = _trend_df(60, start_price=100.0, pct_per_bar=-0.01)  # падает ~1%/бар
        state = _SymbolState(symbol="BTC/USDT", df=df, next_idx=_WARMUP)
        self._open_long(state, entry_i=_WARMUP, entry_price=100.0, sl=95.0, tp=200.0)
        portfolio = _Portfolio(balance=10_000.0)

        _advance_symbol(state, _WARMUP + 10, portfolio, 0.001, 0.0002, "15m")

        assert state.open_trade is None
        assert len(portfolio.closed_trades) == 1
        trade = portfolio.closed_trades[0]
        assert trade["closed_by"] == "sl"
        assert trade["pnl"] < 0
        assert portfolio.trade_returns[0] < 0

    def test_tp_hit_closes_trade_with_positive_pnl_and_maker_rate(self):
        df = _trend_df(60, start_price=100.0, pct_per_bar=0.01)  # растёт ~1%/бар
        state = _SymbolState(symbol="BTC/USDT", df=df, next_idx=_WARMUP)
        self._open_long(state, entry_i=_WARMUP, entry_price=100.0, sl=50.0, tp=110.0)
        portfolio = _Portfolio(balance=10_000.0)

        _advance_symbol(state, _WARMUP + 10, portfolio, 0.001, 0.0002, "15m")

        assert state.open_trade is None
        trade = portfolio.closed_trades[0]
        assert trade["closed_by"] == "tp"
        assert trade["pnl"] > 0

    def test_no_open_trade_just_advances_index(self):
        df = _make_df(60)
        state = _SymbolState(symbol="BTC/USDT", df=df, next_idx=_WARMUP)
        portfolio = _Portfolio(balance=10_000.0)

        _advance_symbol(state, _WARMUP + 5, portfolio, 0.001, 0.0002, "15m")

        assert state.next_idx == _WARMUP + 6
        assert portfolio.closed_trades == []

    def test_flat_price_never_hits_far_sl_tp(self):
        df = _make_df(60)
        state = _SymbolState(symbol="BTC/USDT", df=df, next_idx=_WARMUP)
        self._open_long(state, entry_i=_WARMUP, entry_price=100.0, sl=50.0, tp=200.0)
        portfolio = _Portfolio(balance=10_000.0)

        _advance_symbol(state, len(df) - 1, portfolio, 0.001, 0.0002, "15m")

        assert state.open_trade is not None
        assert portfolio.closed_trades == []


class TestForceCloseTail:
    def test_force_closes_remaining_open_trade(self):
        df = _make_df(60)
        state = _SymbolState(symbol="BTC/USDT", df=df, next_idx=_WARMUP)
        state.open_trade = {
            "action": "buy",
            "entry_price": 100.0,
            "stop_loss": 50.0,
            "take_profit": 200.0,
            "direction": 1,
            "quantity": 1.0,
            "position_usdt": 100.0,
            "entry_comm": 0.1,
            "entry_i": _WARMUP,
        }
        portfolio = _Portfolio(balance=10_000.0)

        _force_close_tail(state, portfolio, 0.001, "15m")

        assert state.open_trade is None
        assert len(portfolio.closed_trades) == 1
        assert portfolio.closed_trades[0]["closed_by"] == "force"

    def test_noop_when_already_flat(self):
        df = _make_df(60)
        state = _SymbolState(symbol="BTC/USDT", df=df, next_idx=_WARMUP)
        portfolio = _Portfolio(balance=10_000.0)

        _force_close_tail(state, portfolio, 0.001, "15m")

        assert portfolio.closed_trades == []


# ---------------------------------------------------------------------------
# _build_checkpoints
# ---------------------------------------------------------------------------


class TestBuildCheckpoints:
    def test_respects_warmup_and_no_lookahead(self):
        states = {
            "BTC/USDT": _SymbolState(
                symbol="BTC/USDT", df=_make_df(150), next_idx=_WARMUP
            ),
            "ETH/USDT": _SymbolState(
                symbol="ETH/USDT", df=_make_df(120), next_idx=_WARMUP
            ),
        }
        checkpoints = _build_checkpoints(states, n_samples=5)

        assert len(checkpoints) <= 5
        assert len(checkpoints) > 0
        for cp in checkpoints:
            for symbol, idx in cp.items():
                df = states[symbol].df
                assert idx >= _WARMUP
                assert idx + 1 < len(df)

    def test_too_short_history_returns_empty(self):
        states = {
            "BTC/USDT": _SymbolState(
                symbol="BTC/USDT", df=_make_df(_WARMUP), next_idx=_WARMUP
            ),
        }
        assert _build_checkpoints(states, n_samples=5) == []

    def test_shorter_symbol_still_produces_checkpoints(self):
        # ETH истории меньше — на последних чекпоинтах референсного (более
        # длинного) символа для ETH индекса может не хватить, но общий
        # список чекпоинтов не должен ломаться.
        states = {
            "BTC/USDT": _SymbolState(
                symbol="BTC/USDT", df=_make_df(200), next_idx=_WARMUP
            ),
            "ETH/USDT": _SymbolState(
                symbol="ETH/USDT", df=_make_df(60), next_idx=_WARMUP
            ),
        }
        checkpoints = _build_checkpoints(states, n_samples=10)
        assert len(checkpoints) > 0


# ---------------------------------------------------------------------------
# _run_ai_round
# ---------------------------------------------------------------------------


class _FakeAI:
    def __init__(self, recs, enabled=True):
        self._recs = recs
        self.enabled = enabled
        self.calls = []

    async def analyze(self, snapshots, balance):
        self.calls.append((list(snapshots), balance))
        return self._recs


@pytest.mark.asyncio
class TestRunAiRound:
    async def test_opens_trade_for_flat_eligible_symbol(self):
        btc_df = _make_df(60)
        eth_df = _make_df(60)
        states = {
            "BTC/USDT": _SymbolState(symbol="BTC/USDT", df=btc_df, next_idx=_WARMUP),
            "ETH/USDT": _SymbolState(symbol="ETH/USDT", df=eth_df, next_idx=_WARMUP),
        }
        checkpoint_idx = {"BTC/USDT": 55, "ETH/USDT": 55}
        ai = _FakeAI(
            [
                {
                    "symbol": "BTC/USDT",
                    "action": "buy",
                    "strategy": "ai",
                    "confidence": 0.8,
                    "stop_loss": 90.0,
                    "take_profit": 120.0,
                }
            ]
        )
        scanner = MarketScanner(api=None, data_loader=None)
        portfolio = _Portfolio(balance=10_000.0)

        n_sent = await _run_ai_round(
            ai, scanner, states, checkpoint_idx, portfolio, 0.001, 0.02, "15m"
        )

        assert n_sent == 2  # оба символа были "плоскими" и попали в батч
        assert states["BTC/USDT"].open_trade is not None
        assert states["ETH/USDT"].open_trade is None  # AI не дал по нему рекомендацию
        assert len(ai.calls) == 1  # ОДИН batch-вызов на цикл, не по символу
        assert portfolio.balance < 10_000.0  # зарезервирован капитал под сделку

    async def test_skips_symbol_with_open_trade(self):
        btc_df = _make_df(60)
        states = {
            "BTC/USDT": _SymbolState(symbol="BTC/USDT", df=btc_df, next_idx=_WARMUP),
        }
        states["BTC/USDT"].open_trade = {
            "action": "buy",
            "entry_price": 100.0,
            "stop_loss": 90.0,
            "take_profit": 120.0,
            "direction": 1,
            "quantity": 1.0,
            "position_usdt": 100.0,
            "entry_comm": 0.1,
            "entry_i": _WARMUP,
        }
        ai = _FakeAI([])
        scanner = MarketScanner(api=None, data_loader=None)
        portfolio = _Portfolio(balance=10_000.0)

        n_sent = await _run_ai_round(
            ai, scanner, states, {"BTC/USDT": 55}, portfolio, 0.001, 0.02, "15m"
        )

        assert n_sent == 0
        assert ai.calls == []  # батч пуст — AI вообще не вызывался

    async def test_hold_action_does_not_open_trade(self):
        states = {
            "BTC/USDT": _SymbolState(
                symbol="BTC/USDT", df=_make_df(60), next_idx=_WARMUP
            ),
        }
        ai = _FakeAI([{"symbol": "BTC/USDT", "action": "hold", "confidence": 0.9}])
        scanner = MarketScanner(api=None, data_loader=None)
        portfolio = _Portfolio(balance=10_000.0)

        await _run_ai_round(
            ai, scanner, states, {"BTC/USDT": 55}, portfolio, 0.001, 0.02, "15m"
        )

        assert states["BTC/USDT"].open_trade is None
        assert portfolio.balance == 10_000.0


# ---------------------------------------------------------------------------
# run_isolation_backtest — end-to-end с моками
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunIsolationBacktest:
    async def test_disabled_ai_returns_none(self, monkeypatch):
        async def fake_load_data(symbol, timeframe, months):
            return _make_df(150)

        monkeypatch.setattr(
            "src.research.ai_signal_isolation_backtest.load_data", fake_load_data
        )
        monkeypatch.setattr(
            "src.research.ai_signal_isolation_backtest.AIAnalyzer",
            lambda: _FakeAI([], enabled=False),
        )

        result = await run_isolation_backtest(["BTC/USDT"], "15m", 6, n_samples=3)
        assert result is None

    async def test_no_data_returns_none(self, monkeypatch):
        async def fake_load_data(symbol, timeframe, months):
            return pd.DataFrame()

        monkeypatch.setattr(
            "src.research.ai_signal_isolation_backtest.load_data", fake_load_data
        )

        result = await run_isolation_backtest(["BTC/USDT"], "15m", 6, n_samples=3)
        assert result is None

    async def test_flat_price_far_sl_tp_forces_close_at_tail(self, monkeypatch):
        """Плоская цена + AI-рекомендация с далёкими SL/TP → сделка ни разу
        не закрывается по SL/TP до конца данных, закрывается принудительно
        в хвосте (по одной на символ, closed_by='force')."""

        async def fake_load_data(symbol, timeframe, months):
            return _make_df(150)

        monkeypatch.setattr(
            "src.research.ai_signal_isolation_backtest.load_data", fake_load_data
        )
        monkeypatch.setattr(
            "src.research.ai_signal_isolation_backtest.AIAnalyzer",
            lambda: _FakeAI(
                [
                    {
                        "symbol": "BTC/USDT",
                        "action": "buy",
                        "strategy": "ai",
                        "confidence": 0.8,
                        "stop_loss": 50.0,
                        "take_profit": 200.0,
                    }
                ],
                enabled=True,
            ),
        )

        portfolio = await run_isolation_backtest(["BTC/USDT"], "15m", 6, n_samples=3)

        assert portfolio is not None
        assert len(portfolio.closed_trades) == 1
        assert portfolio.closed_trades[0]["closed_by"] == "force"


# ---------------------------------------------------------------------------
# Отчётность
# ---------------------------------------------------------------------------


class TestReporting:
    def _portfolio_with_trades(self):
        portfolio = _Portfolio(balance=10_100.0)
        portfolio.trade_returns = [
            0.05,
            -0.02,
            0.03,
            -0.01,
            0.02,
            0.04,
            -0.03,
            0.01,
            0.02,
            0.01,
        ]
        portfolio.closed_trades = [
            {
                "symbol": "BTC/USDT",
                "entry_i": 50,
                "exit_i": 55,
                "action": "buy",
                "entry_price": 100.0,
                "exit_price": 105.0,
                "pnl": 5.0,
                "return_pct": r,
                "closed_by": "tp" if r > 0 else "sl",
            }
            for r in portfolio.trade_returns
        ]
        return portfolio

    def test_print_report_no_trades_does_not_crash(self, capsys):
        _print_report(_Portfolio(balance=10_000.0), ["BTC/USDT"], "15m", 6)
        out = capsys.readouterr().out
        assert "Сделок не открыто" in out

    def test_print_report_with_trades_does_not_crash(self, capsys):
        _print_report(self._portfolio_with_trades(), ["BTC/USDT"], "15m", 6)
        out = capsys.readouterr().out
        assert "Всего сделок: 10" in out

    def test_save_json_writes_expected_fields(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        portfolio = self._portfolio_with_trades()

        _save_json(portfolio, ["BTC/USDT"], "15m", 6)

        out_path = tmp_path / "data" / "ai_isolation_backtest_results.json"
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["symbols"] == ["BTC/USDT"]
        assert data["total_trades"] == 10
        assert len(data["trade_returns"]) == 10
        assert "trades" in data
