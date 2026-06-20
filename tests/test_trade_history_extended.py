"""
Расширенные тесты src/trade_history.py.

Покрывают строки, не охваченные test_trade_history.py:
- get_backtest_stats (lines 14-38)
- get_expected_value (lines 187-218)
- get_summary (lines 241-266)
- record_close на несуществующий trade_id (line 126-127)
- get_recent_trades / get_trade_count (все ветки)
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from src.trade_history import TradeHistory, get_backtest_stats

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db():
    """Временная SQLite БД для каждого теста."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    th = TradeHistory(db_path=path)
    yield th
    os.unlink(path)


@pytest.fixture
def backtest_json(tmp_path):
    """Создаёт временный backtest_results.json и возвращает путь к директории."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    results = {
        "results": [
            {
                "strategy": "ema_crossover",
                "win_rate": 0.65,
                "total_trades": 120,
                "expected_value": 0.012,
                "total_return_pct": 23.5,
            },
            {
                "strategy": "rsi_momentum",
                "win_rate": 0.58,
                "total_trades": 80,
                "expected_value": 0.008,
                "total_return_pct": 11.2,
            },
        ]
    }
    (data_dir / "backtest_results.json").write_text(json.dumps(results))
    return tmp_path


# ---------------------------------------------------------------------------
# get_backtest_stats
# ---------------------------------------------------------------------------


class TestGetBacktestStats:
    """Тесты функции get_backtest_stats."""

    def test_returns_correct_stats_for_known_strategy(self, backtest_json, monkeypatch):
        """get_backtest_stats возвращает правильные данные для известной стратегии."""
        monkeypatch.chdir(backtest_json)
        stats = get_backtest_stats("ema_crossover")
        assert stats["win_rate"] == pytest.approx(0.65)
        assert stats["total_trades"] == 120
        assert stats["ev"] == pytest.approx(0.012)
        assert stats["total_return_pct"] == pytest.approx(23.5)

    def test_returns_zero_dict_for_unknown_strategy(self, backtest_json, monkeypatch):
        """get_backtest_stats возвращает нулевой dict для неизвестной стратегии."""
        monkeypatch.chdir(backtest_json)
        stats = get_backtest_stats("nonexistent_strategy")
        assert stats == {
            "win_rate": 0.0,
            "total_trades": 0,
            "ev": 0.0,
            "total_return_pct": 0.0,
        }

    def test_returns_zero_dict_when_file_missing(self, tmp_path, monkeypatch):
        """get_backtest_stats возвращает нулевой dict если файл отсутствует."""
        monkeypatch.chdir(tmp_path)
        stats = get_backtest_stats("ema_crossover")
        assert stats["win_rate"] == 0.0
        assert stats["total_trades"] == 0

    def test_returns_zero_dict_on_invalid_json(self, tmp_path, monkeypatch):
        """get_backtest_stats обрабатывает повреждённый JSON без исключений."""
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "backtest_results.json").write_text("not valid json{{{")
        stats = get_backtest_stats("ema_crossover")
        assert stats["win_rate"] == 0.0

    def test_result_has_all_required_keys(self, backtest_json, monkeypatch):
        """get_backtest_stats всегда возвращает dict с нужными ключами."""
        monkeypatch.chdir(backtest_json)
        stats = get_backtest_stats("rsi_momentum")
        required_keys = {"win_rate", "total_trades", "ev", "total_return_pct"}
        assert required_keys.issubset(stats.keys())


# ---------------------------------------------------------------------------
# record_open / record_close
# ---------------------------------------------------------------------------


class TestRecordOpenClose:
    """Тесты открытия и закрытия сделок."""

    @pytest.mark.asyncio
    async def test_record_open_returns_positive_id(self, db):
        """record_open возвращает положительный целочисленный ID."""
        tid = await db.record_open("BTC/USDT", "ema", "buy", 50000.0, 0.01, 0.8)
        assert isinstance(tid, int)
        assert tid > 0

    @pytest.mark.asyncio
    async def test_record_close_computes_pnl_for_sell(self, db):
        """record_close корректно рассчитывает PnL для sell-позиции."""
        tid = await db.record_open("BTC/USDT", "rsi", "sell", 50000.0, 0.01, 0.75)
        await db.record_close(tid, exit_price=48000.0)
        summary = await db.get_summary()
        assert summary["total_pnl"] > 0  # продали дорого, купили дешевле

    @pytest.mark.asyncio
    async def test_record_close_nonexistent_id_does_not_raise(self, db):
        """record_close на несуществующий ID не бросает исключение."""
        await db.record_close(99999, exit_price=50000.0)  # не должно упасть

    @pytest.mark.asyncio
    async def test_commission_accumulates_on_close(self, db):
        """Комиссия при закрытии суммируется с entry-комиссией."""
        tid = await db.record_open(
            "ETH/USDT", "macd", "buy", 3000.0, 0.1, 0.8, commission=0.5
        )
        await db.record_close(tid, exit_price=3100.0, commission=0.5)
        summary = await db.get_summary()
        assert summary["total_commissions"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_multiple_opens_increment_ids(self, db):
        """Несколько record_open возвращают разные ID."""
        ids = []
        for price in [50000, 51000, 52000]:
            tid = await db.record_open("BTC/USDT", "ema", "buy", price, 0.01, 0.8)
            ids.append(tid)
        assert len(set(ids)) == 3


# ---------------------------------------------------------------------------
# get_win_rate
# ---------------------------------------------------------------------------


class TestGetWinRateExtended:
    """Расширенные тесты get_win_rate."""

    @pytest.mark.asyncio
    async def test_win_rate_filtered_by_strategy(self, db):
        """win_rate фильтруется по имени стратегии."""
        # 2 прибыльные сделки ema
        for price_out in [61000, 62000]:
            tid = await db.record_open("BTC/USDT", "ema", "buy", 60000.0, 0.01, 0.8)
            await db.record_close(tid, exit_price=price_out)
        # 1 убыточная rsi
        tid = await db.record_open("BTC/USDT", "rsi", "buy", 60000.0, 0.01, 0.8)
        await db.record_close(tid, exit_price=59000.0)

        ema_wr = await db.get_win_rate("ema")
        rsi_wr = await db.get_win_rate("rsi")
        assert ema_wr == pytest.approx(1.0)
        assert rsi_wr == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_win_rate_all_strategies(self, db):
        """win_rate без фильтра охватывает все стратегии."""
        for strat, price_out in [("ema", 61000), ("rsi", 59000)]:
            tid = await db.record_open("BTC/USDT", strat, "buy", 60000.0, 0.01, 0.8)
            await db.record_close(tid, exit_price=price_out)
        wr = await db.get_win_rate()
        assert wr == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_win_rate_respects_lookback(self, db):
        """Параметр lookback ограничивает число учитываемых сделок."""
        for price_out in [61000, 62000, 63000, 59000, 59000]:
            tid = await db.record_open("BTC/USDT", "ema", "buy", 60000.0, 0.01, 0.8)
            await db.record_close(tid, exit_price=price_out)
        wr_all = await db.get_win_rate(lookback=5)
        wr_two = await db.get_win_rate(lookback=2)
        # Последние 2 — оба убыточные
        assert wr_two == pytest.approx(0.0)
        assert wr_all == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# get_expected_value
# ---------------------------------------------------------------------------


class TestGetExpectedValue:
    """Тесты расчёта ожидаемого значения (EV)."""

    @pytest.mark.asyncio
    async def test_ev_zero_when_no_trades(self, db):
        """EV = 0.0 при отсутствии закрытых сделок."""
        ev = await db.get_expected_value()
        assert ev == 0.0

    @pytest.mark.asyncio
    async def test_ev_positive_for_profitable_strategy(self, db):
        """EV > 0 когда большинство сделок прибыльные."""
        # 3 прибыльные, 1 убыточная
        for price_out in [61000, 62000, 63000]:
            tid = await db.record_open("BTC/USDT", "ema", "buy", 60000.0, 0.01, 0.8)
            await db.record_close(tid, exit_price=price_out)
        tid = await db.record_open("BTC/USDT", "ema", "buy", 60000.0, 0.01, 0.8)
        await db.record_close(tid, exit_price=59000.0)

        ev = await db.get_expected_value()
        assert ev > 0

    @pytest.mark.asyncio
    async def test_ev_negative_for_losing_strategy(self, db):
        """EV < 0 когда большинство сделок убыточные."""
        for price_out in [59000, 58000, 57000]:
            tid = await db.record_open("BTC/USDT", "ema", "buy", 60000.0, 0.01, 0.8)
            await db.record_close(tid, exit_price=price_out)
        tid = await db.record_open("BTC/USDT", "ema", "buy", 60000.0, 0.01, 0.8)
        await db.record_close(tid, exit_price=61000.0)

        ev = await db.get_expected_value()
        assert ev < 0

    @pytest.mark.asyncio
    async def test_ev_filtered_by_strategy(self, db):
        """get_expected_value поддерживает фильтрацию по стратегии."""
        tid = await db.record_open("BTC/USDT", "ema", "buy", 60000.0, 0.01, 0.8)
        await db.record_close(tid, exit_price=65000.0)
        ev = await db.get_expected_value(strategy="ema")
        assert isinstance(ev, float)


# ---------------------------------------------------------------------------
# get_summary
# ---------------------------------------------------------------------------


class TestGetSummary:
    """Тесты агрегированной статистики get_summary."""

    @pytest.mark.asyncio
    async def test_summary_empty_db(self, db):
        """get_summary возвращает нули для пустой БД."""
        s = await db.get_summary()
        assert s["total_trades"] == 0
        assert s["closed_trades"] == 0
        assert s["win_rate"] == 0.0
        assert s["total_pnl"] == 0.0

    @pytest.mark.asyncio
    async def test_summary_with_open_and_closed(self, db):
        """get_summary корректно считает открытые и закрытые сделки."""
        # Открытая
        await db.record_open("BTC/USDT", "ema", "buy", 60000.0, 0.01, 0.8)
        # Закрытая прибыльная
        tid = await db.record_open("ETH/USDT", "rsi", "buy", 3000.0, 0.1, 0.75)
        await db.record_close(tid, exit_price=3200.0)

        s = await db.get_summary()
        assert s["total_trades"] == 2
        assert s["closed_trades"] == 1
        assert s["win_rate"] == pytest.approx(1.0)
        assert s["total_pnl"] > 0

    @pytest.mark.asyncio
    async def test_summary_has_required_keys(self, db):
        """get_summary всегда содержит нужные ключи."""
        s = await db.get_summary()
        required = {
            "total_trades",
            "closed_trades",
            "win_rate",
            "total_pnl",
            "total_commissions",
        }
        assert required.issubset(s.keys())

    @pytest.mark.asyncio
    async def test_summary_total_commissions(self, db):
        """get_summary суммирует все комиссии."""
        tid = await db.record_open(
            "BTC/USDT", "ema", "buy", 60000.0, 0.01, 0.8, commission=1.0
        )
        await db.record_close(tid, exit_price=61000.0, commission=1.0)
        s = await db.get_summary()
        assert s["total_commissions"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# get_trade_count (расширенные случаи)
# ---------------------------------------------------------------------------


class TestGetTradeCountExtended:
    """Расширенные тесты get_trade_count."""

    @pytest.mark.asyncio
    async def test_trade_count_zero_for_new_db(self, db):
        """Новая БД → trade_count = 0."""
        assert await db.get_trade_count() == 0

    @pytest.mark.asyncio
    async def test_trade_count_only_counts_closed(self, db):
        """get_trade_count учитывает только закрытые сделки."""
        await db.record_open("BTC/USDT", "ema", "buy", 60000.0, 0.01, 0.8)
        assert await db.get_trade_count() == 0

        tid = await db.record_open("BTC/USDT", "ema", "buy", 60000.0, 0.01, 0.8)
        await db.record_close(tid, exit_price=61000.0)
        assert await db.get_trade_count() == 1

    @pytest.mark.asyncio
    async def test_trade_count_respects_lookback(self, db):
        """Параметр lookback ограничивает количество учитываемых сделок."""
        for _ in range(10):
            tid = await db.record_open("BTC/USDT", "ema", "buy", 60000.0, 0.01, 0.8)
            await db.record_close(tid, exit_price=61000.0)

        assert await db.get_trade_count(lookback=5) == 5
        assert await db.get_trade_count(lookback=20) == 10
