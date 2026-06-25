"""Тесты для SignalCombiner — логика объединения сигналов SAC и AI."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import Config
from src.signal_combiner import SignalCombiner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_snapshot(symbol="BTC/USDT", price=30000.0, atr=300.0):
    return {
        "symbol": symbol,
        "price": price,
        "atr": atr,
        "volume_ratio": 1.0,
        "indicators": {"rsi": 55.0, "macd": 0.1, "bb_width": 0.04},
    }


def _make_config(**kwargs):
    """Возвращает MagicMock, имитирующий Config с нужными атрибутами."""
    cfg = MagicMock()
    cfg.MODE = kwargs.get("MODE", "ai")
    cfg.MIN_SIGNAL_CONFIDENCE = kwargs.get("MIN_SIGNAL_CONFIDENCE", 0.65)
    cfg.PAPER_TRADING = kwargs.get("PAPER_TRADING", True)
    return cfg


def make_rec(
    symbol="BTC/USDT",
    action="buy",
    confidence=0.80,
    strategy="ai",
    reasoning="test",
):
    return {
        "symbol": symbol,
        "action": action,
        "confidence": confidence,
        "strategy": strategy,
        "reasoning": reasoning,
        "entry": 30000.0,
        "stop_loss": 29500.0,
        "take_profit": 31000.0,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ai():
    ai = MagicMock()
    ai.analyze = AsyncMock(return_value=[])
    return ai


@pytest.fixture
def mock_dqn_signal():
    """Патчит SACSignal внутри signal_combiner на время каждого теста."""
    with patch("src.signal_combiner.SACSignal") as MockSAC:  # noqa: N806
        instance = MagicMock()
        instance.get_signal.return_value = {
            "action": "hold",
            "confidence": 0.0,
            "source": "sac",
        }
        MockSAC.return_value = instance
        yield instance


@pytest.fixture
def combiner(mock_ai, mock_dqn_signal):
    """SignalCombiner с замоканными AI и SAC."""
    return SignalCombiner(ai=mock_ai)


# ---------------------------------------------------------------------------
# SignalCombiner.__init__()
# ---------------------------------------------------------------------------


class TestInit:
    def test_init_stores_ai(self, mock_ai, mock_dqn_signal):
        combiner = SignalCombiner(ai=mock_ai)
        assert combiner.ai is mock_ai

    def test_init_no_rc_uses_config_mode(self, mock_ai, mock_dqn_signal, caplog):
        cfg = _make_config(MODE="hybrid")
        with patch("src.signal_combiner.Config", cfg):
            combiner = SignalCombiner(ai=mock_ai)
        assert combiner._rc is None

    def test_init_with_rc_calls_get_mode(self, mock_ai, mock_dqn_signal):
        rc = MagicMock()
        rc.get_mode.return_value = "dqn"
        combiner = SignalCombiner(ai=mock_ai, rc=rc)
        assert combiner._rc is rc
        rc.get_mode.assert_called_once()

    def test_init_sac_instantiated(self, mock_ai, mock_dqn_signal):
        combiner = SignalCombiner(ai=mock_ai)
        assert combiner.sac is mock_dqn_signal


# ---------------------------------------------------------------------------
# combine() — mode routing
# ---------------------------------------------------------------------------


async def test_local_mode_returns_empty(combiner):
    snap = make_snapshot()
    with patch.object(Config, "MODE", "local"):
        result = await combiner.combine([snap], balance=10000)
    assert result == []


async def test_ai_mode_delegates_to_ai(combiner, mock_ai):
    expected = [{"symbol": "BTC/USDT", "action": "buy", "confidence": 0.9}]
    mock_ai.analyze.return_value = expected
    snap = make_snapshot()
    with patch.object(Config, "MODE", "ai"):
        result = await combiner.combine([snap], balance=10000)
    assert result == expected
    mock_ai.analyze.assert_awaited_once()


async def test_dqn_mode_skips_hold(combiner, mock_dqn_signal):
    mock_dqn_signal.get_signal.return_value = {
        "action": "hold",
        "confidence": 0.9,
        "source": "sac",
    }
    snap = make_snapshot()
    with patch.object(Config, "MODE", "dqn"):
        result = await combiner.combine([snap], balance=10000)
    assert result == []


async def test_dqn_mode_returns_buy_above_threshold(combiner, mock_dqn_signal):
    mock_dqn_signal.get_signal.return_value = {
        "action": "buy",
        "confidence": 0.8,
        "source": "sac",
    }
    snap = make_snapshot()
    with patch.object(Config, "MODE", "dqn"):
        result = await combiner.combine([snap], balance=10000)
    assert len(result) == 1
    assert result[0]["action"] == "buy"
    assert result[0]["symbol"] == "BTC/USDT"


async def test_dqn_mode_skips_low_confidence(combiner, mock_dqn_signal):
    """conf=0.3 ниже MIN_SIGNAL_CONFIDENCE=0.65 → пустой результат."""
    mock_dqn_signal.get_signal.return_value = {
        "action": "buy",
        "confidence": 0.3,
        "source": "sac",
    }
    snap = make_snapshot()
    with patch.object(Config, "MODE", "dqn"):
        result = await combiner.combine([snap], balance=10000)
    assert result == []


async def test_unknown_mode_falls_back_to_ai(combiner, mock_ai):
    """Неизвестный MODE → fallback на AI."""
    expected = [make_rec()]
    mock_ai.analyze.return_value = expected
    snap = make_snapshot()
    with patch.object(Config, "MODE", "unknown_mode"):
        result = await combiner.combine([snap], balance=10000)
    assert result == expected
    mock_ai.analyze.assert_awaited_once()


async def test_rc_overrides_mode(mock_ai, mock_dqn_signal):
    """RuntimeConfig.get_mode() имеет приоритет над Config.MODE."""
    rc = MagicMock()
    rc.get_mode.return_value = "local"
    combiner = SignalCombiner(ai=mock_ai, rc=rc)
    snap = make_snapshot()
    result = await combiner.combine([snap], balance=10000)
    assert result == []


async def test_combine_applies_market_context_filter(combiner, mock_ai):
    """market_context передан → фильтр применяется."""
    mock_ai.analyze.return_value = [make_rec(confidence=0.70)]
    snap = make_snapshot()
    # long_overheated снижает buy conf на 0.15 → 0.70 - 0.15 = 0.55 < 0.65
    ctx = {"funding_signal": "long_overheated"}
    with patch.object(Config, "MODE", "ai"):
        result = await combiner.combine([snap], balance=10000, market_context=ctx)
    assert result == []


async def test_combine_applies_sentiment_filter(combiner, mock_ai):
    """sentiment передан → фильтр применяется."""
    mock_ai.analyze.return_value = [make_rec(confidence=0.70)]
    snap = make_snapshot()
    # score < -0.6 → buy dropped
    sentiment = {"BTC/USDT": -0.7}
    with patch.object(Config, "MODE", "ai"):
        result = await combiner.combine([snap], balance=10000, sentiment=sentiment)
    assert result == []


async def test_combine_generates_context_signals_when_no_recs(combiner, mock_ai):
    """Когда recs пустые но market_context есть — контекстные сигналы генерируются."""
    mock_ai.analyze.return_value = []
    snap = make_snapshot(symbol="BTC/USDT", price=30000.0, atr=300.0)
    ctx = {
        "BTC/USDT": {
            "funding_signal": "short_overheated",
            "funding_rate": -0.003,
        }
    }
    with patch.object(Config, "MODE", "ai"):
        result = await combiner.combine([snap], balance=10000, market_context=ctx)
    # short_overheated → contrarian buy
    assert any(r["action"] == "buy" for r in result)


async def test_combine_merges_context_with_existing_recs(combiner, mock_ai):
    """Когда recs и context_recs есть — merge происходит."""
    mock_ai.analyze.return_value = [
        make_rec(symbol="BTC/USDT", action="buy", confidence=0.80)
    ]
    snap = make_snapshot(symbol="BTC/USDT", price=30000.0, atr=300.0)
    ctx = {
        "BTC/USDT": {
            "funding_signal": "short_overheated",
            "funding_rate": -0.003,
        }
    }
    with patch.object(Config, "MODE", "ai"):
        result = await combiner.combine([snap], balance=10000, market_context=ctx)
    # Confidence должен быть увеличен или сигналы объединены
    assert len(result) >= 1


# ---------------------------------------------------------------------------
# _sl_tp()
# ---------------------------------------------------------------------------


class TestSlTp:
    def test_sl_tp_buy(self):
        sl, tp = SignalCombiner._sl_tp(price=100.0, atr=2.0, action="buy")
        assert sl < 100.0 < tp
        assert sl == pytest.approx(100.0 - 1.5 * 2.0)
        assert tp == pytest.approx(100.0 + 3.0 * 2.0)

    def test_sl_tp_sell(self):
        sl, tp = SignalCombiner._sl_tp(price=100.0, atr=2.0, action="sell")
        assert tp < 100.0 < sl
        assert sl == pytest.approx(100.0 + 1.5 * 2.0)
        assert tp == pytest.approx(100.0 - 3.0 * 2.0)

    def test_sl_tp_rounded_to_6_decimals(self):
        sl, tp = SignalCombiner._sl_tp(price=0.12345678, atr=0.001, action="buy")
        # round(x, 6) means at most 6 decimal places
        assert len(str(sl).split(".")[-1]) <= 6
        assert len(str(tp).split(".")[-1]) <= 6

    def test_sl_tp_buy_ratio_2x(self):
        """TP should be 2× the SL distance (1.5 vs 3.0)."""
        price, atr = 50000.0, 500.0
        sl, tp = SignalCombiner._sl_tp(price, atr, "buy")
        sl_dist = price - sl
        tp_dist = tp - price
        assert tp_dist == pytest.approx(sl_dist * 2, rel=1e-5)

    def test_sl_tp_sell_ratio_2x(self):
        price, atr = 50000.0, 500.0
        sl, tp = SignalCombiner._sl_tp(price, atr, "sell")
        sl_dist = sl - price
        tp_dist = price - tp
        assert tp_dist == pytest.approx(sl_dist * 2, rel=1e-5)

    def test_sl_tp_zero_atr(self):
        """atr=0 → sl==tp==price."""
        sl, tp = SignalCombiner._sl_tp(price=100.0, atr=0.0, action="buy")
        assert sl == 100.0
        assert tp == 100.0


# ---------------------------------------------------------------------------
# _sac_only()
# ---------------------------------------------------------------------------


class TestSacOnly:
    def _make_combiner(self, sac_signal, min_conf=0.65):
        mock_ai = MagicMock()
        cfg = _make_config(MIN_SIGNAL_CONFIDENCE=min_conf)
        with patch("src.signal_combiner.SACSignal") as MockSAC:
            instance = MagicMock()
            instance.get_signal.return_value = sac_signal
            MockSAC.return_value = instance
            with patch("src.signal_combiner.Config", cfg):
                return SignalCombiner(ai=mock_ai)

    def test_hold_signal_skipped(self):
        c = self._make_combiner({"action": "hold", "confidence": 0.9})
        cfg = _make_config(MIN_SIGNAL_CONFIDENCE=0.65)
        with patch("src.signal_combiner.Config", cfg):
            result = c._sac_only([make_snapshot()], balance=10000)
        assert result == []

    def test_low_confidence_skipped(self):
        c = self._make_combiner({"action": "buy", "confidence": 0.3})
        cfg = _make_config(MIN_SIGNAL_CONFIDENCE=0.65)
        with patch("src.signal_combiner.Config", cfg):
            result = c._sac_only([make_snapshot()], balance=10000)
        assert result == []

    def test_buy_signal_included(self):
        c = self._make_combiner({"action": "buy", "confidence": 0.80})
        cfg = _make_config(MIN_SIGNAL_CONFIDENCE=0.65)
        with patch("src.signal_combiner.Config", cfg):
            result = c._sac_only([make_snapshot()], balance=10000)
        assert len(result) == 1
        assert result[0]["action"] == "buy"
        assert result[0]["strategy"] == "sac"
        assert result[0]["symbol"] == "BTC/USDT"

    def test_sell_signal_included(self):
        c = self._make_combiner({"action": "sell", "confidence": 0.85})
        cfg = _make_config(MIN_SIGNAL_CONFIDENCE=0.65)
        with patch("src.signal_combiner.Config", cfg):
            result = c._sac_only([make_snapshot()], balance=10000)
        assert len(result) == 1
        assert result[0]["action"] == "sell"

    def test_sl_tp_set_correctly(self):
        c = self._make_combiner({"action": "buy", "confidence": 0.80})
        snap = make_snapshot(price=30000.0, atr=300.0)
        cfg = _make_config(MIN_SIGNAL_CONFIDENCE=0.65)
        with patch("src.signal_combiner.Config", cfg):
            result = c._sac_only([snap], balance=10000)
        assert result[0]["stop_loss"] == pytest.approx(30000.0 - 1.5 * 300.0)
        assert result[0]["take_profit"] == pytest.approx(30000.0 + 3.0 * 300.0)

    def test_missing_atr_uses_default(self):
        """Если atr отсутствует в снапшоте → используется price*0.02."""
        c = self._make_combiner({"action": "buy", "confidence": 0.80})
        snap = {"symbol": "BTC/USDT", "price": 10000.0}  # no atr
        cfg = _make_config(MIN_SIGNAL_CONFIDENCE=0.65)
        with patch("src.signal_combiner.Config", cfg):
            result = c._sac_only([snap], balance=10000)
        assert len(result) == 1
        expected_atr = 10000.0 * 0.02
        assert result[0]["stop_loss"] == pytest.approx(10000.0 - 1.5 * expected_atr)

    def test_multiple_snapshots(self):
        """Несколько снапшотов — каждый обрабатывается."""
        mock_ai = MagicMock()
        cfg = _make_config(MIN_SIGNAL_CONFIDENCE=0.65)
        call_count = 0

        def side_effect(snap, balance):
            nonlocal call_count
            call_count += 1
            if snap["symbol"] == "BTC/USDT":
                return {"action": "buy", "confidence": 0.80}
            return {"action": "hold", "confidence": 0.0}

        with patch("src.signal_combiner.SACSignal") as MockSAC:
            instance = MagicMock()
            instance.get_signal.side_effect = side_effect
            MockSAC.return_value = instance
            with patch("src.signal_combiner.Config", cfg):
                c = SignalCombiner(ai=mock_ai)

        snaps = [make_snapshot("BTC/USDT"), make_snapshot("ETH/USDT", price=2000.0)]
        with patch("src.signal_combiner.Config", cfg):
            result = c._sac_only(snaps, balance=10000)

        assert len(result) == 1
        assert result[0]["symbol"] == "BTC/USDT"

    def test_reasoning_contains_confidence(self):
        c = self._make_combiner({"action": "buy", "confidence": 0.82})
        cfg = _make_config(MIN_SIGNAL_CONFIDENCE=0.65)
        with patch("src.signal_combiner.Config", cfg):
            result = c._sac_only([make_snapshot()], balance=10000)
        assert "82%" in result[0]["reasoning"]

    def test_empty_snapshots_returns_empty(self):
        c = self._make_combiner({"action": "buy", "confidence": 0.80})
        cfg = _make_config(MIN_SIGNAL_CONFIDENCE=0.65)
        with patch("src.signal_combiner.Config", cfg):
            result = c._sac_only([], balance=10000)
        assert result == []


# ---------------------------------------------------------------------------
# _hybrid()
# ---------------------------------------------------------------------------


class TestHybrid:
    def test_hybrid_agree_buy(self, combiner, mock_ai, mock_dqn_signal):
        """И DQN, и AI говорят buy → итоговый conf = 0.4*dqn + 0.6*ai."""
        dqn_conf = 0.75
        ai_conf = 0.80
        mock_dqn_signal.get_signal.return_value = {
            "action": "buy",
            "confidence": dqn_conf,
            "source": "sac",
        }
        ai_rec = {
            "symbol": "BTC/USDT",
            "action": "buy",
            "confidence": ai_conf,
            "strategy": "momentum",
            "reasoning": "trend up",
            "entry": 30000.0,
            "stop_loss": 29500.0,
            "take_profit": 31000.0,
        }
        mock_ai.analyze.return_value = [ai_rec]
        snap = make_snapshot()

        with patch.object(Config, "MODE", "hybrid"):
            result = await_sync(combiner.combine([snap], balance=10000))

        assert len(result) == 1
        expected_conf = round(dqn_conf * 0.4 + ai_conf * 0.6, 3)
        assert result[0]["confidence"] == expected_conf
        assert "hybrid" in result[0]["strategy"]

    def test_hybrid_disagree_returns_empty(self, combiner, mock_ai, mock_dqn_signal):
        mock_dqn_signal.get_signal.return_value = {
            "action": "buy",
            "confidence": 0.85,
            "source": "sac",
        }
        ai_rec = make_rec(action="sell", confidence=0.80)
        mock_ai.analyze.return_value = [ai_rec]
        snap = make_snapshot()

        with patch.object(Config, "MODE", "hybrid"):
            result = await_sync(combiner.combine([snap], balance=10000))

        assert result == []

    def test_hybrid_ai_silent_high_dqn_conf(self, combiner, mock_ai, mock_dqn_signal):
        mock_dqn_signal.get_signal.return_value = {
            "action": "buy",
            "confidence": 0.85,
            "source": "sac",
        }
        mock_ai.analyze.return_value = []
        snap = make_snapshot()

        with patch.object(Config, "MODE", "hybrid"):
            result = await_sync(combiner.combine([snap], balance=10000))

        assert len(result) == 1
        assert result[0]["action"] == "buy"
        assert result[0]["confidence"] == 0.85

    def test_hybrid_ai_silent_low_dqn_conf(self, combiner, mock_ai, mock_dqn_signal):
        mock_dqn_signal.get_signal.return_value = {
            "action": "buy",
            "confidence": 0.5,
            "source": "sac",
        }
        mock_ai.analyze.return_value = []
        snap = make_snapshot()

        with patch.object(Config, "MODE", "hybrid"):
            result = await_sync(combiner.combine([snap], balance=10000))

        assert result == []

    def test_hybrid_hold_from_sac_skipped(self, combiner, mock_ai, mock_dqn_signal):
        mock_dqn_signal.get_signal.return_value = {
            "action": "hold",
            "confidence": 0.95,
        }
        mock_ai.analyze.return_value = [make_rec(action="buy")]
        with patch.object(Config, "MODE", "hybrid"):
            result = await_sync(combiner.combine([make_snapshot()], balance=10000))
        assert result == []

    def test_hybrid_regime_weights_trending_up(
        self, combiner, mock_ai, mock_dqn_signal
    ):
        """trending_up → w_sac=0.5, w_ai=0.5."""
        dqn_conf = 0.80
        ai_conf = 0.70
        mock_dqn_signal.get_signal.return_value = {
            "action": "buy",
            "confidence": dqn_conf,
        }
        mock_ai.analyze.return_value = [make_rec(action="buy", confidence=ai_conf)]
        snap = make_snapshot()
        with patch.object(Config, "MODE", "hybrid"):
            result = await_sync(
                combiner.combine(
                    [snap], balance=10000, regimes={"BTC/USDT": "trending_up"}
                )
            )
        expected = round(dqn_conf * 0.5 + ai_conf * 0.5, 3)
        assert result[0]["confidence"] == expected

    def test_hybrid_regime_weights_trending_down(
        self, combiner, mock_ai, mock_dqn_signal
    ):
        """trending_down → w_sac=0.3, w_ai=0.7."""
        dqn_conf = 0.80
        ai_conf = 0.70
        mock_dqn_signal.get_signal.return_value = {
            "action": "buy",
            "confidence": dqn_conf,
        }
        mock_ai.analyze.return_value = [make_rec(action="buy", confidence=ai_conf)]
        snap = make_snapshot()
        with patch.object(Config, "MODE", "hybrid"):
            result = await_sync(
                combiner.combine(
                    [snap], balance=10000, regimes={"BTC/USDT": "trending_down"}
                )
            )
        expected = round(dqn_conf * 0.3 + ai_conf * 0.7, 3)
        assert result[0]["confidence"] == expected

    def test_hybrid_combined_below_min_conf_dropped(
        self, combiner, mock_ai, mock_dqn_signal
    ):
        """combined conf ниже MIN_SIGNAL_CONFIDENCE → пропуск."""
        mock_dqn_signal.get_signal.return_value = {"action": "buy", "confidence": 0.40}
        mock_ai.analyze.return_value = [make_rec(action="buy", confidence=0.50)]
        snap = make_snapshot()
        cfg = _make_config(MIN_SIGNAL_CONFIDENCE=0.65)
        with patch("src.signal_combiner.Config", cfg):
            result = await_sync(combiner._hybrid([snap], balance=10000))
        assert result == []

    def test_hybrid_reasoning_includes_sac_and_regime(
        self, combiner, mock_ai, mock_dqn_signal
    ):
        mock_dqn_signal.get_signal.return_value = {"action": "buy", "confidence": 0.80}
        mock_ai.analyze.return_value = [
            make_rec(action="buy", confidence=0.80, reasoning="uptrend detected")
        ]
        cfg = _make_config(MIN_SIGNAL_CONFIDENCE=0.65)
        with patch("src.signal_combiner.Config", cfg):
            result = await_sync(
                combiner._hybrid([make_snapshot()], balance=10000, regime="ranging")
            )
        assert "SAC" in result[0]["reasoning"]
        assert "ranging" in result[0]["reasoning"]

    def test_hybrid_reasoning_ai_silent_format(
        self, combiner, mock_ai, mock_dqn_signal
    ):
        mock_dqn_signal.get_signal.return_value = {"action": "sell", "confidence": 0.85}
        mock_ai.analyze.return_value = []
        cfg = _make_config(MIN_SIGNAL_CONFIDENCE=0.65)
        with patch("src.signal_combiner.Config", cfg):
            result = await_sync(combiner._hybrid([make_snapshot()], balance=10000))
        assert "AI silent" in result[0]["reasoning"]

    def test_hybrid_strategy_includes_hybrid_prefix(
        self, combiner, mock_ai, mock_dqn_signal
    ):
        mock_dqn_signal.get_signal.return_value = {"action": "buy", "confidence": 0.80}
        mock_ai.analyze.return_value = [
            make_rec(action="buy", confidence=0.80, strategy="ema_cross")
        ]
        cfg = _make_config(MIN_SIGNAL_CONFIDENCE=0.65)
        with patch("src.signal_combiner.Config", cfg):
            result = await_sync(combiner._hybrid([make_snapshot()], balance=10000))
        assert result[0]["strategy"].startswith("hybrid(")
        assert "sac" in result[0]["strategy"]

    def test_hybrid_chronos_disabled_paper_trading(self, mock_ai, mock_dqn_signal):
        """PAPER_TRADING=True → Chronos не используется."""
        rc = MagicMock()
        rc.get_mode.return_value = "hybrid"
        rc.get_chronos_enabled.return_value = True
        mock_dqn_signal.get_signal.return_value = {"action": "buy", "confidence": 0.80}
        mock_ai.analyze.return_value = [make_rec(action="buy", confidence=0.80)]
        cfg = _make_config(PAPER_TRADING=True, MIN_SIGNAL_CONFIDENCE=0.65)
        with patch("src.signal_combiner.Config", cfg):
            c = SignalCombiner(ai=mock_ai, rc=rc)
            result = await_sync(c._hybrid([make_snapshot()], balance=10000))
        # Should still return result (Chronos not triggered)
        assert len(result) == 1

    def test_hybrid_chronos_disagrees_drops_signal(self, mock_ai, mock_dqn_signal):
        """Chronos disagrees → сигнал пропускается."""
        import pandas as pd

        rc = MagicMock()
        rc.get_mode.return_value = "hybrid"
        rc.get_chronos_enabled.return_value = True
        mock_dqn_signal.get_signal.return_value = {"action": "buy", "confidence": 0.80}
        mock_ai.analyze.return_value = [make_rec(action="buy", confidence=0.80)]

        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        market_data = {"BTC/USDT": df}

        cfg = _make_config(PAPER_TRADING=False, MIN_SIGNAL_CONFIDENCE=0.65)
        with patch("src.signal_combiner.Config", cfg), patch(
            "src.signal_combiner.chronos_analyzer"
        ) as mock_chronos:
            mock_chronos.predict_direction.return_value = "down"  # buy expects "up"
            c = SignalCombiner(ai=mock_ai, rc=rc)
            result = await_sync(
                c._hybrid([make_snapshot()], balance=10000, market_data=market_data)
            )

        assert result == []

    def test_hybrid_chronos_agrees_keeps_signal(self, mock_ai, mock_dqn_signal):
        """Chronos agrees → сигнал проходит."""
        import pandas as pd

        rc = MagicMock()
        rc.get_mode.return_value = "hybrid"
        rc.get_chronos_enabled.return_value = True
        mock_dqn_signal.get_signal.return_value = {"action": "buy", "confidence": 0.80}
        mock_ai.analyze.return_value = [make_rec(action="buy", confidence=0.80)]

        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        market_data = {"BTC/USDT": df}

        cfg = _make_config(PAPER_TRADING=False, MIN_SIGNAL_CONFIDENCE=0.65)
        with patch("src.signal_combiner.Config", cfg), patch(
            "src.signal_combiner.chronos_analyzer"
        ) as mock_chronos:
            mock_chronos.predict_direction.return_value = "up"  # buy expects "up"
            c = SignalCombiner(ai=mock_ai, rc=rc)
            result = await_sync(
                c._hybrid([make_snapshot()], balance=10000, market_data=market_data)
            )

        assert len(result) == 1

    def test_hybrid_chronos_neutral_keeps_signal(self, mock_ai, mock_dqn_signal):
        """Chronos neutral → сигнал проходит."""
        import pandas as pd

        rc = MagicMock()
        rc.get_mode.return_value = "hybrid"
        rc.get_chronos_enabled.return_value = True
        mock_dqn_signal.get_signal.return_value = {"action": "buy", "confidence": 0.80}
        mock_ai.analyze.return_value = [make_rec(action="buy", confidence=0.80)]

        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        market_data = {"BTC/USDT": df}

        cfg = _make_config(PAPER_TRADING=False, MIN_SIGNAL_CONFIDENCE=0.65)
        with patch("src.signal_combiner.Config", cfg), patch(
            "src.signal_combiner.chronos_analyzer"
        ) as mock_chronos:
            mock_chronos.predict_direction.return_value = "neutral"
            c = SignalCombiner(ai=mock_ai, rc=rc)
            result = await_sync(
                c._hybrid([make_snapshot()], balance=10000, market_data=market_data)
            )

        assert len(result) == 1


# ---------------------------------------------------------------------------
# _apply_market_context_filter()
# ---------------------------------------------------------------------------


class TestApplyMarketContextFilter:
    def _filter(self, recs, ctx, min_conf=0.65):
        mock_ai = MagicMock()
        cfg = _make_config(MIN_SIGNAL_CONFIDENCE=min_conf)
        with patch("src.signal_combiner.SACSignal"), patch(
            "src.signal_combiner.Config", cfg
        ):
            c = SignalCombiner(ai=mock_ai)
        with patch("src.signal_combiner.Config", cfg):
            return c._apply_market_context_filter(recs, ctx)

    # ── funding ──────────────────────────────────────────────────────────────

    def test_long_overheated_reduces_buy_conf(self):
        recs = [make_rec(action="buy", confidence=0.80)]
        result = self._filter(recs, {"funding_signal": "long_overheated"})
        assert result[0]["confidence"] == pytest.approx(0.80 - 0.15, abs=1e-3)

    def test_long_overheated_boosts_sell_conf(self):
        recs = [make_rec(action="sell", confidence=0.80)]
        result = self._filter(recs, {"funding_signal": "long_overheated"})
        assert result[0]["confidence"] == pytest.approx(
            min(0.95, 0.80 + 0.05), abs=1e-3
        )

    def test_short_overheated_reduces_sell_conf(self):
        recs = [make_rec(action="sell", confidence=0.80)]
        result = self._filter(recs, {"funding_signal": "short_overheated"})
        assert result[0]["confidence"] == pytest.approx(0.80 - 0.15, abs=1e-3)

    def test_short_overheated_boosts_buy_conf(self):
        recs = [make_rec(action="buy", confidence=0.80)]
        result = self._filter(recs, {"funding_signal": "short_overheated"})
        assert result[0]["confidence"] == pytest.approx(
            min(0.95, 0.80 + 0.05), abs=1e-3
        )

    # ── fear_greed ────────────────────────────────────────────────────────────

    def test_extreme_greed_reduces_buy(self):
        recs = [make_rec(action="buy", confidence=0.80)]
        result = self._filter(recs, {"fear_greed_signal": "extreme_greed"})
        assert result[0]["confidence"] == pytest.approx(0.80 - 0.10, abs=1e-3)

    def test_extreme_fear_reduces_sell(self):
        recs = [make_rec(action="sell", confidence=0.80)]
        result = self._filter(recs, {"fear_greed_signal": "extreme_fear"})
        assert result[0]["confidence"] == pytest.approx(0.80 - 0.10, abs=1e-3)

    def test_extreme_greed_doesnt_affect_sell(self):
        recs = [make_rec(action="sell", confidence=0.80)]
        result = self._filter(recs, {"fear_greed_signal": "extreme_greed"})
        assert result[0]["confidence"] == pytest.approx(0.80, abs=1e-3)

    # ── OI ───────────────────────────────────────────────────────────────────

    def test_oi_bearish_reduces_buy(self):
        recs = [make_rec(action="buy", confidence=0.80)]
        result = self._filter(recs, {"oi_signal": "oi_bearish"})
        assert result[0]["confidence"] == pytest.approx(0.80 - 0.08, abs=1e-3)

    def test_oi_bearish_no_effect_on_sell(self):
        recs = [make_rec(action="sell", confidence=0.80)]
        result = self._filter(recs, {"oi_signal": "oi_bearish"})
        assert result[0]["confidence"] == pytest.approx(0.80, abs=1e-3)

    # ── liquidation ──────────────────────────────────────────────────────────

    def test_long_liquidation_reduces_buy(self):
        recs = [make_rec(action="buy", confidence=0.80)]
        result = self._filter(recs, {"liquidation_pressure": "long_liquidation"})
        assert result[0]["confidence"] == pytest.approx(0.80 - 0.12, abs=1e-3)

    def test_short_squeeze_reduces_sell(self):
        recs = [make_rec(action="sell", confidence=0.80)]
        result = self._filter(recs, {"liquidation_pressure": "short_squeeze"})
        assert result[0]["confidence"] == pytest.approx(0.80 - 0.12, abs=1e-3)

    # ── basis ────────────────────────────────────────────────────────────────

    def test_greed_premium_reduces_buy(self):
        recs = [make_rec(action="buy", confidence=0.80)]
        result = self._filter(recs, {"basis_signal": "greed_premium"})
        assert result[0]["confidence"] == pytest.approx(0.80 - 0.10, abs=1e-3)

    def test_backwardation_reduces_sell(self):
        recs = [make_rec(action="sell", confidence=0.80)]
        result = self._filter(recs, {"basis_signal": "backwardation"})
        assert result[0]["confidence"] == pytest.approx(0.80 - 0.10, abs=1e-3)

    # ── google_trends ─────────────────────────────────────────────────────────

    def test_retail_fomo_reduces_buy(self):
        recs = [make_rec(action="buy", confidence=0.80)]
        result = self._filter(recs, {"google_trends_signal": "retail_fomo"})
        assert result[0]["confidence"] == pytest.approx(0.80 - 0.08, abs=1e-3)

    def test_retail_absent_reduces_sell(self):
        recs = [make_rec(action="sell", confidence=0.80)]
        result = self._filter(recs, {"google_trends_signal": "retail_absent"})
        assert result[0]["confidence"] == pytest.approx(0.80 - 0.08, abs=1e-3)

    # ── pcr ───────────────────────────────────────────────────────────────────

    def test_greed_calls_reduces_buy(self):
        recs = [make_rec(action="buy", confidence=0.80)]
        result = self._filter(recs, {"pcr_signal": "greed_calls"})
        assert result[0]["confidence"] == pytest.approx(0.80 - 0.10, abs=1e-3)

    def test_fear_puts_reduces_sell(self):
        recs = [make_rec(action="sell", confidence=0.80)]
        result = self._filter(recs, {"pcr_signal": "fear_puts"})
        assert result[0]["confidence"] == pytest.approx(0.80 - 0.10, abs=1e-3)

    # ── orderbook ─────────────────────────────────────────────────────────────

    def test_ask_dominant_reduces_buy(self):
        recs = [make_rec(action="buy", confidence=0.80)]
        result = self._filter(recs, {"ob_signal": "ask_dominant"})
        assert result[0]["confidence"] == pytest.approx(0.80 - 0.10, abs=1e-3)

    def test_bid_dominant_reduces_sell(self):
        recs = [make_rec(action="sell", confidence=0.80)]
        result = self._filter(recs, {"ob_signal": "bid_dominant"})
        assert result[0]["confidence"] == pytest.approx(0.80 - 0.10, abs=1e-3)

    # ── iv ───────────────────────────────────────────────────────────────────

    def test_put_skew_reduces_buy(self):
        recs = [make_rec(action="buy", confidence=0.80)]
        result = self._filter(recs, {"iv_signal": "put_skew"})
        assert result[0]["confidence"] == pytest.approx(0.80 - 0.08, abs=1e-3)

    def test_call_skew_reduces_sell(self):
        recs = [make_rec(action="sell", confidence=0.80)]
        result = self._filter(recs, {"iv_signal": "call_skew"})
        assert result[0]["confidence"] == pytest.approx(0.80 - 0.08, abs=1e-3)

    # ── macro ─────────────────────────────────────────────────────────────────

    def test_macro_bearish_reduces_buy(self):
        recs = [make_rec(action="buy", confidence=0.80)]
        result = self._filter(recs, {"macro_signal": "macro_bearish"})
        assert result[0]["confidence"] == pytest.approx(0.80 - 0.07, abs=1e-3)

    def test_macro_bearish_no_effect_on_sell(self):
        recs = [make_rec(action="sell", confidence=0.80)]
        result = self._filter(recs, {"macro_signal": "macro_bearish"})
        assert result[0]["confidence"] == pytest.approx(0.80, abs=1e-3)

    # ── etf ──────────────────────────────────────────────────────────────────

    def test_etf_outflow_reduces_buy(self):
        recs = [make_rec(action="buy", confidence=0.80)]
        result = self._filter(recs, {"etf_signal": "etf_outflow"})
        assert result[0]["confidence"] == pytest.approx(0.80 - 0.08, abs=1e-3)

    def test_etf_inflow_reduces_sell(self):
        recs = [make_rec(action="sell", confidence=0.80)]
        result = self._filter(recs, {"etf_signal": "etf_inflow"})
        assert result[0]["confidence"] == pytest.approx(0.80 - 0.08, abs=1e-3)

    # ── reddit ───────────────────────────────────────────────────────────────

    def test_reddit_bearish_reduces_buy(self):
        recs = [make_rec(action="buy", confidence=0.80)]
        result = self._filter(recs, {"reddit_signal": "reddit_bearish"})
        assert result[0]["confidence"] == pytest.approx(0.80 - 0.05, abs=1e-3)

    # ── stablecoin ────────────────────────────────────────────────────────────

    def test_stablecoin_outflow_reduces_buy(self):
        recs = [make_rec(action="buy", confidence=0.80)]
        result = self._filter(recs, {"stablecoin_signal": "stablecoin_outflow"})
        assert result[0]["confidence"] == pytest.approx(0.80 - 0.06, abs=1e-3)

    # ── combined drops below min_conf ─────────────────────────────────────────

    def test_cumulative_adjustments_drop_below_min(self):
        """Multiple penalties combined drop rec below threshold → excluded."""
        recs = [make_rec(action="buy", confidence=0.70)]
        ctx = {
            "funding_signal": "long_overheated",  # -0.15
            "fear_greed_signal": "extreme_greed",  # -0.10
        }
        # 0.70 - 0.15 - 0.10 = 0.45 < 0.65
        result = self._filter(recs, ctx)
        assert result == []

    def test_neutral_context_passes_through(self):
        recs = [make_rec(action="buy", confidence=0.80)]
        result = self._filter(recs, {})
        assert len(result) == 1
        assert result[0]["confidence"] == pytest.approx(0.80, abs=1e-3)

    def test_per_symbol_context(self):
        """Per-symbol map: сигнал BTC получает BTC контекст."""
        recs = [
            make_rec(symbol="BTC/USDT", action="buy", confidence=0.80),
            make_rec(symbol="ETH/USDT", action="buy", confidence=0.80),
        ]
        ctx = {
            "BTC/USDT": {"funding_signal": "long_overheated"},
            "ETH/USDT": {},
        }
        result = self._filter(recs, ctx)
        # BTC: 0.80 - 0.15 = 0.65 → passes exactly
        btc = next(r for r in result if r["symbol"] == "BTC/USDT")
        eth = next(r for r in result if r["symbol"] == "ETH/USDT")
        assert btc["confidence"] == pytest.approx(0.65, abs=1e-3)
        assert eth["confidence"] == pytest.approx(0.80, abs=1e-3)

    def test_rec_is_not_mutated(self):
        """Оригинальный dict не должен мутироваться."""
        original = make_rec(action="buy", confidence=0.80)
        original_copy = dict(original)
        self._filter([original], {"funding_signal": "long_overheated"})
        assert original == original_copy

    def test_confidence_capped_at_095(self):
        """Буст не должен превышать 0.95."""
        recs = [make_rec(action="sell", confidence=0.93)]
        result = self._filter(recs, {"funding_signal": "long_overheated"})
        assert result[0]["confidence"] <= 0.95


# ---------------------------------------------------------------------------
# _apply_sentiment_filter()
# ---------------------------------------------------------------------------


class TestApplySentimentFilter:
    def _filter(self, recs, sentiment, min_conf=0.65):
        mock_ai = MagicMock()
        cfg = _make_config(MIN_SIGNAL_CONFIDENCE=min_conf)
        with patch("src.signal_combiner.SACSignal"), patch(
            "src.signal_combiner.Config", cfg
        ):
            c = SignalCombiner(ai=mock_ai)
        with patch("src.signal_combiner.Config", cfg):
            return c._apply_sentiment_filter(recs, sentiment)

    def test_very_negative_drops_buy(self):
        """score < -0.6 → buy полностью блокируется."""
        recs = [make_rec(action="buy", confidence=0.99)]
        result = self._filter(recs, {"BTC/USDT": -0.7})
        assert result == []

    def test_very_negative_keeps_sell(self):
        """score < -0.6 не блокирует sell."""
        recs = [make_rec(action="sell", confidence=0.80)]
        result = self._filter(recs, {"BTC/USDT": -0.7})
        assert len(result) == 1

    def test_moderately_negative_reduces_buy(self):
        """score < -0.3 → buy conf -= 0.12."""
        recs = [make_rec(action="buy", confidence=0.80)]
        result = self._filter(recs, {"BTC/USDT": -0.4})
        assert result[0]["confidence"] == pytest.approx(0.80 - 0.12, abs=1e-3)

    def test_moderately_negative_below_threshold_drops(self):
        recs = [make_rec(action="buy", confidence=0.70)]
        # 0.70 - 0.12 = 0.58 < 0.65
        result = self._filter(recs, {"BTC/USDT": -0.4})
        assert result == []

    def test_positive_sell_reduces_conf(self):
        """score > 0.5 → sell conf -= 0.10."""
        recs = [make_rec(action="sell", confidence=0.80)]
        result = self._filter(recs, {"BTC/USDT": 0.6})
        assert result[0]["confidence"] == pytest.approx(0.80 - 0.10, abs=1e-3)

    def test_very_positive_buy_boosts_conf(self):
        """score > 0.7 → buy conf += 0.05, capped at 0.95."""
        recs = [make_rec(action="buy", confidence=0.80)]
        result = self._filter(recs, {"BTC/USDT": 0.8})
        assert result[0]["confidence"] == pytest.approx(
            min(0.95, 0.80 + 0.05), abs=1e-3
        )

    def test_very_positive_buy_capped_at_095(self):
        recs = [make_rec(action="buy", confidence=0.93)]
        result = self._filter(recs, {"BTC/USDT": 0.8})
        assert result[0]["confidence"] <= 0.95

    def test_neutral_sentiment_passes_through(self):
        recs = [make_rec(action="buy", confidence=0.80)]
        result = self._filter(recs, {"BTC/USDT": 0.0})
        assert result[0]["confidence"] == pytest.approx(0.80, abs=1e-3)

    def test_missing_symbol_uses_zero_score(self):
        """Символ не найден в sentiment → score=0 → не меняет."""
        recs = [make_rec(symbol="XRP/USDT", action="buy", confidence=0.80)]
        result = self._filter(recs, {"BTC/USDT": -0.8})
        assert len(result) == 1

    def test_rec_not_mutated(self):
        original = make_rec(action="buy", confidence=0.80)
        original_copy = dict(original)
        self._filter([original], {"BTC/USDT": 0.8})
        assert original == original_copy

    def test_empty_recs(self):
        result = self._filter([], {"BTC/USDT": 0.5})
        assert result == []

    def test_empty_sentiment(self):
        recs = [make_rec(action="buy", confidence=0.80)]
        result = self._filter(recs, {})
        assert result[0]["confidence"] == pytest.approx(0.80, abs=1e-3)


# ---------------------------------------------------------------------------
# _generate_context_signals()
# ---------------------------------------------------------------------------


class TestGenerateContextSignals:
    def _gen(self, snapshots, ctx, min_conf=0.65):
        mock_ai = MagicMock()
        cfg = _make_config(MIN_SIGNAL_CONFIDENCE=min_conf)
        with patch("src.signal_combiner.SACSignal"), patch(
            "src.signal_combiner.Config", cfg
        ):
            c = SignalCombiner(ai=mock_ai)
        with patch("src.signal_combiner.Config", cfg):
            return c._generate_context_signals(snapshots, ctx)

    def test_funding_long_overheated_generates_sell(self):
        snap = make_snapshot(price=30000.0, atr=300.0)
        ctx = {"funding_signal": "long_overheated", "funding_rate": 0.003}
        result = self._gen([snap], ctx)
        assert any(r["action"] == "sell" for r in result)

    def test_funding_short_overheated_generates_buy(self):
        snap = make_snapshot(price=30000.0, atr=300.0)
        ctx = {"funding_signal": "short_overheated", "funding_rate": -0.003}
        result = self._gen([snap], ctx)
        assert any(r["action"] == "buy" for r in result)

    def test_long_liquidation_generates_sell(self):
        snap = make_snapshot(price=30000.0, atr=300.0)
        ctx = {"liquidation_pressure": "long_liquidation"}
        result = self._gen([snap], ctx)
        sells = [r for r in result if r["action"] == "sell"]
        assert len(sells) >= 1
        assert sells[0]["confidence"] == pytest.approx(0.72, abs=1e-3)

    def test_short_squeeze_generates_buy(self):
        snap = make_snapshot(price=30000.0, atr=300.0)
        ctx = {"liquidation_pressure": "short_squeeze"}
        result = self._gen([snap], ctx)
        buys = [r for r in result if r["action"] == "buy"]
        assert len(buys) >= 1
        assert buys[0]["confidence"] == pytest.approx(0.70, abs=1e-3)

    def test_bid_dominant_ob_generates_buy(self):
        snap = make_snapshot(price=30000.0, atr=300.0)
        ctx = {"ob_signal": "bid_dominant", "ob_imbalance": 0.6}
        result = self._gen([snap], ctx)
        buys = [r for r in result if r["action"] == "buy"]
        assert len(buys) >= 1

    def test_ask_dominant_ob_generates_sell(self):
        snap = make_snapshot(price=30000.0, atr=300.0)
        ctx = {"ob_signal": "ask_dominant", "ob_imbalance": -0.6}
        result = self._gen([snap], ctx)
        sells = [r for r in result if r["action"] == "sell"]
        assert len(sells) >= 1

    def test_ob_bid_below_threshold_no_signal(self):
        """ob_imbalance=0.3 < 0.4 → нет сигнала."""
        snap = make_snapshot(price=30000.0, atr=300.0)
        ctx = {"ob_signal": "bid_dominant", "ob_imbalance": 0.3}
        result = self._gen([snap], ctx)
        ob_buys = [r for r in result if "Orderbook" in r.get("reasoning", "")]
        assert len(ob_buys) == 0

    def test_pcr_greed_calls_generates_sell(self):
        snap = make_snapshot(price=30000.0, atr=300.0)
        ctx = {"pcr_signal": "greed_calls", "pcr": 0.5}
        result = self._gen([snap], ctx)
        sells = [
            r
            for r in result
            if r["action"] == "sell" and "PCR" in r.get("reasoning", "")
        ]
        assert len(sells) == 1
        assert sells[0]["confidence"] == pytest.approx(0.65, abs=1e-3)

    def test_pcr_fear_puts_generates_buy(self):
        # confidence=0.63 is below default min_conf=0.65, so use lower threshold
        snap = make_snapshot(price=30000.0, atr=300.0)
        ctx = {"pcr_signal": "fear_puts", "pcr": 1.8}
        result = self._gen([snap], ctx, min_conf=0.60)
        buys = [
            r
            for r in result
            if r["action"] == "buy" and "PCR" in r.get("reasoning", "")
        ]
        assert len(buys) == 1
        assert buys[0]["confidence"] == pytest.approx(0.63, abs=1e-3)

    def test_extreme_fear_greed_generates_buy(self):
        snap = make_snapshot(price=30000.0, atr=300.0)
        ctx = {"fear_greed": 5}
        result = self._gen([snap], ctx)
        buys = [
            r
            for r in result
            if r["action"] == "buy" and "Fear" in r.get("reasoning", "")
        ]
        assert len(buys) == 1
        assert buys[0]["confidence"] == pytest.approx(0.68, abs=1e-3)

    def test_extreme_greed_generates_sell(self):
        snap = make_snapshot(price=30000.0, atr=300.0)
        ctx = {"fear_greed": 95}
        result = self._gen([snap], ctx)
        sells = [
            r
            for r in result
            if r["action"] == "sell" and "Fear" in r.get("reasoning", "")
        ]
        assert len(sells) == 1
        assert sells[0]["confidence"] == pytest.approx(0.67, abs=1e-3)

    def test_basis_greed_premium_above_2pct_generates_sell(self):
        snap = make_snapshot(price=30000.0, atr=300.0)
        ctx = {"basis_signal": "greed_premium", "basis_pct": 3.0}
        result = self._gen([snap], ctx)
        sells = [
            r
            for r in result
            if r["action"] == "sell" and "basis" in r.get("reasoning", "")
        ]
        assert len(sells) == 1

    def test_basis_greed_premium_below_2pct_no_signal(self):
        snap = make_snapshot(price=30000.0, atr=300.0)
        ctx = {"basis_signal": "greed_premium", "basis_pct": 1.5}
        result = self._gen([snap], ctx)
        sells = [
            r
            for r in result
            if r["action"] == "sell" and "basis" in r.get("reasoning", "")
        ]
        assert len(sells) == 0

    def test_btc_etf_outflow_generates_sell(self):
        snap = make_snapshot(symbol="BTC/USDT", price=30000.0, atr=300.0)
        ctx = {"etf_signal": "etf_outflow", "etf_flow": -200.0}
        result = self._gen([snap], ctx)
        sells = [
            r
            for r in result
            if r["action"] == "sell" and "ETF" in r.get("reasoning", "")
        ]
        assert len(sells) == 1

    def test_btc_etf_inflow_generates_buy(self):
        snap = make_snapshot(symbol="BTC/USDT", price=30000.0, atr=300.0)
        ctx = {"etf_signal": "etf_inflow", "etf_flow": 200.0}
        result = self._gen([snap], ctx)
        buys = [
            r
            for r in result
            if r["action"] == "buy" and "ETF" in r.get("reasoning", "")
        ]
        assert len(buys) == 1

    def test_non_btc_etf_no_signal(self):
        """ETF signals only apply to BTC."""
        snap = make_snapshot(symbol="ETH/USDT", price=2000.0, atr=20.0)
        ctx = {"etf_signal": "etf_outflow", "etf_flow": -200.0}
        result = self._gen([snap], ctx)
        etf_signals = [r for r in result if "ETF" in r.get("reasoning", "")]
        assert len(etf_signals) == 0

    def test_btc_etf_outflow_below_100m_no_signal(self):
        snap = make_snapshot(symbol="BTC/USDT", price=30000.0, atr=300.0)
        ctx = {"etf_signal": "etf_outflow", "etf_flow": -50.0}
        result = self._gen([snap], ctx)
        etf_sells = [r for r in result if "ETF" in r.get("reasoning", "")]
        assert len(etf_sells) == 0

    def test_btc_etf_inflow_below_150m_no_signal(self):
        snap = make_snapshot(symbol="BTC/USDT", price=30000.0, atr=300.0)
        ctx = {"etf_signal": "etf_inflow", "etf_flow": 100.0}
        result = self._gen([snap], ctx)
        etf_buys = [r for r in result if "ETF" in r.get("reasoning", "")]
        assert len(etf_buys) == 0

    def test_signal_has_required_fields(self):
        snap = make_snapshot(price=30000.0, atr=300.0)
        ctx = {"liquidation_pressure": "long_liquidation"}
        result = self._gen([snap], ctx)
        for r in result:
            assert "symbol" in r
            assert "action" in r
            assert "confidence" in r
            assert "entry" in r
            assert "stop_loss" in r
            assert "take_profit" in r
            assert "strategy" in r
            assert r["strategy"] == "context_signal"

    def test_zero_price_skipped(self):
        snap = {"symbol": "BTC/USDT", "price": 0.0, "atr": 0.0}
        ctx = {"liquidation_pressure": "long_liquidation"}
        result = self._gen([snap], ctx)
        assert result == []

    def test_per_symbol_context_applied_correctly(self):
        """Per-symbol map: BTC context применяется только к BTC."""
        btc_snap = make_snapshot(symbol="BTC/USDT", price=30000.0, atr=300.0)
        eth_snap = make_snapshot(symbol="ETH/USDT", price=2000.0, atr=20.0)
        ctx = {
            "BTC/USDT": {"liquidation_pressure": "long_liquidation"},
            "ETH/USDT": {},
        }
        result = self._gen([btc_snap, eth_snap], ctx)
        btc_sigs = [r for r in result if r["symbol"] == "BTC/USDT"]
        eth_sigs = [r for r in result if r["symbol"] == "ETH/USDT"]
        assert len(btc_sigs) >= 1
        assert len(eth_sigs) == 0

    def test_empty_snapshots_returns_empty(self):
        result = self._gen([], {"liquidation_pressure": "long_liquidation"})
        assert result == []

    def test_sl_tp_calculated_in_signals(self):
        snap = make_snapshot(price=30000.0, atr=300.0)
        ctx = {"liquidation_pressure": "long_liquidation"}
        result = self._gen([snap], ctx)
        for r in result:
            if r["action"] == "sell":
                assert r["stop_loss"] == pytest.approx(30000.0 + 1.5 * 300.0)
                assert r["take_profit"] == pytest.approx(30000.0 - 3.0 * 300.0)


# ---------------------------------------------------------------------------
# _merge_with_context()
# ---------------------------------------------------------------------------


class TestMergeWithContext:
    def _merge(self, existing, context_recs):
        mock_ai = MagicMock()
        cfg = _make_config()
        with patch("src.signal_combiner.SACSignal"), patch(
            "src.signal_combiner.Config", cfg
        ):
            c = SignalCombiner(ai=mock_ai)
        return c._merge_with_context(existing, context_recs)

    def test_same_symbol_same_action_boosts_confidence(self):
        existing = [make_rec(symbol="BTC/USDT", action="buy", confidence=0.80)]
        context = [make_rec(symbol="BTC/USDT", action="buy", confidence=0.70)]
        result = self._merge(existing, context)
        assert len(result) == 1
        assert result[0]["confidence"] == pytest.approx(
            min(0.95, 0.80 + 0.05), abs=1e-3
        )

    def test_same_symbol_different_action_adds_new(self):
        existing = [make_rec(symbol="BTC/USDT", action="buy", confidence=0.80)]
        context = [make_rec(symbol="BTC/USDT", action="sell", confidence=0.70)]
        result = self._merge(existing, context)
        assert len(result) == 2

    def test_different_symbol_adds_new(self):
        existing = [make_rec(symbol="BTC/USDT", action="buy", confidence=0.80)]
        context = [make_rec(symbol="ETH/USDT", action="buy", confidence=0.70)]
        result = self._merge(existing, context)
        assert len(result) == 2

    def test_result_sorted_by_confidence_desc(self):
        existing = [
            make_rec(symbol="BTC/USDT", action="buy", confidence=0.70),
            make_rec(symbol="ETH/USDT", action="buy", confidence=0.90),
        ]
        context = [
            make_rec(symbol="XRP/USDT", action="buy", confidence=0.75),
        ]
        result = self._merge(existing, context)
        confidences = [r["confidence"] for r in result]
        assert confidences == sorted(confidences, reverse=True)

    def test_empty_existing_adds_all_context(self):
        context = [
            make_rec(symbol="BTC/USDT", action="buy", confidence=0.80),
            make_rec(symbol="ETH/USDT", action="sell", confidence=0.70),
        ]
        result = self._merge([], context)
        assert len(result) == 2

    def test_empty_context_returns_existing(self):
        existing = [make_rec(symbol="BTC/USDT", action="buy", confidence=0.80)]
        result = self._merge(existing, [])
        assert len(result) == 1
        assert result[0]["confidence"] == pytest.approx(0.80, abs=1e-3)

    def test_confidence_capped_at_095_on_boost(self):
        existing = [make_rec(symbol="BTC/USDT", action="buy", confidence=0.93)]
        context = [make_rec(symbol="BTC/USDT", action="buy", confidence=0.80)]
        result = self._merge(existing, context)
        assert result[0]["confidence"] <= 0.95

    def test_original_not_mutated(self):
        original = make_rec(symbol="BTC/USDT", action="buy", confidence=0.80)
        existing = [original]
        context = [make_rec(symbol="BTC/USDT", action="buy", confidence=0.70)]
        self._merge(existing, context)
        # Original dict should not be modified (merge creates copies)
        assert original["confidence"] == pytest.approx(0.80, abs=1e-3)

    def test_multiple_context_same_key_last_wins(self):
        """Multiple context recs with same (symbol, action): each boost applies."""
        existing = [make_rec(symbol="BTC/USDT", action="buy", confidence=0.80)]
        context = [
            make_rec(symbol="BTC/USDT", action="buy", confidence=0.70),
            make_rec(symbol="BTC/USDT", action="buy", confidence=0.65),
        ]
        result = self._merge(existing, context)
        # Two boosts: 0.80 + 0.05 + 0.05 = 0.90
        assert result[0]["confidence"] == pytest.approx(
            min(0.95, 0.80 + 0.05 + 0.05), abs=1e-3
        )

    def test_empty_both_returns_empty(self):
        result = self._merge([], [])
        assert result == []


# ---------------------------------------------------------------------------
# Async helper (used in sync test classes)
# ---------------------------------------------------------------------------


def await_sync(coro):
    """Run a coroutine synchronously in tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# async tests must be top-level for pytest-asyncio
# ---------------------------------------------------------------------------


async def test_hybrid_agree_buy(combiner, mock_ai, mock_dqn_signal):
    """И DQN, и AI говорят buy → итоговый conf = 0.4*dqn + 0.6*ai."""
    dqn_conf = 0.75
    ai_conf = 0.80
    mock_dqn_signal.get_signal.return_value = {
        "action": "buy",
        "confidence": dqn_conf,
        "source": "sac",
    }
    ai_rec = {
        "symbol": "BTC/USDT",
        "action": "buy",
        "confidence": ai_conf,
        "strategy": "momentum",
        "reasoning": "trend up",
        "entry": 30000.0,
        "stop_loss": 29500.0,
        "take_profit": 31000.0,
    }
    mock_ai.analyze.return_value = [ai_rec]
    snap = make_snapshot()

    with patch.object(Config, "MODE", "hybrid"):
        result = await combiner.combine([snap], balance=10000)

    assert len(result) == 1
    expected_conf = round(dqn_conf * 0.4 + ai_conf * 0.6, 3)
    assert result[0]["confidence"] == expected_conf
    assert "hybrid" in result[0]["strategy"]


async def test_hybrid_disagree_returns_empty(combiner, mock_ai, mock_dqn_signal):
    """DQN=buy, AI=sell → hold (нет сигнала)."""
    mock_dqn_signal.get_signal.return_value = {
        "action": "buy",
        "confidence": 0.85,
        "source": "sac",
    }
    ai_rec = {
        "symbol": "BTC/USDT",
        "action": "sell",
        "confidence": 0.80,
        "strategy": "reversal",
        "reasoning": "overbought",
        "entry": 30000.0,
        "stop_loss": 30500.0,
        "take_profit": 29000.0,
    }
    mock_ai.analyze.return_value = [ai_rec]
    snap = make_snapshot()

    with patch.object(Config, "MODE", "hybrid"):
        result = await combiner.combine([snap], balance=10000)

    assert result == []


async def test_hybrid_ai_silent_high_dqn_conf(combiner, mock_ai, mock_dqn_signal):
    """AI ничего не вернул, DQN conf=0.85 >= 0.80 → используем сигнал DQN."""
    mock_dqn_signal.get_signal.return_value = {
        "action": "buy",
        "confidence": 0.85,
        "source": "sac",
    }
    mock_ai.analyze.return_value = []
    snap = make_snapshot()

    with patch.object(Config, "MODE", "hybrid"):
        result = await combiner.combine([snap], balance=10000)

    assert len(result) == 1
    assert result[0]["action"] == "buy"
    assert result[0]["confidence"] == 0.85


async def test_hybrid_ai_silent_low_dqn_conf(combiner, mock_ai, mock_dqn_signal):
    """AI ничего не вернул, DQN conf=0.5 < 0.80 → пропускаем."""
    mock_dqn_signal.get_signal.return_value = {
        "action": "buy",
        "confidence": 0.5,
        "source": "sac",
    }
    mock_ai.analyze.return_value = []
    snap = make_snapshot()

    with patch.object(Config, "MODE", "hybrid"):
        result = await combiner.combine([snap], balance=10000)

    assert result == []


def test_sl_tp_buy():
    sl, tp = SignalCombiner._sl_tp(price=100.0, atr=2.0, action="buy")
    assert sl < 100.0 < tp
    assert sl == pytest.approx(100.0 - 1.5 * 2.0)
    assert tp == pytest.approx(100.0 + 3.0 * 2.0)


def test_sl_tp_sell():
    sl, tp = SignalCombiner._sl_tp(price=100.0, atr=2.0, action="sell")
    assert tp < 100.0 < sl
    assert sl == pytest.approx(100.0 + 1.5 * 2.0)
    assert tp == pytest.approx(100.0 - 3.0 * 2.0)
