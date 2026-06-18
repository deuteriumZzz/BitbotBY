"""Тесты для SignalCombiner — логика объединения сигналов DQN и AI."""

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
    """Patch DQNSignal inside signal_combiner for the duration of each test."""
    with patch("src.signal_combiner.DQNSignal") as MockDQN:
        instance = MagicMock()
        instance.get_signal.return_value = {
            "action": "hold",
            "confidence": 0.0,
            "source": "dqn",
        }
        MockDQN.return_value = instance
        yield instance


@pytest.fixture
def combiner(mock_ai, mock_dqn_signal):
    """SignalCombiner with mocked AI and DQN."""
    return SignalCombiner(ai=mock_ai)


# ---------------------------------------------------------------------------
# Tests
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
        "source": "dqn",
    }
    snap = make_snapshot()
    with patch.object(Config, "MODE", "dqn"):
        result = await combiner.combine([snap], balance=10000)
    assert result == []


async def test_dqn_mode_returns_buy_above_threshold(combiner, mock_dqn_signal):
    mock_dqn_signal.get_signal.return_value = {
        "action": "buy",
        "confidence": 0.8,
        "source": "dqn",
    }
    snap = make_snapshot()
    with patch.object(Config, "MODE", "dqn"):
        result = await combiner.combine([snap], balance=10000)
    assert len(result) == 1
    assert result[0]["action"] == "buy"
    assert result[0]["symbol"] == "BTC/USDT"


async def test_dqn_mode_skips_low_confidence(combiner, mock_dqn_signal):
    """conf=0.3 is below MIN_SIGNAL_CONFIDENCE=0.65 → empty result."""
    mock_dqn_signal.get_signal.return_value = {
        "action": "buy",
        "confidence": 0.3,
        "source": "dqn",
    }
    snap = make_snapshot()
    with patch.object(Config, "MODE", "dqn"):
        result = await combiner.combine([snap], balance=10000)
    assert result == []


async def test_hybrid_agree_buy(combiner, mock_ai, mock_dqn_signal):
    """Both DQN and AI say buy → combined conf = 0.4*dqn + 0.6*ai."""
    dqn_conf = 0.75
    ai_conf = 0.80
    mock_dqn_signal.get_signal.return_value = {
        "action": "buy",
        "confidence": dqn_conf,
        "source": "dqn",
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
    """DQN=buy, AI=sell → hold (no signal)."""
    mock_dqn_signal.get_signal.return_value = {
        "action": "buy",
        "confidence": 0.85,
        "source": "dqn",
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
    """AI returns nothing, DQN conf=0.85 >= 0.80 → use DQN signal."""
    mock_dqn_signal.get_signal.return_value = {
        "action": "buy",
        "confidence": 0.85,
        "source": "dqn",
    }
    mock_ai.analyze.return_value = []
    snap = make_snapshot()

    with patch.object(Config, "MODE", "hybrid"):
        result = await combiner.combine([snap], balance=10000)

    assert len(result) == 1
    assert result[0]["action"] == "buy"
    assert result[0]["confidence"] == 0.85


async def test_hybrid_ai_silent_low_dqn_conf(combiner, mock_ai, mock_dqn_signal):
    """AI returns nothing, DQN conf=0.5 < 0.80 → skip."""
    mock_dqn_signal.get_signal.return_value = {
        "action": "buy",
        "confidence": 0.5,
        "source": "dqn",
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
