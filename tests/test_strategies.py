import pytest

from src.indicators import calculate_technical_indicators
from src.strategies import STRATEGY_REGISTRY
from tests.conftest import make_ohlcv


def prepared_df(n=100, trend=0.002):
    df = make_ohlcv(n=n, trend=trend)
    return calculate_technical_indicators(df)


@pytest.mark.parametrize("name", list(STRATEGY_REGISTRY.keys()))
def test_strategy_returns_valid_signal(name):
    """Every strategy must return action and confidence."""
    strategy = STRATEGY_REGISTRY[name]()
    df = prepared_df(n=100)
    signal = strategy.generate_signal(df)
    assert "action" in signal, f"{name}: missing 'action'"
    assert signal["action"] in (
        "buy",
        "sell",
        "hold",
    ), f"{name}: invalid action '{signal['action']}'"
    assert "confidence" in signal, f"{name}: missing 'confidence'"
    conf = signal["confidence"]
    assert 0.0 <= conf <= 1.0, f"{name}: confidence {conf} out of range"


@pytest.mark.parametrize("name", list(STRATEGY_REGISTRY.keys()))
def test_strategy_handles_short_data(name):
    """Strategy must not crash on minimal data (< warmup)."""
    strategy = STRATEGY_REGISTRY[name]()
    df = prepared_df(n=10)
    try:
        signal = strategy.generate_signal(df)
        assert signal["action"] in ("buy", "sell", "hold")
    except Exception as e:
        pytest.fail(f"{name} crashed on short data: {e}")


@pytest.mark.parametrize("name", list(STRATEGY_REGISTRY.keys()))
def test_strategy_handles_flat_market(name):
    """Strategy must not crash on flat price data."""
    strategy = STRATEGY_REGISTRY[name]()
    df = make_ohlcv(n=100, trend=0.0)
    df["close"] = 100.0
    df = calculate_technical_indicators(df)
    try:
        signal = strategy.generate_signal(df)
        assert signal["action"] in ("buy", "sell", "hold")
    except Exception as e:
        pytest.fail(f"{name} crashed on flat market: {e}")
