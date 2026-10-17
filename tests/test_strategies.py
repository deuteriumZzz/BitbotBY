import pytest

from src.indicators import calculate_technical_indicators
from src.strategies import STRATEGY_REGISTRY
from tests.conftest import make_ohlcv


def prepared_df(n=100, trend=0.002):
    df = make_ohlcv(n=n, trend=trend)
    return calculate_technical_indicators(df)


@pytest.mark.parametrize("name", list(STRATEGY_REGISTRY.keys()))
def test_strategy_returns_valid_signal(name):
    """Каждая стратегия должна возвращать action и confidence."""
    strategy = STRATEGY_REGISTRY[name]()
    df = prepared_df(n=100)
    signal = strategy.generate_signal(df)
    assert "action" in signal, f"{name}: отсутствует 'action'"
    assert signal["action"] in (
        "buy",
        "sell",
        "hold",
    ), f"{name}: недопустимый action '{signal['action']}'"
    assert "confidence" in signal, f"{name}: отсутствует 'confidence'"
    conf = signal["confidence"]
    assert 0.0 <= conf <= 1.0, f"{name}: confidence {conf} вне диапазона"


@pytest.mark.parametrize("name", list(STRATEGY_REGISTRY.keys()))
def test_strategy_handles_short_data(name):
    """Стратегия не должна падать на минимальных данных (< warmup)."""
    strategy = STRATEGY_REGISTRY[name]()
    df = prepared_df(n=10)
    try:
        signal = strategy.generate_signal(df)
        assert signal["action"] in ("buy", "sell", "hold")
    except Exception as e:
        pytest.fail(f"{name} упала на коротких данных: {e}")


@pytest.mark.parametrize("name", list(STRATEGY_REGISTRY.keys()))
def test_strategy_handles_flat_market(name):
    """Стратегия не должна падать на флэтовых ценовых данных."""
    strategy = STRATEGY_REGISTRY[name]()
    df = make_ohlcv(n=100, trend=0.0)
    df["close"] = 100.0
    df = calculate_technical_indicators(df)
    try:
        signal = strategy.generate_signal(df)
        assert signal["action"] in ("buy", "sell", "hold")
    except Exception as e:
        pytest.fail(f"{name} упала на флэтовом рынке: {e}")


def test_volume_spike_is_one_strategy_with_profile_threshold():
    from src.strategies import (
        LEGACY_ALIASES,
        VolumeSpikeStrategy,
        create_strategy,
        set_active_profile,
    )

    assert not [n for n in STRATEGY_REGISTRY if n.startswith("volume_spike_")]
    # Старые имена из истории сделок и Redis по-прежнему создают рабочую стратегию.
    for old in LEGACY_ALIASES:
        assert create_strategy(old).name == "volume_spike"
    try:
        got = {}
        for profile in ("bluechip", "altcoin", "meme"):
            set_active_profile(profile)
            got[profile] = VolumeSpikeStrategy.threshold()
        assert got == {"bluechip": 2.0, "altcoin": 3.0, "meme": 5.0}
        set_active_profile("")
        from config import Config

        assert VolumeSpikeStrategy.threshold() == Config.VOLUME_SPIKE_THRESHOLD
    finally:
        set_active_profile("")


def test_only_sell_only_strategies_are_futures_only():
    # Стратегии, которые умеют только продавать, на споте входов не дают.
    assert sorted(
        n for n, c in STRATEGY_REGISTRY.items() if c.markets == "futures"
    ) == [
        "bear_hedge",
        "pump_exit",
    ]
    assert all(
        c.markets in ("spot", "futures", "both") for c in STRATEGY_REGISTRY.values()
    )
