"""Стратегии из Argent-Systems (src/strategies_ported.py)."""

import numpy as np
import pandas as pd
import pytest

from src.strategies import STRATEGY_REGISTRY, create_strategy
from src.strategies_ported import PORTED_STRATEGIES as FIRST
from src.strategies_ported import PortedStrategy
from src.strategies_ported_ind import IND_STRATEGIES

PORTED_STRATEGIES = {**FIRST, **IND_STRATEGIES}


def _walk(n: int = 1100, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.006, n)))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.003, n)) * close
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": rng.uniform(50, 150, n),
        }
    )


def test_all_ports_are_registered_with_market_flags():
    assert len(FIRST) == 15 and len(IND_STRATEGIES) == 34
    assert len(PORTED_STRATEGIES) == 49
    assert all(n in STRATEGY_REGISTRY for n in PORTED_STRATEGIES)
    assert STRATEGY_REGISTRY["bear_hedge"].markets == "futures"
    assert STRATEGY_REGISTRY["shooting_star"].markets == "futures"
    assert all(
        c.markets == "both"
        for n, c in PORTED_STRATEGIES.items()
        if n not in ("bear_hedge", "shooting_star")
    )


@pytest.mark.parametrize("name", list(PORTED_STRATEGIES))
def test_weight_is_causal(name):
    """Вес на префиксе = префикс веса на всей истории: иначе prepare() в бэктесте
    подсмотрел бы будущее."""
    df = _walk()
    s = create_strategy(name)
    full = s.weights(df)
    assert set(full.unique()) <= {-1.0, 0.0, 1.0}
    for k in (600, 900):
        part = s.weights(df.iloc[:k])
        assert (part.values == full.iloc[:k].values).all(), name


@pytest.mark.parametrize("name", list(PORTED_STRATEGIES))
def test_prepared_signals_match_live_signals(name):
    """Бэктест (prepare) и живой путь (хвост истории) дают один и тот же сигнал."""
    df = _walk()
    live, fast = create_strategy(name), create_strategy(name)
    fast.prepare(df)
    for k in range(400, 1100, 7):
        w = df.iloc[:k]
        assert live.generate_signal(w) == fast.generate_signal(w), (name, k)


def test_signal_only_at_entry_and_not_while_holding():
    class Stub(PortedStrategy):
        name = "stub"
        min_bars = 3

        def __init__(self, series):
            super().__init__()
            self.series = series

        def weights(self, df):
            return pd.Series(self.series[: len(df)], index=df.index)

    df = _walk(6)
    #                 вход   держим  выход  вход в шорт  разворот
    w = [0, 0, 1, 1, 0, -1]
    got = [Stub(w).generate_signal(df.iloc[: i + 1])["action"] for i in range(2, 6)]
    assert got == ["buy", "hold", "hold", "sell"]


def test_too_short_history_is_hold():
    sig = create_strategy("adx_trend_birth").generate_signal(_walk(100))
    assert sig["action"] == "hold"


def test_confidence_is_the_same_for_all_ports():
    # Своей самооценки у портов нет: завышенная уверенность выигрывала бы турнир.
    assert {c.confidence for c in PORTED_STRATEGIES.values()} == {0.70}
