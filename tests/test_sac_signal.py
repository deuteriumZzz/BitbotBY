"""
Тесты инференса SAC: _snap_to_obs() и SACSignal.

Проверяют:
- форму и тип выходного вектора наблюдения
- корректное использование реальных OHLCV из snap["ohlcv"]
- fallback к snap["price"] при отсутствии ohlcv
- декодирование строкового MACD ("bullish"/"bearish")
- нормализацию через norm_stats
- поведение SACSignal без модели (загрузки нет → hold)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from reinforcement_learning.rl_env import OBS_DIM
from src.dqn_signal import SACSignal, _snap_to_obs

# ── Фикстуры снэпшотов ────────────────────────────────

_SNAP_MINIMAL = {
    "symbol": "BTC/USDT",
    "price": 50000.0,
    "atr": 500.0,
    "volume_ratio": 1.2,
    "indicators": {
        "rsi": 55.0,
        "macd": "bullish",
        "bb_width": 0.04,
    },
}

_SNAP_WITH_OHLCV = {
    **_SNAP_MINIMAL,
    "ohlcv": {
        "open": 49800.0,
        "high": 50500.0,
        "low": 49500.0,
        "close": 50000.0,
        "volume": 1234.5,
        "macd": 150.0,
        "macd_signal": 120.0,
    },
}

_SNAP_BEARISH = {
    **_SNAP_MINIMAL,
    "indicators": {**_SNAP_MINIMAL["indicators"], "macd": "bearish"},
}


# ── Тесты _snap_to_obs ────────────────────────────────


def test_obs_shape_and_dtype():
    obs = _snap_to_obs(_SNAP_MINIMAL, norm_stats=None, balance=5000.0)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


def test_ohlcv_values_used_when_present():
    obs = _snap_to_obs(_SNAP_WITH_OHLCV, norm_stats=None, balance=5000.0)
    close = 50000.0
    assert obs[0] == pytest.approx(49800.0 / close - 1.0)  # open_rel = -0.004
    assert obs[1] == pytest.approx(50500.0 / close - 1.0)  # high_rel = +0.010
    assert obs[2] == pytest.approx(49500.0 / close - 1.0)  # low_rel  = -0.010
    assert obs[3] == pytest.approx(0.0)  # in_position = 0
    expected_vol = math.log1p(1234.5) / 15.0
    assert obs[4] == pytest.approx(expected_vol, rel=1e-4)
    assert obs[6] == pytest.approx(150.0 / close)  # macd_norm
    assert obs[7] == pytest.approx(120.0 / close)  # macd_sig_norm


def test_fallback_to_price_without_ohlcv():
    obs = _snap_to_obs(_SNAP_MINIMAL, norm_stats=None, balance=5000.0)
    # fallback: open/high/low все = price → relative = 0.0
    assert obs[0] == pytest.approx(0.0)  # open_rel
    assert obs[1] == pytest.approx(0.0)  # high_rel
    assert obs[2] == pytest.approx(0.0)  # low_rel
    assert obs[3] == pytest.approx(0.0)  # in_position


def test_macd_string_bullish_maps_to_plus_one():
    obs = _snap_to_obs(_SNAP_MINIMAL, norm_stats=None, balance=5000.0)
    # macd="bullish" → 1.0 / close
    assert obs[6] == pytest.approx(1.0 / 50000.0)


def test_macd_string_bearish_maps_to_minus_one():
    obs = _snap_to_obs(_SNAP_BEARISH, norm_stats=None, balance=5000.0)
    assert obs[6] == pytest.approx(-1.0 / 50000.0)


def test_atr_normalized():
    obs = _snap_to_obs(_SNAP_MINIMAL, norm_stats=None, balance=5000.0)
    # atr=500, close=50000 → atr_norm = 500/50000 = 0.01
    assert obs[21] == pytest.approx(500.0 / 50000.0)


def test_rsi_normalized():
    obs = _snap_to_obs(_SNAP_MINIMAL, norm_stats=None, balance=5000.0)
    assert obs[5] == pytest.approx(55.0 / 100.0)


def test_volume_fallback_uses_training_mean():
    norm_stats = {"volume": [80000.0, 5000.0]}
    obs = _snap_to_obs(_SNAP_MINIMAL, norm_stats=norm_stats, balance=5000.0)
    # Нет ohlcv → используем mean=80000; volume log-transformed: log1p(80000)/15.0
    # (volume is NOT z-scored — only log-transformed)
    expected_vol = math.log1p(80000.0) / 15.0
    assert obs[4] == pytest.approx(expected_vol, rel=1e-4)


def test_portfolio_features_set_correctly():
    from config import Config

    obs = _snap_to_obs(_SNAP_MINIMAL, norm_stats=None, balance=3000.0)
    initial = max(Config.INITIAL_BALANCE, 1.0)
    # obs[11] = balance / initial_balance
    assert obs[11] == pytest.approx(3000.0 / initial)
    # obs[12] = (position * price) / initial_balance = 0 (no position)
    assert obs[12] == pytest.approx(0.0)
    # obs[13] = (balance + 0) / initial_balance
    assert obs[13] == pytest.approx(3000.0 / initial)


# ── Тесты SACSignal ───────────────────────────────────


def test_dqn_signal_not_loaded_without_model_file():
    sig = SACSignal()
    assert sig.loaded is False


def test_dqn_signal_returns_hold_when_not_loaded():
    sig = SACSignal()
    result = sig.get_signal(_SNAP_MINIMAL, balance=5000.0)
    assert result["action"] == "hold"
    assert result["confidence"] == 0.0
    assert result["source"] == "sac"


def test_dqn_signal_hold_with_ohlcv_snap_not_loaded():
    sig = SACSignal()
    result = sig.get_signal(_SNAP_WITH_OHLCV, balance=5000.0)
    assert result["action"] == "hold"
