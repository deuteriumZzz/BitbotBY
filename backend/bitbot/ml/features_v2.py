"""Признаки SAC v2 (Phase 7): причинные (без look-ahead), одинаковые для обучения и инференса.

Признак в строке t использует только данные до t включительно: скользящие окна и
EWM с `adjust=False`, никаких сдвигов в будущее. Это проверяется тестом: усечение
ряда справа не меняет уже посчитанные признаки. Для инференса достаточно хвоста
`MIN_TAIL` свечей: память EWM (span ≤ 50) к этому моменту затухает до пренебрежимых
значений (тест сверяет хвост с полным расчётом).

Решение принимается на close свечи t; env_v2 исполняет по open t+1.
Версию (`FEATURE_VERSION` из dataset.py) менять при любом изменении набора или формул.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from bitbot.ml.dataset import FEATURE_VERSION  # noqa: F401 — единая версия пайплайна

WARMUP = 64  # первые строки недостоверны (окна/EWM не прогреты) и отбрасываются
MIN_TAIL = 300  # достаточно для инференса: EWM-память к этому моменту ≈ 0
CLIP = 5.0

FEATURE_NAMES: List[str] = [
    "ret_1",
    "ret_4",
    "ret_16",
    "vol_16",
    "vol_64",
    "rsi",
    "macd_hist",
    "bb_pos",
    "atr_pct",
    "vol_z",
    "range_pct",
    "close_vs_ema50",
    "funding",
]
OBS_EXTRA = ["exposure", "equity_pnl"]  # добавляются средой/инференсом к признакам
_PERCENT = (
    "ret_1",
    "ret_4",
    "ret_16",
    "vol_16",
    "vol_64",
    "atr_pct",
    "range_pct",
    "macd_hist",
)


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Таблица признаков (первые WARMUP строк отброшены). Индекс сохраняется."""
    need = {"open", "high", "low", "close", "volume"}
    if not need <= set(df.columns):
        raise ValueError(f"нужны колонки {sorted(need)}")
    c, h, l, v = (df[k].astype(float) for k in ("close", "high", "low", "volume"))
    if (c <= 0).any():
        raise ValueError("цены должны быть положительными")
    lr = np.log(c).diff()
    ema12, ema26 = (
        c.ewm(span=12, adjust=False).mean(),
        c.ewm(span=26, adjust=False).mean(),
    )
    macd = ema12 - ema26
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    sma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(
        axis=1
    )
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    vmean, vstd = v.rolling(48).mean(), v.rolling(48).std()
    if "funding_rate" in df:
        funding = df["funding_rate"].astype(float).fillna(0.0)
    else:
        funding = pd.Series(0.0, index=df.index)

    f = pd.DataFrame(
        {
            "ret_1": lr,
            "ret_4": np.log(c).diff(4),
            "ret_16": np.log(c).diff(16),
            "vol_16": lr.rolling(16).std(),
            "vol_64": lr.rolling(64).std(),
            "rsi": (_rsi(c) - 50.0) / 50.0,
            "macd_hist": hist / c,
            "bb_pos": (c - sma20) / (2 * sd20.replace(0, np.nan)),
            "atr_pct": atr / c,
            "vol_z": (v - vmean) / vstd.replace(0, np.nan),
            "range_pct": (h - l) / c,
            "close_vs_ema50": c / c.ewm(span=50, adjust=False).mean() - 1.0,
            "funding": funding,
        },
        index=df.index,
    )[FEATURE_NAMES]
    f = f.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # доли → проценты: порядок величин признаков ~1, как у остальных
    for k in _PERCENT + ("funding",):
        f[k] = f[k] * 100.0
    return f.clip(-CLIP, CLIP).iloc[WARMUP:].astype(np.float32)


def prepare(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """(признаки, open, close, funding) — выровнены по одним и тем же строкам для `TradingEnvV2`."""
    feats = build_features(df)
    aligned = df.loc[feats.index]
    if "funding_rate" in aligned:
        funding = aligned["funding_rate"].astype(float).fillna(0.0).to_numpy()
    else:
        funding = np.zeros(len(aligned))
    return (
        feats,
        aligned["open"].astype(float).to_numpy(),
        aligned["close"].astype(float).to_numpy(),
        funding,
    )


def latest_features(df_tail: pd.DataFrame) -> Optional[np.ndarray]:
    """Признаки последней свечи по хвосту истории; None, если истории недостаточно."""
    if len(df_tail) < WARMUP + 2:
        return None
    return build_features(df_tail.iloc[-MIN_TAIL:]).iloc[-1].to_numpy(dtype=np.float32)
