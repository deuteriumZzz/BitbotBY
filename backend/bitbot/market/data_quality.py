"""Data Quality: статус пригодности рыночных данных.

Чистые функции без обращений к сети: принимают DataFrame свечей или контекст
рынка и возвращают `DataQuality`. Статусы: VALID, STALE, INCOMPLETE, INVALID,
UNAVAILABLE (приоритет при нескольких проблемах: INVALID > UNAVAILABLE >
INCOMPLETE > STALE).

Сами по себе проверки ничего не блокируют: бот пишет статусы в снэпшот и
Decision Ledger; блокировка входов — опция `DATA_QUALITY_GATE` (по умолчанию
выключена), потому что ложное «устарело» остановило бы торговлю целиком.
TTL внешних источников контекста хранится в `market_context.py` (по источнику).
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

VALID = "VALID"
STALE = "STALE"
INCOMPLETE = "INCOMPLETE"
INVALID = "INVALID"
UNAVAILABLE = "UNAVAILABLE"

_SEVERITY = {VALID: 0, STALE: 1, INCOMPLETE: 2, UNAVAILABLE: 3, INVALID: 4}
_REQUIRED = ("open", "high", "low", "close", "volume")
_TAIL = 50  # сколько последних свечей проверяем на целостность
MIN_ROWS = 50  # SMA-50 — самый медленный индикатор
_TF_RE = re.compile(r"^(\d+)([mhdw])$")
_UNIT = {"m": 60, "h": 3600, "d": 86400, "w": 604800}


@dataclass(frozen=True)
class DataQuality:
    status: str
    reasons: Tuple[str, ...] = ()
    age_s: Optional[float] = None  # возраст открытия последней свечи, секунд
    last_candle_time: Optional[str] = None
    n_rows: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def timeframe_seconds(timeframe: str) -> Optional[int]:
    """'15m' → 900; неизвестный формат → None."""
    m = _TF_RE.match(str(timeframe))
    return int(m.group(1)) * _UNIT[m.group(2)] if m else None


def last_candle_epoch(df: pd.DataFrame) -> Optional[float]:
    """Unix-время (с) открытия последней свечи или None, если индекс не временной."""
    try:
        idx = df.index[-1]
        if hasattr(idx, "timestamp"):
            ts = pd.Timestamp(idx)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")  # индекс из ccxt — UTC без tz
            return float(ts.timestamp())
        value = float(idx)
        return value / 1000 if value > 1e11 else None
    except Exception:
        return None


def _worst(status_a: str, status_b: str) -> str:
    return status_a if _SEVERITY[status_a] >= _SEVERITY[status_b] else status_b


def check_ohlcv(
    df: Optional[pd.DataFrame],
    timeframe: str,
    *,
    now: Optional[float] = None,
    stale_after_bars: float = 2.0,
    min_rows: int = MIN_ROWS,
) -> DataQuality:
    """Статус свечей. Последняя (формирующаяся) свеча — норма; «устарело» — когда
    она открыта раньше, чем `stale_after_bars` таймфреймов назад."""
    if df is None or len(df) == 0:
        return DataQuality(UNAVAILABLE, ("empty",))
    n = len(df)
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        return DataQuality(INCOMPLETE, tuple(f"missing:{c}" for c in missing), n_rows=n)

    status = VALID
    reasons = []

    def flag(level: str, reason: str) -> None:
        nonlocal status
        status = _worst(status, level)
        reasons.append(reason)

    try:
        tail = df[list(_REQUIRED)].tail(_TAIL).astype(float)
    except (TypeError, ValueError):
        return DataQuality(INVALID, ("non_numeric_ohlcv",), n_rows=n)
    values = tail.to_numpy()
    if not np.isfinite(values).all():
        flag(INVALID, "nan_or_inf")
    else:
        if (tail[["open", "high", "low", "close"]] <= 0).any().any():
            flag(INVALID, "non_positive_price")
        if (tail["high"] < tail["low"]).any():
            flag(INVALID, "high_below_low")
        eps = 1e-9
        if (
            (tail["close"] > tail["high"] * (1 + eps))
            | (tail["close"] < tail["low"] * (1 - eps))
        ).any():
            flag(INVALID, "close_outside_range")
        if (tail["volume"] < 0).any():
            flag(INVALID, "negative_volume")
    tail_index = df.index[-_TAIL:]
    if not tail_index.is_unique:
        flag(INVALID, "duplicate_timestamps")
    elif not tail_index.is_monotonic_increasing:
        flag(INVALID, "index_not_monotonic")

    if n < min_rows:
        flag(INCOMPLETE, f"rows<{min_rows}")

    epoch = last_candle_epoch(df)
    age: Optional[float] = None
    candle_iso: Optional[str] = None
    if epoch is None:
        reasons.append("no_time_index")
    else:
        now_s = time.time() if now is None else now
        age = now_s - epoch
        candle_iso = pd.Timestamp(epoch, unit="s", tz="UTC").isoformat()
        tf = timeframe_seconds(timeframe)
        if tf:
            if age < -0.5 * tf:
                flag(INVALID, "candle_from_future")
            elif age > stale_after_bars * tf:
                flag(STALE, f"age>{stale_after_bars:g}x{timeframe}")
    return DataQuality(
        status, tuple(reasons), None if age is None else round(age, 1), candle_iso, n
    )


def check_context(ctx: Optional[Dict[str, Any]]) -> DataQuality:
    """Статус контекста рынка по `data_status.failed_sources` (см. MarketContext)."""
    if not ctx:
        return DataQuality(UNAVAILABLE, ("empty",))
    failed = list((ctx.get("data_status") or {}).get("failed_sources") or [])
    if "all" in failed:
        return DataQuality(UNAVAILABLE, ("all",))
    if failed:
        return DataQuality(INCOMPLETE, tuple(f"failed:{s}" for s in failed))
    return DataQuality(VALID)


def make_snapshot_id(
    symbol: str, timeframe: str, candle_time: Optional[str], close: float
) -> str:
    """Детерминированный id входных данных: тот же символ/ТФ/свеча/цена → тот же id."""
    raw = f"{symbol}|{timeframe}|{candle_time}|{close:.10g}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]
