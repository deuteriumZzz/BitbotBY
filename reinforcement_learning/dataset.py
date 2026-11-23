"""Dataset Builder для SAC (Phase 7, ADR-001): манифест, Parquet+ZSTD, нормализация только по train.

Версия датасета = sha256 содержимого + версия пайплайна признаков + параметры.
Любая модель в реестре ссылается на `dataset_version`, поэтому обучение
воспроизводимо, а смена данных/признаков не остаётся незамеченной.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

FEATURE_VERSION = "v2.0"  # менять при любом изменении расчёта признаков


def check_causal(df: pd.DataFrame) -> None:
    """Индекс — время: строго возрастает, без дублей и NaN в признаках."""
    if not isinstance(df.index, pd.DatetimeIndex) and not np.issubdtype(df.index.dtype, np.number):
        raise ValueError("индекс должен быть временем или числом")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("индекс должен строго возрастать без дублей")
    if df.isna().any().any():
        raise ValueError("NaN в данных: заполните или отбросьте до сборки датасета")


def content_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(",".join(map(str, df.columns)).encode())
    h.update(np.ascontiguousarray(df.to_numpy(dtype=float)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(df.index.astype("int64"))).tobytes())
    return h.hexdigest()


def build_manifest(df: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    check_causal(df)
    params = params or {}
    digest = hashlib.sha256(
        (content_hash(df) + FEATURE_VERSION + json.dumps(params, sort_keys=True, default=str)).encode()
    ).hexdigest()[:16]
    return {
        "dataset_version": digest,
        "feature_version": FEATURE_VERSION,
        "rows": int(len(df)),
        "columns": list(map(str, df.columns)),
        "params": params,
    }


def save_parquet(
    df: pd.DataFrame, path: str, params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Parquet+ZSTD и `<path>.manifest.json` рядом."""
    manifest = build_manifest(df, params)
    df.to_parquet(path, compression="zstd")
    with open(path + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def load_parquet(path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Загружает и проверяет, что содержимое совпадает с манифестом (не подменили файл)."""
    df = pd.read_parquet(path)
    with open(path + ".manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    now = build_manifest(df, manifest.get("params"))
    if now["dataset_version"] != manifest["dataset_version"]:
        raise ValueError("датасет не совпадает со своим манифестом")
    return df, manifest


def fit_normalizer(train: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Среднее/σ только по train — иначе статистика будущего просочится в обучение."""
    return {
        c: {"mean": float(train[c].mean()), "std": float(train[c].std(ddof=0)) or 1.0}
        for c in train.columns
    }


def apply_normalizer(df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    out = df.copy()
    for c, s in stats.items():
        out[c] = (out[c] - s["mean"]) / s["std"]
    return out


def save_normalizer(stats: Dict[str, Dict[str, float]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
