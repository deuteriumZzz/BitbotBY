"""Обучение SAC v2 (Phase 7): walk-forward на env_v2, кандидат в реестр — без допуска к торговле.

Конвейер: OHLCV → `features_v2` (причинные признаки) → расширяющееся окно с embargo
(`evaluation.walk_forward_splits`) → на каждом фолде нормализация ТОЛЬКО по train,
обучение SAC на `TradingEnvV2` (шорт, комиссия, проскальзывание, funding, исполнение по
open следующего бара), детерминированный прогон на тесте → доходности ПОСЛЕ издержек
против лучшего базиса (buy&hold / вне рынка, выбор лучшего задним числом — строго к
модели; базис без издержек) → `promotion_verdict`. Модель последнего фолда сохраняется и
вносится в реестр как `candidate` с доказательствами; продвигать её дальше может только
человек (`model_registry.promote`). Вердикт ворот — информация, не разрешение.

Все числовые параметры — ПРЕДЛОЖЕНИЯ владельцу. Один актив на запуск: многоактивное
обучение и подбор гиперпараметров — отдельные задачи. Это не торговый эдж: результат на
случайном блуждании не должен проходить ворота.

Запуск:
    python -m reinforcement_learning.train_sac_v2 --csv data.csv --symbol BTC/USDT \
        --timeframe 15m --timesteps 20000 --folds 4
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from reinforcement_learning import dataset as ds
from reinforcement_learning import evaluation as ev
from reinforcement_learning.env_v2 import DEFAULT_FEE, DEFAULT_SLIPPAGE, TradingEnvV2
from reinforcement_learning.features_v2 import FEATURE_NAMES, prepare
from reinforcement_learning.model_registry import (
    ModelRegistry,
    PromotionError,
    default_registry_path,
)

PERIODS_PER_YEAR = {
    "1m": 525_600,
    "5m": 105_120,
    "15m": 35_040,
    "30m": 17_520,
    "1h": 8_760,
    "4h": 2_190,
    "1d": 365,
}


def load_ohlcv(path: str) -> pd.DataFrame:
    """CSV/Parquet с колонками open, high, low, close, volume (+ timestamp|индекс, funding_rate)."""
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    df.columns = [str(c).lower() for c in df.columns]
    for name in ("timestamp", "time", "date", "datetime"):
        if name in df.columns:
            col = df.pop(name)
            unit = "ms" if np.issubdtype(col.dtype, np.number) else None
            df.index = pd.to_datetime(col, unit=unit, utc=True)
            break
    df = df.sort_index()
    return df[~df.index.duplicated(keep="last")]


def rollout(model: Any, env: TradingEnvV2) -> np.ndarray:
    """Детерминированный прогон политики; капитал по барам (включая начальный)."""
    obs, _ = env.reset()
    equity = [env.sim.equity]
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        equity.append(info["equity"])
        if terminated or truncated:
            return np.asarray(equity, float)


def _normalized(feats: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    return ds.apply_normalizer(feats, stats).to_numpy(dtype=np.float32)


def _register(
    reg: ModelRegistry,
    model_id: str,
    path: str,
    manifest: Dict[str, Any],
    params: Dict[str, Any],
    metrics: Dict[str, Any],
) -> str:
    kw = dict(
        dataset_version=manifest["dataset_version"],
        feature_version=ds.FEATURE_VERSION,
        params=params,
        metrics=metrics,
    )
    try:
        reg.register(model_id, path, **kw)
    except (
        PromotionError
    ):  # та же модель уже внесена — не затираем, делаем новую запись
        model_id += f"_{int(time.time())}"
        reg.register(model_id, path, **kw)
    return model_id


def run(
    df: pd.DataFrame,
    *,
    timesteps: int = 20_000,
    n_folds: int = 4,
    min_train_frac: float = 0.5,
    embargo: int = 16,
    seed: int = 0,
    out_path: str = "models/sac_v2.zip",
    symbol: str = "UNKNOWN",
    timeframe: str = "15m",
    fee: float = DEFAULT_FEE,
    slippage: float = DEFAULT_SLIPPAGE,
    register: bool = True,
) -> Dict[str, Any]:
    from stable_baselines3 import SAC  # noqa: PLC0415

    feats, o, c, f = prepare(df)
    n = len(feats)
    splits = ev.walk_forward_splits(n, n_folds, int(n * min_train_frac), embargo)
    ppy = PERIODS_PER_YEAR.get(timeframe, 35_040)
    folds: List[Dict[str, Any]] = []
    reports: List[Dict[str, Any]] = []
    model: Any = None
    stats: Dict[str, Dict[str, float]] = {}

    for k, (tr, te) in enumerate(splits):
        stats = ds.fit_normalizer(feats.iloc[tr])  # нормализация только по train
        env_tr = TradingEnvV2(
            _normalized(feats.iloc[tr], stats),
            o[tr],
            c[tr],
            f[tr],
            fee=fee,
            slippage=slippage,
        )
        model = SAC(
            "MlpPolicy",
            env_tr,
            seed=seed + k,
            verbose=0,
            device="cpu",
            learning_starts=max(100, min(1000, timesteps // 4)),
            batch_size=64,
            buffer_size=max(10_000, min(200_000, timesteps * 2)),
        )
        model.learn(total_timesteps=timesteps)
        env_te = TradingEnvV2(
            _normalized(feats.iloc[te], stats),
            o[te],
            c[te],
            f[te],
            fee=fee,
            slippage=slippage,
        )
        equity = rollout(model, env_te)
        model_ret = np.diff(equity) / equity[:-1]
        bases = ev.baseline_returns(c[te])
        best = max(bases.values(), key=lambda r: float(np.sum(r[: len(model_ret)])))
        folds.append(
            {"model": model_ret, "baseline": best[: len(model_ret)], "equity": equity}
        )
        m = ev.metrics(equity, ppy)
        reports.append(
            {
                "fold": k,
                "train_rows": int(len(tr)),
                "test_rows": int(len(te)),
                **{x: round(v, 5) for x, v in m.items()},
                "buy_and_hold": round(
                    float(np.sum(bases["buy_and_hold"][: len(model_ret)])), 5
                ),
            }
        )

    verdict = ev.promotion_verdict(folds)
    params = {
        "timesteps": timesteps,
        "folds": n_folds,
        "embargo": embargo,
        "fee": fee,
        "slippage": slippage,
        "symbol": symbol,
        "timeframe": timeframe,
        "seed": seed,
    }
    manifest = ds.build_manifest(feats, params)
    base_path = out_path[:-4] if out_path.endswith(".zip") else out_path
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    model.save(base_path)  # SB3 добавляет .zip
    ds.save_normalizer(stats, base_path + "_norm_stats.json")
    meta = {
        "env": "v2",
        "feature_version": ds.FEATURE_VERSION,
        "feature_names": FEATURE_NAMES,
        "obs_dim": len(FEATURE_NAMES) + 2,
        "dataset_version": manifest["dataset_version"],
        **params,
    }
    with open(base_path + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    model_id: Optional[str] = None
    if register:
        reg = ModelRegistry(default_registry_path(out_path))
        model_id = f"sac_v2_{symbol.replace('/', '')}_{timeframe}_{manifest['dataset_version'][:8]}_s{seed}"
        model_id = _register(
            reg,
            model_id,
            base_path + ".zip",
            manifest,
            params,
            {"verdict": verdict, "folds": reports},
        )
    return {
        "model_id": model_id,
        "model_path": base_path + ".zip",
        "dataset_version": manifest["dataset_version"],
        "verdict": verdict,
        "folds": reports,
        "note": "стадия candidate: продвижение только вручную; вердикт — информация, не допуск",
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Обучение SAC v2 (walk-forward, кандидат в реестр)"
    )
    p.add_argument("--csv", required=True)
    p.add_argument("--symbol", default="UNKNOWN")
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--timesteps", type=int, default=20_000)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--min-train-frac", type=float, default=0.5)
    p.add_argument("--embargo", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="models/sac_v2.zip")
    p.add_argument("--fee", type=float, default=DEFAULT_FEE)
    p.add_argument("--slippage", type=float, default=DEFAULT_SLIPPAGE)
    p.add_argument("--no-register", action="store_true")
    a = p.parse_args(argv)

    from src.infra.heavy_job import _threads, lower_priority  # noqa: PLC0415

    lower_priority()()  # обучение не должно мешать боту на той же машине
    import torch  # noqa: PLC0415

    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", _threads())))
    res = run(
        load_ohlcv(a.csv),
        timesteps=a.timesteps,
        n_folds=a.folds,
        min_train_frac=a.min_train_frac,
        embargo=a.embargo,
        seed=a.seed,
        out_path=a.out,
        symbol=a.symbol,
        timeframe=a.timeframe,
        fee=a.fee,
        slippage=a.slippage,
        register=not a.no_register,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
