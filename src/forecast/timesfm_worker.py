"""Воркер TimesFM: отдельный долгоживущий процесс (запускается интерпретатором Python ≥ 3.10).

Бот работает на Python 3.9, а `timesfm` требует ≥ 3.10, поэтому модель живёт в
своём окружении (`.venv-timesfm`) и общается с ботом строками JSON через stdin/stdout.
Побочный эффект — нагрузка PyTorch не делит процесс и event loop с торговым циклом.

Протокол: при старте печатает {"ready": true}; на запрос
{"id", "series": [...], "horizon": h} отвечает
{"id", "median": [...], "mean": [...], "lo": [...], "hi": [...]} (lo/hi = квантили 0.1/0.9)
или {"id", "error": "..."}. Один запрос — одна строка. Логи — только в stderr.

ОБХОДНОЙ ПУТЬ torch: в torch 2.2.2 (последний для macOS Intel) `scaled_dot_product_attention`
возвращает NaN для полностью замаскированных строк; новые версии дают нули. Без
`nan_to_num` модель отдаёт NaN во всём прогнозе (проверено 2026-11-23).
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

MAX_CONTEXT = 512
MAX_HORIZON = 128
MODEL_ID = "google/timesfm-2.5-200m-pytorch"  # веса Apache-2.0, 925 МБ


def _patch_sdpa() -> None:
    import torch  # noqa: PLC0415

    orig = torch.nn.functional.scaled_dot_product_attention

    def safe(*a: Any, **k: Any) -> Any:
        return torch.nan_to_num(orig(*a, **k), nan=0.0)

    torch.nn.functional.scaled_dot_product_attention = safe


def load_model() -> Any:
    import timesfm  # noqa: PLC0415
    import torch  # noqa: PLC0415

    threads = os.getenv("TIMESFM_THREADS")
    if threads and threads.isdigit():
        torch.set_num_threads(int(threads))
    _patch_sdpa()
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID)
    model.compile(
        timesfm.ForecastConfig(
            max_context=MAX_CONTEXT,
            max_horizon=MAX_HORIZON,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=False,
            fix_quantile_crossing=True,
        )
    )
    return model


def handle_request(model: Any, req: Dict[str, Any]) -> Dict[str, Any]:
    """Один запрос → один ответ. Не бросает исключений: ошибка уходит в поле `error`."""
    rid = req.get("id")
    try:
        import numpy as np  # noqa: PLC0415

        series = req.get("series")
        horizon = int(req.get("horizon", 0))
        if not isinstance(series, list) or len(series) < 8:
            return {"id": rid, "error": "series: нужен список из ≥ 8 чисел"}
        if not 1 <= horizon <= MAX_HORIZON:
            return {"id": rid, "error": f"horizon ∈ [1, {MAX_HORIZON}]"}
        x = np.asarray(series[-MAX_CONTEXT:], dtype=np.float32)
        if not np.isfinite(x).all():
            return {"id": rid, "error": "series содержит NaN/inf"}
        point, q = model.forecast(horizon=horizon, inputs=[x])
        out: Dict[str, List[float]] = {
            "median": q[0, :horizon, 5].tolist(),
            "mean": point[0, :horizon].tolist(),
            "lo": q[0, :horizon, 1].tolist(),
            "hi": q[0, :horizon, 9].tolist(),
        }
        if any(not math.isfinite(v) for arr in out.values() for v in arr):
            return {"id": rid, "error": "модель вернула NaN/inf"}
        return {"id": rid, **out}
    except Exception as exc:  # noqa: BLE001 — воркер не должен падать от одного запроса
        return {"id": rid, "error": f"{type(exc).__name__}: {exc}"}


def serve(model: Any, stdin: Any = None, stdout: Any = None) -> None:
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    stdout.write(json.dumps({"ready": True}) + "\n")
    stdout.flush()
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req: Optional[Dict[str, Any]] = json.loads(line)
        except ValueError:
            stdout.write(json.dumps({"id": None, "error": "bad json"}) + "\n")
            stdout.flush()
            continue
        stdout.write(json.dumps(handle_request(model, req or {})) + "\n")
        stdout.flush()


if __name__ == "__main__":
    serve(load_model())
