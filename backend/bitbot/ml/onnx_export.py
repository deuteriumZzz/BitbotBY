"""Экспорт актора SAC в ONNX и проверка совпадения с PyTorch (реестр идей №3).

Экспортируется ДЕТЕРМИНИРОВАННЫЙ актор (то, что делает `predict(deterministic=True)`): вход
`obs [batch, obs_dim]`, выход `action [batch, act_dim]` в [-1, 1]. `verify` сравнивает выходы
ONNX Runtime и PyTorch на случайных наблюдениях — без совпадения файл использовать нельзя.
Зачем: инференс без PyTorch в боевом процессе (меньше памяти и зависимостей); путь не подключён
к боту — `SACV2Policy` по-прежнему грузит .zip. Зависимости `onnx`, `onnxruntime` необязательны.
CLI:  python -m bitbot.ml.onnx_export models/sac_v2.zip [models/sac_v2.onnx]
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional

import numpy as np


def _load(model_path: str) -> Any:
    from stable_baselines3 import SAC  # noqa: PLC0415

    return SAC.load(model_path, device="cpu")


def export_actor(
    model_path: str, out_path: Optional[str] = None, opset: int = 17
) -> str:
    import torch  # noqa: PLC0415

    model = _load(model_path)
    actor = model.policy.actor.eval()
    obs_dim = int(model.observation_space.shape[0])
    out_path = (
        out_path
        or (model_path[:-4] if model_path.endswith(".zip") else model_path) + ".onnx"
    )

    class Deterministic(torch.nn.Module):
        def __init__(self, a: Any) -> None:
            super().__init__()
            self.a = a

        def forward(self, obs: Any) -> Any:
            return self.a(obs, deterministic=True)

    torch.onnx.export(
        Deterministic(actor),
        torch.zeros(1, obs_dim),
        out_path,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=opset,
    )
    return out_path


def verify(
    model_path: str, onnx_path: str, n: int = 200, atol: float = 1e-4, seed: int = 0
) -> Dict[str, Any]:
    import onnxruntime as ort  # noqa: PLC0415
    import torch  # noqa: PLC0415

    model = _load(model_path)
    actor = model.policy.actor.eval()
    obs_dim = int(model.observation_space.shape[0])
    obs = np.random.default_rng(seed).normal(size=(n, obs_dim)).astype(np.float32)
    with torch.no_grad():
        ref = actor(torch.as_tensor(obs), deterministic=True).numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    got = sess.run(["action"], {"obs": obs})[0]
    max_diff = float(np.max(np.abs(ref - got)))
    return {
        "n": n,
        "max_abs_diff": max_diff,
        "ok": bool(max_diff <= atol),
        "obs_dim": obs_dim,
    }


class OnnxPolicy:
    """Инференс без PyTorch: `action(obs)` → массив [-1, 1]."""

    def __init__(self, onnx_path: str) -> None:
        import onnxruntime as ort  # noqa: PLC0415

        self._s = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def action(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, np.float32)
        x = x.reshape(1, -1) if x.ndim == 1 else x
        return self._s.run(["action"], {"obs": x})[0]


def main(argv: Optional[list] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("Использование: python -m bitbot.ml.onnx_export <model.zip> [out.onnx]")
        return 2
    out = export_actor(args[0], args[1] if len(args) > 1 else None)
    res = verify(args[0], out)
    print(out, res)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
