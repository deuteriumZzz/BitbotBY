"""ONNX-экспорт актора SAC: выходы ONNX Runtime совпадают с PyTorch."""

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from reinforcement_learning import onnx_export as ox  # noqa: E402
from reinforcement_learning.env_v2 import TradingEnvV2  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    from stable_baselines3 import SAC

    rng = np.random.default_rng(0)
    n = 200
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    env = TradingEnvV2(rng.normal(size=(n, 13)).astype(np.float32), close, close.copy())
    model = SAC(
        "MlpPolicy",
        env,
        seed=0,
        verbose=0,
        device="cpu",
        learning_starts=20,
        batch_size=16,
        buffer_size=1000,
    )
    model.learn(60)
    path = str(tmp_path_factory.mktemp("onnx") / "tiny.zip")
    model.save(path[:-4])
    return path


def test_export_matches_pytorch_and_runs_without_torch_api(tiny_model):
    out = ox.export_actor(tiny_model)
    assert out.endswith(".onnx")
    res = ox.verify(tiny_model, out, n=100)
    assert res["ok"] and res["max_abs_diff"] < 1e-4 and res["obs_dim"] == 15
    pol = ox.OnnxPolicy(out)
    a1 = pol.action(np.zeros(15, np.float32))
    a2 = pol.action(np.zeros((4, 15), np.float32))  # динамический batch
    assert a1.shape == (1, 1) and a2.shape == (4, 1)
    assert np.all(np.abs(a2) <= 1.0)  # действие SAC в [-1, 1]


def test_cli_exit_codes(tiny_model, tmp_path, capsys):
    assert ox.main([]) == 2
    assert ox.main([tiny_model, str(tmp_path / "x.onnx")]) == 0
    assert "max_abs_diff" in capsys.readouterr().out
