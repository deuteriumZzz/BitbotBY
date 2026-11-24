"""Изоляция нагрузки: ограничения тяжёлых процессов, воркер и клиент TimesFM (без torch)."""

import io
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from src.forecast import timesfm_worker as tw
from src.forecast.timesfm_client import TimesFMWorker
from src.infra import heavy_job as hj


# ── ограничения ресурсов ──────────────────────────────────────────────────────
def test_limit_env_caps_all_thread_vars_and_does_not_mutate_input(monkeypatch):
    monkeypatch.setenv("HEAVY_JOB_THREADS", "3")
    base = {"A": "1"}
    env = hj.limit_env(base)
    assert (
        env["OMP_NUM_THREADS"]
        == env["MKL_NUM_THREADS"]
        == env["TORCH_NUM_THREADS"]
        == "3"
    )
    assert env["A"] == "1" and "OMP_NUM_THREADS" not in base
    assert hj.limit_env({}, threads=2)["TIMESFM_THREADS"] == "2"


def test_default_threads_at_least_one_and_nice_clamped(monkeypatch):
    monkeypatch.delenv("HEAVY_JOB_THREADS", raising=False)
    assert int(hj.limit_env({})["OMP_NUM_THREADS"]) >= 1
    monkeypatch.setenv("HEAVY_JOB_NICE", "99")
    assert hj.nice_level() == 19
    monkeypatch.setenv("HEAVY_JOB_NICE", "abc")
    assert hj.nice_level() == 10


def test_child_process_runs_with_lower_priority(monkeypatch):
    monkeypatch.setenv("HEAVY_JOB_NICE", "10")
    out = subprocess.run(
        [sys.executable, "-c", "import os; print(os.nice(0))"],
        capture_output=True,
        text=True,
        preexec_fn=hj.lower_priority(),
    )
    assert int(out.stdout.strip()) >= 10  # приоритет ребёнка ниже; родитель не затронут


# ── протокол воркера (фейковая модель) ────────────────────────────────────────
class FakeModel:
    def forecast(self, horizon, inputs):
        x = inputs[0]
        point = np.full((1, horizon), float(x[-1]), dtype=np.float32)
        q = np.stack(
            [np.full(horizon, x[-1] + d) for d in np.linspace(-1, 1, 10)], axis=1
        )[None]
        return point, q


def req(**kw):
    base = {"id": 1, "series": list(range(20)), "horizon": 4}
    base.update(kw)
    return base


def test_handle_request_shapes_and_quantile_order():
    r = tw.handle_request(FakeModel(), req())
    assert r["id"] == 1 and len(r["median"]) == 4 and len(r["lo"]) == 4
    assert all(lo <= m <= hi for lo, m, hi in zip(r["lo"], r["median"], r["hi"]))


@pytest.mark.parametrize(
    "bad",
    [
        req(series=[1, 2]),
        req(series="x"),
        req(horizon=0),
        req(horizon=tw.MAX_HORIZON + 1),
        req(series=[1.0] * 7 + [float("nan")]),
    ],
)
def test_handle_request_rejects_bad_input(bad):
    assert "error" in tw.handle_request(FakeModel(), bad)


def test_handle_request_never_raises_and_guards_nan_output():
    class Boom:
        def forecast(self, **_):
            raise RuntimeError("x")

    class NanModel:
        def forecast(self, horizon, inputs):
            return np.full((1, horizon), np.nan), np.full((1, horizon, 10), np.nan)

    assert "RuntimeError" in tw.handle_request(Boom(), req())["error"]
    assert "NaN" in tw.handle_request(NanModel(), req())["error"]


def test_serve_protocol_over_streams():
    stdin = io.StringIO("\n".join([json.dumps(req(id=7)), "not json", ""]) + "\n")
    stdout = io.StringIO()
    tw.serve(FakeModel(), stdin, stdout)
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert (
        lines[0] == {"ready": True}
        and lines[1]["id"] == 7
        and lines[2]["error"] == "bad json"
    )


# ── клиент (фейковый воркер-процесс) ──────────────────────────────────────────
def worker_script(tmp_path, body):
    p = tmp_path / "fake_worker.py"
    p.write_text(textwrap.dedent(body))
    return [sys.executable, str(p)]


GOOD = """
import sys, json
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    r = json.loads(line)
    h = r["horizon"]; last = r["series"][-1]
    print(json.dumps({"id": r["id"], "median": [last]*h, "mean": [last]*h,
                      "lo": [last-1]*h, "hi": [last+1]*h}), flush=True)
"""


def test_client_roundtrip_and_reuse_of_one_process(tmp_path):
    w = TimesFMWorker(
        cmd=worker_script(tmp_path, GOOD), start_timeout_s=10, request_timeout_s=10
    )
    try:
        a = w.forecast([1.0, 2.0, 3.0], 3)
        pid = w._proc.pid
        b = w.forecast([5.0, 6.0], 2)
        assert a["median"] == [3.0, 3.0, 3.0] and b["hi"] == [7.0, 7.0]
        assert w._proc.pid == pid  # модель не перезагружается на каждый запрос
        r = w.as_forecaster()(np.array([1.0, 2.0, 9.0]), 2)
        assert list(r["median"]) == [9.0, 9.0]
    finally:
        w.close()
    assert not w.alive()


def test_client_returns_none_when_worker_never_ready_and_respects_cooldown(tmp_path):
    cmd = worker_script(tmp_path, "import time; time.sleep(30)")
    w = TimesFMWorker(cmd=cmd, start_timeout_s=0.3, cooldown_s=60)
    assert w.forecast([1.0] * 10, 2) is None
    assert w._start() is False  # в паузе после сбоя повторный запуск не пытается


def test_client_handles_timeout_crash_and_error_response(tmp_path):
    hang = worker_script(
        tmp_path, GOOD.replace("h = r[", "import time; time.sleep(30); h = r[")
    )
    w = TimesFMWorker(
        cmd=hang, start_timeout_s=10, request_timeout_s=0.3, cooldown_s=60
    )
    assert w.forecast([1.0] * 10, 2) is None and not w.alive()

    crash = worker_script(tmp_path, GOOD.replace("h = r[", "sys.exit(1); h = r["))
    w2 = TimesFMWorker(
        cmd=crash, start_timeout_s=10, request_timeout_s=5, cooldown_s=60
    )
    assert w2.forecast([1.0] * 10, 2) is None

    err = worker_script(
        tmp_path,
        GOOD.replace('"median": [last]*h,', '"error": "boom", "median": [last]*h,'),
    )
    w3 = TimesFMWorker(cmd=err, start_timeout_s=10, request_timeout_s=5)
    try:
        assert w3.forecast([1.0] * 10, 2) is None
    finally:
        w3.close()


def test_unavailable_forecaster_raises_instead_of_fake_answer():
    w = TimesFMWorker(cmd=["/nonexistent/python"], cooldown_s=60)
    with pytest.raises(RuntimeError):
        w.as_forecaster()(np.arange(20.0), 3)


# ── реальная модель (по желанию: RUN_TIMESFM=1, нужен .venv-timesfm и веса) ───
@pytest.mark.skipif(
    not os.getenv("RUN_TIMESFM") or not os.path.exists(".venv-timesfm/bin/python"),
    reason="реальный TimesFM: RUN_TIMESFM=1 и .venv-timesfm",
)
def test_real_timesfm_beats_naive_on_seasonal_series():
    x = np.sin(np.arange(400) * 2 * np.pi / 24)
    truth = np.sin(np.arange(400, 424) * 2 * np.pi / 24)
    w = TimesFMWorker(start_timeout_s=600, request_timeout_s=120)
    try:
        r = w.forecast(x.tolist(), 24)
        assert r is not None
        assert (
            np.abs(np.array(r["median"]) - truth).mean()
            < np.abs(x[-1] - truth).mean() / 5
        )
    finally:
        w.close()
