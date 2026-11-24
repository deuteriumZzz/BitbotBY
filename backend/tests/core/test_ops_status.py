"""ops_status / run_config и умолчания флагов «в paper»."""

import os
import subprocess
import sys
from types import SimpleNamespace

from bitbot.config.flags import FLAGS
from bitbot.workers import ops as ops_tasks


def _flag_defaults(**env):
    e = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PAPER_TRADING", "WS_SHADOW_ENABLED", "AI_SANITIZE_PROMPT")
    }
    e.update(env)
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "from bitbot.config.settings import Config as C; print(C.WS_SHADOW_ENABLED, C.AI_SANITIZE_PROMPT)",
        ],
        env=e,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return out


def test_flags_follow_paper_mode():
    assert _flag_defaults(PAPER_TRADING="true") == ["True", "True"]
    assert _flag_defaults(PAPER_TRADING="false") == ["False", "False"]


def test_explicit_env_overrides_paper_default():
    assert _flag_defaults(PAPER_TRADING="true", WS_SHADOW_ENABLED="false") == [
        "False",
        "True",
    ]


def _bot(**kw):
    return SimpleNamespace(ws_stream=None, _last_backup=None, **kw)


def test_ops_status_shape_without_ws():
    st = ops_tasks.ops_status(_bot())
    assert set(st["flags"]) == {f.key for f in FLAGS}
    assert all(isinstance(v, bool) for v in st["flags"].values())
    assert st["ws"] is None and st["backup"] is None and st["emergency"] == 0
    assert st["mode"] in ("paper", "live")


def test_run_config_record_writes_ledger_and_swallows_errors():
    calls = []
    bot = _bot(_ledger_write=lambda kind, **kw: calls.append((kind, kw)))
    rec = ops_tasks.run_config_record(bot)
    assert calls[0][0] == "run_config" and calls[0][1]["flags"] == rec["flags"]

    def boom(*a, **k):
        raise RuntimeError("x")

    ops_tasks.run_config_record(_bot(_ledger_write=boom))  # не падает
