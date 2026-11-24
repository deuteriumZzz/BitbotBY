"""Реестр флагов совпадает с настоящими умолчаниями Config в paper и в live."""

import json
import os
import subprocess
import sys

from bitbot.config.flags import FLAGS

_CODE = (
    "import json; from bitbot.config.settings import Config as C; "
    "from bitbot.config.flags import FLAGS; "
    "print(json.dumps({f.key: getattr(C, f.key) for f in FLAGS}))"
)


def _defaults(paper: str) -> dict:
    keys = {f.key for f in FLAGS} | {"PAPER_TRADING"}
    env = {k: v for k, v in os.environ.items() if k not in keys}
    out = subprocess.run(
        [sys.executable, "-c", _CODE],
        env={**env, "PAPER_TRADING": paper},
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_registry_matches_config_defaults():
    assert _defaults("true") == {f.key: f.paper for f in FLAGS}
    assert _defaults("false") == {f.key: f.live for f in FLAGS}


def test_live_status_means_on_in_both_modes():
    assert {f.status for f in FLAGS} <= {"shadow", "gated", "live"}
    assert all(f.paper and f.live for f in FLAGS if f.status == "live")
