"""Режим бота для кода и дочерних процессов — явно, а не через копию в os.environ.

Раньше профиль и тип рынка копировались в os.environ: после рестарта копия
терялась (сделки altcoin писались в общий файл опыта), а сделки meme попадали
в файл bluechip, потому что путь к файлу считался в трёх местах по-разному.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from bitbot.config import runtime
from bitbot.config.settings import Config


@pytest.fixture(autouse=True)
def _reset_active():
    yield
    runtime.set_active_profile("")
    runtime.set_active_market_type("")


def test_experiences_file_per_profile():
    assert runtime.experiences_path("altcoin") == "data/experiences_altcoin.jsonl"
    assert runtime.experiences_path("meme") == "data/experiences_meme.jsonl"
    assert runtime.experiences_path("bluechip") == "data/experiences.jsonl"
    assert runtime.experiences_path("") == "data/experiences.jsonl"


def test_child_env_follows_what_the_bot_runs_now():
    runtime.set_active_profile("meme")
    runtime.set_active_market_type("spot")
    with patch.object(Config, "PAPER_TRADING", True):
        assert runtime.child_env() == {
            "PAPER_TRADING": "true",
            "MARKET_TYPE": "spot",
            "SAC_PROFILE": "meme",
            "EXPERIENCES_PATH": "data/experiences_meme.jsonl",
        }


def test_runtime_switches_do_not_touch_process_env(monkeypatch):
    for k in ("SAC_PROFILE", "EXPERIENCES_PATH", "MARKET_TYPE"):
        monkeypatch.delenv(k, raising=False)
    redis = MagicMock()
    redis.redis_client.get.return_value = None
    redis.redis_client.smembers.return_value = set()
    rc = runtime.RuntimeConfig(redis_client=redis)

    assert rc.apply_market_profile("altcoin") and rc.set_market_type("spot")

    assert runtime.active_profile() == "altcoin"
    assert runtime.active_market_type() == "spot"
    assert not {"SAC_PROFILE", "EXPERIENCES_PATH", "MARKET_TYPE"} & set(os.environ)
