"""Пульс бота для дашборда: src/health_server.write_heartbeat → data.check_healthcheck."""

import os
from datetime import datetime, timedelta

from src.dashboard import data
from src.infra.health_server import write_heartbeat


def test_fresh_heartbeat_makes_dashboard_see_the_bot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert data.check_healthcheck() is False  # файла ещё нет
    write_heartbeat(os.path.join("data", "healthcheck.txt"))
    assert data.check_healthcheck() is True
    assert not os.path.exists(os.path.join("data", "healthcheck.txt.tmp"))


def test_stale_heartbeat_means_bot_is_silent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data")
    old = (datetime.now() - timedelta(seconds=300)).isoformat(timespec="seconds")
    with open(os.path.join("data", "healthcheck.txt"), "w") as f:
        f.write(old)
    assert data.check_healthcheck() is False
