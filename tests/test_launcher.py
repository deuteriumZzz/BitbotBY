"""Лаунчер (launcher/launch.py): ключи, токен, окружение compose."""

import importlib.util
import os
import stat
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

_SPEC = importlib.util.spec_from_file_location(
    "launch", Path(__file__).resolve().parent.parent / "launcher" / "launch.py"
)
launch = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(launch)


def test_master_key_is_valid_fernet_and_persists(tmp_path):
    first = launch.load_state(tmp_path)
    Fernet(first["secrets_key"].encode())  # не бросает: формат верный
    assert launch.load_state(tmp_path) == first  # тот же ключ при следующем запуске


@pytest.mark.skipif(os.name == "nt", reason="права файлов в Windows другие")
def test_state_file_is_private(tmp_path):
    launch.load_state(tmp_path)
    mode = stat.S_IMODE(os.stat(tmp_path / "state.json").st_mode)
    assert mode == 0o600


def test_broken_state_is_replaced(tmp_path):
    (tmp_path / "state.json").write_text("не json")
    assert launch.load_state(tmp_path)["secrets_key"]


def test_login_token_is_long_enough_and_unique():
    a, b = launch.new_login_token(), launch.new_login_token()
    assert len(a) >= 32 and a != b


def test_compose_env_and_url(tmp_path):
    text = launch.compose_env_text("KEY", "TOK", 8099, tmp_path)
    assert "BITBOT_SECRETS_KEY=KEY" in text and "BITBOT_LOCAL_LOGIN_TOKEN=TOK" in text
    assert "BITBOT_PORT=8099" in text and f"BITBOT_DATA={tmp_path}" in text
    assert "BITBOT_DATA" not in launch.compose_env_text("K", "T", 8080)
    assert launch.login_url(8099, "TOK") == "http://localhost:8099/app/#local=TOK"


def test_compose_command_uses_env_file(tmp_path):
    cmd = launch.compose_cmd(["ps"], tmp_path)
    assert cmd[:2] == ["docker", "compose"] and cmd[-1] == "ps"
    assert str(tmp_path / "compose.env") in cmd


def test_secrets_never_reach_git_or_docker_context():
    root = Path(__file__).resolve().parent.parent
    assert ".launcher/" in (root / ".gitignore").read_text()
    assert ".launcher/" in (root / ".dockerignore").read_text()


def test_compose_is_paper_and_loopback_only():
    text = (
        Path(__file__).resolve().parent.parent / "docker-compose.local.yml"
    ).read_text()
    assert '"127.0.0.1:' in text and 'PAPER_TRADING: "true"' in text
    assert "0.0.0.0" not in text
