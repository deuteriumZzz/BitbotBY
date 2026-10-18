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


def test_pick_port_skips_busy_ones():
    busy = {8080, 8081}
    assert launch.pick_port(8080, lambda p: p in busy) == 8082
    assert launch.pick_port(8080, lambda p: False) == 8080
    assert launch.pick_port(8080, lambda p: True) == 8080  # всё занято: прежний порт


def test_autostart_templates_point_at_the_launcher(tmp_path):
    script = tmp_path / "launch.py"
    plist = launch.launchd_plist(
        "/usr/bin/python3", script, tmp_path, tmp_path / "l.log"
    )
    assert "RunAtLoad" in plist and str(script) in plist and "--no-browser" in plist
    assert launch.LAUNCHD_LABEL in plist
    unit = launch.systemd_unit("/usr/bin/python3", script, tmp_path)
    assert f"ExecStart=/usr/bin/python3 {script} start --no-browser" in unit
    assert "ExecStop=/usr/bin/python3" in unit
    bat = launch.windows_startup_bat("C:\\py\\python.exe", script, tmp_path)
    assert "start --no-browser" in bat and bat.startswith("@echo off")


def test_state_keeps_other_keys_when_saved(tmp_path):
    state = launch.load_state(tmp_path)
    key = state["secrets_key"]
    state["keep_awake"] = "1"
    launch.save_state(state, tmp_path)
    again = launch.load_state(tmp_path)
    assert again["secrets_key"] == key and again["keep_awake"] == "1"


def test_awake_not_running_without_pid_file(tmp_path):
    assert launch.awake_running(tmp_path) is False
    (tmp_path / launch.AWAKE_PID).write_text("not-a-number")
    assert launch.awake_running(tmp_path) is False


def test_tunnel_url_is_found_in_cloudflared_output():
    log = "INF Requesting new quick Tunnel\nINF |  https://funny-name-1234.trycloudflare.com  |\n"
    assert launch.find_tunnel_url(log) == "https://funny-name-1234.trycloudflare.com"
    assert launch.find_tunnel_url("https://example.com no tunnel here") is None


def test_public_address_adds_miniapp_url_and_closes_reads():
    plain = launch.compose_env_text("K", "T", 8080)
    assert "WEBAPP_URL" not in plain and "DASHBOARD_API_KEY" not in plain
    text = launch.compose_env_text(
        "K", "T", 8080, None, "https://x.trycloudflare.com", "KEY"
    )
    assert "WEBAPP_URL=https://x.trycloudflare.com/app/" in text
    assert "DASHBOARD_API_KEY=KEY" in text


def test_compose_passes_the_miniapp_address_and_key_through():
    text = (
        Path(__file__).resolve().parent.parent / "docker-compose.local.yml"
    ).read_text()
    assert "WEBAPP_URL: ${WEBAPP_URL:-}" in text
    assert "DASHBOARD_API_KEY: ${DASHBOARD_API_KEY:-}" in text


def test_stale_tunnel_file_is_not_alive(tmp_path):
    (tmp_path / launch.TUNNEL_FILE).write_text(
        '{"pid": 99999999, "url": "https://a.trycloudflare.com"}'
    )
    assert launch.tunnel_alive(launch.tunnel_info(tmp_path)) is False
    assert launch.tunnel_info(tmp_path / "missing") == {}


def test_env_for_import_drops_comments_and_empty_values():
    text = "# c\nGROQ_API_KEY=abc\nEMPTY=\n  # x=1\nMODE=ai\n"
    assert launch.env_for_import(text) == "GROQ_API_KEY=abc\nMODE=ai\n"
