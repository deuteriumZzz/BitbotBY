#!/usr/bin/env python3
"""Лаунчер BitbotBY: один запуск — и приложение открыто на этом компьютере.

    python launcher/launch.py          # запустить (по умолчанию) и открыть приложение
    python launcher/launch.py stop     # остановить
    python launcher/launch.py status   # что запущено
    python launcher/launch.py logs     # последние логи
    python launcher/launch.py url      # новая ссылка входа, если окно закрыли

Нужны только Docker и Python 3 (стандартная библиотека, ничего ставить не надо).
Бот стартует в Paper. Ключи API вводятся в самом приложении и хранятся зашифрованными.
Мастер-ключ шифрования лаунчер создаёт сам и кладёт в .launcher/state.json (права 600,
в git и в образ Docker не попадает).
"""

import base64
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / ".launcher"
COMPOSE_FILE = ROOT / "docker-compose.local.yml"
DEFAULT_PORT = 8080


def say(text: str = "") -> None:
    print(text, flush=True)


def write_private(path: Path, text: str) -> None:
    """Файл с правами только для владельца (ключи и токены)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    try:
        os.chmod(path, 0o600)  # если файл уже был с другими правами
    except OSError:
        pass  # Windows: права не те же, не критично


def new_master_key() -> str:
    """Ключ в формате Fernet: 32 случайных байта в urlsafe base64."""
    return base64.urlsafe_b64encode(os.urandom(32)).decode()


def load_state(state_dir: Path = STATE_DIR) -> Dict[str, str]:
    """Мастер-ключ живёт между запусками: без него зашифрованные ключи не прочитать."""
    path = state_dir / "state.json"
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(state, dict) and state.get("secrets_key"):
                return {k: str(v) for k, v in state.items()}
        except ValueError:
            pass
    state = {"secrets_key": new_master_key()}
    write_private(path, json.dumps(state))
    return state


def new_login_token() -> str:
    return secrets.token_urlsafe(32)  # 43 символа, сервер требует от 32


def compose_env_text(
    secrets_key: str, token: str, port: int, data_dir: Optional[Path] = None
) -> str:
    lines = [
        f"BITBOT_SECRETS_KEY={secrets_key}",
        f"BITBOT_LOCAL_LOGIN_TOKEN={token}",
        f"BITBOT_PORT={port}",
    ]
    if data_dir is not None:
        lines += [
            f"BITBOT_DATA={data_dir / 'data'}",
            f"BITBOT_MODELS={data_dir / 'models'}",
            f"BITBOT_LOGS={data_dir / 'logs'}",
        ]
    return "\n".join(lines) + "\n"


def login_url(port: int, token: str) -> str:
    return f"http://localhost:{port}/app/#local={token}"


def compose_cmd(args: List[str], state_dir: Path = STATE_DIR) -> List[str]:
    return [
        "docker",
        "compose",
        "-f",
        str(COMPOSE_FILE),
        "--env-file",
        str(state_dir / "compose.env"),
        *args,
    ]


def run(cmd: List[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(ROOT), **kw)


def docker_ready() -> bool:
    try:
        return run(["docker", "info"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def ensure_docker() -> bool:
    """Docker установлен и запущен (на macOS и Windows запускаем Docker Desktop)."""
    try:
        run(["docker", "--version"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        say(
            "Не найден Docker. Установите Docker Desktop: "
            "https://www.docker.com/products/docker-desktop/"
        )
        say("После установки запустите это снова.")
        return False
    if docker_ready():
        return True
    say("Docker не запущен, запускаю Docker Desktop…")
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", "-a", "Docker"])
        elif sys.platform.startswith("win"):
            exe = os.path.join(
                os.environ.get("ProgramFiles", r"C:\Program Files"),
                "Docker",
                "Docker",
                "Docker Desktop.exe",
            )
            subprocess.Popen([exe])
    except OSError:
        pass
    for _ in range(40):  # до ~2 минут
        time.sleep(3)
        if docker_ready():
            return True
    say("Docker так и не запустился. Откройте Docker Desktop вручную и повторите.")
    return False


def wait_healthy(port: int, timeout: int = 600) -> bool:
    """Ждём, пока дашборд ответит; первая сборка образа занимает несколько минут."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=3
            ) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(2)
    return False


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def own_dashboard_running() -> bool:
    try:
        out = run(
            ["docker", "ps", "-q", "--filter", "name=bitbotby-dashboard-1"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return bool(out.stdout.strip())


def pick_port(preferred: int, busy: Optional[Callable[[int], bool]] = None) -> int:
    """Первый свободный порт начиная с желаемого (8080 часто занят)."""
    is_busy = busy or (lambda p: not port_free(p))
    for port in range(preferred, preferred + 30):
        if not is_busy(port):
            return port
    return preferred


def previous_port() -> Optional[int]:
    try:
        for line in (
            (STATE_DIR / "compose.env").read_text(encoding="utf-8").splitlines()
        ):
            if line.startswith("BITBOT_PORT="):
                return int(line.split("=", 1)[1])
    except (OSError, ValueError):
        pass
    return None


def prepare(port: int, data_dir: Optional[Path]) -> str:
    """Пишет compose.env с новым токеном входа; возвращает ссылку для браузера."""
    state = load_state()
    token = new_login_token()
    write_private(
        STATE_DIR / "compose.env",
        compose_env_text(state["secrets_key"], token, port, data_dir),
    )
    return login_url(port, token)


def cmd_start(port: int, data_dir: Optional[Path], open_browser: bool = True) -> int:
    if not ensure_docker():
        return 1
    # Если наш дашборд уже работает, остаёмся на его порту; иначе берём свободный.
    previous = previous_port()
    if previous is not None and own_dashboard_running():
        port = previous
    else:
        chosen = pick_port(port)
        if chosen != port:
            say(f"Порт {port} занят другой программой, использую {chosen}.")
        port = chosen
    url = prepare(port, data_dir)
    say("Запускаю BitbotBY (Paper: сделки виртуальные).")
    say(
        "Первая сборка занимает 5–15 минут: скачиваются пакеты. "
        "Следующие запуски быстрые."
    )
    if run(compose_cmd(["up", "-d", "--build"])).returncode != 0:
        say("Не получилось запустить. Логи: python launcher/launch.py logs")
        return 1
    if not wait_healthy(port):
        say(
            "Дашборд не ответил вовремя. "
            "Посмотрите логи: python launcher/launch.py logs"
        )
        return 1
    say(f"\nГотово. Приложение: http://localhost:{port}/app/")
    say(
        "Ключи (Bybit, AI) вводите в приложении: Настройки → Подключения. "
        "Они хранятся зашифрованными."
    )
    say("Остановить: python launcher/launch.py stop")
    if open_browser:
        webbrowser.open(url)
    else:
        say(f"Ссылка входа (одноразовая): {url}")
    return 0


def cmd_url(port: int, data_dir: Optional[Path]) -> int:
    """Новая ссылка входа: токен меняется, поэтому пересоздаётся только дашборд."""
    if not ensure_docker():
        return 1
    url = prepare(port, data_dir)
    if run(compose_cmd(["up", "-d", "dashboard"])).returncode != 0 or not wait_healthy(
        port, 120
    ):
        say("Не удалось перезапустить дашборд.")
        return 1
    webbrowser.open(url)
    return 0


def main(argv: List[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    flags = {
        a.split("=", 1)[0]: a.split("=", 1)[1]
        for a in argv
        if a.startswith("--") and "=" in a
    }
    action = args[0] if args else "start"
    port = int(os.getenv("BITBOT_PORT", str(DEFAULT_PORT)))
    data_dir = Path(flags["--data-dir"]).resolve() if "--data-dir" in flags else None
    if (
        action in ("stop", "status", "logs")
        and not (STATE_DIR / "compose.env").exists()
    ):
        say("Ещё не запускалось: нечего показывать.")
        return 0
    if action == "start":
        return cmd_start(port, data_dir, open_browser="--no-browser" not in argv)
    if action == "url":
        return cmd_url(port, data_dir)
    if action == "stop":
        return run(compose_cmd(["down"])).returncode
    if action == "status":
        return run(compose_cmd(["ps"])).returncode
    if action == "logs":
        return run(compose_cmd(["logs", "--tail", "80"])).returncode
    say(__doc__ or "")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
