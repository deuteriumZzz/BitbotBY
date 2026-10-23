#!/usr/bin/env python3
"""Лаунчер BitbotBY: один запуск — и приложение открыто на этом компьютере.

    python launcher/launch.py          # запустить (по умолчанию) и открыть приложение
    python launcher/launch.py stop     # остановить
    python launcher/launch.py status   # что запущено
    python launcher/launch.py logs     # последние логи
    python launcher/launch.py url      # новая ссылка входа, если окно закрыли
    python launcher/launch.py autostart on|off|status   # запуск при входе в систему
    python launcher/launch.py awake on|off|status       # не давать компьютеру уснуть
    python launcher/launch.py import-env [--env=путь] [--overwrite]   # ключи из .env → хранилище
    python launcher/launch.py phone on|off|status       # MiniApp в Telegram: адрес до компьютера
    python3 launcher/launch.py server --domain=bot.example.com   # на сервере (VPS), см. docs/DEPLOY_SERVER.md

Нужны только Docker и Python 3 (стандартная библиотека, ничего ставить не надо).
Бот стартует в Paper. Ключи API вводятся в самом приложении и хранятся зашифрованными.
Мастер-ключ шифрования лаунчер создаёт сам и кладёт в .launcher/state.json (права 600,
в git и в образ Docker не попадает).
"""

import base64
import contextlib
import getpass
import io
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
    secrets_key: str,
    token: str,
    port: int,
    data_dir: Optional[Path] = None,
    webapp_url: str = "",
    api_key: str = "",
) -> str:
    lines = [
        f"BITBOT_SECRETS_KEY={secrets_key}",
        f"BITBOT_LOCAL_LOGIN_TOKEN={token}",
        f"BITBOT_PORT={port}",
    ]
    if webapp_url:
        # MiniApp без публичного адреса не открыть, а с адресом закрываем чтение без входа.
        lines += [f"WEBAPP_URL={webapp_url}/app/", f"DASHBOARD_API_KEY={api_key}"]
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


def prepare(port: int, data_dir: Optional[Path], webapp_url: str = "") -> str:
    """Пишет compose.env с новым токеном входа; возвращает ссылку для браузера."""
    state = load_state()
    token = new_login_token()
    write_private(
        STATE_DIR / "compose.env",
        compose_env_text(
            state["secrets_key"],
            token,
            port,
            data_dir,
            webapp_url,
            state.get("api_key", "") if webapp_url else "",
        ),
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
    public = public_url(port, start=True)
    url = prepare(port, data_dir, public)
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
    if public:
        say(f"MiniApp в Telegram: {public}/app/ (кнопка «Терминал» в вашем боте)")
        say(
            "Чтобы MiniApp заработал, подключите Telegram в приложении (токен и chat id)."
        )
    say(
        "Ключи (Bybit, AI) вводите в приложении: Настройки → Подключения. "
        "Они хранятся зашифрованными."
    )
    say("Остановить: python launcher/launch.py stop")
    if load_state().get("keep_awake") == "1":
        spawn_awake()
    spawn_helper(port)
    if open_browser:
        webbrowser.open(url)
    else:
        say(f"Ссылка входа (одноразовая): {url}")
    return 0


def public_url(port: int, start: bool = False) -> str:
    """Публичный https-адрес для MiniApp, если включён «phone on»; иначе пусто."""
    state = load_state()
    if state.get("phone") != "1":
        return ""
    if state.get("webapp_url"):
        return state["webapp_url"]
    info = tunnel_info()
    if tunnel_alive(info) and info.get("url"):
        return str(info["url"])
    return (start_tunnel(port) or "") if start else ""


def cmd_url(port: int, data_dir: Optional[Path]) -> int:
    """Новая ссылка входа: токен меняется, поэтому пересоздаётся только дашборд."""
    if not ensure_docker():
        return 1
    url = prepare(port, data_dir, public_url(port))
    if run(compose_cmd(["up", "-d", "dashboard"])).returncode != 0 or not wait_healthy(
        port, 120
    ):
        say("Не удалось перезапустить дашборд.")
        return 1
    webbrowser.open(url)
    return 0


# ── Доступ с телефона: адрес для MiniApp через туннель Cloudflare ────────────
# MiniApp открывается Telegram по https, поэтому телефону нужен адрес до компьютера.
# Быстрый туннель cloudflared даёт его без аккаунта. Адрес публичный и меняется при
# каждом новом туннеле, поэтому вместе с ним включается DASHBOARD_API_KEY: без входа
# через Telegram (MiniApp) или ссылку лаунчера данные приложения не отдаются.

_TUNNEL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
TUNNEL_FILE = "tunnel.json"


def find_tunnel_url(text: str) -> Optional[str]:
    m = _TUNNEL_RE.search(text)
    return m.group(0) if m else None


def tunnel_info(state_dir: Path = STATE_DIR) -> Dict[str, object]:
    try:
        info = json.loads((state_dir / TUNNEL_FILE).read_text(encoding="utf-8"))
        return info if isinstance(info, dict) else {}
    except (OSError, ValueError):
        return {}


def tunnel_alive(info: Dict[str, object]) -> bool:
    try:
        os.kill(int(str(info.get("pid", ""))), 0)
        return True
    except (OSError, ValueError):
        return False


def stop_tunnel(state_dir: Path = STATE_DIR) -> None:
    info = tunnel_info(state_dir)
    if tunnel_alive(info):
        try:
            os.kill(int(str(info["pid"])), 15)
        except OSError:
            pass
    (state_dir / TUNNEL_FILE).unlink(missing_ok=True)


def start_tunnel(port: int, wait: int = 45) -> Optional[str]:
    """Адрес https для MiniApp; None, если cloudflared нет или туннель не поднялся."""
    info = tunnel_info()
    if tunnel_alive(info) and info.get("port") == port and info.get("url"):
        return str(info["url"])
    stop_tunnel()
    exe = shutil.which("cloudflared")
    if not exe:
        say("Для доступа с телефона нужен cloudflared: brew install cloudflared")
        say("(или другой https-адрес до этого компьютера: phone on --url=https://…)")
        return None
    log_path = STATE_DIR / "tunnel.log"
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [exe, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=not sys.platform.startswith("win"),
        )
    deadline = time.time() + wait
    while time.time() < deadline:
        url = find_tunnel_url(log_path.read_text(encoding="utf-8", errors="replace"))
        if url:
            write_private(
                STATE_DIR / TUNNEL_FILE,
                json.dumps({"pid": proc.pid, "url": url, "port": port}),
            )
            return url
        if proc.poll() is not None:
            break
        time.sleep(1)
    say("Туннель не поднялся. Смотрите .launcher/tunnel.log")
    proc.terminate()
    return None


def cmd_phone(action: str, port: int, flags: Dict[str, str]) -> int:
    state = load_state()
    if action == "status":
        info = tunnel_info()
        on = state.get("phone") == "1"
        say("Доступ с телефона: " + ("включён" if on else "выключен"))
        if on and tunnel_alive(info):
            say(f"Адрес MiniApp: {info.get('url')}/app/")
        return 0
    if action == "off":
        state["phone"] = "0"
        state.pop("webapp_url", None)
        save_state(state)
        stop_tunnel()
        if (STATE_DIR / "compose.env").exists() and ensure_docker():
            prepare(port, None)
            run(compose_cmd(["up", "-d"]))
        say("Доступ с телефона выключен, адрес закрыт.")
        return 0
    if action != "on":
        say("Используйте: phone on [--url=https://…] | off | status")
        return 2
    if "--url" in flags:
        if not flags["--url"].startswith("https://"):
            say("Адрес должен начинаться с https://")
            return 2
        state["webapp_url"] = flags["--url"].rstrip("/")
    state["phone"] = "1"
    state.setdefault("api_key", secrets.token_urlsafe(32))
    save_state(state)
    say("Включаю доступ с телефона и перезапускаю бота с новым адресом…")
    return cmd_start(port, None, open_browser=False)


def env_for_import(text: str) -> str:
    """Из .env берём только строки KEY=VALUE: комментарии и пустое не нужны."""
    keep = [
        ln
        for ln in text.splitlines()
        if "=" in ln and not ln.lstrip().startswith("#") and ln.split("=", 1)[1].strip()
    ]
    return "\n".join(keep) + "\n"


def cmd_import_env(env_path: Path, overwrite: bool) -> int:
    """Переносит ключи из .env в зашифрованное хранилище приложения."""
    if not env_path.exists():
        say(f"Не нашёл {env_path}. Укажите файл: import-env --env=/путь/к/.env")
        return 2
    if not (STATE_DIR / "compose.env").exists() or not own_dashboard_running():
        say("Сначала запустите BitbotBY (BitbotBY.command), потом повторите.")
        return 2
    say("Переношу ключи из .env в зашифрованное хранилище (значения не показываются)…")
    cmd = compose_cmd(["exec", "-T", "dashboard", "python", "-m", "src.secrets_import"])
    if overwrite:
        cmd.append("--overwrite")
    proc = run(
        cmd,
        input=env_for_import(env_path.read_text(encoding="utf-8")),
        text=True,
        capture_output=True,
    )
    say((proc.stdout or "").strip() or (proc.stderr or "").strip()[-300:])
    if proc.returncode != 0:
        return proc.returncode
    run(compose_cmd(["restart", "bot"]), capture_output=True)
    say(
        "Бот перезапущен и подхватит ключи. Проверка у сервисов: Настройки → Подключения."
    )
    say("Теперь ключи из .env можно удалить — они уже в хранилище.")
    return 0


# ── Сервер (VPS): docker-compose.server.yml + Caddy с HTTPS ───────────────────
#
# Одна команда на сервере: готовит .server/server.env (мастер-ключ хранилища и ключ
# панели — один раз и навсегда), запускает всё и, если Telegram ещё не подключён,
# спрашивает токен бота и находит ваш чат по коду. Ключи уходят в зашифрованное
# хранилище тем же src.secrets_import, что и import-env; .env не нужен.

SERVER_DIR = ROOT / ".server"
SERVER_COMPOSE = ROOT / "docker-compose.server.yml"
_DOMAIN = re.compile(
    r"(?=.{4,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}"
)


def read_env_file(path: Path) -> Dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    return dict(ln.split("=", 1) for ln in lines if "=" in ln)


def server_env_text(domain: str, secrets_key: str, api_key: str) -> str:
    return (
        "\n".join(
            [
                f"DOMAIN={domain}",
                f"WEBAPP_URL=https://{domain}/app/",
                f"BITBOT_SECRETS_KEY={secrets_key}",
                f"DASHBOARD_API_KEY={api_key}",
            ]
        )
        + "\n"
    )


def prepare_server(domain: str, server_dir: Path = SERVER_DIR) -> Path:
    """Пишет server.env; ключи прошлого запуска сохраняются — иначе хранилище не прочитать."""
    path = server_dir / "server.env"
    old = read_env_file(path)
    write_private(
        path,
        server_env_text(
            domain,
            old.get("BITBOT_SECRETS_KEY") or new_master_key(),
            old.get("DASHBOARD_API_KEY") or secrets.token_urlsafe(32),
        ),
    )
    return path


def server_compose(args: List[str], server_dir: Path = SERVER_DIR) -> List[str]:
    return [
        "docker",
        "compose",
        "-f",
        str(SERVER_COMPOSE),
        "--env-file",
        str(server_dir / "server.env"),
        *args,
    ]


def chat_by_code(payload: Dict[str, object], code: str) -> Optional[str]:
    """Личный чат, приславший боту ровно этот код (ответ getUpdates)."""
    for upd in payload.get("result") or []:  # type: ignore[union-attr]
        msg = upd.get("message") or {}
        chat = msg.get("chat") or {}
        if chat.get("type") == "private" and str(msg.get("text") or "").strip() == code:
            return str(chat["id"])
    return None


def wait_chat(token: str, code: str, timeout: int = 300) -> Optional[str]:
    """Ждёт код в личке бота. Токен живёт только в памяти и нигде не печатается."""
    url = f"https://api.telegram.org/bot{token}/getUpdates?timeout=0"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                chat = chat_by_code(json.loads(r.read().decode()), code)
            if chat:
                return chat
        except urllib.error.HTTPError as e:
            if e.code in (401, 404):
                say("Токен не подошёл: проверьте его в @BotFather.")
                return None
            if e.code == 409:
                say("Этого бота уже опрашивает другая программа — остановите её.")
                return None
        except (OSError, ValueError):
            pass
        time.sleep(3)
    return None


def server_telegram_ready() -> bool:
    check = (
        "from src import secrets_store as s;"
        "print(bool(s.get_value('TELEGRAM_BOT_TOKEN') and s.get_value('TELEGRAM_CHAT_ID')))"
    )
    proc = run(
        server_compose(["exec", "-T", "dashboard", "python", "-c", check]),
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() == "True"


def setup_server_telegram() -> bool:
    say(
        "\nПодключим Telegram — через него вы входите в приложение и получаете уведомления."
    )
    say("1. В Telegram откройте @BotFather, команда /newbot, скопируйте токен бота.")
    token = getpass.getpass("2. Вставьте токен (ввод не показывается): ").strip()
    if not token:
        say("Токен пустой. Подключите позже: эта же команда спросит снова.")
        return False
    code = f"{secrets.randbelow(10**6):06d}"
    say(f"3. Откройте своего бота в Telegram и отправьте ему: {code}")
    say("   Жду до 5 минут…")
    chat = wait_chat(token, code)
    if not chat:
        say("Код не пришёл. Запустите команду ещё раз.")
        return False
    proc = run(
        server_compose(
            [
                "exec",
                "-T",
                "dashboard",
                "python",
                "-m",
                "src.secrets_import",
                "--overwrite",
            ]
        ),
        input=f"TELEGRAM_BOT_TOKEN={token}\nTELEGRAM_CHAT_ID={chat}\n",
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        say("Не получилось сохранить ключ. Логи: docker compose … logs dashboard")
        return False
    run(server_compose(["restart", "bot"]), capture_output=True)
    say("Telegram подключён: токен и чат — в зашифрованном хранилище.")
    return True


def cmd_server(flags: Dict[str, str]) -> int:
    domain = (
        (
            (
                flags.get("--domain")
                or read_env_file(SERVER_DIR / "server.env").get("DOMAIN")
            )
            or ""
        )
        .strip()
        .lower()
    )
    if not _DOMAIN.fullmatch(domain):
        say(
            "Укажите домен сервера: python3 launcher/launch.py server --domain=bot.example.com"
        )
        return 2
    if not docker_ready():
        say("Docker не найден или не запущен: https://docs.docker.com/engine/install/")
        return 1
    prepare_server(domain)
    say(f"Запускаю BitbotBY на {domain} (Paper: сделки виртуальные).")
    say("Первая сборка занимает 5–15 минут, сертификат HTTPS Caddy получит сам.")
    if run(server_compose(["up", "-d", "--build", "--wait"])).returncode != 0:
        say("Не получилось запустить. Логи: python3 launcher/launch.py server-logs")
        return 1
    if not server_telegram_ready() and not setup_server_telegram():
        return 1
    say(f"\nГотово. Приложение: https://{domain}/app/ — вход по коду из вашего бота.")
    say("В самом боте появится кнопка «Терминал» (MiniApp).")
    say("Ключи Bybit и AI вводите в приложении: Настройки → Подключения.")
    say("Сохраните копию .server/server.env: без мастер-ключа ключи не прочитать.")
    return 0


# ── Автозапуск при входе в систему и «не засыпать» ───────────────────────────
# Оба режима включаются явной командой: это изменение настроек вашей системы.

LAUNCHD_LABEL = "by.bitbotby.launcher"
AWAKE_PID = "awake.pid"


def save_state(state: Dict[str, str], state_dir: Path = STATE_DIR) -> None:
    write_private(state_dir / "state.json", json.dumps(state))


def launchd_plist(
    python: str, script: Path, root: Path, log: Path
) -> str:  # noqa: E501
    """Агент входа macOS: при входе в систему поднимает бота без окна браузера."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>{LAUNCHD_LABEL}</string>
  <key>ProgramArguments</key><array>
    <string>{python}</string><string>{script}</string><string>start</string><string>--no-browser</string>
  </array>
  <key>WorkingDirectory</key><string>{root}</string>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>{log}</string>
  <key>StandardErrorPath</key><string>{log}</string>
</dict></plist>
"""


def systemd_unit(python: str, script: Path, root: Path) -> str:
    return f"""[Unit]
Description=BitbotBY (локальный запуск)
After=network-online.target docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory={root}
ExecStart={python} {script} start --no-browser
ExecStop={python} {script} stop

[Install]
WantedBy=default.target
"""


def windows_startup_bat(python: str, script: Path, root: Path) -> str:
    return (
        f'@echo off\r\ncd /d "{root}"\r\n"{python}" "{script}" start --no-browser\r\n'
    )


def autostart_file() -> Path:
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
    if sys.platform.startswith("win"):
        appdata = Path(os.environ.get("APPDATA", str(home)))
        return appdata / "Microsoft/Windows/Start Menu/Programs/Startup/BitbotBY.bat"
    return home / ".config" / "systemd" / "user" / "bitbotby.service"


def cmd_autostart(action: str) -> int:
    path = autostart_file()
    script = Path(__file__).resolve()
    if action == "status":
        say("Автозапуск при входе: " + ("включён" if path.exists() else "выключен"))
        return 0
    if action == "off":
        if sys.platform == "darwin" and path.exists():
            run(
                ["launchctl", "bootout", f"gui/{os.getuid()}", str(path)],
                capture_output=True,
            )
        elif sys.platform.startswith("linux") and path.exists():
            run(
                ["systemctl", "--user", "disable", "--now", "bitbotby.service"],
                capture_output=True,
            )
        path.unlink(missing_ok=True)
        say("Автозапуск выключен.")
        return 0
    if action != "on":
        say("Используйте: autostart on | off | status")
        return 2
    path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        path.write_text(
            launchd_plist(sys.executable, script, ROOT, STATE_DIR / "autostart.log"),
            encoding="utf-8",
        )
        run(
            ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)],
            capture_output=True,
        )
    elif sys.platform.startswith("win"):
        path.write_text(
            windows_startup_bat(sys.executable, script, ROOT), encoding="utf-8"
        )
    else:
        path.write_text(systemd_unit(sys.executable, script, ROOT), encoding="utf-8")
        run(
            ["systemctl", "--user", "enable", "--now", "bitbotby.service"],
            capture_output=True,
        )
    say("Автозапуск включён: бот будет подниматься сам при входе в систему.")
    say(
        "Docker Desktop тоже должен запускаться при входе "
        "(его настройка «Start Docker Desktop when you sign in»)."
    )
    return 0


def awake_running(state_dir: Path = STATE_DIR) -> bool:
    try:
        pid = int((state_dir / AWAKE_PID).read_text().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def spawn_awake() -> None:
    """Фоновый процесс, который не даёт компьютеру уснуть, пока работает бот."""
    if awake_running():
        return
    kw: Dict[str, object] = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
    }
    if sys.platform.startswith("win"):
        kw["creationflags"] = 0x00000008  # DETACHED_PROCESS
    else:
        kw["start_new_session"] = True
    me = str(Path(__file__).resolve())
    subprocess.Popen([sys.executable, me, "awake-run"], **kw)  # type: ignore[call-overload]


def bot_running() -> bool:
    try:
        out = run(
            ["docker", "ps", "-q", "--filter", "name=bitbotby-bot-1"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return True  # не можем проверить — не отпускаем сон раньше времени
    return bool(out.stdout.strip())


def cmd_awake_run() -> int:
    # «awake off» шлёт SIGTERM: без обработчика Python не выполнит finally,
    # и caffeinate останется работать сиротой.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    write_private(STATE_DIR / AWAKE_PID, str(os.getpid()))
    guard: Optional[subprocess.Popen] = None  # type: ignore[type-arg]
    try:
        if sys.platform == "darwin":
            guard = subprocess.Popen(["caffeinate", "-i"])
        elif sys.platform.startswith("win"):
            import ctypes  # noqa: PLC0415

            keep = 0x80000001  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
            ctypes.windll.kernel32.SetThreadExecutionState(keep)  # type: ignore[attr-defined]
        elif shutil.which("systemd-inhibit"):
            guard = subprocess.Popen(
                [
                    "systemd-inhibit",
                    "--what=idle:sleep",
                    "--why=BitbotBY работает",
                    "sleep",
                    "infinity",
                ]
            )
        while bot_running():
            time.sleep(60)
    finally:
        if guard is not None:
            guard.terminate()
        (STATE_DIR / AWAKE_PID).unlink(missing_ok=True)
    return 0


# ── Помощник: переключатели компьютера из приложения ──────────────────────────
#
# Приложение в Docker компьютером не управляет. Оно кладёт запрос в общую папку
# data (launcher_request.json), помощник на компьютере его забирает и выполняет
# только действия из ALLOWED — никаких произвольных команд. Новых портов нет:
# запрос проходит через вход приложения, как остальные кнопки управления.
# Включить доступ с телефона из приложения нельзя: это открывает приложение
# в интернет, поэтому только командой на самом компьютере.

HELPER_PID = "helper.pid"
REQUEST_FILE = "launcher_request.json"
RESULT_FILE = "launcher_result.json"
REQUEST_MAX_AGE = 120
ALLOWED = {
    ("autostart", True),
    ("autostart", False),
    ("keep_awake", True),
    ("keep_awake", False),
    ("phone", False),
}


def helper_running(state_dir: Path = STATE_DIR) -> bool:
    try:
        os.kill(int((state_dir / HELPER_PID).read_text().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def spawn_helper(port: int) -> None:
    if helper_running():
        return
    kw: Dict[str, object] = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
        "env": {**os.environ, "BITBOT_PORT": str(port)},
    }
    if sys.platform.startswith("win"):
        kw["creationflags"] = 0x00000008  # DETACHED_PROCESS
    else:
        kw["start_new_session"] = True
    me = str(Path(__file__).resolve())
    subprocess.Popen([sys.executable, me, "helper"], **kw)  # type: ignore[call-overload]


def handle_request(
    req: Dict[str, object], port: int, now: Optional[float] = None
) -> Tuple[bool, str]:
    """Один запрос приложения → (получилось, что сказать человеку)."""
    setting, on = req.get("setting"), req.get("on")
    if not isinstance(on, bool) or (setting, on) not in ALLOWED:
        return False, "Это можно сделать только командой на компьютере."
    try:
        at = float(str(req.get("at")))
    except ValueError:
        return False, "Непонятный запрос."
    if (time.time() if now is None else now) - at > REQUEST_MAX_AGE:
        return False, "Запрос устарел — нажмите ещё раз."
    action = "on" if on else "off"
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        if setting == "autostart":
            code = cmd_autostart(action)
        elif setting == "keep_awake":
            code = cmd_awake(action)
        else:
            code = cmd_phone("off", port, {})
    return code == 0, out.getvalue().strip()


def cmd_helper(port: int) -> int:
    """Работает, пока работает бот: забирает запросы приложения из папки data."""
    write_private(STATE_DIR / HELPER_PID, str(os.getpid()))
    data = status_path().parent
    request, result = data / REQUEST_FILE, data / RESULT_FILE
    checked, misses = 0.0, 0
    try:
        # Бот пропал из docker ps два раза подряд — его остановили.
        while misses < 2:
            if time.time() - checked >= 30:
                checked = time.time()
                misses = 0 if bot_running() else misses + 1
                publish_status()
            if request.exists():
                try:
                    req = json.loads(request.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    req = {}
                request.unlink(missing_ok=True)
                req = req if isinstance(req, dict) else {}
                ok, message = handle_request(req, port)
                result.write_text(
                    json.dumps(
                        {
                            "id": str(req.get("id", ""))[:32],
                            "ok": ok,
                            "message": message,
                            "at": int(time.time()),
                        }
                    ),
                    encoding="utf-8",
                )
                publish_status()
            time.sleep(2)
    finally:
        (STATE_DIR / HELPER_PID).unlink(missing_ok=True)
    return 0


def cmd_awake(action: str) -> int:
    state = load_state()
    if action == "status":
        say(
            "Не давать компьютеру уснуть: "
            + ("включено" if state.get("keep_awake") == "1" else "выключено")
            + (", работает сейчас" if awake_running() else "")
        )
        return 0
    if action == "on":
        state["keep_awake"] = "1"
        save_state(state)
        if bot_running():
            spawn_awake()
        say(
            "Включено: пока работает бот, компьютер не уснёт сам. "
            "Закрытая крышка ноутбука всё равно усыпляет его."
        )
        return 0
    if action == "off":
        state["keep_awake"] = "0"
        save_state(state)
        try:
            os.kill(int((STATE_DIR / AWAKE_PID).read_text().strip()), 15)
        except (OSError, ValueError):
            pass
        say("Выключено.")
        return 0
    say("Используйте: awake on | off | status")
    return 2


def status_path(state_dir: Path = STATE_DIR) -> Path:
    """Папка data, которую видит панель в Docker (как в compose.env)."""
    env = state_dir / "compose.env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("BITBOT_DATA="):
                return Path(line.split("=", 1)[1]) / "launcher.json"
    return ROOT / "data" / "launcher.json"


def publish_status(state_dir: Path = STATE_DIR) -> None:
    """Флаги для приложения: из Docker их не видно. Без ключей и адресов."""
    state = load_state(state_dir)
    status = {
        "autostart": autostart_file().exists(),
        "keep_awake": state.get("keep_awake") == "1",
        "phone": state.get("phone") == "1",
        "helper": helper_running(state_dir),
        "at": int(time.time()),
    }
    path = status_path(state_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(status), encoding="utf-8")
    except OSError:
        pass  # приложение просто не покажет флаги


def main(argv: List[str]) -> int:
    try:
        return _run(argv)
    finally:
        args = [a for a in argv if not a.startswith("--")]
        if (args[0] if args else "start") in ("start", "autostart", "awake", "phone"):
            publish_status()


def _run(argv: List[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    flags = {
        a.split("=", 1)[0]: a.split("=", 1)[1]
        for a in argv
        if a.startswith("--") and "=" in a
    }
    action = args[0] if args else "start"
    if action == "server":
        return cmd_server(flags)
    if action == "server-logs":
        return run(server_compose(["logs", "--tail", "80"])).returncode
    if action == "server-stop":
        return run(server_compose(["down"])).returncode
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
    if action == "import-env":
        env = Path(flags["--env"]).expanduser() if "--env" in flags else ROOT / ".env"
        return cmd_import_env(env, "--overwrite" in argv)
    if action == "autostart":
        return cmd_autostart(args[1] if len(args) > 1 else "status")
    if action == "awake":
        return cmd_awake(args[1] if len(args) > 1 else "status")
    if action == "awake-run":
        return cmd_awake_run()
    if action == "helper":
        return cmd_helper(port)
    if action == "phone":
        return cmd_phone(args[1] if len(args) > 1 else "status", port, flags)
    if action == "stop":
        stop_tunnel()
        return run(compose_cmd(["down"])).returncode
    if action == "status":
        return run(compose_cmd(["ps"])).returncode
    if action == "logs":
        return run(compose_cmd(["logs", "--tail", "80"])).returncode
    say(__doc__ or "")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
