"""Команды разработчика `./bitbot`: запуск без Docker, фоновые задачи, тесты, линтеры, проверка.

Заменяют Makefile. Python проекта — .venv (3.11), иначе python3.11 из PATH.
Задачи запускаются по реестру точек входа backend/bitbot/config/entrypoints.py:
имя модуля берётся оттуда, а не пишется здесь второй раз.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
ENTRYPOINTS = BACKEND / "bitbot" / "config" / "entrypoints.py"

REPORTS = {"alpha": "ALPHA_REPORT", "divergence": "LIVE_DIVERGENCE", "ai": "AI_ISOLATION"}

# Дообучение walk-forward (окна 4 месяца, шаг 1 месяц) — как было в `make retrain`.
_RETRAIN = """
import asyncio, os
from bitbot.market.data_loader import DataLoader
from bitbot.ml.train_sac import train_walk_forward
async def run():
    loader = DataLoader()
    await loader.initialize(os.getenv('BYBIT_API_KEY', ''), os.getenv('BYBIT_API_SECRET', ''))
    df = await loader.load_ohlcv(os.getenv('TRADING_SYMBOL', 'BTC/USDT'), '15m', limit=17280)
    train_walk_forward(df)
asyncio.run(run())
"""


def python() -> str:
    venv = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(venv) if venv.exists() else (shutil.which("python3.11") or sys.executable)


def entrypoint(name: str) -> str:
    """Модуль из реестра точек входа (читается как текст: зависимости не нужны)."""
    for node in ast.parse(ENTRYPOINTS.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == name:
            return str(ast.literal_eval(node.value))
    raise SystemExit(f"нет точки входа {name} в {ENTRYPOINTS}")


def _flags(argv: List[str]) -> Dict[str, str]:
    return {
        a[2:].split("=", 1)[0]: (a.split("=", 1)[1] if "=" in a else "1")
        for a in argv
        if a.startswith("--")
    }


def _run(cmd: List[str], env: Optional[Dict[str, str]] = None, cwd: Path = ROOT) -> int:
    # Мастер-ключ хранилища — тот же, что у Docker-запуска (.launcher/state.json): ключи сервисов
    # читаются из зашифрованного хранилища, настройки — из приложения; .env не нужен.
    from launcher import load_state  # noqa: PLC0415 — cli/ уже в sys.path у ./bitbot

    key = {"BITBOT_SECRETS_KEY": load_state()["secrets_key"]}
    # Как в Docker (docker-compose.*.yml): база — paper; реальный счёт включает только
    # переключатель в приложении (кодовое слово), а не отсутствие переменной.
    full = {"PAPER_TRADING": "true", **os.environ, "PYTHONPATH": str(BACKEND), **key, **(env or {})}
    return subprocess.call(cmd, cwd=cwd, env=full)


def _module(name: str, env: Optional[Dict[str, str]] = None, *args: str) -> int:
    return _run([python(), "-m", entrypoint(name), *args], env)


# ── приложение ────────────────────────────────────────────────────────────────


def build_frontend(force: bool = False) -> int:
    """Собирает приложение, если сборки нет или исходники новее (npm сам не нужен руками)."""
    dist = FRONTEND / "dist" / "index.html"
    newest = max(p.stat().st_mtime for p in (FRONTEND / "src").rglob("*") if p.is_file())
    if not force and dist.exists() and dist.stat().st_mtime >= newest:
        return 0
    npm = shutil.which("npm")
    if not npm:
        print("Нужен Node.js 20+ (npm), чтобы собрать приложение: https://nodejs.org")
        return 1
    if not (FRONTEND / "node_modules").exists() and _run([npm, "ci"], cwd=FRONTEND):
        return 1
    print("Собираю приложение…")
    return _run([npm, "run", "build"], cwd=FRONTEND)


def start_native(argv: List[str]) -> int:
    """Бот и приложение на этом компьютере без Docker (нужны Redis и .venv)."""
    f = _flags(argv)
    if build_frontend():
        return 1
    env = {}
    if "paper" in f:
        env["PAPER_TRADING"] = "true"
    if "mode" in f:
        env["MODE"] = f["mode"]
    print("Бот и приложение: http://localhost:8080/app/ (остановить — Ctrl+C)")
    return _module("SUPERVISOR", env, "--with-dashboard")


# ── задачи, тесты, линтеры, проверка ──────────────────────────────────────────


def job(argv: List[str]) -> int:
    what = argv[1] if len(argv) > 1 else ""
    f = _flags(argv)
    env: Dict[str, str] = {}
    if "profile" in f:
        env["SAC_PROFILE"] = f["profile"]
    if what == "train" and "docker" in f:
        import launcher  # noqa: PLC0415 — соседний модуль cli/

        # сервис trainer в compose (профиль train) — тот же модуль из реестра точек входа
        return subprocess.call(launcher.compose_cmd(["--profile", "train", "run", "--rm", "trainer"]))
    if what == "train":
        if "long" in f:
            env["TOTAL_TIMESTEPS"] = "1000000"
        return _module("TRAIN_SAC_V2" if "v2" in f else "TRAIN_SAC", env)
    if what == "tune":
        return _module("TUNE_SAC", env)
    if what == "retrain":
        return _run([python(), "-c", _RETRAIN], env)
    if what == "backtest":
        for k, var in (("symbol", "BT_SYMBOL"), ("months", "BT_MONTHS"), ("timeframe", "BT_TIMEFRAME")):
            if k in f:
                env[var] = f[k]
        return _module("BACKTEST", env)
    if what == "report" and len(argv) > 2 and argv[2] in REPORTS:
        return _module(REPORTS[argv[2]], env)
    print(
        "./bitbot job train [--long] [--v2] [--docker] [--profile=altcoin] | tune [--profile=…] | retrain\n"
        "             backtest [--symbol=ETH/USDT] [--months=6] [--timeframe=15m]\n"
        "             report alpha|divergence|ai"
    )
    return 2


def _run_clean(cmd: List[str], cwd: Path = ROOT) -> int:
    """Тесты и проверки — без рабочего окружения бота: тесты ждут live по умолчанию и не должны
    видеть настоящее хранилище ключей (с ним дашборд считает вход настроенным и отвечает 401)."""
    env = {k: v for k, v in os.environ.items() if k not in ("PAPER_TRADING", "BITBOT_SECRETS_KEY")}
    return subprocess.call(cmd, cwd=cwd, env={**env, "PYTHONPATH": str(BACKEND)})


def test(argv: List[str]) -> int:
    part = argv[1] if len(argv) > 1 else "all"
    code = 0
    if part in ("all", "backend"):
        code |= _run_clean([python(), "-m", "pytest", "backend/tests", "--ignore=backend/tests/integration", "-q"])
    if part in ("all", "frontend"):
        code |= _run_clean(["npx", "vitest", "run"], cwd=FRONTEND)
    return code


def lint(argv: List[str]) -> int:
    py = python()
    steps = [
        ([py, "-m", "ruff", "check", "backend"], ROOT),
        ([py, "-m", "black", "--check", "-q", "backend"], ROOT),
        ([str(Path(py).with_name("lint-imports")) if Path(py).with_name("lint-imports").exists() else "lint-imports"], BACKEND),
        ([py, "-m", "mypy", "bitbot"], BACKEND),
        (["npx", "tsc", "-b"], FRONTEND),
    ]
    return max(_run_clean(cmd, cwd=cwd) for cmd, cwd in steps)


def fmt(argv: List[str]) -> int:
    py = python()
    return _run([py, "-m", "black", "-q", "backend"]) | _run(
        [py, "-m", "ruff", "check", "--fix", "--select", "I", "-q", "backend"]
    )


def check(argv: List[str]) -> int:
    """В Docker — внутри контейнера бота; без Docker — здесь, с мастер-ключом лаунчера."""
    import launcher  # noqa: PLC0415 — соседний модуль cli/

    if (launcher.STATE_DIR / "compose.env").exists():
        ps = subprocess.run(launcher.compose_cmd(["ps", "-q", "bot"]), capture_output=True, text=True)
        if ps.returncode == 0 and ps.stdout.strip():
            return subprocess.call(launcher.compose_cmd(["exec", "-T", "bot", "python", "-m", entrypoint("CHECK")]))
    env = {}
    state = launcher.STATE_DIR / "state.json"
    if not os.getenv("BITBOT_SECRETS_KEY") and state.exists():
        key = json.loads(state.read_text(encoding="utf-8")).get("secrets_key", "")
        if key:
            env["BITBOT_SECRETS_KEY"] = key
    return _module("CHECK", env)


COMMANDS = {"job": job, "test": test, "lint": lint, "fmt": fmt, "check": check}
