"""`./bitbot check`: настройки, Redis, хранилище ключей и база сделок — до запуска бота.

Только читает. Значения ключей не печатает: сколько расшифровано и какие нет
(неверный мастер-ключ BITBOT_SECRETS_KEY — самая частая причина «ключи пропали»).
Код выхода 1, если есть проблема.
"""

from __future__ import annotations

import sys
from typing import List, Tuple

from bitbot.config import secrets
from bitbot.config.registry import effective_config
from bitbot.config.settings import Config
from bitbot.storage import db_health
from bitbot.storage.redis_client import RedisClient


def run() -> List[Tuple[bool, str]]:
    lines: List[Tuple[bool, str]] = []

    if not secrets.enabled():
        lines.append((False, "хранилище ключей: BITBOT_SECRETS_KEY не задан"))
    else:
        stored = sorted(secrets._load_raw()["items"])  # только имена, не значения
        bad = sorted(set(stored) - set(secrets.load_values()))
        lines.append(
            (
                not bad,
                f"хранилище ключей: расшифровано {len(stored) - len(bad)} из {len(stored)}"
                + (f"; не расшифровываются: {', '.join(bad)}" if bad else ""),
            )
        )
        secrets.apply_secrets(Config)

    try:
        effective_config(Config).validate()
        mode = "paper" if Config.PAPER_TRADING else "LIVE"
        lines.append((True, f"настройки: в порядке, режим {mode}"))
    except ValueError as exc:
        lines.append((False, f"настройки: {exc}"))

    try:
        RedisClient().redis_client.ping()
        lines.append((True, f"Redis: {Config.REDIS_HOST}:{Config.REDIS_PORT} отвечает"))
    except Exception as exc:  # noqa: BLE001 — показать причину, не падать
        lines.append((False, f"Redis: недоступен ({exc})"))

    reports = db_health.check_all(quick=True)
    if not reports:
        lines.append((True, "база сделок: ещё не создана (появится с первой сделкой)"))
    for rep in reports:
        ok = bool(rep.get("ok"))
        detail = (
            "цела" if ok else (rep.get("error") or "; ".join(rep.get("messages", [])))
        )
        lines.append((ok, f"база {rep['path']}: {detail}"))
    return lines


def main() -> int:
    lines = run()
    for ok, text in lines:
        print(("✓ " if ok else "✗ ") + text)
    return 0 if all(ok for ok, _ in lines) else 1


if __name__ == "__main__":
    sys.exit(main())
