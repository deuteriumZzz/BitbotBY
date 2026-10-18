"""Разовый перенос ключей из .env в зашифрованное хранилище.

Запускается лаунчером внутри контейнера (`launch.py import-env`): строки `KEY=VALUE`
приходят через stdin, значения нигде не печатаются — только названия полей. Уже
сохранённые в хранилище значения не перезаписываются (если не просили иначе).
"""

import sys
from typing import Dict, List, Tuple

from src import secrets_store


def parse_env(text: str) -> Dict[str, str]:
    """Простой разбор .env: комментарии, кавычки и `# …` после значения."""
    out: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if value[:1] in ('"', "'") and value.count(value[0]) >= 2:
            value = value[1 : value.index(value[0], 1)]
        else:
            value = value.split(" #", 1)[0].strip()
        if key and value:
            out[key] = value
    return out


def import_values(
    values: Dict[str, str], overwrite: bool = False
) -> Tuple[Dict[str, List[str]], List[str]]:
    """Сохраняет ключи по сервисам. Возвращает ({сервис: поля}, пропущенные поля)."""
    stored = secrets_store.load_values()
    done: Dict[str, List[str]] = {}
    skipped: List[str] = []
    for service in secrets_store.SERVICES:
        fresh = {
            f.env: values[f.env]
            for f in service.fields
            if values.get(f.env) and (overwrite or f.env not in stored)
        }
        if not fresh:
            continue
        errors = secrets_store.check_input(service, fresh)
        skipped += [env for env in errors]
        ok = {k: v for k, v in fresh.items() if k not in errors}
        if ok:
            secrets_store.save(
                service,
                ok,
                None,
                "warn",
                "Перенесено из .env, ещё не проверено.",
                "env",
            )
            done[service.id] = sorted(ok)
    return done, skipped


def main() -> int:
    if not secrets_store.enabled():
        print("BITBOT_SECRETS_KEY не задан: хранилище ключей выключено.")
        return 3
    done, skipped = import_values(
        parse_env(sys.stdin.read()), "--overwrite" in sys.argv
    )
    for service, fields in done.items():
        print(f"{service}: {', '.join(fields)}")
    if skipped:
        print("Пропущено (не подошёл формат): " + ", ".join(sorted(skipped)))
    print(f"Перенесено сервисов: {len(done)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
