"""Ключи Redis — только в bitbot/config/keys.py: строка "bot:…" в другом месте кода — ошибка.

API и воркеры — разные процессы; если ключ написан строкой в двух местах, опечатка
в одном из них тихо разрывает связь (приложение читает пустоту). Поэтому ключи
живут в одном файле, а этот тест не даёт строке-ключу появиться где-то ещё.
"""

import ast
from pathlib import Path

import bitbot
from bitbot.config import keys

ROOT = Path(bitbot.__file__).parent
KEYS_FILE = Path(keys.__file__)


def test_no_redis_key_strings_outside_keys_py():
    offenders = []
    for path in ROOT.rglob("*.py"):
        if path == KEYS_FILE:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith("bot:")
            ):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} {node.value}")
    assert not offenders, "ключ Redis вне config/keys.py:\n" + "\n".join(offenders)


def test_key_values_are_unique():
    values = [v for k, v in vars(keys).items() if k.isupper() and isinstance(v, str)]
    assert len(values) == len(set(values))
