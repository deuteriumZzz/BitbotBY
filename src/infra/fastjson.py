"""JSON с необязательным ускорением через orjson (реестр идей №5: «только по профилю»).

Профиль (AUDIT_REPORT §10r): сериализация записи Decision Ledger — десятки микросекунд, цикл
бота — 30 с, поэтому ускорение НЕ нужно для торгового пути. Модуль используется там, где поток
событий может вырасти (шина событий); при отсутствии orjson работает стандартный json с тем же
результатом (поведение идентично: Unicode не экранируется, неизвестные типы → строка).
"""

from __future__ import annotations

import json
from typing import Any

try:  # необязательная зависимость
    import orjson
except ImportError:  # pragma: no cover
    orjson = None  # type: ignore[assignment]


def dumps(obj: Any) -> str:
    if orjson is not None:
        return orjson.dumps(obj, default=str).decode()
    return json.dumps(obj, ensure_ascii=False, default=str, separators=(",", ":"))


def loads(raw: Any) -> Any:
    if orjson is not None:
        return orjson.loads(raw)
    return json.loads(raw)


def backend() -> str:
    return "orjson" if orjson is not None else "json"
