"""Реестр настроек бота — единый источник для UI «Все параметры» и /api/config.

Каждая настройка описана один раз: подпись по-русски, тип, границы и способ
применения. «Сразу» (hot) — через RuntimeConfig или ключ Redis, который бот
читает каждый цикл. Остальные хранятся в Redis-хэше `bot:config_overrides` и
применяются при старте бота (`apply_overrides`), до `Config.validate()`.

Приоритет значения: изменение из UI > переменная окружения > значение по умолчанию.
Ключи API здесь не хранятся — для них отдельное зашифрованное хранилище.

DQN_WEIGHT, AI_WEIGHT, DQN_SOLO_CONFIDENCE, AI_STRATEGY_SELECTION
намеренно не включены: код их не читает (веса Hybrid зашиты по режиму рынка
в signal_combiner.py), и ручка в UI ничего бы не меняла.

Пакет: model.py — описание настройки и помощники; trading/risk/models/feeds/ops —
таблица по темам, table.py — она целиком в порядке приложения; values.py — значения
(приведение типа, переопределения из приложения, правка, история).
"""

from bitbot.config.registry.model import (  # noqa: F401
    BASIC,
    HOT_KEYS,
    Setting,
)
from bitbot.config.registry.table import BY_KEY, SETTINGS  # noqa: F401
from bitbot.config.registry.values import (  # noqa: F401 — публичный API пакета
    HISTORY_MAX,
    apply_overrides,
    apply_patch,
    coerce,
    current_values,
    effective_config,
    load_history,
    load_overrides,
    record_history,
    validate_patch,
)
