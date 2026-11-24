"""
Настройка структурированного логирования для торгового бота.

JSON-формат в production, человекочитаемый формат в development.
Поддерживает фильтрацию API-ключей через _SecretFilter. Каждая строка лога
торгового цикла несёт его trace_id — тот же, что в журнале решений.
"""

from __future__ import annotations

import contextvars
import json
import logging
import logging.handlers
import os
import re
import sys
from datetime import datetime

# Секреты по форме — ловят то, что не зарегистрировано: токен бота в URL Telegram
# (его пишут ошибки httpx), Bearer-токены, пары «ключ=значение» с именами секретов.
_SECRET_PATTERNS = (
    (re.compile(r"\bbot\d{5,}:[A-Za-z0-9_-]{30,}"), "bot***"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", re.I), "Bearer ***"),
    (
        re.compile(
            r"(?i)(\b(?:api[_-]?key|apikey|api[_-]?secret|secret|passphrase|password|"
            r"passwd|token|access[_-]?token|sign|signature|x-bapi-api-key|x-bapi-sign|"
            r"x-api-key|authorization)\b[\"']?\s*[:=]\s*[\"']?)([^\s\"'&,;}]{6,})"
        ),
        r"\1***",
    ),
)


# trace_id торгового цикла. Ставит DecisionLedger.begin_cycle; его получают все
# строки лога цикла и задачи, запущенные из цикла. Фоновые задачи, созданные до
# цикла (монитор позиций), живут в своём контексте и метки не получают.
TRACE_ID: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")


class _TraceFilter(logging.Filter):
    """Добавляет к записи trace_id цикла: поле для JSON, метку для текста."""

    def filter(self, record: logging.LogRecord) -> bool:
        tid = TRACE_ID.get()
        if tid:
            record.trace_id = tid
        record.trace_tag = f" [{tid}]" if tid else ""
        return True


def mask_secrets(text: str) -> str:
    """Текст без зарегистрированных секретов и без значений, похожих на секреты."""
    if not text:
        return text
    for secret in _SecretFilter._secrets:
        if secret in text:
            text = text.replace(secret, "***")
    for rx, repl in _SECRET_PATTERNS:
        text = rx.sub(repl, text)
    return text


class _SecretFilter(logging.Filter):
    """Маскирует значения API-ключей в лог-записях перед отправкой."""

    _secrets: list = []

    @classmethod
    def register(cls, *values: str) -> None:
        """
        Добавляет секретные значения для маскировки в логах (прежние остаются).

        :param values: Строки длиннее 8 символов, которые нужно скрыть.
        """
        for v in values:
            if v and len(v) > 8 and v not in cls._secrets:
                cls._secrets.append(v)

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Заменяет секреты на *** в сообщении лога.

        Трейсбеки и extra-поля маскирует _MaskingFormatter уже в готовой строке.

        :param record: Запись лога.
        :return: Всегда True (запись разрешена, но может быть изменена).
        """
        msg = record.getMessage()
        masked = mask_secrets(msg)
        if masked != msg:
            record.msg = masked
            record.args = ()
        return True


class _MaskingFormatter(logging.Formatter):
    """Обёртка над форматтером: маскирует всю готовую строку, включая трейсбек."""

    def __init__(self, inner: logging.Formatter) -> None:
        super().__init__()
        self._inner = inner

    def format(self, record: logging.LogRecord) -> str:
        return mask_secrets(self._inner.format(record))


def install_masking(handler: logging.Handler) -> None:
    """Включает маскировку секретов на обработчике (идемпотентно)."""
    fmt = handler.formatter or logging.Formatter()
    if not isinstance(fmt, _MaskingFormatter):
        handler.setFormatter(_MaskingFormatter(fmt))
    if not any(isinstance(f, _SecretFilter) for f in handler.filters):
        handler.addFilter(_SecretFilter())


class JSONFormatter(logging.Formatter):
    """Форматирует лог-записи в JSON для структурированного логирования."""

    # Стандартные поля LogRecord — не дублируем их в extra
    _SKIP = frozenset(
        {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "taskName",
            "trace_tag",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        """
        Преобразует запись лога в JSON-строку.

        :param record: Запись лога.
        :return: JSON-строка с полями time, level, logger, message
            и дополнительными полями из extra={}.
        """
        data: dict = {
            "time": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        for key, val in record.__dict__.items():
            if key not in self._SKIP:
                data[key] = val
        return json.dumps(data, ensure_ascii=False, default=str)


def setup_logging(
    level: str = "INFO",
    json_logs: bool = False,
) -> None:
    """
    Настраивает корневой логгер для всего приложения.

    Добавляет StreamHandler (stdout) и FileHandler (logs/trading.log).
    FileHandler всегда использует JSON-формат для машинной обработки.

    :param level: Уровень логирования (DEBUG, INFO, WARNING, ERROR).
    :param json_logs: True — JSON-формат в stdout (для prod),
        False — текстовый формат (для dev).
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s%(trace_tag)s — %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    install_masking(handler)
    handler.addFilter(_TraceFilter())
    root.handlers.clear()
    root.addHandler(handler)

    os.makedirs("logs", exist_ok=True)
    # Rotate at 10 MB, keep 5 backups → max ~50 MB on disk.
    fh = logging.handlers.RotatingFileHandler(
        "logs/trading.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(JSONFormatter())
    install_masking(fh)
    fh.addFilter(_TraceFilter())
    root.addHandler(fh)
