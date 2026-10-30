"""
Настройка структурированного логирования для торгового бота.

JSON-формат в production, человекочитаемый формат в development.
Поддерживает фильтрацию API-ключей через _SecretFilter.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime


class _SecretFilter(logging.Filter):
    """Маскирует значения API-ключей в лог-записях перед отправкой."""

    _secrets: list = []

    @classmethod
    def register(cls, *values: str) -> None:
        """
        Регистрирует секретные значения для маскировки в логах.

        :param values: Строки длиннее 8 символов, которые нужно скрыть.
        """
        cls._secrets = [v for v in values if v and len(v) > 8]

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Заменяет секретные значения на *** в сообщении лога.

        :param record: Запись лога.
        :return: Всегда True (запись разрешена, но может быть изменена).
        """
        if self._secrets:
            msg = record.getMessage()
            for secret in self._secrets:
                if secret in msg:
                    msg = msg.replace(secret, "***")
                    record.msg = msg
                    record.args = ()
        return True


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
                "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    secret_filter = _SecretFilter()
    handler.addFilter(secret_filter)
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
    fh.addFilter(secret_filter)
    root.addHandler(fh)
