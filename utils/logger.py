import json
import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional

from config import Config


class JSONFormatter(logging.Formatter):
    """
    Кастомный форматтер для структурированного логирования в формате JSON.

    Этот класс наследуется от logging.Formatter и переопределяет метод format для создания
    структурированных логов в формате JSON. Лог включает основные поля, такие как timestamp,
    уровень, имя логгера, сообщение, модуль, функция и строка. Если присутствует информация
    об исключении, она добавляется в лог. Также поддерживает дополнительные данные через
    атрибут 'extra' в записи лога.

    Логика: Форматирует запись лога в словарь и сериализует его в JSON-строку. Обработка
    исключений осуществляется через formatException базового класса. Дополнительные данные
    из 'extra' объединяются с основными полями.

    Обработка ошибок: Не поднимает исключений; в случае проблем с сериализацией JSON
    может возникнуть исключение json.JSONDecodeError, но это обрабатывается на уровне
    логгера.

    :param record: Запись лога от logging (logging.LogRecord).
    :return: Строка в формате JSON с данными лога (str).
    """

    def format(self, record):
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra data if present
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data)


def setup_logger(
    name: str = __name__, level: int = logging.INFO, log_file: Optional[str] = None
) -> logging.Logger:
    """
    Настраивает и конфигурирует логгер для приложения.

    Создает логгер с заданным именем и уровнем логирования. Избегает дублирования обработчиков,
    если они уже существуют. Добавляет консольный обработчик с простым форматтером и,
    опционально, файловый обработчик с ротацией и JSON-форматтером. Также добавляет
    пользовательский уровень TRACE для более детальной отладки.

    Логика: Проверяет наличие обработчиков; если их нет, создает и настраивает консольный
    и файловый (если указан) обработчики. Создает директорию для логов, если необходимо.
    Определяет метод trace для логгера.

    Обработка ошибок: Не поднимает исключений; в случае проблем с созданием директории
    или файла может возникнуть OSError, но это обрабатывается на уровне ОС.

    :param name: Имя логгера (str, по умолчанию __name__).
    :param level: Уровень логирования (int, по умолчанию logging.INFO).
    :param log_file: Путь к файлу логов (Optional[str], по умолчанию None).
    :return: Настроенный логгер (logging.Logger).
    """

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create formatters
    json_formatter = JSONFormatter()
    simple_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # File handler if log file specified
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)

    # Add trace level for more detailed debugging
    logging.TRACE = logging.DEBUG - 5
    logging.addLevelName(logging.TRACE, "TRACE")

    def trace(self, message, *args, **kwargs):
        if self.isEnabledFor(logging.TRACE):
            self._log(logging.TRACE, message, args, **kwargs)

    logging.Logger.trace = trace

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Получает настроенный экземпляр логгера.

    Если имя не указано, использует APP_NAME из конфигурации. Если логгер не настроен,
    автоматически настраивает его с использованием setup_logger, указывая файл логов
    на основе конфигурации.

    Логика: Проверяет наличие обработчиков у логгера; если их нет, вызывает setup_logger
    с параметрами из Config.

    Обработка ошибок: Не поднимает исключений; зависит от setup_logger.

    :param name: Имя логгера (str, по умолчанию None).
    :return: Экземпляр логгера (logging.Logger).
    """

    if name is None:
        name = Config.APP_NAME

    logger = logging.getLogger(name)

    # If logger not configured, set it up
    if not logger.handlers:
        log_file = os.path.join(Config.LOGS_DIR, f"{Config.APP_NAME}.log")
        logger = setup_logger(name, Config.LOG_LEVEL, log_file)

    return logger


class LoggerAdapter:
    """
    Адаптер для добавления контекстной информации в логи.

    Этот класс обертывает логгер и позволяет добавлять или удалять контекстную информацию,
    которая будет включена в каждый лог. Поддерживает методы логирования с дополнительным
    контекстом.

    Логика: Хранит словарь extra для контекста. Методы add_context и remove_context
    модифицируют этот словарь. Методы логирования (log, debug и т.д.) передают extra
    в логгер.

    Обработка ошибок: Не поднимает исключений; ошибки логирования обрабатываются
    базовым логгером.

    :param logger: Экземпляр логгера для обертки (logging.Logger).
    :param extra: Начальный словарь контекста (dict, по умолчанию пустой).
    """

    def __init__(self, logger: logging.Logger, extra: dict = None):
        self.logger = logger
        self.extra = extra or {}

    def add_context(self, **kwargs):
        """
        Добавляет контекстную информацию в логи.

        Обновляет словарь extra новыми ключ-значениями.

        :param kwargs: Ключ-значения для добавления в контекст.
        :return: Сам объект адаптера для цепочки вызовов (LoggerAdapter).
        """
        self.extra.update(kwargs)
        return self

    def remove_context(self, *keys):
        """
        Удаляет контекстную информацию из логов.

        Убирает указанные ключи из словаря extra.

        :param keys: Ключи для удаления из контекста.
        :return: Сам объект адаптера для цепочки вызовов (LoggerAdapter).
        """
        for key in keys:
            self.extra.pop(key, None)
        return self

    def log(self, level, msg, *args, **kwargs):
        """
        Логирует сообщение с контекстной информацией.

        Передает уровень, сообщение и контекст в базовый логгер.

        :param level: Уровень логирования (int).
        :param msg: Сообщение для логирования (str).
        :param args: Дополнительные аргументы.
        :param kwargs: Дополнительные ключ-значения.
        """
        if self.extra:
            kwargs["extra"] = {"extra": self.extra}
        self.logger.log(level, msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        """
        Логирует сообщение на уровне DEBUG с контекстом.

        :param msg: Сообщение для логирования (str).
        :param args: Дополнительные аргументы.
        :param kwargs: Дополнительные ключ-значения.
        """
        self.log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """
        Логирует сообщение на уровне INFO с контекстом.

        :param msg: Сообщение для логирования (str).
        :param args: Дополнительные аргументы.
        :param kwargs: Дополнительные ключ-значения.
        """
        self.log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """
        Логирует сообщение на уровне WARNING с контекстом.

        :param msg: Сообщение для логирования (str).
        :param args: Дополнительные аргументы.
        :param kwargs: Дополнительные ключ-значения.
        """
        self.log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """
        Логирует сообщение на уровне ERROR с контекстом.

        :param msg: Сообщение для логирования (str).
        :param args: Дополнительные аргументы.
        :param kwargs: Дополнительные ключ-значения.
        """
        self.log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """
        Логирует сообщение на уровне CRITICAL с контекстом.

        :param msg: Сообщение для логирования (str).
        :param args: Дополнительные аргументы.
        :param kwargs: Дополнительные ключ-значения.
        """
        self.log(logging.CRITICAL, msg, *args, **kwargs)

    def trace(self, msg, *args, **kwargs):
        """
        Логирует сообщение на уровне TRACE с контекстом.

        Проверяет наличие метода trace у логгера перед вызовом.

        :param msg: Сообщение для логирования (str).
        :param args: Дополнительные аргументы.
        :param kwargs: Дополнительные ключ-значения.
        """
        if hasattr(self.logger, "trace"):
            if self.extra:
                kwargs["extra"] = {"extra": self.extra}
            self.logger.trace(msg, *args, **kwargs)


# Global logger instance
logger = get_logger()


def log_performance(metrics: dict, strategy: str = None):
    """
    Логирует метрики производительности.

    Записывает метрики производительности в лог на уровне INFO, включая стратегию,
    если указана.

    Логика: Создает словарь extra с метриками и стратегией, затем логирует сообщение.

    Обработка ошибок: Не поднимает исключений; зависит от логгера.

    :param metrics: Словарь с метриками производительности (dict).
    :param strategy: Название стратегии (str, по умолчанию None).
    """
    extra = {"metrics": metrics}
    if strategy:
        extra["strategy"] = strategy

    logger.info("Performance metrics", extra={"extra": extra})


def log_trade(trade_data: dict):
    """
    Логирует исполнение сделки.

    Записывает данные о сделке в лог на уровне INFO.

    Логика: Передает trade_data как extra в логгер.

    Обработка ошибок: Не поднимает исключений; зависит от логгера.

    :param trade_data: Словарь с данными о сделке (dict).
    """
    logger.info("Trade executed", extra={"extra": trade_data})


def log_market_data(symbol: str, data: dict):
    """
    Логирует обновление рыночных данных.

    Записывает обновление данных для символа на уровне DEBUG.

    Логика: Включает символ и данные в extra.

    Обработка ошибок: Не поднимает исключений; зависит от логгера.

    :param symbol: Символ актива (str).
    :param data: Словарь с рыночными данными (dict).
    """
    logger.debug(
        f"Market data update for {symbol}",
        extra={"extra": {"symbol": symbol, "data": data}},
    )


def log_portfolio_update(portfolio_state: dict):
    """
    Логирует обновление портфеля.

    Записывает состояние портфеля в лог на уровне INFO.

    Логика: Передает portfolio_state как extra.

    Обработка ошибок: Не поднимает исключений; зависит от логгера.

    :param portfolio_state: Словарь с состоянием портфеля (dict).
    """
    logger.info("Portfolio updated", extra={"extra": portfolio_state})


def setup_error_handler():
    """
    Настраивает глобальный обработчик ошибок.

    Заменяет sys.excepthook на функцию, которая логирует необработанные исключения
    на уровне CRITICAL, игнорируя KeyboardInterrupt.

    Логика: Определяет внутреннюю функцию handle_exception, которая проверяет тип
    исключения и логирует его, затем устанавливает ее как excepthook.

    Обработка ошибок: Не поднимает исключений; логирует ошибки через логгер.
    """
    import sys

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.critical(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = handle_exception


# Setup error handler when module is imported
setup_error_handler()
