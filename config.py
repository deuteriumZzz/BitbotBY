import os
from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    """
    Конфигурационный класс для приложения криптовалютной торговли.

    Этот класс содержит все необходимые параметры для настройки Redis, API Bybit, 
    торговых стратегий, данных и новостей. Он использует переменные окружения для 
    гибкой конфигурации и предоставляет значения по умолчанию. Класс наследуется 
    от dataclass для автоматической генерации методов __init__, __repr__ и т.д.

    Поля:
    - REDIS_URL: URL подключения к Redis (str).
    - REDIS_HOST: Хост Redis (str).
    - REDIS_PORT: Порт Redis (int).
    - REDIS_DB: Номер базы данных Redis (int).
    - BYBIT_API_KEY: API-ключ для Bybit (str).
    - BYBIT_API_SECRET: Секретный ключ для Bybit (str).
    - INITIAL_BALANCE: Начальный баланс портфеля (float).
    - RISK_PER_TRADE: Риск на одну сделку (float, доля от баланса).
    - COMMISSION_RATE: Ставка комиссии (float).
    - TRADING_INTERVAL: Интервал торговли в секундах (int).
    - MAX_POSITION_SIZE: Максимальный размер позиции (float, доля от баланса).
    - STOP_LOSS_PERCENT: Процент стоп-лосса (float).
    - ENABLED_STRATEGIES: Список включенных стратегий (List[str]).
    - DEFAULT_STRATEGY: Стратегия по умолчанию (str).
    - DATA_DIR: Директория для данных (str).
    - SYMBOLS: Список символов в формате "BASE/QUOTE" (List[str]).
    - TRADING_SYMBOLS: Список символов для торговли (List[str]).
    - SYMBOL: Текущий торговый символ (str).
    - TIMEFRAME: Таймфрейм для данных (str).
    - NEWS_API_KEY: Ключ для API новостей (str).
    - NEWS_UPDATE_INTERVAL: Интервал обновления новостей в секундах (int).

    Логика: Поля инициализируются из переменных окружения с fallback на значения 
    по умолчанию. Метод __post_init__ устанавливает значения для списков, если они 
    не заданы. Классовый метод from_env создает экземпляр на основе переменных 
    окружения.

    Обработка ошибок: Не поднимает исключений; преобразование типов (например, int 
    или float) может вызвать ValueError, если переменные окружения некорректны, 
    но это обрабатывается на уровне интерпретатора.
    """

    # Redis configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))

    # Bybit API configuration
    BYBIT_API_KEY: str = os.getenv("BYBIT_API_KEY", "")
    BYBIT_API_SECRET: str = os.getenv("BYBIT_API_SECRET", "")

    # Trading configuration
    INITIAL_BALANCE: float = 10000.0
    RISK_PER_TRADE: float = 0.02
    COMMISSION_RATE: float = 0.001
    TRADING_INTERVAL: int = int(os.getenv("TRADING_INTERVAL", 300))
    MAX_POSITION_SIZE: float = 0.1
    STOP_LOSS_PERCENT: float = 0.02

    # Strategy configuration
    ENABLED_STRATEGIES: List[str] = None
    DEFAULT_STRATEGY: str = "ema_crossover"

    # Data configuration
    DATA_DIR: str = "data"
    SYMBOLS: List[str] = None
    TRADING_SYMBOLS: List[str] = None
    SYMBOL: str = os.getenv("TRADING_SYMBOL", "BTCUSDT")  # Добавлено
    TIMEFRAME: str = os.getenv("TIMEFRAME", "15")  # Добавлено

    # News configuration
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    NEWS_UPDATE_INTERVAL: int = 3600

    def __post_init__(self):
        """
        Пост-инициализация для установки значений по умолчанию для списков.

        Этот метод вызывается автоматически после __init__. Он проверяет, не установлены 
        ли SYMBOLS, TRADING_SYMBOLS и ENABLED_STRATEGIES, и присваивает им значения 
        по умолчанию, если они равны None.

        Логика: Для SYMBOLS устанавливается список базовых символов. Для TRADING_SYMBOLS 
        удаляются слэши из SYMBOLS. Для ENABLED_STRATEGIES устанавливается список 
        включенных стратегий.

        Обработка ошибок: Не поднимает исключений; работает только с присваиваниями.

        :return: None
        """
        if self.SYMBOLS is None:
            self.SYMBOLS = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
        if self.TRADING_SYMBOLS is None:
            self.TRADING_SYMBOLS = [s.replace("/", "") for s in self.SYMBOLS]
        if self.ENABLED_STRATEGIES is None:
            self.ENABLED_STRATEGIES = ["ema_crossover", "rsi_momentum"]

    @classmethod
    def from_env(cls) -> "Config":
        """
        Создает экземпляр Config на основе переменных окружения.

        Этот классовый метод считывает все необходимые переменные окружения и 
        создает объект Config с этими значениями, используя значения по умолчанию 
        для отсутствующих переменных.

        Логика: Вызывает конструктор cls с параметрами, полученными из os.getenv, 
        с преобразованием типов где необходимо.

        Обработка ошибок: Может поднять ValueError при некорректном преобразовании 
        типов (например, int(os.getenv(...))), но это обрабатывается на уровне 
        интерпретатора.

        :return: Новый экземпляр Config (Config).
        """
        return cls(
            REDIS_URL=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            REDIS_HOST=os.getenv("REDIS_HOST", "localhost"),
            REDIS_PORT=int(os.getenv("REDIS_PORT", 6379)),
            REDIS_DB=int(os.getenv("REDIS_DB", 0)),
            BYBIT_API_KEY=os.getenv("BYBIT_API_KEY", ""),
            BYBIT_API_SECRET=os.getenv("BYBIT_API_SECRET", ""),
            INITIAL_BALANCE=float(os.getenv("INITIAL_BALANCE", 10000.0)),
            RISK_PER_TRADE=float(os.getenv("RISK_PER_TRADE", 0.02)),
            COMMISSION_RATE=float(os.getenv("COMMISSION_RATE", 0.001)),
            TRADING_INTERVAL=int(os.getenv("TRADING_INTERVAL", 300)),
            SYMBOL=os.getenv("TRADING_SYMBOL", "BTCUSDT"),
            TIMEFRAME=os.getenv("TIMEFRAME", "15"),
            NEWS_API_KEY=os.getenv("NEWS_API_KEY", ""),
        )


# Global config instance
config = Config.from_env()
