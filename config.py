import os
from dataclasses import dataclass
from typing import List


@dataclass
class Config:
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
        if self.SYMBOLS is None:
            self.SYMBOLS = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
        if self.TRADING_SYMBOLS is None:
            self.TRADING_SYMBOLS = [s.replace("/", "") for s in self.SYMBOLS]
        if self.ENABLED_STRATEGIES is None:
            self.ENABLED_STRATEGIES = ["ema_crossover", "rsi_momentum"]

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables"""
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
