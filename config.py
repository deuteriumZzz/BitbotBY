import os
from dataclasses import dataclass


@dataclass
class Config:
    # Redis configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")

    # Trading configuration
    INITIAL_BALANCE: float = 10000.0
    RISK_PER_TRADE: float = 0.02
    COMMISSION_RATE: float = 0.001

    # Data configuration
    DATA_DIR: str = "data"
    SYMBOLS: list = None

    def __post_init__(self):
        if self.SYMBOLS is None:
            self.SYMBOLS = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables"""
        return cls(
            REDIS_HOST=os.getenv("REDIS_HOST", "localhost"),
            REDIS_PORT=int(os.getenv("REDIS_PORT", 6379)),
            REDIS_PASSWORD=os.getenv("REDIS_PASSWORD", ""),
            INITIAL_BALANCE=float(os.getenv("INITIAL_BALANCE", 10000.0)),
            RISK_PER_TRADE=float(os.getenv("RISK_PER_TRADE", 0.02)),
            COMMISSION_RATE=float(os.getenv("COMMISSION_RATE", 0.001)),
        )


# Global config instance
config = Config.from_env()
