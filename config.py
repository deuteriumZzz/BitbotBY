from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """
    Bot configuration. Reads from environment variables
    and .env file automatically. Override in tests via
    Config(SYMBOL="ETH/USDT") without touching os.environ.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="",
    )

    # ── Redis ──────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # ── Bybit ──────────────────────────────────────────
    BYBIT_API_KEY: str = ""
    BYBIT_API_SECRET: str = ""

    # ── Trading ────────────────────────────────────────
    INITIAL_BALANCE: float = 10000.0
    RISK_PER_TRADE: float = 0.02
    COMMISSION_RATE: float = 0.001
    TRADING_INTERVAL: int = 30
    MAX_POSITION_SIZE: float = 0.1
    STOP_LOSS_PERCENT: float = 0.05
    DAILY_LOSS_LIMIT: float = 0.05
    TESTNET: bool = False

    # ── Strategy ───────────────────────────────────────
    ACTIVE_STRATEGY: str = "ema_crossover"
    # Alias used in some modules
    DEFAULT_STRATEGY: str = "ema_crossover"
    AI_STRATEGY_SELECTION: bool = True
    MIN_SIGNAL_CONFIDENCE: float = 0.65
    ENABLED_STRATEGIES: List[str] = [
        "ema_crossover", "rsi_momentum",
        "macd_crossover", "bollinger_bands",
        "scalping", "swing_trading",
        "breakout", "mean_reversion",
        "trend_following",
    ]

    # ── Symbol / Data ──────────────────────────────────
    TRADING_SYMBOL: str = "BTC/USDT"
    TIMEFRAME: str = "15m"
    DATA_DIR: str = "data"
    SYMBOLS: List[str] = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
    TRADING_SYMBOLS: List[str] = [
        "BTCUSDT", "ETHUSDT", "ADAUSDT"
    ]

    # ── AI ─────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = ""
    AI_MODEL: str = "claude-sonnet-4-6"

    # ── Mode ───────────────────────────────────────────
    MODE: str = "ai"
    DQN_MODEL_PATH: str = "models/dqn_model.pth"

    # ── Signal Combiner weights (calibrated) ───────────
    # DQN weight in hybrid mode (technical signals)
    DQN_WEIGHT: float = 0.4
    # AI weight in hybrid mode (fundamental + news)
    AI_WEIGHT: float = 0.6
    # Min DQN confidence to act alone when AI is silent
    DQN_SOLO_CONFIDENCE: float = 0.80

    # ── Scanner ────────────────────────────────────────
    SCAN_TOP_N: int = 20
    AUTO_EXECUTE: bool = False

    # ── News ───────────────────────────────────────────
    NEWS_API_KEY: str = ""
    NEWS_UPDATE_INTERVAL: int = 900

    # ── Telegram ───────────────────────────────────────
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    TELEGRAM_CONFIRM_TIMEOUT: int = 60

    # ── Positions ──────────────────────────────────────
    MAX_POSITIONS: int = 3
    PAPER_TRADING: bool = True
    TRAILING_STOP_ATR_MULT: float = 1.0

    # ── Convenience properties ─────────────────────────
    @property
    def SYMBOL(self) -> str:
        return self.TRADING_SYMBOL

    @field_validator("RISK_PER_TRADE")
    @classmethod
    def risk_must_be_fraction(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError(
                "RISK_PER_TRADE must be between 0 and 1"
            )
        return v

    @field_validator("MIN_SIGNAL_CONFIDENCE")
    @classmethod
    def confidence_range(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError(
                "MIN_SIGNAL_CONFIDENCE must be 0-1"
            )
        return v

    def validate(self) -> None:
        """Legacy compatibility shim."""
        pass

    @classmethod
    def from_env(cls) -> "Config":
        """Legacy compatibility shim."""
        return cls()


# Singleton — imported everywhere as Config
Config = Config()

# Lowercase alias for modules that do `from config import config`
config = Config
