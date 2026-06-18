import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Конфигурация торгового бота. Все параметры читаются из env."""

    # ── Redis ──────────────────────────────────────────────────────────────
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))

    # ── Bybit API ──────────────────────────────────────────────────────────
    BYBIT_API_KEY: str = os.getenv("BYBIT_API_KEY", "")
    BYBIT_API_SECRET: str = os.getenv("BYBIT_API_SECRET", "")

    # ── Trading ────────────────────────────────────────────────────────────
    INITIAL_BALANCE: float = float(
        os.getenv("INITIAL_BALANCE", "10000.0")
    )
    RISK_PER_TRADE: float = float(
        os.getenv("RISK_PER_TRADE", "0.02")
    )
    COMMISSION_RATE: float = float(
        os.getenv("COMMISSION_RATE", "0.001")
    )
    # Интервал между итерациями цикла в секундах (30 = real-time)
    TRADING_INTERVAL: int = int(
        os.getenv("TRADING_INTERVAL", "30")
    )
    MAX_POSITION_SIZE: float = float(
        os.getenv("MAX_POSITION_SIZE", "0.1")
    )
    STOP_LOSS_PERCENT: float = float(
        os.getenv("STOP_LOSS_PERCENT", "0.05")
    )
    # Max daily loss as fraction of initial balance (5% = 0.05)
    DAILY_LOSS_LIMIT: float = float(
        os.getenv("DAILY_LOSS_LIMIT", "0.05")
    )
    # True → use Bybit testnet (safe for testing)
    TESTNET: bool = (
        os.getenv("TESTNET", "false").lower() == "true"
    )

    # ── Strategy ───────────────────────────────────────────────────────────
    ENABLED_STRATEGIES: List[str] = field(default_factory=lambda: [
        "ema_crossover",
        "rsi_momentum",
        "macd_crossover",
        "bollinger_bands",
        "scalping",
        "swing_trading",
        "breakout",
        "mean_reversion",
        "trend_following",
    ])
    DEFAULT_STRATEGY: str = os.getenv(
        "ACTIVE_STRATEGY", "ema_crossover"
    )
    # True → AI выбирает стратегию автоматически
    AI_STRATEGY_SELECTION: bool = (
        os.getenv("AI_STRATEGY_SELECTION", "false").lower() == "true"
    )
    # Минимальный confidence для исполнения сигнала (0.0–1.0)
    MIN_SIGNAL_CONFIDENCE: float = float(
        os.getenv("MIN_SIGNAL_CONFIDENCE", "0.65")
    )

    # ── Data / Symbols ─────────────────────────────────────────────────────
    DATA_DIR: str = "data"
    SYMBOLS: List[str] = field(
        default_factory=lambda: ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
    )
    TRADING_SYMBOLS: List[str] = field(
        default_factory=lambda: ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    )
    # Основной символ в формате ccxt spot: BTC/USDT
    SYMBOL: str = os.getenv("TRADING_SYMBOL", "BTC/USDT")
    # Таймфрейм в формате ccxt: 1m, 5m, 15m, 1h, 4h, 1d
    TIMEFRAME: str = os.getenv("TIMEFRAME", "15m")

    # ── AI (Claude API) ────────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    AI_MODEL: str = os.getenv("AI_MODEL", "claude-sonnet-4-6")

    # ── Mode ───────────────────────────────────────────────────────────────
    # local  → только правила (9 стратегий, без внешних API)
    # dqn    → только обученная DQN-модель
    # ai     → только Claude API
    # hybrid → DQN + Claude должны согласиться
    MODE: str = os.getenv("MODE", "ai")
    # Путь к файлу весов DQN (создаётся train_dqn.py)
    DQN_MODEL_PATH: str = os.getenv(
        "DQN_MODEL_PATH", "models/dqn_model.pth"
    )

    # ── Market Scanner ─────────────────────────────────────────────────────
    # Сколько монет сканировать (топ по объёму)
    SCAN_TOP_N: int = int(os.getenv("SCAN_TOP_N", "20"))
    # True → автоматически исполнять топ-1 рекомендацию
    AUTO_EXECUTE: bool = (
        os.getenv("AUTO_EXECUTE", "false").lower() == "true"
    )

    # ── News ───────────────────────────────────────────────────────────────
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    # Интервал обновления новостей в секундах (900 = 15 мин)
    NEWS_UPDATE_INTERVAL: int = int(
        os.getenv("NEWS_UPDATE_INTERVAL", "900")
    )

    @classmethod
    def from_env(cls) -> "Config":
        """Создаёт экземпляр Config из переменных окружения."""
        return cls(
            REDIS_URL=os.getenv(
                "REDIS_URL", "redis://localhost:6379/0"
            ),
            REDIS_HOST=os.getenv("REDIS_HOST", "localhost"),
            REDIS_PORT=int(os.getenv("REDIS_PORT", 6379)),
            REDIS_DB=int(os.getenv("REDIS_DB", 0)),
            BYBIT_API_KEY=os.getenv("BYBIT_API_KEY", ""),
            BYBIT_API_SECRET=os.getenv("BYBIT_API_SECRET", ""),
            INITIAL_BALANCE=float(
                os.getenv("INITIAL_BALANCE", "10000.0")
            ),
            RISK_PER_TRADE=float(
                os.getenv("RISK_PER_TRADE", "0.02")
            ),
            COMMISSION_RATE=float(
                os.getenv("COMMISSION_RATE", "0.001")
            ),
            TRADING_INTERVAL=int(
                os.getenv("TRADING_INTERVAL", "30")
            ),
            SYMBOL=os.getenv("TRADING_SYMBOL", "BTC/USDT"),
            TIMEFRAME=os.getenv("TIMEFRAME", "15m"),
            NEWS_API_KEY=os.getenv("NEWS_API_KEY", ""),
            DEFAULT_STRATEGY=os.getenv(
                "ACTIVE_STRATEGY", "ema_crossover"
            ),
            AI_STRATEGY_SELECTION=(
                os.getenv(
                    "AI_STRATEGY_SELECTION", "false"
                ).lower() == "true"
            ),
            MIN_SIGNAL_CONFIDENCE=float(
                os.getenv("MIN_SIGNAL_CONFIDENCE", "0.65")
            ),
            ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY", ""),
            AI_MODEL=os.getenv("AI_MODEL", "claude-sonnet-4-6"),
            MODE=os.getenv("MODE", "ai"),
            DQN_MODEL_PATH=os.getenv(
                "DQN_MODEL_PATH", "models/dqn_model.pth"
            ),
            SCAN_TOP_N=int(os.getenv("SCAN_TOP_N", "20")),
            AUTO_EXECUTE=(
                os.getenv("AUTO_EXECUTE", "false").lower() == "true"
            ),
            NEWS_UPDATE_INTERVAL=int(
                os.getenv("NEWS_UPDATE_INTERVAL", "900")
            ),
            DAILY_LOSS_LIMIT=float(
                os.getenv("DAILY_LOSS_LIMIT", "0.05")
            ),
            TESTNET=(
                os.getenv("TESTNET", "false").lower() == "true"
            ),
        )

    def validate(self):
        """Валидирует конфигурацию.

        :raises ValueError: Если параметры за пределами допустимых значений.
        """
        if self.INITIAL_BALANCE <= 0:
            raise ValueError("INITIAL_BALANCE must be positive")
        if not 0 < self.RISK_PER_TRADE < 1:
            raise ValueError(
                "RISK_PER_TRADE must be between 0 and 1"
            )
        if self.MAX_POSITION_SIZE <= 0 or self.MAX_POSITION_SIZE > 1:
            raise ValueError(
                "MAX_POSITION_SIZE must be between 0 and 1"
            )
        if self.COMMISSION_RATE < 0:
            raise ValueError("COMMISSION_RATE must be non-negative")
        if self.TRADING_INTERVAL <= 0:
            raise ValueError("TRADING_INTERVAL must be positive")


# Global config instance
config = Config.from_env()
