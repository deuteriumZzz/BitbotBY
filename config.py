import os
from dataclasses import dataclass, field
from typing import List

# Stablecoins excluded from scanning and trading
STABLECOIN_BASES: frozenset[str] = frozenset(
    {
        "USDC",
        "BUSD",
        "FDUSD",
        "TUSD",
        "DAI",
        "USDE",
        "PYUSD",
        "FRAX",
        "LUSD",
        "GUSD",
        "USDP",
        "SUSD",
        "CUSD",
        "USDJ",
        "HUSD",
        "EURS",
        "EURT",
        "USDD",
        "CRVUSD",
        "MKUSD",
    }
)


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
    INITIAL_BALANCE: float = float(os.getenv("INITIAL_BALANCE", "10000.0"))
    RISK_PER_TRADE: float = float(os.getenv("RISK_PER_TRADE", "0.02"))
    COMMISSION_RATE: float = float(os.getenv("COMMISSION_RATE", "0.001"))
    # Интервал между итерациями цикла в секундах (30 = real-time)
    TRADING_INTERVAL: int = int(os.getenv("TRADING_INTERVAL", "30"))
    MAX_POSITION_SIZE: float = float(os.getenv("MAX_POSITION_SIZE", "0.1"))
    STOP_LOSS_PERCENT: float = float(os.getenv("STOP_LOSS_PERCENT", "0.05"))

    # ── Strategy ───────────────────────────────────────────────────────────
    ENABLED_STRATEGIES: List[str] = field(
        default_factory=lambda: [
            "ema_crossover",
            "rsi_momentum",
            "macd_crossover",
            "bollinger_bands",
            "scalping",
            "swing_trading",
            "breakout",
            "mean_reversion",
            "trend_following",
        ]
    )
    DEFAULT_STRATEGY: str = os.getenv("ACTIVE_STRATEGY", "ema_crossover")
    # True → AI выбирает стратегию автоматически
    AI_STRATEGY_SELECTION: bool = (
        os.getenv("AI_STRATEGY_SELECTION", "false").lower() == "true"
    )
    # Минимальный confidence для исполнения сигнала (0.0–1.0)
    MIN_SIGNAL_CONFIDENCE: float = float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.65"))

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

    # ── AI провайдеры ─────────────────────────────────────────────────────
    # auto      → Claude → DeepSeek → OpenAI → local (первый найденный ключ)
    # anthropic → только Claude
    # deepseek  → только DeepSeek
    # openai    → только ChatGPT
    AI_PROVIDER: str = os.getenv("AI_PROVIDER", "auto")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    AI_MODEL: str = os.getenv("AI_MODEL", "claude-sonnet-4-6")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # ── Mode ───────────────────────────────────────────────────────────────
    # local  → только правила (9 стратегий, без внешних API)
    # dqn    → SAC-модель (train_sac.py)
    # ai     → AI-провайдер (Claude / DeepSeek / OpenAI)
    # hybrid → SAC + AI должны согласиться; расхождение → hold
    MODE: str = os.getenv("MODE", "ai")
    # Путь к SAC-модели (создаётся train_sac.py)
    SAC_MODEL_PATH: str = os.getenv("SAC_MODEL_PATH", "models/sac_model.zip")

    # ── Market Scanner ─────────────────────────────────────────────────────
    # Сколько монет сканировать (топ по объёму)
    SCAN_TOP_N: int = int(os.getenv("SCAN_TOP_N", "20"))
    # True → автоматически исполнять топ-1 рекомендацию
    AUTO_EXECUTE: bool = os.getenv("AUTO_EXECUTE", "false").lower() == "true"

    # ── News ───────────────────────────────────────────────────────────────
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    # Интервал обновления новостей в секундах (900 = 15 мин)
    NEWS_UPDATE_INTERVAL: int = int(os.getenv("NEWS_UPDATE_INTERVAL", "900"))

    # ── Telegram ───────────────────────────────────────────────────────────
    # Токен BotFather: /newbot → скопировать токен
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    # Chat ID (личный или группа): @userinfobot → узнать id
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    # Секунд ждать подтверждения (потом → авто-исполнение)
    TELEGRAM_CONFIRM_TIMEOUT: int = int(os.getenv("TELEGRAM_CONFIRM_TIMEOUT", "60"))

    # ── Risk Management ────────────────────────────────────────────────────
    # Максимальный дневной убыток: доля баланса (0.05 = 5%)
    DAILY_LOSS_LIMIT: float = float(os.getenv("DAILY_LOSS_LIMIT", "0.05"))
    # True → использовать Bybit testnet
    TESTNET: bool = os.getenv("TESTNET", "false").lower() == "true"

    # ── Signal Combiner weights ────────────────────────────────────────────
    # Вес SAC сигнала в hybrid режиме (DQN_WEIGHT + AI_WEIGHT = 1.0)
    DQN_WEIGHT: float = float(os.getenv("DQN_WEIGHT", "0.4"))
    AI_WEIGHT: float = float(os.getenv("AI_WEIGHT", "0.6"))
    # Минимальный confidence SAC для соло-исполнения (без AI)
    DQN_SOLO_CONFIDENCE: float = float(os.getenv("DQN_SOLO_CONFIDENCE", "0.80"))

    # ── Position Management ────────────────────────────────────────────────
    # Максимум одновременных открытых позиций
    MAX_POSITIONS: int = int(os.getenv("MAX_POSITIONS", "3"))
    # True → симулировать сделки без реальных ордеров
    PAPER_TRADING: bool = os.getenv("PAPER_TRADING", "false").lower() == "true"
    # Trailing SL: двигать SL вслед за ценой (кратно ATR)
    TRAILING_STOP_ATR_MULT: float = float(os.getenv("TRAILING_STOP_ATR_MULT", "1.0"))
    # Circuit breaker: остановить бота после N подряд убыточных сделок (0 = выключен)
    CIRCUIT_BREAKER_LOSSES: int = int(os.getenv("CIRCUIT_BREAKER_LOSSES", "3"))

    @classmethod
    def from_env(cls) -> "Config":
        """Создаёт экземпляр Config из переменных окружения."""
        return cls(
            REDIS_URL=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            REDIS_HOST=os.getenv("REDIS_HOST", "localhost"),
            REDIS_PORT=int(os.getenv("REDIS_PORT", 6379)),
            REDIS_DB=int(os.getenv("REDIS_DB", 0)),
            BYBIT_API_KEY=os.getenv("BYBIT_API_KEY", ""),
            BYBIT_API_SECRET=os.getenv("BYBIT_API_SECRET", ""),
            INITIAL_BALANCE=float(os.getenv("INITIAL_BALANCE", "10000.0")),
            RISK_PER_TRADE=float(os.getenv("RISK_PER_TRADE", "0.02")),
            COMMISSION_RATE=float(os.getenv("COMMISSION_RATE", "0.001")),
            TRADING_INTERVAL=int(os.getenv("TRADING_INTERVAL", "30")),
            SYMBOL=os.getenv("TRADING_SYMBOL", "BTC/USDT"),
            TIMEFRAME=os.getenv("TIMEFRAME", "15m"),
            NEWS_API_KEY=os.getenv("NEWS_API_KEY", ""),
            DEFAULT_STRATEGY=os.getenv("ACTIVE_STRATEGY", "ema_crossover"),
            AI_STRATEGY_SELECTION=(
                os.getenv("AI_STRATEGY_SELECTION", "false").lower() == "true"
            ),
            MIN_SIGNAL_CONFIDENCE=float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.65")),
            AI_PROVIDER=os.getenv("AI_PROVIDER", "auto"),
            ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY", ""),
            AI_MODEL=os.getenv("AI_MODEL", "claude-sonnet-4-6"),
            DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY", ""),
            DEEPSEEK_MODEL=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", ""),
            OPENAI_MODEL=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            MODE=os.getenv("MODE", "ai"),
            SAC_MODEL_PATH=os.getenv("SAC_MODEL_PATH", "models/sac_model.zip"),
            SCAN_TOP_N=int(os.getenv("SCAN_TOP_N", "20")),
            AUTO_EXECUTE=(os.getenv("AUTO_EXECUTE", "false").lower() == "true"),
            NEWS_UPDATE_INTERVAL=int(os.getenv("NEWS_UPDATE_INTERVAL", "900")),
            TELEGRAM_BOT_TOKEN=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            TELEGRAM_CHAT_ID=os.getenv("TELEGRAM_CHAT_ID", ""),
            TELEGRAM_CONFIRM_TIMEOUT=int(os.getenv("TELEGRAM_CONFIRM_TIMEOUT", "60")),
            MAX_POSITIONS=int(os.getenv("MAX_POSITIONS", "3")),
            PAPER_TRADING=(os.getenv("PAPER_TRADING", "false").lower() == "true"),
            TRAILING_STOP_ATR_MULT=float(os.getenv("TRAILING_STOP_ATR_MULT", "1.0")),
            CIRCUIT_BREAKER_LOSSES=int(os.getenv("CIRCUIT_BREAKER_LOSSES", "3")),
        )

    def validate(self) -> None:
        """Валидирует конфигурацию и обязательные секреты.

        :raises ValueError: Если параметры за пределами допустимых значений
            или отсутствуют обязательные переменные окружения.
        """
        # Reject stablecoin as main trading symbol
        base = self.SYMBOL.split("/")[0]
        if base in STABLECOIN_BASES:
            raise ValueError(
                f"TRADING_SYMBOL={self.SYMBOL} is a stablecoin — set a real asset."
            )

        if self.INITIAL_BALANCE <= 0:
            raise ValueError("INITIAL_BALANCE must be positive")
        if not 0 < self.RISK_PER_TRADE < 1:
            raise ValueError("RISK_PER_TRADE must be between 0 and 1")
        if self.MAX_POSITION_SIZE <= 0 or self.MAX_POSITION_SIZE > 1:
            raise ValueError("MAX_POSITION_SIZE must be between 0 and 1")
        if self.COMMISSION_RATE < 0:
            raise ValueError("COMMISSION_RATE must be non-negative")
        if self.TRADING_INTERVAL <= 0:
            raise ValueError("TRADING_INTERVAL must be positive")

        # Bybit API keys required for live trading
        if not self.PAPER_TRADING:
            missing = []
            if not self.BYBIT_API_KEY:
                missing.append("BYBIT_API_KEY")
            if not self.BYBIT_API_SECRET:
                missing.append("BYBIT_API_SECRET")
            if missing:
                raise ValueError(
                    f"Missing required env vars for live trading: "
                    f"{', '.join(missing)}. "
                    f"Set them in .env or use PAPER_TRADING=true."
                )

        # AI provider validation
        if self.AI_PROVIDER not in ("auto", "anthropic", "deepseek", "openai"):
            raise ValueError(
                f"AI_PROVIDER={self.AI_PROVIDER!r} is invalid. "
                "Use: auto | anthropic | deepseek | openai"
            )
        if self.MODE in ("ai", "hybrid"):
            has_any_key = bool(
                self.ANTHROPIC_API_KEY or self.DEEPSEEK_API_KEY or self.OPENAI_API_KEY
            )
            provider_key = {
                "anthropic": self.ANTHROPIC_API_KEY,
                "deepseek": self.DEEPSEEK_API_KEY,
                "openai": self.OPENAI_API_KEY,
            }.get(self.AI_PROVIDER, "")
            if self.AI_PROVIDER == "auto" and not has_any_key:
                raise ValueError(
                    "AI_PROVIDER=auto: нужен хотя бы один ключ — "
                    "ANTHROPIC_API_KEY, DEEPSEEK_API_KEY или OPENAI_API_KEY."
                )
            if self.AI_PROVIDER != "auto" and not provider_key:
                raise ValueError(
                    f"AI_PROVIDER={self.AI_PROVIDER}: ключ не задан. "
                    "Получить: console.anthropic.com / "
                    "platform.deepseek.com / platform.openai.com"
                )

        # SAC model file required for dqn/hybrid modes
        if self.MODE in ("dqn", "hybrid"):
            if not os.path.exists(self.SAC_MODEL_PATH):
                raise ValueError(
                    f"SAC model not found: {self.SAC_MODEL_PATH}. "
                    f"Train it first: make train"
                )


# Global config instance
config = Config.from_env()
