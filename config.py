import os
from dataclasses import dataclass, field
from typing import List

# Стейблкоины, исключённые из сканирования и торговли
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
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))

    # ── Bybit API ──────────────────────────────────────────────────────────
    BYBIT_API_KEY: str = os.getenv("BYBIT_API_KEY", "")
    BYBIT_API_SECRET: str = os.getenv("BYBIT_API_SECRET", "")
    # "spot"   — споровой рынок (только лонги)
    # "linear" — бессрочные фьючерсы (нужен reduceOnly для SL/TP)
    MARKET_TYPE: str = os.getenv("MARKET_TYPE", "linear")
    # Плечо для фьючерсной торговли (1 = без плеча, 3 = 3x)
    # Используется как fallback когда ATR недоступен или LEVERAGE_MODE=fixed
    LEVERAGE: int = int(os.getenv("LEVERAGE", "3"))
    # Режим управления плечом: fixed | volatility | full
    LEVERAGE_MODE: str = os.getenv("LEVERAGE_MODE", "volatility")
    # Волатильность-таргетинг: целевой риск портфеля на одно ATR-движение (0.01 = 1%)
    LEVERAGE_TARGET_RISK: float = float(os.getenv("LEVERAGE_TARGET_RISK", "0.01"))
    LEVERAGE_MAX: int = int(os.getenv("LEVERAGE_MAX", "5"))
    LEVERAGE_MIN: int = int(os.getenv("LEVERAGE_MIN", "1"))

    # ── Trading ────────────────────────────────────────────────────────────
    INITIAL_BALANCE: float = float(os.getenv("INITIAL_BALANCE", "10000.0"))
    RISK_PER_TRADE: float = float(os.getenv("RISK_PER_TRADE", "0.02"))
    COMMISSION_RATE: float = float(os.getenv("COMMISSION_RATE", "0.001"))
    # Интервал между итерациями цикла в секундах (30 = real-time)
    TRADING_INTERVAL: int = int(os.getenv("TRADING_INTERVAL", "30"))
    MAX_POSITION_SIZE: float = float(os.getenv("MAX_POSITION_SIZE", "0.1"))
    STOP_LOSS_PERCENT: float = float(os.getenv("STOP_LOSS_PERCENT", "0.05"))

    # ── Strategy ───────────────────────────────────────────────────────────
    DEFAULT_STRATEGY: str = os.getenv("ACTIVE_STRATEGY", "ema_crossover")
    # Минимальный confidence для исполнения сигнала (0.0–1.0)
    MIN_SIGNAL_CONFIDENCE: float = float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.65"))

    # ── Data / Symbols ─────────────────────────────────────────────────────
    DATA_DIR: str = "data"
    SYMBOLS: List[str] = field(
        default_factory=lambda: ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
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
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    MAX_DRAWDOWN_PERCENT: float = float(os.getenv("MAX_DRAWDOWN_PERCENT", "0.2"))

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

    # ── Position Management ────────────────────────────────────────────────
    # Максимум одновременных открытых позиций
    MAX_POSITIONS: int = int(os.getenv("MAX_POSITIONS", "3"))
    # True → симулировать сделки без реальных ордеров
    PAPER_TRADING: bool = os.getenv("PAPER_TRADING", "false").lower() == "true"
    # Trailing SL: двигать SL вслед за ценой (кратно ATR)
    TRAILING_STOP_ATR_MULT: float = float(os.getenv("TRAILING_STOP_ATR_MULT", "1.0"))
    # Circuit breaker: остановить бота после N подряд убыточных сделок (0 = выключен)
    CIRCUIT_BREAKER_LOSSES: int = int(os.getenv("CIRCUIT_BREAKER_LOSSES", "3"))

    # ── Monitoring ────────────────────────────────────────────────────────────
    # Порт HTTP health-сервера (GET /health, GET /metrics)
    # 0 = выключен
    HEALTH_PORT: int = int(os.getenv("HEALTH_PORT", "8081"))

    # ── Correlation filter ─────────────────────────────────────────────────
    # Максимальная |корреляция| log-returns между одновременными позициями.
    # Новый сигнал блокируется если его корреляция с любой открытой позицией
    # превышает порог. 0.0 = фильтр выключен.
    MAX_CORRELATION: float = float(os.getenv("MAX_CORRELATION", "0.7"))
    # Окно расчёта корреляции в свечах
    CORRELATION_WINDOW: int = int(os.getenv("CORRELATION_WINDOW", "50"))

    # ── Partial TP (УЛУЧШЕНИЕ 3) ───────────────────────────────────────────
    # Частичная фиксация прибыли: закрыть PARTIAL_TP_FRACTION позиции при
    # достижении PARTIAL_TP_TRIGGER цели и перенести SL на breakeven.
    PARTIAL_TP_ENABLED: bool = os.getenv("PARTIAL_TP_ENABLED", "true").lower() == "true"
    # Доля пути до TP при которой срабатывает частичная фиксация (0.6 = 60%)
    PARTIAL_TP_TRIGGER: float = float(os.getenv("PARTIAL_TP_TRIGGER", "0.6"))
    # Доля позиции для закрытия при срабатывании (0.5 = 50%)
    PARTIAL_TP_FRACTION: float = float(os.getenv("PARTIAL_TP_FRACTION", "0.5"))

    # ── Trading hours filter (УЛУЧШЕНИЕ 4) ────────────────────────────────
    # Торговые часы UTC. Пустая строка = круглосуточно.
    # Формат "8-22": торговать только с 8:00 до 22:00 UTC.
    TRADING_HOURS: str = os.getenv("TRADING_HOURS", "")

    # ── Liquidity filter (УЛУЧШЕНИЕ 5) ────────────────────────────────────
    # Минимальный 24h объём в USDT для допуска к торговле
    MIN_VOLUME_USDT: float = float(os.getenv("MIN_VOLUME_USDT", "1000000"))
    # Максимальный допустимый спред (ask-bid)/mid в процентах
    MAX_SPREAD_PCT: float = float(os.getenv("MAX_SPREAD_PCT", "0.3"))

    # ── Drawdown scaling (УЛУЧШЕНИЕ 6) ────────────────────────────────────
    # Уменьшать размер позиции при просадке от пика баланса
    DRAWDOWN_SCALE_ENABLED: bool = (
        os.getenv("DRAWDOWN_SCALE_ENABLED", "true").lower() == "true"
    )
    # Порог просадки для активации масштабирования (0.10 = 10%)
    DRAWDOWN_SCALE_THRESHOLD: float = float(
        os.getenv("DRAWDOWN_SCALE_THRESHOLD", "0.10")
    )
    # Множитель размера позиции при просадке (0.5 = 50% от расчётного)
    DRAWDOWN_SCALE_FACTOR: float = float(os.getenv("DRAWDOWN_SCALE_FACTOR", "0.5"))

    # ── Online Learning ────────────────────────────────────────────────────
    # Режим автообучения SAC-модели на реальных сделках:
    #   disabled — выключено (вручную: make train)
    #   online   — gradient steps после каждой сделки [РИСКОВАННО, только эксперименты]
    #   periodic — полный ретрейн в фоне каждые N сделок [РЕКОМЕНДУЕТСЯ]
    #   hybrid   — periodic + динамические веса стратегий в реальном времени
    ONLINE_LEARNING_MODE: str = os.getenv("ONLINE_LEARNING_MODE", "periodic")
    # Количество закрытых сделок до запуска переобучения (для periodic/hybrid)
    ONLINE_LEARNING_TRIGGER: int = int(os.getenv("ONLINE_LEARNING_TRIGGER", "50"))
    # Шаги градиента за один online-апдейт (для режима online)
    ONLINE_LEARNING_GRADIENT_STEPS: int = int(
        os.getenv("ONLINE_LEARNING_GRADIENT_STEPS", "50")
    )

    # ── Backtest ───────────────────────────────────────────────────────────
    # Доля данных, отложенная для out-of-sample теста (0.2 = последние 20%)
    BACKTEST_HOLDOUT_RATIO: float = float(os.getenv("BT_HOLDOUT_RATIO", "0.2"))

    # ── Macro blackout (УЛУЧШЕНИЕ 7) ──────────────────────────────────────
    # Ключ Finnhub API для получения экономического календаря
    FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY", "")
    # Блокировать торговлю за 30 мин до / после крупных макро-событий
    MACRO_BLACKOUT_ENABLED: bool = (
        os.getenv("MACRO_BLACKOUT_ENABLED", "true").lower() == "true"
    )

    def validate(self) -> None:
        """Валидирует конфигурацию и обязательные секреты.

        :raises ValueError: Если параметры за пределами допустимых значений
            или отсутствуют обязательные переменные окружения.
        """
        # Отклоняем стейблкоин как основной торговый символ
        base = self.SYMBOL.split("/")[0]
        if base in STABLECOIN_BASES:
            raise ValueError(
                f"TRADING_SYMBOL={self.SYMBOL} is a stablecoin — set a real asset."
            )

        if self.INITIAL_BALANCE <= 0:
            raise ValueError("INITIAL_BALANCE must be positive")
        if getattr(self, "LEVERAGE", 1) < 1:
            raise ValueError("LEVERAGE must be >= 1")
        if not (0 < getattr(self, "STOP_LOSS_PERCENT", 0.02) < 1):
            raise ValueError("STOP_LOSS_PERCENT must be between 0 and 1")
        if not 0 < self.RISK_PER_TRADE < 1:
            raise ValueError("RISK_PER_TRADE must be between 0 and 1")
        if self.MAX_POSITION_SIZE <= 0 or self.MAX_POSITION_SIZE > 1:
            raise ValueError("MAX_POSITION_SIZE must be between 0 and 1")
        if self.COMMISSION_RATE < 0:
            raise ValueError("COMMISSION_RATE must be non-negative")
        if self.TRADING_INTERVAL <= 0:
            raise ValueError("TRADING_INTERVAL must be positive")

        # Ключи Bybit API обязательны для торговли в реальном режиме
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

        # Валидация AI-провайдера
        if self.AI_PROVIDER not in ("auto", "anthropic", "deepseek", "openai", "groq"):
            raise ValueError(
                f"AI_PROVIDER={self.AI_PROVIDER!r} is invalid. "
                "Use: auto | anthropic | openai | deepseek | groq"
            )
        if self.MODE in ("ai", "hybrid"):
            has_any_key = bool(
                self.ANTHROPIC_API_KEY
                or self.OPENAI_API_KEY
                or self.DEEPSEEK_API_KEY
                or self.GROQ_API_KEY
            )
            provider_key = {
                "anthropic": self.ANTHROPIC_API_KEY,
                "openai": self.OPENAI_API_KEY,
                "deepseek": self.DEEPSEEK_API_KEY,
                "groq": self.GROQ_API_KEY,
            }.get(self.AI_PROVIDER, "")
            if self.AI_PROVIDER == "auto" and not has_any_key:
                raise ValueError(
                    "AI_PROVIDER=auto: нужен хотя бы один ключ — "
                    "ANTHROPIC_API_KEY, OPENAI_API_KEY, "
                    "DEEPSEEK_API_KEY или GROQ_API_KEY."
                )
            if self.AI_PROVIDER != "auto" and not provider_key:
                raise ValueError(
                    f"AI_PROVIDER={self.AI_PROVIDER}: ключ не задан. "
                    "Получить: console.anthropic.com / "
                    "platform.deepseek.com / platform.openai.com"
                )

        # Файл SAC-модели обязателен для режимов dqn/hybrid
        if self.MODE in ("dqn", "hybrid"):
            if not os.path.exists(self.SAC_MODEL_PATH):
                raise ValueError(
                    f"SAC model not found: {self.SAC_MODEL_PATH}. "
                    f"Train it first: make train"
                )


# Глобальный экземпляр конфигурации
config = Config()
