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
        "USD1",
        "XAUT",
        "PAXG",
        "AEUR",
        "BIDR",
        "BVND",
    }
)

# Токены не подходящие для нормальной торговли:
# leveraged ETF, commodity трекеры, синтетика
JUNK_BASES: frozenset[str] = frozenset(
    {
        "SOXL",
        "SOXS",
        "TQQQ",
        "SQQQ",
        "NVDL",
        "TSLL",
        "TSLS",
        "XAU",
        "XAG",
        "XOI",
        "SLX",
        "SPCXX",
        "BSB",
        "BILL",
        "XPL",
        "ASTER",
        # Недоступны на Bybit linear
        "ZEC",
        "LAB",
        "SPCX",
        "M",
        "MU",
        "BEAT",
        "FARTCOIN",
        "SNDK",
    }
)

# Тикеры с такими префиксами — мусор (1000PEPE, 1000SHIB и т.д.)
JUNK_PREFIXES: tuple[str, ...] = ("1000", "10000", "100000")

# Мемкоины — исключаются из всех сканов независимо от режима сезона
MEME_COIN_BASES: frozenset[str] = frozenset(
    {
        "DOGE",
        "SHIB",
        "PEPE",
        "FLOKI",
        "BONK",
        "WIF",
        "MEME",
        "NEIRO",
        "MOG",
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
    # Порог просадки для LEVERAGE_MODE=full: плечо снижается по мере просадки
    # (не ниже 30% от расчётного). Не путать с MAX_DRAWDOWN_PERCENT (hard halt).
    # Не задан → старый MAX_DRAWDOWN_PERCENT (раньше он обслуживал оба порога).
    LEVERAGE_DD_THRESHOLD: float = float(
        os.getenv("LEVERAGE_DD_THRESHOLD", os.getenv("MAX_DRAWDOWN_PERCENT", "0.15"))
    )

    # ── Trading ────────────────────────────────────────────────────────────
    INITIAL_BALANCE: float = float(os.getenv("INITIAL_BALANCE", "10000.0"))
    RISK_PER_TRADE: float = float(os.getenv("RISK_PER_TRADE", "0.02"))
    COMMISSION_RATE: float = float(os.getenv("COMMISSION_RATE", "0.001"))
    # Интервал между итерациями цикла в секундах (30 = real-time)
    TRADING_INTERVAL: int = int(os.getenv("TRADING_INTERVAL", "30"))
    MAX_POSITION_SIZE: float = float(os.getenv("MAX_POSITION_SIZE", "0.1"))
    STOP_LOSS_PERCENT: float = float(os.getenv("STOP_LOSS_PERCENT", "0.05"))
    # Жёсткий кап на совокупную notional-экспозицию ВСЕХ открытых позиций
    # (с учётом плеча) как множитель баланса. 0 или меньше — отключено.
    # По умолчанию 1.5x — заметно туже, чем текущий незащищённый худший случай
    # (MAX_POSITIONS x Half-Kelly 20% x LEVERAGE_MAX может доходить до ~3x).
    MAX_PORTFOLIO_EXPOSURE: float = float(os.getenv("MAX_PORTFOLIO_EXPOSURE", "1.5"))

    # ── Strategy ───────────────────────────────────────────────────────────
    DEFAULT_STRATEGY: str = os.getenv("ACTIVE_STRATEGY", "ema_crossover")
    # Минимальный confidence для исполнения сигнала (0.0–1.0)
    MIN_SIGNAL_CONFIDENCE: float = float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.65"))
    # Порог для paper режима — ниже чтобы накопить больше сделок для обучения SAC
    MIN_SIGNAL_CONFIDENCE_PAPER: float = float(
        os.getenv("MIN_SIGNAL_CONFIDENCE_PAPER", "0.60")
    )

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
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    # Бесплатные fallback-провайдеры — последними в цепочке (после платных
    # и Groq/Gemini), чтобы не расходовать их скромные дневные лимиты, пока
    # ещё доступны более щедрые/качественные провайдеры выше по списку.
    # OpenRouter free-tier (":free" модели): 50 запросов/сутки без депозита,
    # 1000/сутки после разового пополнения от $10 (openrouter.ai/docs).
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL: str = os.getenv(
        "OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"
    )
    # NVIDIA NIM (build.nvidia.com) — точные бесплатные лимиты не публикуются
    # в доках, зависят от аккаунта; проверять в личном кабинете build.nvidia.com.
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
    NVIDIA_MODEL: str = os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
    # Hard drawdown halt (trading_bot): закрыть всё и пауза DRAWDOWN_HALT_HOURS
    MAX_DRAWDOWN_PERCENT: float = float(os.getenv("MAX_DRAWDOWN_PERCENT", "0.2"))
    PUMPPORTAL_API_KEY: str = os.getenv("PUMPPORTAL_API_KEY", "")

    # ── Mode ───────────────────────────────────────────────────────────────
    # local  → только правила (9 стратегий, без внешних API)
    # dqn    → SAC-модель (train_sac.py)
    # ai     → AI-провайдер (Claude / DeepSeek / OpenAI)
    # hybrid → SAC + AI должны согласиться; расхождение → hold
    MODE: str = os.getenv("MODE", "ai")
    # Путь к SAC-модели (создаётся train_sac.py).
    # bluechip / пусто → sac_model.zip (дефолт), altcoin → sac_model_altcoin.zip
    SAC_MODEL_PATH: str = os.getenv(
        "SAC_MODEL_PATH",
        (
            "models/sac_model_altcoin.zip"
            if os.getenv("SAC_PROFILE") == "altcoin"
            else "models/sac_model.zip"
        ),
    )

    # ── Market Scanner ─────────────────────────────────────────────────────
    # Сколько монет сканировать (топ по объёму)
    SCAN_TOP_N: int = int(os.getenv("SCAN_TOP_N", "20"))
    # True → автоматически исполнять топ-1 рекомендацию
    AUTO_EXECUTE: bool = os.getenv("AUTO_EXECUTE", "false").lower() == "true"
    # Когда AI/SAC/Hybrid не дали ни одного сигнала, бот может открыть сделку по
    # локальным стратегиям. По умолчанию выключено: у стратегий нет доказанного
    # преимущества, а сделки смешивались бы с сделками модели. В Local не влияет.
    LOCAL_FALLBACK: bool = os.getenv("LOCAL_FALLBACK", "false").lower() == "true"

    # ── News ───────────────────────────────────────────────────────────────
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    # Интервал обновления новостей в секундах (900 = 15 мин)
    NEWS_UPDATE_INTERVAL: int = int(os.getenv("NEWS_UPDATE_INTERVAL", "900"))

    # ── Twitter/X Sentiment ────────────────────────────────────────────────
    # Bearer Token из Twitter Developer Portal (Basic план $100/мес)
    # Без токена — TwitterAnalyzer молча возвращает 0.0, если не задан ещё и
    # TWSCRAPE_ACCOUNTS ниже.
    TWITTER_BEARER_TOKEN: str = os.getenv("TWITTER_BEARER_TOKEN", "")
    # Максимум твитов на запрос (экономим лимит Basic: 10K/мес)
    TWITTER_MAX_RESULTS: int = int(os.getenv("TWITTER_MAX_RESULTS", "10"))
    # Fallback без официального API: twscrape (github.com/vladkens/twscrape)
    # ротирует несколько X-аккаунтов вместо платного Bearer Token.
    # Используется ТОЛЬКО если TWITTER_BEARER_TOKEN не задан.
    # Формат: "user:pass:email:email_pass,user2:pass2:email2:email_pass2"
    TWSCRAPE_ACCOUNTS: str = os.getenv("TWSCRAPE_ACCOUNTS", "")
    # Путь к SQLite-базе twscrape с сессионными куки залогиненных аккаунтов
    TWSCRAPE_DB_PATH: str = os.getenv("TWSCRAPE_DB_PATH", "data/twscrape_accounts.db")

    # ── Telegram Sentiment (Telethon MTProto) ──────────────────────────────
    # Получить на my.telegram.org → API development tools
    TELEGRAM_SENTIMENT_API_ID: str = os.getenv("TELEGRAM_SENTIMENT_API_ID", "")
    TELEGRAM_SENTIMENT_API_HASH: str = os.getenv("TELEGRAM_SENTIMENT_API_HASH", "")
    # Номер телефона для авторизации Telethon (формат: +375291234567)
    TELEGRAM_SENTIMENT_PHONE: str = os.getenv("TELEGRAM_SENTIMENT_PHONE", "")
    # Публичные каналы через запятую (без @)
    TELEGRAM_SENTIMENT_CHANNELS: str = os.getenv(
        "TELEGRAM_SENTIMENT_CHANNELS",
        "CryptoMoonAlerts,pumpsignalscrypto,memecoinsnews",
    )

    # ── Sentiment Poller ───────────────────────────────────────────────────
    # Интервал опроса Twitter + Telegram в секундах
    SENTIMENT_POLL_INTERVAL: int = int(os.getenv("SENTIMENT_POLL_INTERVAL", "300"))
    # TTL кэша сентимента в Redis в секундах
    SENTIMENT_CACHE_TTL: int = int(os.getenv("SENTIMENT_CACHE_TTL", "600"))
    # Порог buy/sell для SentimentStrategy (-1.0..1.0)
    SENTIMENT_BUY_THRESHOLD: float = float(os.getenv("SENTIMENT_BUY_THRESHOLD", "0.4"))
    SENTIMENT_SELL_THRESHOLD: float = float(
        os.getenv("SENTIMENT_SELL_THRESHOLD", "-0.4")
    )

    # ── Telegram ───────────────────────────────────────────────────────────
    # Токен BotFather: /newbot → скопировать токен
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    # Chat ID (личный или группа): @userinfobot → узнать id
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    # Секунд на решение по сделке в Live без авто-сделок: 10–120 (дольше —
    # цена уходит и сигнал устаревает)
    TELEGRAM_CONFIRM_TIMEOUT: int = int(os.getenv("TELEGRAM_CONFIRM_TIMEOUT", "60"))
    # Если не ответили: execute — открыть сделку, skip — пропустить
    CONFIRM_ON_TIMEOUT: str = os.getenv("CONFIRM_ON_TIMEOUT", "execute")

    # ── Risk Management ────────────────────────────────────────────────────
    # Максимальный дневной убыток: доля баланса (0.05 = 5%)
    DAILY_LOSS_LIMIT: float = float(os.getenv("DAILY_LOSS_LIMIT", "0.05"))
    # Hard drawdown halt: пауза N часов если просадка от пика >= MAX_DRAWDOWN_PERCENT
    DRAWDOWN_HALT_HOURS: float = float(os.getenv("DRAWDOWN_HALT_HOURS", "4"))
    # Сколько циклов подряд просадка должна держаться перед остановкой
    # (защита от flash crash)
    DRAWDOWN_CONFIRM_CYCLES: int = int(os.getenv("DRAWDOWN_CONFIRM_CYCLES", "3"))
    # Volatility circuit breaker: ATR вырос в N раз от медианы → закрыть позиции
    ATR_SPIKE_MULT: float = float(os.getenv("ATR_SPIKE_MULT", "3.0"))
    # Кол-во свечей для расчёта базового ATR (медиана)
    ATR_SPIKE_LOOKBACK: int = int(os.getenv("ATR_SPIKE_LOOKBACK", "20"))
    # Pre-committed kill-switch: если live EV стратегии за последние
    # LIVE_ALPHA_KILL_MIN_TRADES сделок хуже LIVE_ALPHA_KILL_EV_THRESHOLD —
    # новые сделки по этой стратегии блокируются до ручного вмешательства.
    # Решение принято ДО того, как стратегия начала терять деньги, не
    # постфактум, когда просадка уже случилась.
    LIVE_ALPHA_KILL_ENABLED: bool = (
        os.getenv("LIVE_ALPHA_KILL_ENABLED", "true").lower() == "true"
    )
    LIVE_ALPHA_KILL_MIN_TRADES: int = int(os.getenv("LIVE_ALPHA_KILL_MIN_TRADES", "20"))
    LIVE_ALPHA_KILL_EV_THRESHOLD: float = float(
        os.getenv("LIVE_ALPHA_KILL_EV_THRESHOLD", "-0.5")
    )
    # True → использовать Bybit testnet
    TESTNET: bool = os.getenv("TESTNET", "false").lower() == "true"

    # ── Position Management ────────────────────────────────────────────────
    # Максимум одновременных открытых позиций
    MAX_POSITIONS: int = int(os.getenv("MAX_POSITIONS", "3"))
    # True → симулировать сделки без реальных ордеров
    PAPER_TRADING: bool = os.getenv("PAPER_TRADING", "false").lower() == "true"
    # Кодовое слово для переключения Paper ↔ Live через Telegram
    # (пусто = функция отключена)
    TRADING_MODE_PIN: str = os.getenv("TRADING_MODE_PIN", "")
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
    # В paper режиме порог выше — крипто сильно коррелирует в боковике,
    # 0.70 блокирует почти всё; 0.85 позволяет накопить сделки для SAC.
    MAX_CORRELATION_PAPER: float = float(os.getenv("MAX_CORRELATION_PAPER", "0.85"))
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
    # Cooldown: не переоткрывать один и тот же символ N часов после закрытия
    SYMBOL_COOLDOWN_HOURS: float = float(os.getenv("SYMBOL_COOLDOWN_HOURS", "3"))
    # Порог volume spike: во сколько раз текущий объём свечи превышает среднее
    # за последние 20 свечей. 2.5 = аномалия, рекомендуется 2.0–4.0.
    VOLUME_SPIKE_THRESHOLD: float = float(os.getenv("VOLUME_SPIKE_THRESHOLD", "2.5"))
    # Максимальный 24h объём в USDT (0 = без ограничения).
    # Позволяет исключить блючипы при торговле альткоинами.
    MAX_VOLUME_USDT: float = float(os.getenv("MAX_VOLUME_USDT", "0"))
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
        if self.MAX_PORTFOLIO_EXPOSURE < 0:
            raise ValueError("MAX_PORTFOLIO_EXPOSURE must be >= 0 (0 disables the cap)")
        if not (1 <= self.LEVERAGE_MAX <= 20):
            raise ValueError(
                f"LEVERAGE_MAX={self.LEVERAGE_MAX} must be between 1 and 20 — "
                "higher values risk liquidation on small adverse moves."
            )
        if not (0 < self.DAILY_LOSS_LIMIT <= 0.5):
            raise ValueError(
                f"DAILY_LOSS_LIMIT={self.DAILY_LOSS_LIMIT} must be a fraction "
                "in (0, 0.5] — e.g. 0.05 for 5%/day, not 5.0."
            )
        if not (0 < self.MAX_DRAWDOWN_PERCENT <= 0.5):
            raise ValueError(
                f"MAX_DRAWDOWN_PERCENT={self.MAX_DRAWDOWN_PERCENT} must be a "
                "fraction in (0, 0.5] — e.g. 0.2 for 20%, not 20.0."
            )
        if not (0 < self.LEVERAGE_DD_THRESHOLD <= 0.5):
            raise ValueError(
                f"LEVERAGE_DD_THRESHOLD={self.LEVERAGE_DD_THRESHOLD} must be a "
                "fraction in (0, 0.5] — e.g. 0.15 for 15%, not 15.0."
            )
        if self.DAILY_LOSS_LIMIT > self.MAX_DRAWDOWN_PERCENT:
            raise ValueError(
                f"DAILY_LOSS_LIMIT={self.DAILY_LOSS_LIMIT} must not exceed "
                f"MAX_DRAWDOWN_PERCENT={self.MAX_DRAWDOWN_PERCENT} — the daily "
                "breaker should trip before the overall drawdown halt."
            )
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
        _valid_providers = (
            "auto",
            "anthropic",
            "deepseek",
            "openai",
            "groq",
            "gemini",
            "openrouter",
            "nvidia",
        )
        if self.AI_PROVIDER not in _valid_providers:
            raise ValueError(
                f"AI_PROVIDER={self.AI_PROVIDER!r} is invalid. "
                f"Use: {' | '.join(_valid_providers)}"
            )
        if self.MODE in ("ai", "hybrid"):
            provider_key = {
                "anthropic": self.ANTHROPIC_API_KEY,
                "openai": self.OPENAI_API_KEY,
                "deepseek": self.DEEPSEEK_API_KEY,
                "groq": self.GROQ_API_KEY,
                "gemini": self.GEMINI_API_KEY,
                "openrouter": self.OPENROUTER_API_KEY,
                "nvidia": self.NVIDIA_API_KEY,
            }
            has_any_key = any(provider_key.values())
            if self.AI_PROVIDER == "auto" and not has_any_key:
                raise ValueError(
                    "AI_PROVIDER=auto: нужен хотя бы один ключ — "
                    "ANTHROPIC_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY, "
                    "GROQ_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY "
                    "или NVIDIA_API_KEY."
                )
            if self.AI_PROVIDER != "auto" and not provider_key.get(self.AI_PROVIDER):
                raise ValueError(
                    f"AI_PROVIDER={self.AI_PROVIDER}: ключ не задан. "
                    "Получить: console.anthropic.com / platform.deepseek.com / "
                    "platform.openai.com / console.groq.com / "
                    "aistudio.google.com / openrouter.ai / build.nvidia.com"
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
