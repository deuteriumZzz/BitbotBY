"""Все ключи Redis бота в одном месте: и воркеры, и API читают их отсюда.

Значения не меняются — по ним лежат данные в Redis. Ключ, который строится из
префикса (сессия, задача, cooldown), — константа `…_PREFIX`. Новый ключ — сюда;
строка "bot:…" где-то ещё в коде не проходит тест tests/test_redis_keys.py.
"""

# ── Настройки из приложения (RuntimeConfig, реестр настроек) ──────────────

AI_CALLS_PREFIX = "bot:ai_calls:"
AI_CHAIN = "bot:ai_chain"
AI_PROVIDER = "bot:ai_provider"
ALERT_COOLDOWN_PREFIX = "bot:alert:cooldown:"
ALERT_ENABLED_PREFIX = "bot:alert:enabled:"
AUTO_EXECUTE = "bot:auto_execute"
BACKTEST_PROGRESS = "bot:backtest_progress"
CHRONOS_FILTER = "bot:chronos_filter"
FORECASTER = "bot:forecaster"
CONFIG_HISTORY = "bot:config_history"
CONFIG_OVERRIDES = "bot:config_overrides"
CONFIRM_ON_TIMEOUT = "bot:confirm_on_timeout"
CONFIRM_TIMEOUT = "bot:confirm_timeout"
CURRENT_SYMBOLS = "bot:current_symbols"
DEEPSEEK_BALANCE = "bot:deepseek_balance"
DRAWDOWN_CONFIRM_CYCLES = "bot:drawdown_confirm_cycles"
DRAWDOWN_SCALE_ENABLED = "bot:drawdown_scale_enabled"
EXCLUDED_SYMBOLS = "bot:excluded_symbols"
EXPERIMENTAL_STRATEGIES_SEEN = "bot:experimental_strategies_seen"
FIRST_START_DATE = "bot:first_start_date"
# Первый запуск: "0" — бот на паузе, ждёт, пока владелец его настроит и включит; "1" — включён.
SETUP_DONE = "bot:setup_done"
FORCED_SYMBOLS = "bot:forced_symbols"
# Настройки реестра «сразу» без своего геттера (RuntimeConfig.hot).
HOT_PREFIX = "bot:hot:"
JOB_PREFIX = "bot:job:"
LAUNCH_RATE_HISTORY = "bot:launch_rate_history"
LEVERAGE_DD_THRESHOLD = "bot:leverage_dd_threshold"
LEVERAGE_MODE = "bot:leverage_mode"
LEVERAGE_TARGET_RISK = "bot:leverage_target_risk"
LOCAL_FALLBACK = "bot:local_fallback"
LOGIN_LINK_PREFIX = "bot:login_link:"
MARKET_TYPE = "bot:market_type"
MAX_VOLUME_USDT = "bot:max_volume_usdt"
MIN_VOLUME_USDT = "bot:min_volume_usdt"
PAPER_GUARDS_OFF = "bot:paper_guards_off"
PROVIDER_COOLDOWN_PREFIX = "bot:provider_cooldown:"
PROVIDER_STATUS_PREFIX = "bot:provider_status:"
QUIET_HOURS = "bot:quiet_hours"
RISK_PER_TRADE = "bot:risk_per_trade"
SAC_BACKUP_PROFILE = "bot:sac_backup_profile"
SAC_MODEL_BACKUP = "bot:sac_model_backup"
SAC_PROMPTED = "bot:sac_prompted"
SCAN_TOP_N = "bot:scan_top_n"
SEASON_SWITCH_MODE = "bot:season_switch_mode"
SIGNAL_CONFIDENCE_LIVE = "bot:signal_confidence_live"
SIGNAL_CONFIDENCE_PAPER = "bot:signal_confidence_paper"
SL_PERCENT = "bot:sl_percent"
TIMEFRAME = "bot:timeframe"
TOKENS_USED_PREFIX = "bot:tokens_used:"
TP_MULTIPLIER = "bot:tp_multiplier"
TRADING_HOURS = "bot:trading_hours"
TRADING_INTERVAL = "bot:trading_interval"
TRAIN_PROGRESS = "bot:train_progress"
TRAIN_TOP_N = "bot:train_top_n"
TUNE_PROGRESS = "bot:tune_progress"
TUNE_REMINDED = "bot:tune_reminded"

# ── Шина команд и событий ─────────────────────────────────────────────────

CMDRES_PREFIX = "bot:cmdres:"
CMD_PREFIX = "bot:cmd:"
COMMANDS = "bot:commands"
EVENTS = "bot:events"
STREAM_EVENTS = "bot:stream:events"

# ── Приложение: вход и сессии ─────────────────────────────────────────────

DESKTOP_CODE = "bot:desktop_code"
DESKTOP_CODE_SENT = "bot:desktop_code_sent"
DESKTOP_CODE_TRIES = "bot:desktop_code_tries"
DISABLED_STRATEGIES = "bot:disabled_strategies"
FEAR_GREED = "bot:fear_greed"
LAST_AI_PROVIDER = "bot:last_ai_provider"
LAST_CYCLE = "bot:last_cycle"
LOCAL_LOGIN_FAILS = "bot:local_login_fails"
LOCAL_LOGIN_USED_PREFIX = "bot:local_login_used:"
MARKET_PROFILE = "bot:market_profile"
MAX_POSITIONS = "bot:max_positions"
MODE = "bot:mode"
MODE_READINESS = "bot:mode_readiness"
OPS_STATUS = "bot:ops_status"
PAPER_TRADING_OVERRIDE = "bot:paper_trading_override"
PAUSED = "bot:paused"
PENDING_SIGNAL_PREFIX = "bot:pending_signal:"
SEASON_INDEX = "bot:season_index"
SESSION_PREFIX = "bot:session:"
WATCHDOG_ALERT = "bot:watchdog_alert"

# ── Состояние бота для приложения ─────────────────────────────────────────

PIN_FAILS = "bot:pin_fails"

# ── Исполнение и сделки по доверенным каналам ─────────────────────────────

# последний разобранный пост — для приложения
TG_LAST_TRUSTED = "bot:tg:last_trusted"
# канал → id сделки, после которой снято доверие
TG_UNTRUSTED = "bot:tg:untrusted"

# ── Риск ──────────────────────────────────────────────────────────────────

RISK_DAILY_REF_PREFIX = "bot:risk:daily_ref:"

# ── Источники: настроение, Telegram ───────────────────────────────────────

SENTIMENT_PREFIX = "bot:sentiment:"
TELEGRAM_STATUS = "bot:telegram:status"
