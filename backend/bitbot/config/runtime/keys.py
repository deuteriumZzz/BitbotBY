"""Ключи Redis, в которых бот хранит правки настроек из приложения.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


_KEY_MODE = "bot:mode"


_KEY_SCAN_TOP_N = "bot:scan_top_n"


_KEY_PAUSED = "bot:paused"


_KEY_AUTO_EXECUTE = "bot:auto_execute"


_KEY_LOCAL_FALLBACK = "bot:local_fallback"


_KEY_PAPER_GUARDS_OFF = "bot:paper_guards_off"


_KEY_FORCED = "bot:forced_symbols"


_KEY_EXCLUDED = "bot:excluded_symbols"


_KEY_DISABLED_STRATS = "bot:disabled_strategies"


_KEY_EXPERIMENTAL_SEEN = "bot:experimental_strategies_seen"


_KEY_JOB = "bot:job:"


_KEY_TRADING_HOURS = "bot:trading_hours"


_KEY_QUIET_HOURS = "bot:quiet_hours"


_KEY_MAX_POSITIONS = "bot:max_positions"


_KEY_RISK_PER_TRADE = "bot:risk_per_trade"


_KEY_DRAWDOWN_SCALE = "bot:drawdown_scale_enabled"


_KEY_TRAIN_TOP_N = "bot:train_top_n"


_KEY_AI_PROVIDER = "bot:ai_provider"


_KEY_MARKET_TYPE = "bot:market_type"


_KEY_LEVERAGE_MODE = "bot:leverage_mode"


_KEY_LEVERAGE_TARGET_RISK = "bot:leverage_target_risk"


_KEY_LEVERAGE_DD = "bot:leverage_dd_threshold"


_KEY_DRAWDOWN_CONFIRM = "bot:drawdown_confirm_cycles"


_KEY_SAC_PROMPTED = "bot:sac_prompted"


_KEY_CONFIRM_TIMEOUT = "bot:confirm_timeout"


_KEY_CONFIRM_ON_TIMEOUT = "bot:confirm_on_timeout"


_KEY_FIRST_START_DATE = "bot:first_start_date"


_KEY_TUNE_REMINDED = "bot:tune_reminded"


_KEY_CHRONOS = "bot:chronos_filter"


_KEY_MARKET_PROFILE = "bot:market_profile"


_KEY_CURRENT_SYMBOLS = "bot:current_symbols"


_KEY_TIMEFRAME = "bot:timeframe"


_KEY_MIN_VOLUME = "bot:min_volume_usdt"


_KEY_MAX_VOLUME = "bot:max_volume_usdt"


_KEY_PAPER_TRADING = "bot:paper_trading_override"


_KEY_SAC_BACKUP = "bot:sac_model_backup"


_KEY_SAC_BACKUP_PROFILE = "bot:sac_backup_profile"


_KEY_TRAIN_PROGRESS = "bot:train_progress"


_KEY_TUNE_PROGRESS = "bot:tune_progress"


_KEY_BACKTEST_PROGRESS = "bot:backtest_progress"


_KEY_LAST_AI_PROVIDER = "bot:last_ai_provider"


_KEY_SEASON_MODE = "bot:season_switch_mode"


_KEY_SEASON_INDEX = "bot:season_index"


_KEY_FEAR_GREED = "bot:fear_greed"


_KEY_LAUNCH_RATE_HISTORY = "bot:launch_rate_history"


_KEY_CONFIDENCE_LIVE = "bot:signal_confidence_live"


_KEY_CONFIDENCE_PAPER = "bot:signal_confidence_paper"


# Настройки реестра «сразу» без своего геттера (RuntimeConfig.hot).
HOT_PREFIX = "bot:hot:"
