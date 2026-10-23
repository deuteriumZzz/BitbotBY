"""
Runtime-конфигурация поверх Redis.

Позволяет менять настройки бота без перезапуска контейнера.
Все значения хранятся в Redis и переопределяют .env при наличии.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set

from config import Config
from src import ai_providers

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
_KEY_AWAITING_MODE_PIN = "bot:awaiting_mode_pin"
_KEY_AWAITING_ALERT_COOLDOWN = "bot:awaiting_alert_cooldown"
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

_AI_PROVIDERS = frozenset({"auto", *ai_providers.BY_ID})
_LEVERAGE_MODES = frozenset({"fixed", "volatility", "full"})

# Время на решение по сделке. Дольше 2 минут ждать нельзя: цена уходит,
# и сигнал устаревает.
CONFIRM_TIMEOUT_MIN = 10
CONFIRM_TIMEOUT_MAX = 120


def confirm_policy(timeout: Any, on_timeout: Any) -> "tuple[int, str]":
    """(секунд на решение, что без ответа: execute | skip).

    Старое «0» (ручной режим: ждать ответа до часа) читается как потолок
    и пропуск без ответа.
    """
    try:
        sec = int(timeout) if isinstance(timeout, (int, str)) else 60
    except ValueError:
        sec = 60
    if sec == 0:
        return CONFIRM_TIMEOUT_MAX, "skip"
    sec = max(CONFIRM_TIMEOUT_MIN, min(sec, CONFIRM_TIMEOUT_MAX))
    return sec, on_timeout if on_timeout in ("execute", "skip") else "execute"


_MARKET_TYPES = frozenset({"spot", "linear"})

# Пресеты риска — одно определение для Telegram (/risk) и приложения
# (GET /api/risk/presets). Ключи — как в настройках (settings_registry): приложение
# пишет их обычным PATCH /api/config, Telegram — через apply_risk_preset.
RISK_PRESETS: Dict[str, Dict[str, Any]] = {
    "conservative": {
        "MAX_POSITIONS": 2,
        "RISK_PER_TRADE": 0.01,
        "DRAWDOWN_SCALE_ENABLED": True,
        "LEVERAGE_MODE": "fixed",
    },
    "moderate": {
        "MAX_POSITIONS": 3,
        "RISK_PER_TRADE": 0.02,
        "DRAWDOWN_SCALE_ENABLED": True,
        "LEVERAGE_MODE": "volatility",
    },
    "aggressive": {
        "MAX_POSITIONS": 5,
        "RISK_PER_TRADE": 0.04,
        "DRAWDOWN_SCALE_ENABLED": False,
        "LEVERAGE_MODE": "full",
    },
}

# Классификатор bluechip монет. Используется для разделения сканов:
# bluechip-сезон: из топа берём только монеты из этого набора
# altcoin-сезон: из топа исключаем все монеты из этого набора
BLUECHIP_BASES: frozenset[str] = frozenset(
    {
        "BTC",
        "ETH",
        "SOL",
        "XRP",
        "BNB",
        "ADA",
        "AVAX",
        "LINK",
        "DOT",
        "LTC",
        "UNI",
        "ATOM",
        "NEAR",
        "APT",
    }
)

_MARKET_PROFILES = {
    "bluechip": {
        "label": "🔵 Блючипы",
        "scan_top_n": 20,
        "train_top_n": 20,
        "min_volume_usdt": 0,
        "max_volume_usdt": 0,
        "risk_per_trade": 0.02,
        "timeframe": "15m",
        "trading_interval": 900,  # раз в 15 мин = 96 AI запросов/день
        "mode": "hybrid",
        "model_path": "models/sac_model.zip",
        "sl_percent": 0.05,  # SL 5%
        "tp_multiplier": 2.0,  # TP = 10% (консервативно)
    },
    "altcoin": {
        "label": "🟡 Альткоины",
        "scan_top_n": 100,
        "train_top_n": 100,
        "min_volume_usdt": 500_000,
        "max_volume_usdt": 2_000_000_000,
        "risk_per_trade": 0.01,
        "timeframe": "5m",
        "trading_interval": 300,  # раз в 5 мин = 288 AI запросов/день
        "mode": "hybrid",
        "model_path": "models/sac_model_altcoin.zip",
        "sl_percent": 0.04,  # SL 4%
        "tp_multiplier": 4.0,  # TP = 16% (ловим большие альт-движения)
    },
    "meme": {
        "label": "🚀 Мемкоины",
        "scan_top_n": 200,
        "train_top_n": 50,
        "min_volume_usdt": 50_000,
        "max_volume_usdt": 500_000_000,
        "risk_per_trade": 0.005,
        "timeframe": "3m",
        "trading_interval": 90,  # раз в 90 сек — быстрый цикл на коротком ТФ
        "mode": "hybrid",
        "model_path": "models/sac_model_meme.zip",
        "sl_percent": 0.12,  # SL 12% — широкий стоп для волатильных монет
        "tp_multiplier": 6.0,  # TP = 72% (6×SL — ловим памп)
    },
}

_VALID_MODES = {"ai", "local", "hybrid", "dqn"}


class RuntimeConfig:
    """
    Обёртка над Redis для хранения изменяемых во время работы настроек.

    Fallback: если ключ в Redis отсутствует, возвращает значение из Config (.env).
    """

    def __init__(self, redis_client: Any) -> None:
        self._r = redis_client  # RedisClient instance

    def _get(self, key: str) -> str | None:
        try:
            raw = self._r.redis_client.get(key)
            if raw is None:
                return None
            if isinstance(raw, bytes):
                return raw.decode()
            if isinstance(raw, str):
                return raw
            return None  # неожиданный тип (напр. MagicMock в тестах)
        except Exception:
            return None

    def _set(self, key: str, value: str) -> None:
        try:
            self._r.redis_client.set(key, value)
        except Exception as e:
            logger.warning("RuntimeConfig set failed: %s", e)

    # ── Mode ──────────────────────────────────────────────────────────────────

    def get_mode(self) -> str:
        val = self._get(_KEY_MODE)
        return val if val in _VALID_MODES else Config.MODE

    def set_mode(self, mode: str) -> bool:
        if mode not in _VALID_MODES:
            return False
        self._set(_KEY_MODE, mode)
        logger.info("Runtime: режим изменён → %s", mode)
        return True

    # ── Scan top N ────────────────────────────────────────────────────────────

    def get_scan_top_n(self) -> int:
        val = self._get(_KEY_SCAN_TOP_N)
        try:
            n = int(val) if val else Config.SCAN_TOP_N
            return max(1, min(n, 200))
        except (ValueError, TypeError):
            return Config.SCAN_TOP_N

    def set_scan_top_n(self, n: int) -> bool:
        if not (1 <= n <= 200):
            return False
        self._set(_KEY_SCAN_TOP_N, str(n))
        logger.info("Runtime: scan_top_n изменён → %d", n)
        return True

    # ── Pause ─────────────────────────────────────────────────────────────────

    def is_paused(self) -> bool:
        return self._get(_KEY_PAUSED) == "1"

    def set_paused(self, paused: bool) -> None:
        self._set(_KEY_PAUSED, "1" if paused else "0")
        logger.info("Runtime: бот %s", "на паузе" if paused else "возобновлён")

    # ── Auto-execute ──────────────────────────────────────────────────────────

    def get_auto_execute(self) -> bool:
        val = self._get(_KEY_AUTO_EXECUTE)
        if val is None:
            return Config.AUTO_EXECUTE
        return val == "1"

    def set_auto_execute(self, enabled: bool) -> None:
        self._set(_KEY_AUTO_EXECUTE, "1" if enabled else "0")
        logger.info("Runtime: auto_execute → %s", "ВКЛ" if enabled else "ВЫКЛ")

    def get_local_fallback(self) -> bool:
        val = self._get(_KEY_LOCAL_FALLBACK)
        if val is None:
            return Config.LOCAL_FALLBACK
        return val == "1"

    def set_local_fallback(self, enabled: bool) -> None:
        self._set(_KEY_LOCAL_FALLBACK, "1" if enabled else "0")
        logger.info("Runtime: local_fallback → %s", "ВКЛ" if enabled else "ВЫКЛ")

    def get_paper_guards_off(self) -> bool:
        """Сохранённый выбор (в Live не действует, см. paper_guards_off)."""
        val = self._get(_KEY_PAPER_GUARDS_OFF)
        if val is None:
            return Config.PAPER_GUARDS_OFF
        return val == "1"

    def set_paper_guards_off(self, enabled: bool) -> None:
        self._set(_KEY_PAPER_GUARDS_OFF, "1" if enabled else "0")
        logger.warning("Runtime: paper_guards_off → %s", "ВКЛ" if enabled else "ВЫКЛ")

    def paper_guards_off(self) -> bool:
        """True только в Paper и только если защиты выключены. В Live всегда False."""
        return bool(Config.PAPER_TRADING) and self.get_paper_guards_off()

    # ── Symbol overrides ──────────────────────────────────────────────────────

    def _smembers(self, key: str) -> Set[str]:
        try:
            raw = self._r.redis_client.smembers(key)
            return {m.decode() if isinstance(m, bytes) else m for m in raw}
        except Exception:
            return set()

    def get_forced_symbols(self) -> Set[str]:
        return self._smembers(_KEY_FORCED)

    def add_forced_symbol(self, symbol: str) -> None:
        try:
            self._r.redis_client.sadd(_KEY_FORCED, symbol)
            logger.info("Runtime: %s добавлен в список", symbol)
        except Exception as e:
            logger.warning("RuntimeConfig sadd failed: %s", e)

    def remove_forced_symbol(self, symbol: str) -> None:
        try:
            self._r.redis_client.srem(_KEY_FORCED, symbol)
        except Exception as e:
            logger.warning("RuntimeConfig srem failed: %s", e)

    def get_excluded_symbols(self) -> Set[str]:
        return self._smembers(_KEY_EXCLUDED)

    def add_excluded_symbol(self, symbol: str) -> None:
        try:
            self._r.redis_client.sadd(_KEY_EXCLUDED, symbol)
            logger.info("Runtime: %s исключён из сканирования", symbol)
        except Exception as e:
            logger.warning("RuntimeConfig sadd failed: %s", e)

    def remove_excluded_symbol(self, symbol: str) -> None:
        try:
            self._r.redis_client.srem(_KEY_EXCLUDED, symbol)
        except Exception as e:
            logger.warning("RuntimeConfig srem failed: %s", e)

    # ── Risk management ───────────────────────────────────────────────────────

    def get_max_positions(self) -> int:
        val = self._get(_KEY_MAX_POSITIONS)
        try:
            return max(1, min(int(val), 20)) if val else Config.MAX_POSITIONS
        except (ValueError, TypeError):
            return Config.MAX_POSITIONS

    def set_max_positions(self, n: int) -> bool:
        if not (1 <= n <= 20):
            return False
        self._set(_KEY_MAX_POSITIONS, str(n))
        logger.info("Runtime: max_positions → %d", n)
        return True

    def get_risk_per_trade(self) -> float:
        val = self._get(_KEY_RISK_PER_TRADE)
        try:
            v = float(val) if val else Config.RISK_PER_TRADE
            return max(0.001, min(v, 0.5))
        except (ValueError, TypeError):
            return Config.RISK_PER_TRADE

    def set_risk_per_trade(self, v: float) -> bool:
        if not (0.001 <= v <= 0.5):
            return False
        self._set(_KEY_RISK_PER_TRADE, str(v))
        logger.info("Runtime: risk_per_trade → %.1f%%", v * 100)
        return True

    def get_sl_percent(self) -> float:
        """SL % для текущего профиля (bluechip=5%, altcoin=4%)."""
        try:
            val = self._get("bot:sl_percent")
            return float(val) if val else Config.STOP_LOSS_PERCENT
        except (ValueError, TypeError):
            return Config.STOP_LOSS_PERCENT

    def get_tp_multiplier(self) -> float:
        """Множитель TP относительно SL (bluechip=2x, altcoin=4x)."""
        try:
            val = self._get("bot:tp_multiplier")
            return float(val) if val else 2.0
        except (ValueError, TypeError):
            return 2.0

    def get_trading_interval(self) -> int:
        """Интервал торгового цикла в секундах (из профиля или .env)."""
        try:
            val = self._get("bot:trading_interval")
            return int(val) if val else Config.TRADING_INTERVAL
        except (ValueError, TypeError):
            return Config.TRADING_INTERVAL

    def get_signal_confidence(self, paper: bool) -> float:
        key = _KEY_CONFIDENCE_PAPER if paper else _KEY_CONFIDENCE_LIVE
        default = (
            Config.MIN_SIGNAL_CONFIDENCE_PAPER
            if paper
            else Config.MIN_SIGNAL_CONFIDENCE
        )
        val = self._get(key)
        try:
            v = float(val) if val else default
            return max(0.40, min(v, 0.95))
        except (ValueError, TypeError):
            return default

    def set_signal_confidence(self, v: float, paper: bool) -> bool:
        if not (0.40 <= v <= 0.95):
            return False
        key = _KEY_CONFIDENCE_PAPER if paper else _KEY_CONFIDENCE_LIVE
        self._set(key, str(round(v, 2)))
        mode = "paper" if paper else "live"
        logger.info("Runtime: signal_confidence[%s] → %.2f", mode, v)
        return True

    def get_drawdown_scale_enabled(self) -> bool:
        val = self._get(_KEY_DRAWDOWN_SCALE)
        if val is None:
            return Config.DRAWDOWN_SCALE_ENABLED
        return val == "1"

    def set_drawdown_scale_enabled(self, enabled: bool) -> None:
        self._set(_KEY_DRAWDOWN_SCALE, "1" if enabled else "0")
        logger.info("Runtime: drawdown_scale → %s", "ВКЛ" if enabled else "ВЫКЛ")

    def apply_risk_preset(self, name: str) -> bool:
        """Применяет пресет риск-профиля. Возвращает False если имя неизвестно."""
        preset = RISK_PRESETS.get(name)
        if not preset:
            return False
        self.set_max_positions(int(preset["MAX_POSITIONS"]))
        self.set_risk_per_trade(float(preset["RISK_PER_TRADE"]))
        self.set_drawdown_scale_enabled(bool(preset["DRAWDOWN_SCALE_ENABLED"]))
        self.set_leverage_mode(str(preset["LEVERAGE_MODE"]))
        logger.info("Runtime: применён риск-профиль '%s'", name)
        return True

    def get_risk_summary(self) -> dict:
        return {
            "max_positions": self.get_max_positions(),
            "risk_per_trade": self.get_risk_per_trade(),
            "drawdown_scale_enabled": self.get_drawdown_scale_enabled(),
            "drawdown_confirm_cycles": self.get_drawdown_confirm_cycles(),
        }

    def get_drawdown_confirm_cycles(self) -> int:
        """Кол-во циклов подряд для подтверждения просадки перед halt (1–10)."""
        val = self._get(_KEY_DRAWDOWN_CONFIRM)
        try:
            return (
                max(1, min(int(val), 10))
                if val
                else int(getattr(Config, "DRAWDOWN_CONFIRM_CYCLES", 3))
            )
        except (ValueError, TypeError):
            return 3

    def set_drawdown_confirm_cycles(self, n: int) -> bool:
        if not (1 <= n <= 10):
            return False
        self._set(_KEY_DRAWDOWN_CONFIRM, str(n))
        logger.info("Runtime: drawdown_confirm_cycles → %d", n)
        return True

    # ── SAC Training ──────────────────────────────────────────────────────────

    def get_train_top_n(self) -> int:
        """Количество символов для обучения SAC (default: TRAIN_TOP_N или 20)."""
        val = self._get(_KEY_TRAIN_TOP_N)
        try:
            return (
                max(1, min(int(val), 100))
                if val
                else int(__import__("os").getenv("TRAIN_TOP_N", "20"))
            )
        except (ValueError, TypeError):
            return 20

    def set_train_top_n(self, n: int) -> bool:
        if not (1 <= n <= 100):
            return False
        self._set(_KEY_TRAIN_TOP_N, str(n))
        logger.info("Runtime: train_top_n → %d", n)
        return True

    # ── AI Provider ───────────────────────────────────────────────────────────

    def get_ai_provider(self) -> str:
        """Текущий AI-провайдер: auto | anthropic | openai | deepseek | groq."""
        val = self._get(_KEY_AI_PROVIDER)
        if val and val in _AI_PROVIDERS:
            return val
        return getattr(Config, "AI_PROVIDER", "auto")

    def set_ai_provider(self, provider: str) -> bool:
        if provider not in _AI_PROVIDERS:
            return False
        self._set(_KEY_AI_PROVIDER, provider)
        logger.info("Runtime: ai_provider → %s", provider)
        return True

    # ── Market Type (spot / futures) ─────────────────────────────────────────

    def get_market_type(self) -> str:
        """Текущий тип рынка: spot | linear (perpetual futures)."""
        val = self._get(_KEY_MARKET_TYPE)
        if val and val in _MARKET_TYPES:
            return val
        return getattr(Config, "MARKET_TYPE", "linear")

    def set_market_type(self, market_type: str) -> bool:
        if market_type not in _MARKET_TYPES:
            return False
        self._set(_KEY_MARKET_TYPE, market_type)
        # Зеркалит паттерн apply_market_profile()'s SAC_PROFILE — env var
        # даёт коду без ссылки на RuntimeConfig (напр. position_monitor.py
        # при записи experiences.jsonl) увидеть актуальный тип рынка.
        import os as _os

        _os.environ["MARKET_TYPE"] = market_type
        logger.info("Runtime: market_type → %s", market_type)
        return True

    def get_last_ai_provider(self) -> str:
        """Провайдер, который реально ответил последним."""
        return self._get(_KEY_LAST_AI_PROVIDER) or ""

    def set_last_ai_provider(self, provider: str) -> None:
        self._set(_KEY_LAST_AI_PROVIDER, provider)

    def set_provider_status(self, provider: str, status: str) -> None:
        """Сохраняет статус провайдера: 'ok', 'no_balance', 'rate_limit'."""
        self._set(f"bot:provider_status:{provider}", status)

    def get_provider_status(self, provider: str) -> str:
        """Возвращает последний известный статус провайдера."""
        return self._get(f"bot:provider_status:{provider}") or "unknown"

    # ── Цепочка AI-провайдеров (раздел «AI» в приложении) ─────────────────────

    def get_ai_chain(self) -> dict:
        """Режим, порядок, модели и включённые провайдеры; пусто — по умолчанию."""
        raw = self._get("bot:ai_chain")
        try:
            return ai_providers.clean_chain(json.loads(raw)) if raw else {}
        except (ValueError, TypeError):
            return {}

    def set_ai_chain(self, changes: dict) -> dict:
        """Сливает проверенные изменения с текущими; ValueError — неверные."""
        merged = {**self.get_ai_chain(), **ai_providers.clean_chain(changes)}
        self._set("bot:ai_chain", json.dumps(merged))
        logger.info("Runtime: ai_chain → %s", merged)
        return merged

    def set_provider_cooldown(self, provider: str, until: float) -> None:
        """До какого времени (epoch) провайдер на паузе; 0 — паузы нет."""
        self._set(f"bot:provider_cooldown:{provider}", str(int(until)))

    def get_provider_cooldown(self, provider: str) -> float:
        try:
            return float(self._get(f"bot:provider_cooldown:{provider}") or 0)
        except ValueError:
            return 0.0

    def record_ai_call(self, provider: str) -> None:
        """Счётчик запросов к провайдеру за текущие UTC-сутки."""
        today = datetime.now(timezone.utc).date().isoformat()
        key = f"bot:ai_calls:{provider}:{today}"
        try:
            self._r.redis_client.incr(key)
            self._r.redis_client.expire(key, 2 * 24 * 3600)
        except Exception as e:
            logger.warning("RuntimeConfig record_ai_call failed: %s", e)

    def get_ai_calls_today(self, provider: str) -> int:
        today = datetime.now(timezone.utc).date().isoformat()
        try:
            return int(self._get(f"bot:ai_calls:{provider}:{today}") or 0)
        except ValueError:
            return 0

    def record_token_usage(self, provider: str, tokens: int) -> None:
        """Накапливает счётчик токенов за текущие UTC-сутки для провайдера."""
        import datetime as _dt

        today = _dt.datetime.now(_dt.timezone.utc).date().isoformat()
        key = f"bot:tokens_used:{provider}:{today}"
        try:
            self._r.redis_client.incrby(key, tokens)
            self._r.redis_client.expire(key, 2 * 24 * 3600)
        except Exception as e:
            logger.warning("RuntimeConfig record_token_usage failed: %s", e)

    def get_token_usage_today(self, provider: str) -> int:
        """Токенов использовано провайдером за текущие UTC-сутки."""
        import datetime as _dt

        today = _dt.datetime.now(_dt.timezone.utc).date().isoformat()
        val = self._get(f"bot:tokens_used:{provider}:{today}")
        try:
            return int(val) if val else 0
        except ValueError:
            return 0

    def set_deepseek_balance(self, balance_usd: float) -> None:
        """Кэширует баланс DeepSeek в USD (TTL 1 час)."""
        import json as _json
        import time as _time

        try:
            self._r.redis_client.setex(
                "bot:deepseek_balance",
                3600,
                _json.dumps({"balance": balance_usd, "ts": _time.time()}),
            )
        except Exception as e:
            logger.warning("RuntimeConfig set_deepseek_balance failed: %s", e)

    def get_deepseek_balance(self) -> "dict | None":
        """Возвращает {'balance': float, 'ts': float} или None если нет в кэше."""
        import json as _json

        raw = self._get("bot:deepseek_balance")
        if not raw:
            return None
        try:
            return _json.loads(raw)
        except Exception:
            return None

    # ── Leverage ──────────────────────────────────────────────────────────────

    def get_leverage_dd_threshold(self) -> float:
        """Порог просадки для режима плеча full (0.05–0.5).

        Не hard drawdown halt: тот читает Config.MAX_DRAWDOWN_PERCENT напрямую.
        """
        default = float(getattr(Config, "LEVERAGE_DD_THRESHOLD", 0.15))
        val = self._get(_KEY_LEVERAGE_DD)
        try:
            return max(0.05, min(float(val), 0.5)) if val else default
        except (ValueError, TypeError):
            return default

    def set_leverage_dd_threshold(self, pct: float) -> bool:
        if not (0.05 <= pct <= 0.5):
            return False
        self._set(_KEY_LEVERAGE_DD, str(pct))
        logger.info("Runtime: leverage_dd_threshold → %.2f", pct)
        return True

    def get_leverage_mode(self) -> str:
        """Режим плеча: fixed | volatility | full."""
        val = self._get(_KEY_LEVERAGE_MODE)
        if val and val in _LEVERAGE_MODES:
            return val
        return getattr(Config, "LEVERAGE_MODE", "volatility")

    def set_leverage_mode(self, mode: str) -> bool:
        if mode not in _LEVERAGE_MODES:
            return False
        self._set(_KEY_LEVERAGE_MODE, mode)
        logger.info("Runtime: leverage_mode → %s", mode)
        return True

    def get_leverage_target_risk(self) -> float:
        """Целевой риск на ATR-движение для режимов volatility/full."""
        val = self._get(_KEY_LEVERAGE_TARGET_RISK)
        try:
            return (
                max(0.001, min(float(val), 0.1))
                if val
                else float(getattr(Config, "LEVERAGE_TARGET_RISK", 0.01))
            )
        except (ValueError, TypeError):
            return 0.01

    def set_leverage_target_risk(self, risk: float) -> bool:
        if not (0.001 <= risk <= 0.1):
            return False
        self._set(_KEY_LEVERAGE_TARGET_RISK, str(risk))
        logger.info("Runtime: leverage_target_risk → %.3f", risk)
        return True

    # ── Trading hours ─────────────────────────────────────────────────────────

    def get_trading_hours(self) -> str:
        """
        Возвращает временной диапазон торговли в формате "HH-HH" UTC.
        Пустая строка = 24 часа (без ограничений).
        """
        val = self._get(_KEY_TRADING_HOURS)
        if val is None:
            return Config.TRADING_HOURS
        return val

    def set_trading_hours(self, hours: str) -> bool:
        """
        Устанавливает временной диапазон. Формат: "9-17" или "" (сброс).
        Возвращает False при неверном формате.
        """
        if hours in ("", "0"):
            self._set(_KEY_TRADING_HOURS, "")
            logger.info("Runtime: trading_hours сброшен (24/7)")
            return True
        try:
            start, end = map(int, hours.split("-"))
            if not (0 <= start <= 23 and 0 <= end <= 23 and start != end):
                return False
        except (ValueError, AttributeError):
            return False
        self._set(_KEY_TRADING_HOURS, hours)
        logger.info("Runtime: trading_hours → %s UTC", hours)
        return True

    def is_trading_time(self) -> bool:
        """True если сейчас разрешено торговать по временному фильтру (UTC)."""
        import datetime as dt

        hours = self.get_trading_hours()
        if not hours:
            return True
        try:
            start_h, end_h = map(int, hours.split("-"))
            hour = dt.datetime.utcnow().hour
            if start_h < end_h:
                return start_h <= hour < end_h
            # Перенос через полночь: "22-6"
            return hour >= start_h or hour < end_h
        except (ValueError, AttributeError):
            return True

    # ── Strategies ────────────────────────────────────────────────────────────

    def get_disabled_strategies(self) -> Set[str]:
        """Возвращает имена отключённых стратегий. Пустой set = все включены."""
        return self._smembers(_KEY_DISABLED_STRATS)

    def disable_strategy(self, name: str) -> None:
        try:
            self._r.redis_client.sadd(_KEY_DISABLED_STRATS, name)
            logger.info("Runtime: стратегия отключена — %s", name)
        except Exception as e:
            logger.warning("RuntimeConfig sadd failed: %s", e)

    def enable_strategy(self, name: str) -> None:
        try:
            self._r.redis_client.srem(_KEY_DISABLED_STRATS, name)
            logger.info("Runtime: стратегия включена — %s", name)
        except Exception as e:
            logger.warning("RuntimeConfig srem failed: %s", e)

    def ensure_experimental_disabled(self, names: Iterable[str]) -> List[str]:
        """Новые экспериментальные стратегии при первом появлении выключаются.

        Каждое имя обрабатывается один раз: дальше решает пользователь (включил —
        остаётся включённой). Возвращает, какие были выключены сейчас.
        """
        try:
            seen = self._smembers(_KEY_EXPERIMENTAL_SEEN)
            fresh = [n for n in names if n not in seen]
            for name in fresh:
                self.disable_strategy(name)
                self._r.redis_client.sadd(_KEY_EXPERIMENTAL_SEEN, name)
            return fresh
        except Exception as e:
            logger.warning("RuntimeConfig experimental seeding failed: %s", e)
            return []

    def reset_strategies(self, include_experimental: bool = False) -> None:
        """Включить основные стратегии (очистить список отключённых).

        Экспериментальные (без доказанного преимущества) остаются выключенными,
        если явно не попросили include_experimental: одно нажатие не должно
        включать десятки стратегий, которые мы держим выключенными нарочно.
        """
        try:
            self._r.redis_client.delete(_KEY_DISABLED_STRATS)
            logger.info("Runtime: основные стратегии включены")
        except Exception as e:
            logger.warning("RuntimeConfig delete failed: %s", e)
            return
        if include_experimental:
            return
        from src.strategies import STRATEGY_REGISTRY  # noqa: PLC0415

        for name, cls in STRATEGY_REGISTRY.items():
            if cls.experimental:
                self.disable_strategy(name)

    def toggle_strategy(self, name: str) -> bool:
        """Переключает стратегию. Возвращает True если теперь включена."""
        if name in self.get_disabled_strategies():
            self.enable_strategy(name)
            return True
        self.disable_strategy(name)
        return False

    # ── SAC prompt ────────────────────────────────────────────────────────────

    def _confirm(self) -> "tuple[int, str]":
        return confirm_policy(
            self._get(_KEY_CONFIRM_TIMEOUT) or Config.TELEGRAM_CONFIRM_TIMEOUT,
            self._get(_KEY_CONFIRM_ON_TIMEOUT)
            or getattr(Config, "CONFIRM_ON_TIMEOUT", "execute"),
        )

    def get_confirm_timeout(self) -> int:
        """Секунд на решение по сделке: от 10 до 120."""
        return self._confirm()[0]

    def set_confirm_timeout(self, seconds: int) -> bool:
        if not CONFIRM_TIMEOUT_MIN <= seconds <= CONFIRM_TIMEOUT_MAX:
            return False
        self._set(_KEY_CONFIRM_TIMEOUT, str(seconds))
        logger.info("Runtime: confirm_timeout → %dс", seconds)
        return True

    def get_confirm_on_timeout(self) -> str:
        """Если не ответили вовремя: execute — открыть сделку, skip — пропустить."""
        return self._confirm()[1]

    def set_confirm_on_timeout(self, value: str) -> bool:
        if value not in ("execute", "skip"):
            return False
        if self._get(_KEY_CONFIRM_TIMEOUT) == "0":  # старый ручной режим
            self._set(_KEY_CONFIRM_TIMEOUT, str(CONFIRM_TIMEOUT_MAX))
        self._set(_KEY_CONFIRM_ON_TIMEOUT, value)
        logger.info("Runtime: confirm_on_timeout → %s", value)
        return True

    def is_sac_prompted(self, profile: str = "") -> bool:
        """True если пользователь уже получал запрос об обучении SAC для профиля."""
        key = f"{_KEY_SAC_PROMPTED}:{profile}" if profile else _KEY_SAC_PROMPTED
        return self._get(key) == "1"

    def set_sac_prompted(self, profile: str = "") -> None:
        """Отмечаем что запрос об обучении SAC был отправлен для профиля."""
        key = f"{_KEY_SAC_PROMPTED}:{profile}" if profile else _KEY_SAC_PROMPTED
        self._set(key, "1")

    def get_sac_model_path(self) -> str:
        """Путь к SAC модели для текущего профиля. Fallback: Config.SAC_MODEL_PATH."""
        profile = self.get_market_profile()
        if profile and profile in _MARKET_PROFILES:
            return str(_MARKET_PROFILES[profile]["model_path"])
        return Config.SAC_MODEL_PATH

    def get_sac_model_path_for_profile(self, profile: str) -> str:
        """Путь к SAC модели для конкретного профиля."""
        if profile and profile in _MARKET_PROFILES:
            return str(_MARKET_PROFILES[profile]["model_path"])
        return Config.SAC_MODEL_PATH

    # ── First start & tune reminder ───────────────────────────────────────────

    def ensure_first_start_date(self) -> str:
        """Записывает дату первого запуска если ещё не записана. Возвращает дату."""
        import datetime as dt

        existing = self._get(_KEY_FIRST_START_DATE)
        if existing:
            return existing
        today = dt.date.today().isoformat()
        self._set(_KEY_FIRST_START_DATE, today)
        return today

    def days_since_first_start(self) -> int:
        """Количество дней с первого запуска бота."""
        import datetime as dt

        raw = self._get(_KEY_FIRST_START_DATE)
        if not raw:
            return 0
        try:
            first = dt.date.fromisoformat(raw)
            return (dt.date.today() - first).days
        except ValueError:
            return 0

    def is_tune_reminded(self) -> bool:
        """True если напоминание о тюнинге уже отправлялось."""
        return self._get(_KEY_TUNE_REMINDED) == "1"

    def set_tune_reminded(self) -> None:
        """Отмечаем что напоминание о тюнинге было отправлено."""
        self._set(_KEY_TUNE_REMINDED, "1")

    # ── Chronos filter ────────────────────────────────────────────────────────

    def get_chronos_enabled(self) -> bool:
        """True → тройное подтверждение SAC+LLM+Chronos в hybrid режиме."""
        val = self._get(_KEY_CHRONOS)
        return val == "1" if val is not None else False

    def set_chronos_enabled(self, enabled: bool) -> None:
        self._set(_KEY_CHRONOS, "1" if enabled else "0")
        logger.info("Chronos filter %s", "ON" if enabled else "OFF")

    # ── Market Profile ────────────────────────────────────────────────────────

    def get_market_profile(self) -> str:
        """Текущий профиль: 'bluechip' | 'altcoin' | '' (не задан)."""
        return self._get(_KEY_MARKET_PROFILE) or ""

    def ensure_default_market_profile(self, default: str = "bluechip") -> str:
        """Применяет профиль по умолчанию, если он ещё не задан (первый запуск
        или после очистки Redis). Возвращает действующий профиль."""
        current = self.get_market_profile()
        if current:
            from src.strategies import set_active_profile  # noqa: PLC0415

            set_active_profile(current)  # порог volume_spike после рестарта
            return current
        self.apply_market_profile(default)
        return default

    def get_timeframe(self) -> str:
        """Таймфрейм свечей. Fallback: Config.TIMEFRAME."""
        return self._get(_KEY_TIMEFRAME) or Config.TIMEFRAME

    def get_min_volume_usdt(self) -> float:
        """Минимальный 24h объём USDT. Fallback: Config.MIN_VOLUME_USDT."""
        val = self._get(_KEY_MIN_VOLUME)
        try:
            return float(val) if val is not None else Config.MIN_VOLUME_USDT
        except ValueError:
            return Config.MIN_VOLUME_USDT

    def get_max_volume_usdt(self) -> float:
        """Максимальный 24h объём USDT (0 = без ограничения). Fallback: Config."""
        val = self._get(_KEY_MAX_VOLUME)
        try:
            return float(val) if val is not None else Config.MAX_VOLUME_USDT
        except ValueError:
            return Config.MAX_VOLUME_USDT

    def apply_market_profile(self, name: str) -> bool:
        """Применяет профиль рынка. Возвращает False если имя неизвестно."""
        import os as _os

        profile = _MARKET_PROFILES.get(name)
        if not profile:
            return False
        self._set(_KEY_MARKET_PROFILE, name)
        self._set(_KEY_TIMEFRAME, str(profile.get("timeframe", "")))
        self._set(_KEY_MIN_VOLUME, str(profile.get("min_volume_usdt", 0)))
        self._set(_KEY_MAX_VOLUME, str(profile.get("max_volume_usdt", 0)))
        self.set_scan_top_n(int(str(profile.get("scan_top_n", 20))))
        self.set_train_top_n(int(str(profile.get("train_top_n", 20))))
        self.set_risk_per_trade(float(str(profile.get("risk_per_trade", 0.02))))
        self.set_mode(str(profile.get("mode", "ai")))
        self._set("bot:sl_percent", str(profile.get("sl_percent", 0.05)))
        self._set("bot:tp_multiplier", str(profile.get("tp_multiplier", 2.0)))
        self._set("bot:trading_interval", str(profile.get("trading_interval", 60)))
        # Направляем сохранение сделок в профильный файл
        _exp = (
            "data/experiences_altcoin.jsonl"
            if name == "altcoin"
            else (
                "data/experiences_meme.jsonl"
                if name == "meme"
                else "data/experiences.jsonl"
            )
        )
        _os.environ["EXPERIENCES_PATH"] = _exp
        _os.environ["SAC_PROFILE"] = name

        # Переключаем профиль-специфичные стратегии. volume_spike одна для всех
        # профилей: порог берётся из SAC_PROFILE внутри стратегии. Старые
        # профильные имена из списка отключённых убираем — их больше нет.
        from src.strategies import set_active_profile  # noqa: PLC0415

        set_active_profile(name)
        _MEME_ONLY = {"momentum_burst", "pump_exit", "sentiment"}
        for legacy in (
            "volume_spike_bluechip",
            "volume_spike_altcoin",
            "volume_spike_meme",
        ):
            self.enable_strategy(legacy)
        self.enable_strategy("volume_spike")
        for strat_name in _MEME_ONLY:
            if name == "meme":
                self.enable_strategy(strat_name)
            else:
                self.disable_strategy(strat_name)

        logger.info("Runtime: применён профиль рынка '%s'", name)
        return True

    def get_market_profiles_info(self) -> dict:
        return {k: v["label"] for k, v in _MARKET_PROFILES.items()}

    def profile_ids(self) -> list:
        return list(_MARKET_PROFILES)

    def get_market_profile_config(self, name: str) -> dict:
        """Возвращает полный конфиг профиля (train_top_n, timeframe и др.)."""
        return dict(_MARKET_PROFILES.get(name, {}))

    # ── Текущий резолвнутый список монет (для /symbols и дашборда) ────────────

    def set_current_symbols(self, profile: str, symbols: List[str]) -> None:
        """Сохраняет список монет, реально отобранных сканом в этом цикле."""
        payload = {
            "profile": profile,
            "symbols": symbols,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._set(_KEY_CURRENT_SYMBOLS, json.dumps(payload, ensure_ascii=False))

    def get_current_symbols(self) -> Optional[dict]:
        """Возвращает {"profile", "symbols", "updated_at"} или None, если данных нет."""
        raw = self._get(_KEY_CURRENT_SYMBOLS)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return None

    # ── Paper / Live switch ───────────────────────────────────────────────────

    def set_awaiting_mode_pin(self, ttl: int = 120) -> None:
        try:
            self._r.redis_client.setex(_KEY_AWAITING_MODE_PIN, ttl, "1")
        except Exception:
            pass

    def is_awaiting_mode_pin(self) -> bool:
        return self._get(_KEY_AWAITING_MODE_PIN) == "1"

    def clear_awaiting_mode_pin(self) -> None:
        try:
            self._r.redis_client.delete(_KEY_AWAITING_MODE_PIN)
        except Exception:
            pass

    def get_paper_trading_override(self) -> "bool | None":
        val = self._get(_KEY_PAPER_TRADING)
        if val == "1":
            return True
        if val == "0":
            return False
        return None

    def set_paper_trading_override(self, paper: bool) -> None:
        self._set(_KEY_PAPER_TRADING, "1" if paper else "0")

    def apply_paper_trading(self, paper: bool) -> None:
        """Переключает Paper/Live в процессе и запоминает выбор в Redis.

        Общий путь для Telegram-кнопки и команды из приложения; после него
        вызывающий синхронизирует позиции (TradingBot._handle_mode_switch).
        """
        import os as _os

        Config.PAPER_TRADING = paper
        _os.environ["PAPER_TRADING"] = "true" if paper else "false"
        self.set_paper_trading_override(paper)

    # ── Season switch mode ────────────────────────────────────────────────────

    def get_season_switch_mode(self) -> str:
        import os as _os

        val = self._get(_KEY_SEASON_MODE)
        if val in ("alert", "auto"):
            return val
        return _os.getenv("SEASON_SWITCH_MODE", "alert").lower()

    def set_season_switch_mode(self, mode: str) -> bool:
        if mode not in ("alert", "auto"):
            return False
        self._set(_KEY_SEASON_MODE, mode)
        return True

    def set_season_index(self, signal: "str | None", index: dict) -> None:
        """Сохраняет последний результат CoinGecko (TTL 6 часов)."""
        import json as _json
        import time as _time

        data = {
            "signal": signal,
            "altcoin_index": index.get("altcoin_index"),
            "btc_dominance": index.get("btc_dominance"),
            "btc_30d": index.get("btc_30d"),
            "checked_at": _time.time(),
        }
        try:
            self._r.redis_client.setex(_KEY_SEASON_INDEX, 6 * 3600, _json.dumps(data))
        except Exception:
            pass

    def get_season_index(self) -> "dict | None":
        """Читает последний результат CoinGecko из Redis."""
        import json as _json

        raw = self._get(_KEY_SEASON_INDEX)
        if not raw:
            return None
        try:
            return _json.loads(raw)
        except Exception:
            return None

    # ── Фоновые задачи: общий замок для Telegram и приложения ─────────────────

    def try_start_job(self, group: str, source: str, ttl: int) -> bool:
        """Занимает группу задач (sac, backtest, …); False, если уже занята.

        Замок общий для Telegram и приложения, поэтому один и тот же бэктест или
        обучение не запустятся дважды. TTL — страховка на случай падения бота.
        """
        try:
            return bool(
                self._r.redis_client.set(_KEY_JOB + group, source, nx=True, ex=ttl)
            )
        except Exception as e:
            logger.warning("RuntimeConfig job lock failed: %s", e)
            return True  # без Redis замок невозможен — не блокируем запуск

    def end_job(self, group: str) -> None:
        try:
            self._r.redis_client.delete(_KEY_JOB + group)
        except Exception as e:
            logger.warning("RuntimeConfig job unlock failed: %s", e)

    def running_jobs(self, groups: Iterable[str]) -> Dict[str, str]:
        """{группа: источник запуска} для занятых групп."""
        out: Dict[str, str] = {}
        for g in groups:
            val = self._get(_KEY_JOB + g)
            if val:
                out[g] = val
        return out

    def get_sac_backup_path(self) -> str:
        return self._get(_KEY_SAC_BACKUP) or ""

    def set_sac_backup_path(self, path: str, profile: str = "") -> None:
        self._set(_KEY_SAC_BACKUP, path)
        self._set(_KEY_SAC_BACKUP_PROFILE, profile)

    def get_sac_backup_profile(self) -> str:
        """Профиль, чью модель сохранила последняя резервная копия ('' — неизвестно)."""
        return self._get(_KEY_SAC_BACKUP_PROFILE) or ""

    def get_training_profiles(self) -> list:
        """Профили рынка для вкладки «Обучение»: таймфрейм, монеты, файл модели."""
        import datetime as _dt
        import os as _os

        active = self.get_market_profile()
        rows = []
        for pid, cfg in _MARKET_PROFILES.items():
            path = str(cfg["model_path"])
            exists = _os.path.exists(path)
            rows.append(
                {
                    "id": pid,
                    "label": str(cfg["label"]).split(" ", 1)[-1],
                    "timeframe": cfg["timeframe"],
                    "train_top_n": cfg["train_top_n"],
                    "model_path": path,
                    "model_exists": exists,
                    "trained_at": (
                        _dt.datetime.fromtimestamp(
                            _os.path.getmtime(path), _dt.timezone.utc
                        ).isoformat()
                        if exists
                        else None
                    ),
                    "active": pid == active,
                }
            )
        return rows

    # ── Train progress ────────────────────────────────────────────────────────

    def set_train_progress(self, data: dict) -> None:
        import json as _json

        try:
            self._set(_KEY_TRAIN_PROGRESS, _json.dumps(data))
        except Exception:
            pass

    def get_train_progress(self) -> "dict | None":
        import json as _json

        val = self._get(_KEY_TRAIN_PROGRESS)
        if not val:
            return None
        try:
            return _json.loads(val)
        except Exception:
            return None

    def clear_train_progress(self) -> None:
        try:
            self._r.redis_client.delete(_KEY_TRAIN_PROGRESS)
        except Exception:
            pass

    def set_tune_progress(self, data: dict) -> None:
        import json as _json

        try:
            self._set(_KEY_TUNE_PROGRESS, _json.dumps(data))
        except Exception:
            pass

    def get_tune_progress(self) -> "dict | None":
        import json as _json

        val = self._get(_KEY_TUNE_PROGRESS)
        if not val:
            return None
        try:
            return _json.loads(val)
        except Exception:
            return None

    def clear_tune_progress(self) -> None:
        try:
            self._r.redis_client.delete(_KEY_TUNE_PROGRESS)
        except Exception:
            pass

    # ── Backtest progress ─────────────────────────────────────────────────────

    def set_backtest_progress(self, data: dict) -> None:
        import json as _json

        try:
            self._set(_KEY_BACKTEST_PROGRESS, _json.dumps(data))
        except Exception:
            pass

    def get_backtest_progress(self) -> "dict | None":
        import json as _json

        val = self._get(_KEY_BACKTEST_PROGRESS)
        if not val:
            return None
        try:
            return _json.loads(val)
        except Exception:
            return None

    def clear_backtest_progress(self) -> None:
        try:
            self._r.redis_client.delete(_KEY_BACKTEST_PROGRESS)
        except Exception:
            pass

    # ── Fear & Greed ──────────────────────────────────────────────────────────

    def set_fear_greed(self, value: int, label: str) -> None:
        import json as _json

        try:
            self._r.redis_client.setex(
                _KEY_FEAR_GREED,
                6 * 3600,
                _json.dumps({"value": value, "label": label}),
            )
        except Exception:
            pass

    def get_fear_greed(self) -> "dict | None":
        import json as _json

        raw = self._get(_KEY_FEAR_GREED)
        if not raw:
            return None
        try:
            return _json.loads(raw)
        except Exception:
            return None

    # ── Pump.fun launch-rate baseline (для meme-season) ─────────────────────────
    # Абсолютный порог "нормального" темпа лаунчей на pump.fun угадать нельзя —
    # база сама по себе огромна и меняется со временем. Вместо этого копим
    # историю замеров и сравниваем текущий темп с собственным скользящим
    # медианным бейзлайном (см. SeasonDetector.classify).

    _LAUNCH_RATE_MAX_SAMPLES = 42  # ~7 дней при проверке раз в 4 часа

    def record_launch_rate_sample(self, rate: float) -> None:
        """Добавляет замер темпа лаунчей в скользящую историю (капается)."""
        import json as _json

        raw = self._get(_KEY_LAUNCH_RATE_HISTORY)
        try:
            history = _json.loads(raw) if raw else []
        except Exception:
            history = []
        history.append(round(rate, 2))
        history = history[-self._LAUNCH_RATE_MAX_SAMPLES :]
        self._set(_KEY_LAUNCH_RATE_HISTORY, _json.dumps(history))

    def get_launch_rate_baseline(self, min_samples: int = 5) -> "float | None":
        """Медиана накопленной истории. None если данных ещё недостаточно."""
        import json as _json
        import statistics as _statistics

        raw = self._get(_KEY_LAUNCH_RATE_HISTORY)
        if not raw:
            return None
        try:
            history = _json.loads(raw)
        except Exception:
            return None
        if len(history) < min_samples:
            return None
        return float(_statistics.median(history))

    # ── Startup ───────────────────────────────────────────────────────────────

    _ALL_KEYS = (
        _KEY_MODE,
        _KEY_SCAN_TOP_N,
        _KEY_PAUSED,
        _KEY_AUTO_EXECUTE,
        _KEY_LOCAL_FALLBACK,
        _KEY_PAPER_GUARDS_OFF,
        _KEY_FORCED,
        _KEY_EXCLUDED,
        _KEY_DISABLED_STRATS,
        _KEY_TRADING_HOURS,
        _KEY_MAX_POSITIONS,
        _KEY_RISK_PER_TRADE,
        _KEY_DRAWDOWN_SCALE,
        _KEY_TRAIN_TOP_N,
        _KEY_AI_PROVIDER,
        _KEY_MARKET_TYPE,
        _KEY_LEVERAGE_MODE,
        _KEY_LEVERAGE_TARGET_RISK,
        _KEY_LEVERAGE_DD,
        _KEY_DRAWDOWN_CONFIRM,
        _KEY_MARKET_PROFILE,
        _KEY_TIMEFRAME,
        _KEY_MIN_VOLUME,
        _KEY_MAX_VOLUME,
    )

    def reset_to_defaults(self) -> None:
        """
        Сбрасывает все Runtime-настройки, возвращая управление .env.
        Вызывается при старте бота — каждый запуск начинается чисто.
        """
        try:
            self._r.redis_client.delete(*self._ALL_KEYS, _KEY_EXPERIMENTAL_SEEN)
            logger.info("RuntimeConfig: сброс к .env-дефолтам выполнен")
        except Exception as e:
            logger.warning("RuntimeConfig reset failed: %s", e)
            return
        # Экспериментальные стратегии после сброса снова выключены по умолчанию.
        from src.strategies import STRATEGY_REGISTRY  # noqa: PLC0415

        self.ensure_experimental_disabled(
            n for n, c in STRATEGY_REGISTRY.items() if c.experimental
        )

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "mode": self.get_mode(),
            "scan_top_n": self.get_scan_top_n(),
            "paused": self.is_paused(),
            "auto_execute": self.get_auto_execute(),
            "forced_symbols": sorted(self.get_forced_symbols()),
            "excluded_symbols": sorted(self.get_excluded_symbols()),
        }

    # ── Alert settings ────────────────────────────────────────────────────────

    _ALERT_DEFAULTS: dict = {
        "atr_spike": {"label": "⚡ ATR Spike", "cooldown": 10},
        "drawdown": {"label": "📉 Drawdown", "cooldown": 5},
        "daily_limit": {"label": "🔒 Daily limit", "cooldown": 60},
        "bot_stalled": {"label": "🔇 BotStalled", "cooldown": 20},
        "new_signal": {"label": "📊 Новые сигналы", "cooldown": 0},
        "position_open": {"label": "📈 Открытие позиции", "cooldown": 0},
        "position_close": {"label": "💰 Закрытие позиции", "cooldown": 0},
    }

    def get_alert_enabled(self, name: str) -> bool:
        val = self._get(f"bot:alert:enabled:{name}")
        return val != "0"  # включено по умолчанию

    def set_alert_enabled(self, name: str, enabled: bool) -> None:
        self._set(f"bot:alert:enabled:{name}", "1" if enabled else "0")

    def get_alert_cooldown_mins(self, name: str) -> int:
        val = self._get(f"bot:alert:cooldown:{name}")
        if val is not None:
            try:
                return max(0, int(val))
            except ValueError:
                pass
        return self._ALERT_DEFAULTS.get(name, {}).get("cooldown", 5)

    def set_alert_cooldown_mins(self, name: str, minutes: int) -> None:
        self._set(f"bot:alert:cooldown:{name}", str(max(0, minutes)))

    def set_awaiting_alert_cooldown(self, alert_name: str, ttl: int = 60) -> None:
        try:
            self._r.redis_client.setex(_KEY_AWAITING_ALERT_COOLDOWN, ttl, alert_name)
        except Exception:
            pass

    def get_awaiting_alert_cooldown(self) -> str | None:
        val = self._get(_KEY_AWAITING_ALERT_COOLDOWN)
        return val if val else None

    def clear_awaiting_alert_cooldown(self) -> None:
        try:
            self._r.redis_client.delete(_KEY_AWAITING_ALERT_COOLDOWN)
        except Exception:
            pass

    def format_full_status(self, balance: float, paper: bool, all_strats: list) -> str:
        """Форматирует полный статус всех параметров для Telegram."""
        mode = self.get_mode()
        scan_n = self.get_scan_top_n()
        auto = self.get_auto_execute()
        paused = self.is_paused()
        hours = self.get_trading_hours() or "24/7"
        risk = self.get_risk_summary()
        disabled = self.get_disabled_strategies()
        enabled_count = len(all_strats) - len(disabled)

        mode_icon = "paper" if paper else "live"
        pause_line = "⏸ *НА ПАУЗЕ*\n" if paused else ""
        auto_icon = "✅" if auto else "❌"
        dd_icon = "✅" if risk["drawdown_scale_enabled"] else "❌"

        return (
            f"🤖 *BitbotBY запущен* \\[{mode_icon}\\]\n"
            f"{pause_line}\n"
            f"*Режим:* `{mode}` \\| *Символов:* `{scan_n}`\n"
            f"*Баланс:* `${balance:,.2f}`\n\n"
            f"*⚙️ Параметры:*\n"
            f"  Авто\\-сделки: {auto_icon}\n"
            f"  Макс позиций: `{risk['max_positions']}`\n"
            f"  Риск/сделку: `{risk['risk_per_trade'] * 100:.1f}%`\n"
            f"  Защита просадки: {dd_icon}\n"
            f"  Часы торговли: `{hours}` UTC\n"
            f"  Стратегий вкл: `{enabled_count}/{len(all_strats)}`\n\n"
            f"_Нажми кнопку для изменения любого параметра:_"
        )
