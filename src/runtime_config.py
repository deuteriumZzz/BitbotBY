"""
Runtime-конфигурация поверх Redis.

Позволяет менять настройки бота без перезапуска контейнера.
Все значения хранятся в Redis и переопределяют .env при наличии.
"""

from __future__ import annotations

import logging
from typing import Any, Set

from config import Config

logger = logging.getLogger(__name__)

_KEY_MODE = "bot:mode"
_KEY_SCAN_TOP_N = "bot:scan_top_n"
_KEY_PAUSED = "bot:paused"
_KEY_AUTO_EXECUTE = "bot:auto_execute"
_KEY_FORCED = "bot:forced_symbols"
_KEY_EXCLUDED = "bot:excluded_symbols"
_KEY_DISABLED_STRATS = "bot:disabled_strategies"
_KEY_TRADING_HOURS = "bot:trading_hours"
_KEY_MAX_POSITIONS = "bot:max_positions"
_KEY_RISK_PER_TRADE = "bot:risk_per_trade"
_KEY_DRAWDOWN_SCALE = "bot:drawdown_scale_enabled"
_KEY_TRAIN_TOP_N = "bot:train_top_n"
_KEY_AI_PROVIDER = "bot:ai_provider"
_KEY_LEVERAGE_MODE = "bot:leverage_mode"
_KEY_LEVERAGE_TARGET_RISK = "bot:leverage_target_risk"
_KEY_MAX_DRAWDOWN = "bot:max_drawdown_percent"
_KEY_SAC_PROMPTED = "bot:sac_prompted"
_KEY_CONFIRM_TIMEOUT = "bot:confirm_timeout"
_KEY_FIRST_START_DATE = "bot:first_start_date"
_KEY_TUNE_REMINDED = "bot:tune_reminded"
_KEY_CHRONOS = "bot:chronos_filter"
_KEY_MARKET_PROFILE = "bot:market_profile"
_KEY_TIMEFRAME = "bot:timeframe"
_KEY_MIN_VOLUME = "bot:min_volume_usdt"
_KEY_MAX_VOLUME = "bot:max_volume_usdt"
_KEY_AWAITING_MODE_PIN = "bot:awaiting_mode_pin"
_KEY_PAPER_TRADING = "bot:paper_trading_override"
_KEY_SAC_BACKUP = "bot:sac_model_backup"
_KEY_TRAIN_PROGRESS = "bot:train_progress"
_KEY_BACKTEST_PROGRESS = "bot:backtest_progress"
_KEY_LAST_AI_PROVIDER = "bot:last_ai_provider"
_KEY_SEASON_MODE = "bot:season_switch_mode"
_KEY_SEASON_INDEX = "bot:season_index"
_KEY_FEAR_GREED = "bot:fear_greed"

_AI_PROVIDERS = frozenset({"auto", "anthropic", "openai", "deepseek", "groq"})
_LEVERAGE_MODES = frozenset({"fixed", "volatility", "full"})

_RISK_PRESETS = {
    "conservative": {
        "max_positions": 2,
        "risk_per_trade": 0.01,
        "drawdown_scale": True,
    },
    "moderate": {
        "max_positions": 3,
        "risk_per_trade": 0.02,
        "drawdown_scale": True,
    },
    "aggressive": {
        "max_positions": 5,
        "risk_per_trade": 0.04,
        "drawdown_scale": False,
    },
}

_MARKET_PROFILES = {
    "bluechip": {
        "label": "🔵 Блючипы",
        "scan_top_n": 20,
        "train_top_n": 20,
        "min_volume_usdt": 3_000_000_000,
        "max_volume_usdt": 0,
        "risk_per_trade": 0.02,
        "timeframe": "15m",
        "mode": "hybrid",
        "model_path": "models/sac_model.zip",
    },
    "altcoin": {
        "label": "🟡 Альткоины",
        "scan_top_n": 50,
        "train_top_n": 50,
        "min_volume_usdt": 500_000,
        "max_volume_usdt": 2_000_000_000,
        "risk_per_trade": 0.01,
        "timeframe": "5m",
        "mode": "ai",
        "model_path": "models/sac_model_altcoin.zip",
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
        preset = _RISK_PRESETS.get(name)
        if not preset:
            return False
        self.set_max_positions(int(preset["max_positions"]))
        self.set_risk_per_trade(float(preset["risk_per_trade"]))
        self.set_drawdown_scale_enabled(bool(preset["drawdown_scale"]))
        logger.info("Runtime: применён риск-профиль '%s'", name)
        return True

    def get_risk_summary(self) -> dict:
        return {
            "max_positions": self.get_max_positions(),
            "risk_per_trade": self.get_risk_per_trade(),
            "drawdown_scale_enabled": self.get_drawdown_scale_enabled(),
        }

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

    def get_last_ai_provider(self) -> str:
        """Провайдер, который реально ответил последним."""
        return self._get(_KEY_LAST_AI_PROVIDER) or ""

    def set_last_ai_provider(self, provider: str) -> None:
        self._set(_KEY_LAST_AI_PROVIDER, provider)

    # ── Leverage ──────────────────────────────────────────────────────────────

    def get_max_drawdown_percent(self) -> float:
        """Порог просадки для режима плеча full (0.05–0.5)."""
        val = self._get(_KEY_MAX_DRAWDOWN)
        try:
            return (
                max(0.05, min(float(val), 0.5))
                if val
                else float(getattr(Config, "MAX_DRAWDOWN_PERCENT", 0.15))
            )
        except (ValueError, TypeError):
            return 0.15

    def set_max_drawdown_percent(self, pct: float) -> bool:
        if not (0.05 <= pct <= 0.5):
            return False
        self._set(_KEY_MAX_DRAWDOWN, str(pct))
        logger.info("Runtime: max_drawdown_percent → %.2f", pct)
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

    def reset_strategies(self) -> None:
        """Включить все стратегии (очистить список отключённых)."""
        try:
            self._r.redis_client.delete(_KEY_DISABLED_STRATS)
            logger.info("Runtime: все стратегии включены")
        except Exception as e:
            logger.warning("RuntimeConfig delete failed: %s", e)

    def toggle_strategy(self, name: str) -> bool:
        """Переключает стратегию. Возвращает True если теперь включена."""
        if name in self.get_disabled_strategies():
            self.enable_strategy(name)
            return True
        self.disable_strategy(name)
        return False

    # ── SAC prompt ────────────────────────────────────────────────────────────

    def get_confirm_timeout(self) -> int:
        """Таймаут подтверждения сделки в секундах.
        0 = ручной режим (авто-исполнения нет, сделка ждёт Trade).
        10–300 = авто-исполнение через N секунд. Default 60."""
        val = self._get(_KEY_CONFIRM_TIMEOUT)
        try:
            parsed = int(val) if val else 60
            return 0 if parsed == 0 else max(10, min(parsed, 300))
        except (ValueError, TypeError):
            return 60

    def set_confirm_timeout(self, seconds: int) -> bool:
        if seconds != 0 and not (10 <= seconds <= 300):
            return False
        self._set(_KEY_CONFIRM_TIMEOUT, str(seconds))
        label = "ручной" if seconds == 0 else f"{seconds}с"
        logger.info("Runtime: confirm_timeout → %s", label)
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
        # Направляем сохранение сделок в профильный файл
        _os.environ["EXPERIENCES_PATH"] = f"data/experiences_{name}.jsonl"
        _os.environ["SAC_PROFILE"] = name
        logger.info("Runtime: применён профиль рынка '%s'", name)
        return True

    def get_market_profiles_info(self) -> dict:
        return {k: v["label"] for k, v in _MARKET_PROFILES.items()}

    def get_market_profile_config(self, name: str) -> dict:
        """Возвращает полный конфиг профиля (train_top_n, timeframe и др.)."""
        return dict(_MARKET_PROFILES.get(name, {}))

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

    def get_sac_backup_path(self) -> str:
        return self._get(_KEY_SAC_BACKUP) or ""

    def set_sac_backup_path(self, path: str) -> None:
        self._set(_KEY_SAC_BACKUP, path)

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

    # ── Startup ───────────────────────────────────────────────────────────────

    _ALL_KEYS = (
        _KEY_MODE,
        _KEY_SCAN_TOP_N,
        _KEY_PAUSED,
        _KEY_AUTO_EXECUTE,
        _KEY_FORCED,
        _KEY_EXCLUDED,
        _KEY_DISABLED_STRATS,
        _KEY_TRADING_HOURS,
        _KEY_MAX_POSITIONS,
        _KEY_RISK_PER_TRADE,
        _KEY_DRAWDOWN_SCALE,
        _KEY_TRAIN_TOP_N,
        _KEY_AI_PROVIDER,
        _KEY_LEVERAGE_MODE,
        _KEY_LEVERAGE_TARGET_RISK,
        _KEY_MAX_DRAWDOWN,
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
            self._r.redis_client.delete(*self._ALL_KEYS)
            logger.info("RuntimeConfig: сброс к .env-дефолтам выполнен")
        except Exception as e:
            logger.warning("RuntimeConfig reset failed: %s", e)

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
