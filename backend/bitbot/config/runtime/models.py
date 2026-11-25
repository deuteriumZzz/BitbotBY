"""AI и модели: провайдер и цепочка AI, обучение и подсказки SAC, Chronos, прогресс задач.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from bitbot.config import ai_providers, keys
from bitbot.config.runtime.presets import (
    _AI_PROVIDERS,
    _MARKET_PROFILES,
    CONFIRM_TIMEOUT_MAX,
    CONFIRM_TIMEOUT_MIN,
    confirm_policy,
)
from bitbot.config.runtime.risk import RiskSettings
from bitbot.config.settings import Config

logger = logging.getLogger(__name__)


class ModelSettings(RiskSettings):
    # ── SAC Training ──────────────────────────────────────────────────────────
    def get_train_top_n(self) -> int:
        """Количество символов для обучения SAC (default: TRAIN_TOP_N или 20)."""
        val = self._get(keys.TRAIN_TOP_N)
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
        self._set(keys.TRAIN_TOP_N, str(n))
        logger.info("Runtime: train_top_n → %d", n)
        return True

    # ── AI Provider ───────────────────────────────────────────────────────────
    def get_ai_provider(self) -> str:
        """Текущий AI-провайдер: auto | anthropic | openai | deepseek | groq."""
        val = self._get(keys.AI_PROVIDER)
        if val and val in _AI_PROVIDERS:
            return val
        return getattr(Config, "AI_PROVIDER", "auto")

    def set_ai_provider(self, provider: str) -> bool:
        if provider not in _AI_PROVIDERS:
            return False
        self._set(keys.AI_PROVIDER, provider)
        logger.info("Runtime: ai_provider → %s", provider)
        return True

    # ── Цепочка AI-провайдеров (раздел «AI» в приложении) ─────────────────────
    def get_ai_chain(self) -> dict:
        """Режим, порядок, модели и включённые провайдеры; пусто — по умолчанию."""
        raw = self._get(keys.AI_CHAIN)
        try:
            return ai_providers.clean_chain(json.loads(raw)) if raw else {}
        except (ValueError, TypeError):
            return {}

    def set_ai_chain(self, changes: dict) -> dict:
        """Сливает проверенные изменения с текущими; ValueError — неверные."""
        merged = {**self.get_ai_chain(), **ai_providers.clean_chain(changes)}
        self._set(keys.AI_CHAIN, json.dumps(merged))
        logger.info("Runtime: ai_chain → %s", merged)
        return merged

    def set_provider_cooldown(self, provider: str, until: float) -> None:
        """До какого времени (epoch) провайдер на паузе; 0 — паузы нет."""
        self._set(f"{keys.PROVIDER_COOLDOWN_PREFIX}{provider}", str(int(until)))

    def get_provider_cooldown(self, provider: str) -> float:
        try:
            return float(self._get(f"{keys.PROVIDER_COOLDOWN_PREFIX}{provider}") or 0)
        except ValueError:
            return 0.0

    def record_ai_call(self, provider: str) -> None:
        """Счётчик запросов к провайдеру за текущие UTC-сутки."""
        today = datetime.now(timezone.utc).date().isoformat()
        key = f"{keys.AI_CALLS_PREFIX}{provider}:{today}"
        try:
            self._r.redis_client.incr(key)
            self._r.redis_client.expire(key, 2 * 24 * 3600)
        except Exception as e:
            logger.warning("RuntimeConfig record_ai_call failed: %s", e)

    def get_ai_calls_today(self, provider: str) -> int:
        today = datetime.now(timezone.utc).date().isoformat()
        try:
            return int(self._get(f"{keys.AI_CALLS_PREFIX}{provider}:{today}") or 0)
        except ValueError:
            return 0

    def record_token_usage(self, provider: str, tokens: int) -> None:
        """Накапливает счётчик токенов за текущие UTC-сутки для провайдера."""
        import datetime as _dt

        today = _dt.datetime.now(_dt.timezone.utc).date().isoformat()
        key = f"{keys.TOKENS_USED_PREFIX}{provider}:{today}"
        try:
            self._r.redis_client.incrby(key, tokens)
            self._r.redis_client.expire(key, 2 * 24 * 3600)
        except Exception as e:
            logger.warning("RuntimeConfig record_token_usage failed: %s", e)

    def get_token_usage_today(self, provider: str) -> int:
        """Токенов использовано провайдером за текущие UTC-сутки."""
        import datetime as _dt

        today = _dt.datetime.now(_dt.timezone.utc).date().isoformat()
        val = self._get(f"{keys.TOKENS_USED_PREFIX}{provider}:{today}")
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
                keys.DEEPSEEK_BALANCE,
                3600,
                _json.dumps({"balance": balance_usd, "ts": _time.time()}),
            )
        except Exception as e:
            logger.warning("RuntimeConfig set_deepseek_balance failed: %s", e)

    def get_deepseek_balance(self) -> "dict | None":
        """Возвращает {'balance': float, 'ts': float} или None если нет в кэше."""
        import json as _json

        raw = self._get(keys.DEEPSEEK_BALANCE)
        if not raw:
            return None
        try:
            return _json.loads(raw)
        except Exception:
            return None

    # ── SAC prompt ────────────────────────────────────────────────────────────
    def _confirm(self) -> "tuple[int, str]":
        return confirm_policy(
            self._get(keys.CONFIRM_TIMEOUT) or Config.TELEGRAM_CONFIRM_TIMEOUT,
            self._get(keys.CONFIRM_ON_TIMEOUT)
            or getattr(Config, "CONFIRM_ON_TIMEOUT", "execute"),
        )

    def get_confirm_timeout(self) -> int:
        """Секунд на решение по сделке: от 10 до 120."""
        return self._confirm()[0]

    def set_confirm_timeout(self, seconds: int) -> bool:
        if not CONFIRM_TIMEOUT_MIN <= seconds <= CONFIRM_TIMEOUT_MAX:
            return False
        self._set(keys.CONFIRM_TIMEOUT, str(seconds))
        logger.info("Runtime: confirm_timeout → %dс", seconds)
        return True

    def get_confirm_on_timeout(self) -> str:
        """Если не ответили вовремя: execute — открыть сделку, skip — пропустить."""
        return self._confirm()[1]

    def set_confirm_on_timeout(self, value: str) -> bool:
        if value not in ("execute", "skip"):
            return False
        if self._get(keys.CONFIRM_TIMEOUT) == "0":  # старый ручной режим
            self._set(keys.CONFIRM_TIMEOUT, str(CONFIRM_TIMEOUT_MAX))
        self._set(keys.CONFIRM_ON_TIMEOUT, value)
        logger.info("Runtime: confirm_on_timeout → %s", value)
        return True

    def is_sac_prompted(self, profile: str = "") -> bool:
        """True если пользователь уже получал запрос об обучении SAC для профиля."""
        key = f"{keys.SAC_PROMPTED}:{profile}" if profile else keys.SAC_PROMPTED
        return self._get(key) == "1"

    def set_sac_prompted(self, profile: str = "") -> None:
        """Отмечаем что запрос об обучении SAC был отправлен для профиля."""
        key = f"{keys.SAC_PROMPTED}:{profile}" if profile else keys.SAC_PROMPTED
        self._set(key, "1")

    def get_sac_model_path_for_profile(self, profile: str) -> str:
        """Путь к SAC модели для конкретного профиля."""
        if profile and profile in _MARKET_PROFILES:
            return str(_MARKET_PROFILES[profile]["model_path"])
        return Config.SAC_MODEL_PATH

    # ── First start & tune reminder ───────────────────────────────────────────
    def ensure_first_start_date(self) -> str:
        """Записывает дату первого запуска если ещё не записана. Возвращает дату."""
        import datetime as dt

        existing = self._get(keys.FIRST_START_DATE)
        if existing:
            return existing
        today = dt.date.today().isoformat()
        self._set(keys.FIRST_START_DATE, today)
        return today

    def days_since_first_start(self) -> int:
        """Количество дней с первого запуска бота."""
        import datetime as dt

        raw = self._get(keys.FIRST_START_DATE)
        if not raw:
            return 0
        try:
            first = dt.date.fromisoformat(raw)
            return (dt.date.today() - first).days
        except ValueError:
            return 0

    def is_tune_reminded(self) -> bool:
        """True если напоминание о тюнинге уже отправлялось."""
        return self._get(keys.TUNE_REMINDED) == "1"

    def set_tune_reminded(self) -> None:
        """Отмечаем что напоминание о тюнинге было отправлено."""
        self._set(keys.TUNE_REMINDED, "1")

    # ── Chronos filter ────────────────────────────────────────────────────────
    def get_chronos_enabled(self) -> bool:
        """True → тройное подтверждение SAC+LLM+Chronos в hybrid режиме."""
        val = self._get(keys.CHRONOS_FILTER)
        return val == "1" if val is not None else False

    def set_chronos_enabled(self, enabled: bool) -> None:
        self._set(keys.CHRONOS_FILTER, "1" if enabled else "0")
        logger.info("Chronos filter %s", "ON" if enabled else "OFF")

    # ── Train progress ────────────────────────────────────────────────────────
    def set_train_progress(self, data: dict) -> None:
        import json as _json

        try:
            self._set(keys.TRAIN_PROGRESS, _json.dumps(data))
        except Exception:
            pass

    def get_train_progress(self) -> "dict | None":
        import json as _json

        val = self._get(keys.TRAIN_PROGRESS)
        if not val:
            return None
        try:
            return _json.loads(val)
        except Exception:
            return None

    def clear_train_progress(self) -> None:
        try:
            self._r.redis_client.delete(keys.TRAIN_PROGRESS)
        except Exception:
            pass

    def set_tune_progress(self, data: dict) -> None:
        import json as _json

        try:
            self._set(keys.TUNE_PROGRESS, _json.dumps(data))
        except Exception:
            pass

    def get_tune_progress(self) -> "dict | None":
        import json as _json

        val = self._get(keys.TUNE_PROGRESS)
        if not val:
            return None
        try:
            return _json.loads(val)
        except Exception:
            return None

    def clear_tune_progress(self) -> None:
        try:
            self._r.redis_client.delete(keys.TUNE_PROGRESS)
        except Exception:
            pass

    # ── Backtest progress ─────────────────────────────────────────────────────
    def set_backtest_progress(self, data: dict) -> None:
        import json as _json

        try:
            self._set(keys.BACKTEST_PROGRESS, _json.dumps(data))
        except Exception:
            pass

    def get_backtest_progress(self) -> "dict | None":
        import json as _json

        val = self._get(keys.BACKTEST_PROGRESS)
        if not val:
            return None
        try:
            return _json.loads(val)
        except Exception:
            return None

    def clear_backtest_progress(self) -> None:
        try:
            self._r.redis_client.delete(keys.BACKTEST_PROGRESS)
        except Exception:
            pass
