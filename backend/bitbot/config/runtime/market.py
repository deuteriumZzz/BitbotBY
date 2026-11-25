"""Рынок: тип рынка, профиль монет, стратегии, сезон, Fear & Greed, Pump.fun.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Set

from bitbot.config import keys
from bitbot.config.runtime.active import (
    set_active_market_type,
    set_active_profile,
)
from bitbot.config.runtime.models import ModelSettings
from bitbot.config.runtime.presets import _MARKET_PROFILES, _MARKET_TYPES
from bitbot.config.settings import Config

logger = logging.getLogger(__name__)


class MarketSettings(ModelSettings):
    # ── Pump.fun launch-rate baseline (для meme-season) ─────────────────────────
    # Абсолютный порог "нормального" темпа лаунчей на pump.fun угадать нельзя —
    # база сама по себе огромна и меняется со временем. Вместо этого копим
    # историю замеров и сравниваем текущий темп с собственным скользящим
    # медианным бейзлайном (см. SeasonDetector.classify).
    _LAUNCH_RATE_MAX_SAMPLES = 42  # ~7 дней при проверке раз в 4 часа

    # ── Market Type (spot / futures) ─────────────────────────────────────────
    def get_market_type(self) -> str:
        """Текущий тип рынка: spot | linear (perpetual futures)."""
        val = self._get(keys.MARKET_TYPE)
        if val and val in _MARKET_TYPES:
            return val
        return getattr(Config, "MARKET_TYPE", "linear")

    def set_market_type(self, market_type: str) -> bool:
        if market_type not in _MARKET_TYPES:
            return False
        self._set(keys.MARKET_TYPE, market_type)
        set_active_market_type(market_type)
        logger.info("Runtime: market_type → %s", market_type)
        return True

    def get_last_ai_provider(self) -> str:
        """Провайдер, который реально ответил последним."""
        return self._get(keys.LAST_AI_PROVIDER) or ""

    def set_last_ai_provider(self, provider: str) -> None:
        self._set(keys.LAST_AI_PROVIDER, provider)

    def set_provider_status(self, provider: str, status: str) -> None:
        """Сохраняет статус провайдера: 'ok', 'no_balance', 'rate_limit'."""
        self._set(f"{keys.PROVIDER_STATUS_PREFIX}{provider}", status)

    def get_provider_status(self, provider: str) -> str:
        """Возвращает последний известный статус провайдера."""
        return self._get(f"{keys.PROVIDER_STATUS_PREFIX}{provider}") or "unknown"

    # ── Strategies ────────────────────────────────────────────────────────────
    def get_disabled_strategies(self) -> Set[str]:
        """Возвращает имена отключённых стратегий. Пустой set = все включены."""
        return self._smembers(keys.DISABLED_STRATEGIES)

    def disable_strategy(self, name: str) -> None:
        try:
            self._r.redis_client.sadd(keys.DISABLED_STRATEGIES, name)
            logger.info("Runtime: стратегия отключена — %s", name)
        except Exception as e:
            logger.warning("RuntimeConfig sadd failed: %s", e)

    def enable_strategy(self, name: str) -> None:
        try:
            self._r.redis_client.srem(keys.DISABLED_STRATEGIES, name)
            logger.info("Runtime: стратегия включена — %s", name)
        except Exception as e:
            logger.warning("RuntimeConfig srem failed: %s", e)

    def ensure_experimental_disabled(self, names: Iterable[str]) -> List[str]:
        """Новые экспериментальные стратегии при первом появлении выключаются.

        Каждое имя обрабатывается один раз: дальше решает пользователь (включил —
        остаётся включённой). Возвращает, какие были выключены сейчас.
        """
        try:
            seen = self._smembers(keys.EXPERIMENTAL_STRATEGIES_SEEN)
            fresh = [n for n in names if n not in seen]
            for name in fresh:
                self.disable_strategy(name)
                self._r.redis_client.sadd(keys.EXPERIMENTAL_STRATEGIES_SEEN, name)
            return fresh
        except Exception as e:
            logger.warning("RuntimeConfig experimental seeding failed: %s", e)
            return []

    def reset_strategies(self, keep_off: Iterable[str]) -> None:
        """Включить все стратегии, кроме `keep_off` (очистить список отключённых).

        `keep_off` — обычно экспериментальные (без доказанного преимущества,
        `strategies.experimental_names()`): одно нажатие не должно включать
        десятки стратегий, которые мы держим выключенными нарочно. Список
        передаёт вызывающий: настройки не знают о стратегиях (слой ниже).
        """
        try:
            self._r.redis_client.delete(keys.DISABLED_STRATEGIES)
            logger.info("Runtime: основные стратегии включены")
        except Exception as e:
            logger.warning("RuntimeConfig delete failed: %s", e)
            return
        for name in keep_off:
            self.disable_strategy(name)

    def toggle_strategy(self, name: str) -> bool:
        """Переключает стратегию. Возвращает True если теперь включена."""
        if name in self.get_disabled_strategies():
            self.enable_strategy(name)
            return True
        self.disable_strategy(name)
        return False

    def get_sac_model_path(self) -> str:
        """Путь к SAC модели для текущего профиля. Fallback: Config.SAC_MODEL_PATH."""
        profile = self.get_market_profile()
        if profile and profile in _MARKET_PROFILES:
            return str(_MARKET_PROFILES[profile]["model_path"])
        return Config.SAC_MODEL_PATH

    # ── Market Profile ────────────────────────────────────────────────────────
    def get_market_profile(self) -> str:
        """Текущий профиль: 'bluechip' | 'altcoin' | '' (не задан)."""
        return self._get(keys.MARKET_PROFILE) or ""

    def ensure_default_market_profile(self, default: str = "bluechip") -> str:
        """Применяет профиль по умолчанию, если он ещё не задан (первый запуск
        или после очистки Redis). Возвращает действующий профиль."""
        current = self.get_market_profile()
        if current:
            set_active_profile(current)  # порог volume_spike после рестарта
            return current
        self.apply_market_profile(default)
        return default

    def get_timeframe(self) -> str:
        """Таймфрейм свечей. Fallback: Config.TIMEFRAME."""
        return self._get(keys.TIMEFRAME) or Config.TIMEFRAME

    def get_min_volume_usdt(self) -> float:
        """Минимальный 24h объём USDT. Fallback: Config.MIN_VOLUME_USDT."""
        val = self._get(keys.MIN_VOLUME_USDT)
        try:
            return float(val) if val is not None else Config.MIN_VOLUME_USDT
        except ValueError:
            return Config.MIN_VOLUME_USDT

    def get_max_volume_usdt(self) -> float:
        """Максимальный 24h объём USDT (0 = без ограничения). Fallback: Config."""
        val = self._get(keys.MAX_VOLUME_USDT)
        try:
            return float(val) if val is not None else Config.MAX_VOLUME_USDT
        except ValueError:
            return Config.MAX_VOLUME_USDT

    def apply_market_profile(self, name: str) -> bool:
        """Применяет профиль рынка. Возвращает False если имя неизвестно."""
        profile = _MARKET_PROFILES.get(name)
        if not profile:
            return False
        self._set(keys.MARKET_PROFILE, name)
        self._set(keys.TIMEFRAME, str(profile.get("timeframe", "")))
        self._set(keys.MIN_VOLUME_USDT, str(profile.get("min_volume_usdt", 0)))
        self._set(keys.MAX_VOLUME_USDT, str(profile.get("max_volume_usdt", 0)))
        self.set_scan_top_n(int(str(profile.get("scan_top_n", 20))))
        self.set_train_top_n(int(str(profile.get("train_top_n", 20))))
        self.set_risk_per_trade(float(str(profile.get("risk_per_trade", 0.02))))
        # «Кто решает» (MODE) профиль не меняет: это выбор владельца. Раньше профиль
        # ставил Hybrid, и без допущенной SAC бот после автосмены по сезону переставал
        # открывать сделки.
        self._set(keys.SL_PERCENT, str(profile.get("sl_percent", 0.05)))
        self._set(keys.TP_MULTIPLIER, str(profile.get("tp_multiplier", 2.0)))
        self._set(keys.TRADING_INTERVAL, str(profile.get("trading_interval", 60)))

        # Переключаем профиль-специфичные стратегии. volume_spike одна для всех
        # профилей: порог берётся из active_profile() внутри стратегии. Старые
        # профильные имена из списка отключённых убираем — их больше нет.
        set_active_profile(name)
        # sentiment — во всех профилях: Telegram-каналы читаются по монетам
        # любого скана (Twitter по-прежнему только в meme).
        _MEME_ONLY = {"momentum_burst", "pump_exit"}
        self.enable_strategy("sentiment")
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
        self._set(keys.CURRENT_SYMBOLS, json.dumps(payload, ensure_ascii=False))

    def get_current_symbols(self) -> Optional[dict]:
        """Возвращает {"profile", "symbols", "updated_at"} или None, если данных нет."""
        raw = self._get(keys.CURRENT_SYMBOLS)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return None

    # ── Season switch mode ────────────────────────────────────────────────────
    def get_season_switch_mode(self) -> str:
        import os as _os

        val = self._get(keys.SEASON_SWITCH_MODE)
        if val in ("alert", "auto"):
            return val
        return _os.getenv("SEASON_SWITCH_MODE", "alert").lower()

    def set_season_switch_mode(self, mode: str) -> bool:
        if mode not in ("alert", "auto"):
            return False
        self._set(keys.SEASON_SWITCH_MODE, mode)
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
            self._r.redis_client.setex(keys.SEASON_INDEX, 6 * 3600, _json.dumps(data))
        except Exception:
            pass

    def try_start_job(self, group: str, source: str, ttl: int) -> bool:
        """Занимает группу задач (sac, backtest, …); False, если уже занята.

        Замок общий для Telegram и приложения, поэтому один и тот же бэктест или
        обучение не запустятся дважды. TTL — страховка на случай падения бота.
        """
        try:
            return bool(
                self._r.redis_client.set(
                    keys.JOB_PREFIX + group, source, nx=True, ex=ttl
                )
            )
        except Exception as e:
            logger.warning("RuntimeConfig job lock failed: %s", e)
            return True  # без Redis замок невозможен — не блокируем запуск

    def end_job(self, group: str) -> None:
        try:
            self._r.redis_client.delete(keys.JOB_PREFIX + group)
        except Exception as e:
            logger.warning("RuntimeConfig job unlock failed: %s", e)

    def running_jobs(self, groups: Iterable[str]) -> Dict[str, str]:
        """{группа: источник запуска} для занятых групп."""
        out: Dict[str, str] = {}
        for g in groups:
            val = self._get(keys.JOB_PREFIX + g)
            if val:
                out[g] = val
        return out

    def get_sac_backup_path(self) -> str:
        return self._get(keys.SAC_MODEL_BACKUP) or ""

    def set_sac_backup_path(self, path: str, profile: str = "") -> None:
        self._set(keys.SAC_MODEL_BACKUP, path)
        self._set(keys.SAC_BACKUP_PROFILE, profile)

    def get_sac_backup_profile(self) -> str:
        """Профиль, чью модель сохранила последняя резервная копия ('' — неизвестно)."""
        return self._get(keys.SAC_BACKUP_PROFILE) or ""

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

    # ── Fear & Greed ──────────────────────────────────────────────────────────
    def set_fear_greed(self, value: int, label: str) -> None:
        import json as _json

        try:
            self._r.redis_client.setex(
                keys.FEAR_GREED,
                6 * 3600,
                _json.dumps({"value": value, "label": label}),
            )
        except Exception:
            pass

    def get_fear_greed(self) -> "dict | None":
        import json as _json

        raw = self._get(keys.FEAR_GREED)
        if not raw:
            return None
        try:
            return _json.loads(raw)
        except Exception:
            return None

    def record_launch_rate_sample(self, rate: float) -> None:
        """Добавляет замер темпа лаунчей в скользящую историю (капается)."""
        import json as _json

        raw = self._get(keys.LAUNCH_RATE_HISTORY)
        try:
            history = _json.loads(raw) if raw else []
        except Exception:
            history = []
        history.append(round(rate, 2))
        history = history[-self._LAUNCH_RATE_MAX_SAMPLES :]
        self._set(keys.LAUNCH_RATE_HISTORY, _json.dumps(history))

    def get_launch_rate_baseline(self, min_samples: int = 5) -> "float | None":
        """Медиана накопленной истории. None если данных ещё недостаточно."""
        import json as _json
        import statistics as _statistics

        raw = self._get(keys.LAUNCH_RATE_HISTORY)
        if not raw:
            return None
        try:
            history = _json.loads(raw)
        except Exception:
            return None
        if len(history) < min_samples:
            return None
        return float(_statistics.median(history))
