"""Настройки AI и моделей: решения AI, обучение SAC и бэктест, сезон рынка.
"""

from __future__ import annotations

import logging
from typing import Tuple

from bitbot.config import ai_providers
from bitbot.config.registry.model import (
    _G_AI,
    _G_SAC,
    _G_SEASON,
    _STRATEGIES,
    Setting,
    _opts,
    _rc,
)

logger = logging.getLogger(__name__)


SETTINGS: Tuple[Setting, ...] = (
    Setting(
        "MODE",
        _G_AI,
        "Кто принимает решение",
        "choice",
        "ai",
        options=(
            ("local", "Локальные стратегии (без AI)"),
            ("ai", "AI"),
            ("dqn", "SAC"),
            ("hybrid", "Hybrid"),
        ),
        **_rc("set_mode", "get_mode"),
    ),
    Setting(
        "LOCAL_FALLBACK",
        _G_AI,
        "Если AI или SAC недоступны — торговать локальными стратегиями",
        "bool",
        False,
        hint="Только на время сбоя: в AI не ответил ни один провайдер, в SAC модель не "
        "загружена. Тогда бот ищет вход по включённым локальным стратегиям (у них нет "
        "доказанного преимущества). Если модель ответила, но входа не увидела, сделки "
        "нет — стратегии её не перебивают. Выключено: модель недоступна — новых "
        "сделок нет. В режиме Local не используется.",
        **_rc("set_local_fallback", "get_local_fallback"),
    ),
    Setting(
        "AI_PROVIDER",
        _G_AI,
        "AI-провайдер",
        "choice",
        "auto",
        options=(
            ("auto", "Авто — по цепочке"),
            *((p.id, p.name) for p in ai_providers.PROVIDERS),
        ),
        hint="Выбранный провайдер идёт первым; остальная цепочка — в разделе AI.",
        **_rc("set_ai_provider", "get_ai_provider"),
    ),
    Setting(
        "CHRONOS_ENABLED",
        _G_AI,
        "Прогноз Chronos в режиме Hybrid",
        "bool",
        False,
        hint="Модель Amazon Chronos чуть повышает или понижает уверенность сигнала.",
        hot_set=lambda rc, v: rc.set_chronos_enabled(v),
        hot_get=lambda rc: rc.get_chronos_enabled(),
    ),
    Setting(
        "FORECASTER",
        _G_AI,
        "Чей прогноз",
        "choice",
        "chronos",
        options=(("chronos", "Chronos (Amazon)"), ("timesfm", "TimesFM (Google)")),
        hint="Обе модели прогнозируют цену по свечам; поправка одна и та же. Меняется сразу.",
        **_rc("set_forecaster", "get_forecaster"),
    ),
    Setting(
        "AI_DAILY_BUDGET",
        _G_AI,
        "AI-запросов в сутки",
        "int",
        200,
        min=0,
        max=10000,
        hint="Сверх лимита — простой анализ настроения без AI.",
    ),
    Setting("AI_MAX_SYMBOLS", _G_AI, "Монет в AI за цикл", "int", 30, min=1, max=100),
    Setting(
        "ACTIVE_STRATEGY",
        _G_AI,
        "Стратегия по умолчанию",
        "choice",
        "ema_crossover",
        options=_opts(*_STRATEGIES),
        attr="DEFAULT_STRATEGY",
    ),
    Setting(
        "SEASON_SWITCH_MODE",
        _G_SEASON,
        "При смене сезона",
        "choice",
        "alert",
        options=(("alert", "Спросить в Telegram"), ("auto", "Переключить самому")),
        **_rc("set_season_switch_mode", "get_season_switch_mode"),
    ),
    Setting(
        "SEASON_CHECK_INTERVAL_H",
        _G_SEASON,
        "Проверять сезон каждые",
        "float",
        4.0,
        min=1,
        max=48,
        unit="ч",
    ),
    Setting(
        "ONLINE_LEARNING_MODE",
        _G_SAC,
        "Автообучение",
        "choice",
        "disabled",
        options=(
            ("disabled", "Выключено"),
            ("periodic", "Периодически"),
            ("online", "После каждой сделки"),
            ("hybrid", "Периодически + веса стратегий"),
        ),
        hint="«После каждой сделки» — рискованно, только для экспериментов.",
    ),
    Setting(
        "ONLINE_LEARNING_TRIGGER",
        _G_SAC,
        "Переобучать каждые",
        "int",
        50,
        min=10,
        max=1000,
        unit="сделок",
    ),
    Setting(
        "ONLINE_LEARNING_GRADIENT_STEPS",
        _G_SAC,
        "Шагов за онлайн-обновление",
        "int",
        50,
        min=1,
        max=50,
    ),
    Setting(
        "TRAIN_TOP_N",
        _G_SAC,
        "Монет для автодообучения",
        "int",
        20,
        min=1,
        max=100,
        hint="Только для автоматического дообучения. "
        "Ручное обучение берёт число монет из профиля рынка.",
        **_rc("set_train_top_n", "get_train_top_n"),
    ),
    Setting(
        "TRAIN_MIN_CANDLES",
        _G_SAC,
        "Минимум свечей на монету",
        "int",
        2880,
        min=500,
        max=50000,
        step=100,
    ),
    Setting(
        "TOTAL_TIMESTEPS",
        _G_SAC,
        "Шагов обучения",
        "int",
        50000,
        min=1000,
        max=5_000_000,
        step=1000,
    ),
    Setting(
        "EXPERIENCES_MAX",
        _G_SAC,
        "Хранить записей опыта",
        "int",
        10000,
        min=1000,
        max=1_000_000,
        step=1000,
    ),
    Setting(
        "BT_HOLDOUT_RATIO",
        _G_SAC,
        "Отложенная выборка в бэктесте",
        "pct",
        0.2,
        min=0.05,
        max=0.5,
        step=0.05,
        hint="Последняя часть истории, на которой проверяется стратегия.",
    ),
    Setting("SAC_MODEL_PATH", _G_SAC, "Файл модели SAC", "str", "models/sac_model.zip"),
)
