"""Настройки источников: новости и соцсети, сделки по доверенным каналам.
"""

from __future__ import annotations

import logging
from typing import Tuple

from bitbot.config.registry.model import (
    _G_NEWS,
    _G_TRUST,
    Setting,
    _hot,
)

logger = logging.getLogger(__name__)


SETTINGS: Tuple[Setting, ...] = (
    Setting(
        "NEWS_UPDATE_INTERVAL",
        _G_NEWS,
        "Обновлять новости каждые",
        "int",
        900,
        min=60,
        max=86400,
        unit="с",
    ),
    Setting(
        "TWITTER_MAX_RESULTS",
        _G_NEWS,
        "Твитов за запрос",
        "int",
        10,
        min=10,
        max=100,
        hint="Меньше — экономнее тариф Basic.",
    ),
    Setting(
        "TELEGRAM_SENTIMENT_CHANNELS",
        _G_NEWS,
        "Telegram-каналы для мониторинга",
        "str",
        "CryptoMoonAlerts,pumpsignalscrypto,memecoinsnews",
        hint=(
            "Через запятую: имя без @ или ссылка t.me/…, приватный канал — "
            "числовой id (-100…). Бот подпишется на них от вашего аккаунта: "
            "так Telegram присылает посты сразу."
        ),
        **_hot("TELEGRAM_SENTIMENT_CHANNELS"),
    ),
    Setting(
        "TELEGRAM_WINDOW_MIN",
        _G_NEWS,
        "Учитывать посты Telegram за",
        "int",
        60,
        min=5,
        max=1440,
        unit="мин",
        hint="Монета оценивается только по свежим постам за это время.",
        **_hot("TELEGRAM_WINDOW_MIN"),
    ),
    Setting(
        "TELEGRAM_TRUSTED_TRADING",
        _G_TRUST,
        "Сделки по доверенным каналам",
        "bool",
        False,
        hint=(
            "Пост доверенного канала сразу становится сделкой — без сканера, "
            "стратегий и AI, в любом профиле. Остаются стоп, свой риск и слоты, "
            "пауза, дневной лимит, спред и режим счёта."
        ),
        **_hot("TELEGRAM_TRUSTED_TRADING"),
    ),
    Setting(
        "TELEGRAM_TRUSTED_CHANNELS",
        _G_TRUST,
        "Доверенные каналы",
        "str",
        "",
        hint=(
            "Через запятую. Удобнее — переключателем «Доверять» в "
            "Настройки → Каналы Telegram."
        ),
        **_hot("TELEGRAM_TRUSTED_CHANNELS"),
    ),
    Setting(
        "TELEGRAM_TRUSTED_RISK",
        _G_TRUST,
        "Риск на сделку по каналу",
        "pct",
        0.005,
        min=0.001,
        max=0.05,
        step=0.001,
        hint="Сколько баланса потеряете, если сработает стоп.",
        **_hot("TELEGRAM_TRUSTED_RISK"),
    ),
    Setting(
        "TELEGRAM_TRUSTED_SLOTS",
        _G_TRUST,
        "Слотов для сделок по каналам",
        "int",
        2,
        min=1,
        max=10,
        hint="Отдельно от обычных слотов бота.",
        **_hot("TELEGRAM_TRUSTED_SLOTS"),
    ),
    Setting(
        "TELEGRAM_TRUSTED_SL",
        _G_TRUST,
        "Стоп, если в посте его нет",
        "pct",
        0.03,
        min=0.005,
        max=0.2,
        step=0.005,
        hint="Без цели в посте тейк — в 2 раза дальше стопа.",
        **_hot("TELEGRAM_TRUSTED_SL"),
    ),
    Setting(
        "TELEGRAM_TRUSTED_DEDUP_MIN",
        _G_TRUST,
        "Одна сделка по монете за",
        "int",
        30,
        min=1,
        max=1440,
        unit="мин",
        hint="Та же монета из нескольких каналов или репост — одна сделка.",
        **_hot("TELEGRAM_TRUSTED_DEDUP_MIN"),
    ),
    Setting(
        "TELEGRAM_TRUSTED_UNTRUST",
        _G_TRUST,
        "Снимать доверие после убытков подряд",
        "int",
        0,
        min=0,
        max=20,
        hint="0 — никогда. Об отключении канала придёт сообщение в Telegram.",
        **_hot("TELEGRAM_TRUSTED_UNTRUST"),
    ),
    Setting(
        "SENTIMENT_POLL_INTERVAL",
        _G_NEWS,
        "Опрашивать Twitter каждые",
        "int",
        300,
        min=60,
        max=3600,
        unit="с",
        hint="Посты Telegram приходят сразу, без опроса.",
    ),
    Setting(
        "SENTIMENT_CACHE_TTL",
        _G_NEWS,
        "Хранить оценку настроения",
        "int",
        600,
        min=60,
        max=7200,
        unit="с",
    ),
    Setting(
        "SENTIMENT_BUY_THRESHOLD",
        _G_NEWS,
        "Порог покупки по настроению",
        "float",
        0.4,
        min=0,
        max=1,
        step=0.05,
    ),
    Setting(
        "SENTIMENT_SELL_THRESHOLD",
        _G_NEWS,
        "Порог продажи по настроению",
        "float",
        -0.4,
        min=-1,
        max=0,
        step=0.05,
    ),
)
