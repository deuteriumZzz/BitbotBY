"""Пресеты и справочники настроек: риск, профили монет, допустимые значения, правила подтверждения.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from bitbot.config import ai_providers

logger = logging.getLogger(__name__)


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


def _hour_in(hours: str, hour: int) -> Optional[bool]:
    """Попадает ли час в диапазон «HH-HH» (с переносом через полночь: «22-6»).

    None — диапазон пустой или в неверном формате.
    """
    try:
        start, end = (int(x) for x in hours.split("-"))
    except (ValueError, AttributeError):
        return None
    if not (0 <= start <= 23 and 0 <= end <= 23 and start != end):
        return None
    return start <= hour < end if start < end else hour >= start or hour < end


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
