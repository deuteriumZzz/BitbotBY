"""Реестр флагов поведения: что делает флаг, умолчание в paper и в live, статус.

Статус:
  shadow — только наблюдает (журнал, метрики, тревоги), на сделки не влияет;
  gated  — меняет решения или учёт; включается только после проверки:
           в paper (там, где умолчание True) или отдельным решением владельца;
  live   — включён в обоих режимах (защиты и проверенная логика выхода).

Значение флага — `Config.<ключ>` (переменная окружения главнее умолчания);
этот файл ничего не включает сам. Тест tests/test_flags_registry.py запускает
Config с PAPER_TRADING=true и false и сверяет умолчания с этой таблицей.
Здесь — флаги, которые включают или выключают целую функцию; обычные настройки
(пороги, проценты, списки) — в config/registry.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class Flag:
    key: str
    what: str
    paper: bool  # умолчание в paper
    live: bool  # умолчание в live
    status: str  # shadow | gated | live


FLAGS: Tuple[Flag, ...] = (
    # ── наблюдение ───────────────────────────────────────────────────────────
    Flag(
        "RISK_GATE_SHADOW",
        "вердикт единого Risk Gate по каждому входу — в журнал, без блокировки",
        True,
        True,
        "shadow",
    ),
    Flag(
        "WS_SHADOW_ENABLED",
        "публичный WS Bybit параллельно REST: метрики и тревога при тишине",
        True,
        False,
        "shadow",
    ),
    Flag(
        "EVENT_STREAM_ENABLED",
        "копии записей журнала отметками в Redis Stream (шина событий)",
        False,
        False,
        "shadow",
    ),
    # ── меняют решения или учёт ──────────────────────────────────────────────
    Flag(
        "AI_SANITIZE_PROMPT",
        "заголовки новостей очищаются и помечаются как недоверенный текст",
        True,
        False,
        "gated",
    ),
    Flag(
        "AI_STRICT_VALIDATION",
        "ответ LLM проходит строгую схему: лишние поля и выдуманные цены — отказ",
        True,
        False,
        "gated",
    ),
    Flag(
        "DATA_QUALITY_GATE",
        "нет входа по символу со свечами не VALID (устарели, битые, неполные)",
        True,
        False,
        "gated",
    ),
    Flag(
        "AI_RAG_CONTEXT",
        "в промпт AI добавляется, чем заканчивались похожие решения",
        False,
        False,
        "gated",
    ),
    Flag(
        "FUNDING_IN_PNL",
        "выплаты funding входят в результат сделки",
        False,
        False,
        "gated",
    ),
    Flag(
        "SAC_V2_ENABLED",
        "SAC v2 вместо v1; торгует только после допуска в реестре моделей",
        False,
        False,
        "gated",
    ),
    Flag(
        "RECONCILE_AUTOFIX",
        "бот сам исправляет расхождение позиций с биржей (иначе только сообщает)",
        False,
        False,
        "gated",
    ),
    Flag(
        "LOCAL_FALLBACK",
        "сделка по локальным стратегиям, когда AI/SAC не дали сигнала",
        False,
        False,
        "gated",
    ),
    Flag(
        "TELEGRAM_TRUSTED_TRADING",
        "сделки по постам доверенных Telegram-каналов",
        False,
        False,
        "gated",
    ),
    Flag(
        "PAPER_GUARDS_OFF",
        "без защит портфеля, чтобы быстрее копить сделки; в live игнорируется",
        False,
        False,
        "gated",
    ),
    # ── включены везде ───────────────────────────────────────────────────────
    Flag(
        "SAC_REQUIRE_APPROVAL",
        "SAC торгует только после допуска в реестре моделей",
        True,
        True,
        "live",
    ),
    Flag(
        "LIVE_ALPHA_KILL_ENABLED",
        "стратегия с плохим live EV блокируется до ручного решения",
        True,
        True,
        "live",
    ),
    Flag(
        "DRAWDOWN_SCALE_ENABLED",
        "размер позиции уменьшается при просадке от пика баланса",
        True,
        True,
        "live",
    ),
    Flag(
        "MACRO_BLACKOUT_ENABLED",
        "нет торговли за 30 мин до и после крупных макро-событий",
        True,
        True,
        "live",
    ),
    Flag(
        "PARTIAL_TP_ENABLED",
        "частичная фиксация прибыли и перенос SL в безубыток",
        True,
        True,
        "live",
    ),
    Flag(
        "DB_BACKUP_ENABLED",
        "ежедневная проверенная копия базы сделок",
        True,
        True,
        "live",
    ),
)

BY_KEY: Dict[str, Flag] = {f.key: f for f in FLAGS}


def current(config: Any) -> Dict[str, bool]:
    """Значения всех флагов сейчас (для карточки «Защита и наблюдение» и журнала)."""
    return {f.key: getattr(config, f.key, False) is True for f in FLAGS}
