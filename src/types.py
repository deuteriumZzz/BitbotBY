"""
Общие доменные типы торгового бота.

Использование TypedDict вместо Dict[str, Any] для _monitored делает
схему позиции явной и проверяемой mypy. Поля Required позволяют поймать
отсутствующие поля в точке создания, а не во время выполнения SL/TP логики.
"""

from __future__ import annotations

from typing import Optional

from typing_extensions import Required, TypedDict


class PositionRecord(TypedDict, total=False):
    """
    Одна открытая позиция в TradingBot._monitored.

    Все поля опциональны (total=False) кроме структурного минимума —
    qty, side, entry — которые Required чтобы mypy ловил неполные
    словари позиций во время компиляции, а не KeyError в рантайме.
    """

    # Identity
    trade_id: Optional[int]  # None когда позиция восстановлена после краша

    # Structural — Required: неполные словари — ошибка типа, а не KeyError
    qty: Required[float]
    side: Required[str]  # "buy" | "sell"
    entry: Required[float]

    # Уровни риска при открытии; 0.0 = "не задано" (восстановленные позиции)
    stop_loss: float
    take_profit: float

    # Состояние трейлинг-стопа: ATR при входе — шаг; peak отслеживает максимум
    atr: float
    peak_price: float

    # Снэпшот сохраняется при закрытии для буфера опыта SAC
    snap: Optional[dict]

    # Записывается при открытии для атрибуции PnL и диагностики сайзинга
    balance_at_entry: float

    # ID условных ордеров на бирже — отменяются при закрытии ботом
    # чтобы предотвратить двойное исполнение.
    exchange_sl_id: Optional[str]
    exchange_tp_id: Optional[str]
