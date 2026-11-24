"""Funding в PnL сделки (AUDIT M-10): оценка для paper и выплаты с биржи для live.

Включается флагом `FUNDING_IN_PNL` (по умолчанию ВЫКЛЮЧЕН: меняет учёт результата сделок
и статистику, на которой считаются Kelly и метрики). Эффект на PnL: минус — заплатили,
плюс — получили.

* paper: оценка `estimate_effect` по ставке `PAPER_FUNDING_RATE_8H` и сроку удержания.
  Лонг платит при положительной ставке, шорт получает (как на бирже).
* live: сумма выплат из `fetch_funding_history`. ccxt/Bybit отдают `amount = execFee`
  (комиссионная семантика: положительное — уплачено, отрицательное — получено), поэтому
  эффект на PnL = −Σ amount. ЗНАК НЕ ПРОВЕРЕН НА РЕАЛЬНОЙ БИРЖЕ — проверить на testnet до
  включения флага. Защита: эффект, превышающий `MAX_SHARE_OF_MARGIN` маржи, считается
  аномалией и не применяется.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

FUNDING_PERIOD_S = 8 * 3600
MAX_SHARE_OF_MARGIN = 0.5  # предложение: funding > 50% маржи за сделку — аномалия


def estimate_effect(
    side: str, qty: float, entry: float, exit_price: float, hold_s: float, rate_8h: float
) -> float:
    """Оценка эффекта funding на PnL за срок удержания (минус — заплатили)."""
    if qty <= 0 or hold_s <= 0 or rate_8h == 0:
        return 0.0
    notional = qty * (entry + exit_price) / 2.0
    sign = 1.0 if side in ("buy", "long") else -1.0  # лонг платит при rate > 0
    return -sign * notional * rate_8h * (hold_s / FUNDING_PERIOD_S)


def effect_from_history(entries: Iterable[Dict[str, Any]]) -> float:
    """Эффект на PnL по записям `fetch_funding_history` (amount > 0 — уплачено)."""
    total = 0.0
    for e in entries:
        try:
            total += float(e.get("amount") or 0.0)
        except (TypeError, ValueError):
            continue
    return -total


def is_sane(effect: float, margin: float) -> bool:
    return margin <= 0 or abs(effect) <= MAX_SHARE_OF_MARGIN * margin


async def live_effect(api: Any, symbol: str, since_ms: int, limit: int = 100) -> Optional[float]:
    """Выплаты funding по символу с момента открытия; None — биржа не ответила."""
    try:
        entries = await api.exchange.fetch_funding_history(symbol, since=since_ms, limit=limit)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 — отсутствие данных не должно ломать закрытие
        logger.warning("funding history for %s unavailable: %s", symbol, exc)
        return None
    return effect_from_history(entries or [])
