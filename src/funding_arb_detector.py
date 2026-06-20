"""Детектор funding arbitrage для Bybit perpetual рынка."""

from __future__ import annotations

import logging
import time
from typing import Dict

logger = logging.getLogger(__name__)

_FUNDING_ALERT_COOLDOWN: float = 8 * 3600  # 8 часов (интервал funding payments)
_FUNDING_THRESHOLD: float = 0.0005  # 0.05% per 8h
_FUNDING_EXTREME: float = 0.001  # 0.10% per 8h


class FundingArbDetector:
    """
    Детектирует возможности funding arbitrage.

    Условие: funding_rate > 0.05% (0.0005) за 8ч ≈ 54% годовых.
    Алерт не чаще 1 раза в 8 часов на символ.
    """

    def __init__(self) -> None:
        self._last_alert_at: Dict[str, float] = {}

    async def check_and_alert(
        self,
        symbol: str,
        funding_rate: float,
        telegram: object,
    ) -> None:
        if abs(funding_rate) < _FUNDING_THRESHOLD:
            return

        now = time.monotonic()
        last = self._last_alert_at.get(symbol, 0.0)
        if now - last < _FUNDING_ALERT_COOLDOWN:
            return

        self._last_alert_at[symbol] = now
        annualized = abs(funding_rate) * 3 * 365 * 100
        level = "EXTREME 🔥" if abs(funding_rate) >= _FUNDING_EXTREME else "HIGH ⚡"
        base = symbol.split("/")[0]
        if funding_rate > 0:
            direction = f"Long {base} SPOT + Short PERP (longs pay shorts)"
        else:
            direction = f"Short {base} SPOT + Long PERP (shorts pay longs)"
        msg = (
            f"💰 FUNDING ARB {level}\n"
            f"Symbol: {symbol}\n"
            f"Funding: {funding_rate * 100:.4f}% per 8h\n"
            f"APY: ~{annualized:.0f}%\n"
            f"Strategy: {direction}\n"
            f"Risk: Near-zero (delta neutral)"
        )
        try:
            await telegram.notify(msg)  # type: ignore[attr-defined]
            logger.info(
                "FundingArb alert sent: %s rate=%.4f%% APY=%.0f%%",
                symbol,
                funding_rate * 100,
                annualized,
            )
        except Exception as exc:
            logger.warning("FundingArb alert failed for %s: %s", symbol, exc)
