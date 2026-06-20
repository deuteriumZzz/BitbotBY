"""
Макро-событийный фильтр: блокирует новые позиции во время FOMC, CPI, NFP.

Использует Finnhub economic calendar API (бесплатный тир).
Если API недоступен или ключ не задан — фильтр пропускается (fail-open).
Кэш событий обновляется раз в час чтобы не превышать лимиты бесплатного тира.

УЛУЧШЕНИЕ 7: предотвращает открытие позиций в окне ±30 минут вокруг
ключевых макро-событий (FOMC, CPI, NFP, GDP, PCE).
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List

logger = logging.getLogger(__name__)

_BLACKOUT_MINUTES = 30  # минут до и после события
_CACHE_TTL = 3600.0  # кэш на 1 час

# Ключевые события которые блокируем (проверяются по вхождению в название)
_HIGH_IMPACT_KEYWORDS = frozenset(
    {
        "fomc",
        "federal reserve",
        "interest rate decision",
        "cpi",
        "consumer price index",
        "nonfarm",
        "non-farm",
        "nfp",
        "gdp",
        "pce",
    }
)


class MacroCalendar:
    """Получает экономический календарь Finnhub и сигнализирует о blackout-окнах."""

    def __init__(self) -> None:
        self._api_key: str = os.getenv("FINNHUB_API_KEY", "")
        self._cache: List[Dict] = []
        self._cache_ts: float = 0.0

    async def is_blackout(self) -> bool:
        """Возвращает True если сейчас активен blackout (близко к макро-событию).

        При любой ошибке возвращает False (fail-open — не блокируем торговлю).
        """
        if not self._api_key:
            return False
        try:
            events = await self._get_events()
            now_ts = datetime.now(timezone.utc).timestamp()
            blackout_secs = _BLACKOUT_MINUTES * 60

            for event in events:
                event_ts = event.get("_ts", 0)
                diff_secs = event_ts - now_ts
                if abs(now_ts - event_ts) < blackout_secs:
                    logger.warning(
                        "Macro blackout: %s in %d min — no new positions",
                        event.get("event", "unknown"),
                        int(diff_secs / 60),
                    )
                    return True
            return False
        except Exception as exc:
            logger.debug("MacroCalendar error: %s — allowing trade", exc)
            return False

    async def _get_events(self) -> List[Dict]:
        """Возвращает кэшированный список отфильтрованных событий.

        Обновляет кэш из Finnhub API если TTL истёк.
        """
        now = time.monotonic()
        if self._cache and now - self._cache_ts < _CACHE_TTL:
            return self._cache

        import aiohttp  # noqa: PLC0415 — опциональная зависимость

        today = datetime.now(timezone.utc)
        date_from = today.strftime("%Y-%m-%d")
        date_to = (today + timedelta(days=7)).strftime("%Y-%m-%d")

        url = (
            f"https://finnhub.io/api/v1/calendar/economic"
            f"?from={date_from}&to={date_to}&token={self._api_key}"
        )

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    logger.debug(
                        "Finnhub returned HTTP %d — skipping macro filter", resp.status
                    )
                    return []
                data = await resp.json()

        events: List[Dict] = []
        for item in data.get("economicCalendar", []):
            name = (item.get("event") or "").lower()
            impact = (item.get("impact") or "").lower()

            # Только события высокого импакта из нашего списка
            if impact != "high":
                continue
            if not any(kw in name for kw in _HIGH_IMPACT_KEYWORDS):
                continue

            date_str = item.get("date") or ""
            time_str = item.get("time") or "00:00:00"
            try:
                dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
                dt = dt.replace(tzinfo=timezone.utc)
                events.append({"event": item.get("event"), "_ts": dt.timestamp()})
            except ValueError:
                continue

        self._cache = events
        self._cache_ts = now
        logger.debug(
            "MacroCalendar: fetched %d high-impact events (next 7 days)", len(events)
        )
        return events
