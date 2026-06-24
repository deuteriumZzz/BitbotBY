"""
Детектор рыночного сезона (BTC Season / Altcoin Season).

Использует CoinGecko бесплатный API:
  - /global → BTC dominance
  - /coins/markets → топ-100 монет, 30d возврат

Логика:
  Alt Season:   altcoin_index >= 75  AND btc_dominance < 45%
  BTC Season:   altcoin_index <= 25  OR  btc_dominance > 52%

Гистерезис: сигнал подтверждается 2 проверки подряд (~8ч при CHECK_INTERVAL_H=4)
перед отправкой алерта. Повторный алерт — не раньше чем через COOLDOWN_H часов.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_COINGECKO_GLOBAL = "https://api.coingecko.com/api/v3/global"
_COINGECKO_MARKETS = (
    "https://api.coingecko.com/api/v3/coins/markets"
    "?vs_currency=usd&order=market_cap_desc&per_page=100&page=1"
    "&price_change_percentage=30d"
)

_ALT_INDEX_HI = 75  # % монет обогнавших BTC за 30d → Alt Season
_ALT_INDEX_LO = 25  # ниже → BTC Season
_BTC_DOM_ALT = 45.0  # dominance ниже → поддерживает Alt Season
_BTC_DOM_BTC = 52.0  # dominance выше → поддерживает BTC Season

_CONFIRM_STREAK = 2  # проверок подряд для подтверждения
_COOLDOWN_H = 24  # минимум часов между повторными алертами


class SeasonDetector:
    """
    Периодически опрашивает CoinGecko, вычисляет индекс сезона
    и возвращает рекомендацию о смене профиля.
    """

    def __init__(self) -> None:
        self._streak: dict[str, int] = {"altcoin": 0, "bluechip": 0}
        self._last_alert: dict[str, float] = {"altcoin": 0.0, "bluechip": 0.0}

    async def fetch_data(self) -> dict[str, Any] | None:
        """Забирает BTC dominance и данные топ-100 монет с CoinGecko."""
        try:
            import asyncio as _asyncio

            import aiohttp

            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(_COINGECKO_GLOBAL) as r:
                    if r.status != 200:
                        logger.warning("CoinGecko /global status %d", r.status)
                        return None
                    global_data = await r.json()

                await _asyncio.sleep(1)

                async with session.get(_COINGECKO_MARKETS) as r:
                    if r.status != 200:
                        logger.warning("CoinGecko /markets status %d", r.status)
                        return None
                    markets = await r.json()

            return {"global": global_data, "markets": markets}

        except Exception as exc:
            logger.warning("CoinGecko fetch failed: %s", exc)
            return None

    def compute_index(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """
        Вычисляет altcoin_index и btc_dominance.

        altcoin_index = % монет из топ-100 (кроме BTC и стейблкоинов)
        у которых 30d-доходность выше чем у BTC.
        """
        _STABLECOINS = {"tether", "usd-coin", "dai", "true-usd", "first-digital-usd"}
        try:
            btc_dominance = data["global"]["data"]["market_cap_percentage"].get(
                "btc", 0.0
            )
            markets: list[dict] = data["markets"]

            btc_row = next((c for c in markets if c.get("id") == "bitcoin"), None)
            if not btc_row:
                return None

            btc_30d = btc_row.get("price_change_percentage_30d_in_currency") or 0.0

            alts = [
                c
                for c in markets
                if c.get("id") != "bitcoin" and c.get("id") not in _STABLECOINS
            ]
            if not alts:
                return None

            outperformed = sum(
                1
                for c in alts
                if (c.get("price_change_percentage_30d_in_currency") or 0.0) > btc_30d
            )
            altcoin_index = round(outperformed / len(alts) * 100, 1)

            return {
                "altcoin_index": altcoin_index,
                "btc_dominance": round(btc_dominance, 2),
                "btc_30d": round(btc_30d, 2),
                "alts_count": len(alts),
                "alts_outperformed": outperformed,
            }

        except Exception as exc:
            logger.warning("season compute_index error: %s", exc)
            return None

    def classify(self, index: dict[str, Any]) -> str | None:
        """Возвращает 'altcoin', 'bluechip' или None если сезон неясен."""
        ai = index["altcoin_index"]
        dom = index["btc_dominance"]

        if ai >= _ALT_INDEX_HI and dom < _BTC_DOM_ALT:
            return "altcoin"
        if ai <= _ALT_INDEX_LO or dom > _BTC_DOM_BTC:
            return "bluechip"
        return None

    def should_alert(
        self,
        signal: str | None,
        current_profile: str,
        now: float,
    ) -> bool:
        """
        Возвращает True если нужно отправить алерт о смене сезона.

        Условия:
        - signal есть и отличается от current_profile
        - streak достиг _CONFIRM_STREAK
        - прошло достаточно времени с последнего алерта этого типа
        """
        for key in ("altcoin", "bluechip"):
            if key != signal:
                self._streak[key] = 0

        if signal is None or signal == current_profile:
            return False

        self._streak[signal] = self._streak.get(signal, 0) + 1

        if self._streak[signal] < _CONFIRM_STREAK:
            logger.info(
                "Season signal %s: streak %d/%d — ждём подтверждения",
                signal,
                self._streak[signal],
                _CONFIRM_STREAK,
            )
            return False

        elapsed_h = (now - self._last_alert.get(signal, 0.0)) / 3600
        if elapsed_h < _COOLDOWN_H:
            logger.debug(
                "Season alert %s cooldown: %.1fh осталось",
                signal,
                _COOLDOWN_H - elapsed_h,
            )
            return False

        self._last_alert[signal] = now
        return True

    def format_message(self, target: str, index: dict[str, Any]) -> str:
        """Форматирует Telegram-уведомление об изменении сезона."""
        ai = index["altcoin_index"]
        dom = index["btc_dominance"]
        btc_30d = index["btc_30d"]
        count = index["alts_outperformed"]
        total = index["alts_count"]

        bar_filled = int(ai / 10)
        bar = "█" * bar_filled + "░" * (10 - bar_filled)

        if target == "altcoin":
            header = "🟡 *Обнаружен АЛЬТСЕЗОН*"
            hint = (
                f"{count} из {total} альтов обогнали BTC за 30 дней.\n"
                f"BTC Dominance: *{dom}%*\n\n"
                "Рекомендуется переключиться на профиль *Альткоины*."
            )
        else:
            header = "🔵 *Обнаружен BTC SEASON*"
            hint = (
                f"BTC Dominance вырос до *{dom}%*.\n"
                f"Только {count} из {total} альтов обогнали BTC за 30 дней.\n\n"
                "Рекомендуется переключиться на профиль *Блючипы*."
            )

        return (
            f"{header}\n\n"
            f"Альт-индекс: `[{bar}]` *{ai:.0f}/100*\n"
            f"BTC Dominance: *{dom}%*\n"
            f"BTC 30d: *{btc_30d:+.1f}%*\n\n"
            f"{hint}"
        )
