"""Торговля: режим решений, скан, пауза, автоисполнение, монеты вручную, часы торговли, Paper/Live.
"""

from __future__ import annotations

import logging
from typing import Set

from bitbot.config import keys
from bitbot.config.runtime.base import RuntimeBase
from bitbot.config.runtime.presets import _VALID_MODES, _hour_in
from bitbot.config.settings import Config

logger = logging.getLogger(__name__)


class TradingSettings(RuntimeBase):
    # ── Mode ──────────────────────────────────────────────────────────────────
    def get_mode(self) -> str:
        val = self._get(keys.MODE)
        return val if val in _VALID_MODES else Config.MODE

    def set_mode(self, mode: str) -> bool:
        if mode not in _VALID_MODES:
            return False
        self._set(keys.MODE, mode)
        logger.info("Runtime: режим изменён → %s", mode)
        return True

    # ── Scan top N ────────────────────────────────────────────────────────────
    def get_scan_top_n(self) -> int:
        val = self._get(keys.SCAN_TOP_N)
        try:
            n = int(val) if val else Config.SCAN_TOP_N
            return max(1, min(n, 200))
        except (ValueError, TypeError):
            return Config.SCAN_TOP_N

    def set_scan_top_n(self, n: int) -> bool:
        if not (1 <= n <= 200):
            return False
        self._set(keys.SCAN_TOP_N, str(n))
        logger.info("Runtime: scan_top_n изменён → %d", n)
        return True

    # ── Pause ─────────────────────────────────────────────────────────────────
    def is_paused(self) -> bool:
        return self._get(keys.PAUSED) == "1"

    def set_paused(self, paused: bool) -> None:
        self._set(keys.PAUSED, "1" if paused else "0")
        logger.info("Runtime: бот %s", "на паузе" if paused else "возобновлён")

    def hold_until_setup(self) -> bool:
        """Самый первый запуск: новых входов нет, пока владелец сам не включит бота.

        Срабатывает, только если бот ещё ни разу не запускался (нет даты первого старта),
        поэтому уже работающие установки после обновления не встают на паузу.
        True — пауза поставлена.
        """
        if self._get(keys.FIRST_START_DATE) or self._get(keys.SETUP_DONE) == "1":
            return False
        self.set_paused(True)
        self._set(keys.SETUP_DONE, "0")
        return True

    def mark_setup_done(self) -> None:
        """Владелец включил бота — первый запуск пройден."""
        self._set(keys.SETUP_DONE, "1")

    # ── Auto-execute ──────────────────────────────────────────────────────────
    def get_auto_execute(self) -> bool:
        val = self._get(keys.AUTO_EXECUTE)
        if val is None:
            return Config.AUTO_EXECUTE
        return val == "1"

    def set_auto_execute(self, enabled: bool) -> None:
        self._set(keys.AUTO_EXECUTE, "1" if enabled else "0")
        logger.info("Runtime: auto_execute → %s", "ВКЛ" if enabled else "ВЫКЛ")

    def get_local_fallback(self) -> bool:
        val = self._get(keys.LOCAL_FALLBACK)
        if val is None:
            return Config.LOCAL_FALLBACK
        return val == "1"

    def set_local_fallback(self, enabled: bool) -> None:
        self._set(keys.LOCAL_FALLBACK, "1" if enabled else "0")
        logger.info("Runtime: local_fallback → %s", "ВКЛ" if enabled else "ВЫКЛ")

    def get_paper_guards_off(self) -> bool:
        """Сохранённый выбор (в Live не действует, см. paper_guards_off)."""
        val = self._get(keys.PAPER_GUARDS_OFF)
        if val is None:
            return Config.PAPER_GUARDS_OFF
        return val == "1"

    def set_paper_guards_off(self, enabled: bool) -> None:
        self._set(keys.PAPER_GUARDS_OFF, "1" if enabled else "0")
        logger.warning("Runtime: paper_guards_off → %s", "ВКЛ" if enabled else "ВЫКЛ")

    def paper_guards_off(self) -> bool:
        """True только в Paper и только если защиты выключены. В Live всегда False."""
        return bool(Config.PAPER_TRADING) and self.get_paper_guards_off()

    def get_forced_symbols(self) -> Set[str]:
        return self._smembers(keys.FORCED_SYMBOLS)

    def add_forced_symbol(self, symbol: str) -> None:
        try:
            self._r.redis_client.sadd(keys.FORCED_SYMBOLS, symbol)
            logger.info("Runtime: %s добавлен в список", symbol)
        except Exception as e:
            logger.warning("RuntimeConfig sadd failed: %s", e)

    def remove_forced_symbol(self, symbol: str) -> None:
        try:
            self._r.redis_client.srem(keys.FORCED_SYMBOLS, symbol)
        except Exception as e:
            logger.warning("RuntimeConfig srem failed: %s", e)

    def get_excluded_symbols(self) -> Set[str]:
        return self._smembers(keys.EXCLUDED_SYMBOLS)

    def add_excluded_symbol(self, symbol: str) -> None:
        try:
            self._r.redis_client.sadd(keys.EXCLUDED_SYMBOLS, symbol)
            logger.info("Runtime: %s исключён из сканирования", symbol)
        except Exception as e:
            logger.warning("RuntimeConfig sadd failed: %s", e)

    def remove_excluded_symbol(self, symbol: str) -> None:
        try:
            self._r.redis_client.srem(keys.EXCLUDED_SYMBOLS, symbol)
        except Exception as e:
            logger.warning("RuntimeConfig srem failed: %s", e)

    # ── Trading hours ─────────────────────────────────────────────────────────
    def get_trading_hours(self) -> str:
        """
        Возвращает временной диапазон торговли в формате "HH-HH" UTC.
        Пустая строка = 24 часа (без ограничений).
        """
        val = self._get(keys.TRADING_HOURS)
        if val is None:
            return Config.TRADING_HOURS
        return val

    def set_trading_hours(self, hours: str) -> bool:
        """
        Устанавливает временной диапазон. Формат: "9-17" или "" (сброс).
        Возвращает False при неверном формате.
        """
        if hours in ("", "0"):
            self._set(keys.TRADING_HOURS, "")
            logger.info("Runtime: trading_hours сброшен (24/7)")
            return True
        try:
            start, end = map(int, hours.split("-"))
            if not (0 <= start <= 23 and 0 <= end <= 23 and start != end):
                return False
        except (ValueError, AttributeError):
            return False
        self._set(keys.TRADING_HOURS, hours)
        logger.info("Runtime: trading_hours → %s UTC", hours)
        return True

    def is_trading_time(self) -> bool:
        """True если сейчас разрешено торговать по временному фильтру (UTC)."""
        import datetime as dt

        hours = self.get_trading_hours()
        if not hours:
            return True
        inside = _hour_in(hours, dt.datetime.utcnow().hour)
        return True if inside is None else inside

    # ── Paper / Live switch ───────────────────────────────────────────────────
    def get_paper_trading_override(self) -> "bool | None":
        val = self._get(keys.PAPER_TRADING_OVERRIDE)
        if val == "1":
            return True
        if val == "0":
            return False
        return None

    def set_paper_trading_override(self, paper: bool) -> None:
        self._set(keys.PAPER_TRADING_OVERRIDE, "1" if paper else "0")

    def apply_paper_trading(self, paper: bool) -> None:
        """Переключает Paper/Live в процессе и запоминает выбор в Redis.

        Общий путь для Telegram-кнопки и команды из приложения; после него
        вызывающий синхронизирует позиции (TradingBot._handle_mode_switch).
        """
        Config.PAPER_TRADING = paper
        self.set_paper_trading_override(paper)
