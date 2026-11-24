"""Фоновый монитор открытых позиций: цикл наблюдения каждые 5 с.

Здесь — сам цикл (`run`): цена, dynamic exit, трейлинг, частичная фиксация,
закрытие по SL/TP. Остальное разложено по ролям цепочкой классов (каждый видит
методы нижних): monitor_base (состояние, события, circuit breaker) → booking
(учёт закрытия) → closing (закрытие на бирже) → levels (SL/TP, трейлинг,
Partial TP: при 60% пути до TP закрывается 50% позиции, SL — в безубыток).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Dict

import ccxt

from bitbot.config.constants import MONITOR_MAX_ERRORS
from bitbot.domain.types import PositionRecord
from bitbot.execution.levels import LevelsMixin

logger = logging.getLogger(__name__)


class PositionMonitor(LevelsMixin):
    """
    Следит за открытыми позициями каждые 5 с;
    закрывает при срабатывании SL/TP/трейлинга.

    Хранит счётчик _consecutive_losses для circuit breaker — изоляция от TradingBot
    позволяет сбрасывать счётчик в тестах без перестройки всего бота.
    """

    async def run(
        self,
        is_running: Callable[[], bool],
        monitored: Dict[str, PositionRecord],
        lock: asyncio.Lock,
    ) -> None:
        """
        Цикл опроса: проверяет каждый символ каждые 5 с пока is_running() == True.

        :param is_running: Функция возвращающая True пока бот должен работать.
        :param monitored: Словарь открытых позиций {symbol: PositionRecord}.
        :param lock: asyncio.Lock для потокобезопасного доступа к monitored.
        """
        error_counts: Dict[str, int] = {}
        while is_running():
            async with lock:
                snapshot = list(monitored.items())
            for sym, pos in snapshot:
                # None — placeholder OrderExecutor'а пока ожидается подтверждение.
                if pos is None:
                    continue
                try:
                    price = await self._api.get_current_price(sym)
                    if not price:
                        continue
                    new_pos = self._apply_trailing_stop(sym, pos, price)
                    if new_pos is not pos:
                        async with lock:
                            if sym in monitored and monitored[sym] is not None:
                                monitored[sym] = new_pos
                        pos = new_pos
                    # УЛУЧШЕНИЕ 3: частичная фиксация прибыли
                    pos = await self._check_partial_tp(sym, pos, price, monitored, lock)
                    self._publish_tick(sym, pos, price)
                    try:
                        exit_action, reason = self._check_dynamic_exit(sym, pos)
                        if exit_action == "force_close":
                            # Экстремальные условия — закрываем сразу
                            self._exit_confirm_counts.pop(sym, None)
                            logger.warning(
                                "Dynamic exit (force) %s [%s]: %s",
                                sym,
                                pos.get("side"),
                                reason,
                            )
                            await self._force_close_at_market(
                                sym, pos, price, monitored, lock, reason
                            )
                            continue
                        if exit_action == "tighten_sl":
                            count = self._exit_confirm_counts.get(sym, 0) + 1
                            self._exit_confirm_counts[sym] = count
                            if count >= self._EXIT_CONFIRM_THRESHOLD:
                                # Режим стабильно против позиции — закрываем
                                self._exit_confirm_counts.pop(sym, None)
                                logger.warning(
                                    "Dynamic exit (confirmed x%d) %s [%s]: %s",
                                    count,
                                    sym,
                                    pos.get("side"),
                                    reason,
                                )
                                await self._force_close_at_market(
                                    sym, pos, price, monitored, lock, reason
                                )
                                continue
                            # Первые N-1 циклов — только подтягиваем SL
                            logger.warning(
                                "Dynamic exit (tighten SL %d/%d) %s [%s]: %s",
                                count,
                                self._EXIT_CONFIRM_THRESHOLD,
                                sym,
                                pos.get("side"),
                                reason,
                            )
                            pos = await self._tighten_sl(
                                sym, pos, price, monitored, lock
                            )
                        else:
                            # Условие исчезло — сбрасываем счётчик
                            self._exit_confirm_counts.pop(sym, None)
                    except Exception as _dyn_exc:
                        logger.error(
                            "Dynamic exit check failed for %s: %s", sym, _dyn_exc
                        )
                    await self._check_and_close(sym, pos, price, monitored, lock)
                    error_counts.pop(sym, None)
                except (ccxt.NetworkError, ccxt.RequestTimeout) as exc:
                    # Transient — retry next cycle without alarming
                    count = error_counts.get(sym, 0) + 1
                    error_counts[sym] = count
                    logger.warning(
                        "Monitor network error %s (attempt %d/%d): %s",
                        sym,
                        count,
                        MONITOR_MAX_ERRORS,
                        exc,
                    )
                    if count >= MONITOR_MAX_ERRORS:
                        logger.error(
                            "Dropping %s from monitor after %d network errors — "
                            "will be reconciled on next restart",
                            sym,
                            count,
                        )
                        if self._telegram:
                            msg = (
                                f"⚠️ *{sym}* удалён из мониторинга"
                                f" после {count} ошибок. Проверь вручную!"
                            )
                            _t = asyncio.create_task(self._telegram.notify(msg))
                            self._background_tasks.add(_t)
                            _t.add_done_callback(self._background_tasks.discard)
                        async with lock:
                            monitored.pop(sym, None)
                        error_counts.pop(sym, None)
                except Exception as exc:
                    # Broad catch here is intentional: this loop must never
                    # crash the whole monitor task due to one bad symbol.
                    count = error_counts.get(sym, 0) + 1
                    error_counts[sym] = count
                    logger.error(
                        "Monitor error %s (attempt %d/%d): %s",
                        sym,
                        count,
                        MONITOR_MAX_ERRORS,
                        exc,
                    )
                    if count >= MONITOR_MAX_ERRORS:
                        logger.warning(
                            "Removing %s from monitor after %d errors",
                            sym,
                            count,
                        )
                        if self._telegram:
                            msg = (
                                f"⚠️ *{sym}* удалён из мониторинга"
                                f" после {count} ошибок. Проверь вручную!"
                            )
                            _t = asyncio.create_task(self._telegram.notify(msg))
                            self._background_tasks.add(_t)
                            _t.add_done_callback(self._background_tasks.discard)
                        async with lock:
                            monitored.pop(sym, None)
                        error_counts.pop(sym, None)
            await asyncio.sleep(5)
