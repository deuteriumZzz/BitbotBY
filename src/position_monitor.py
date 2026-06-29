"""
Фоновый монитор позиций: SL/TP, трейлинг-стоп, circuit breaker.

Вынесен из TradingBot чтобы отделить задачу «следить за открытыми
позициями» от основного торгового цикла. TradingBot делегирует
_monitor_positions() сюда; публичная сигнатура метода не изменилась,
поэтому существующие тесты проходят.

УЛУЧШЕНИЕ 3: частичная фиксация прибыли (Partial TP) — при достижении
60% пути до TP закрывается 50% позиции и SL переносится на breakeven.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Callable, Dict, Optional

import ccxt

from config import Config
from src.constants import MONITOR_MAX_ERRORS
from src.types import PositionRecord

if TYPE_CHECKING:
    from src.bybit_api import BybitAPI
    from src.online_learner import OnlineLearner
    from src.portfolio_manager import PortfolioManager
    from src.redis_client import RedisClient
    from src.telegram_notifier import TelegramNotifier
    from src.trade_history import TradeHistory

logger = logging.getLogger(__name__)


class PositionMonitor:
    """
    Следит за открытыми позициями каждые 5 с;
    закрывает при срабатывании SL/TP/трейлинга.

    Хранит счётчик _consecutive_losses для circuit breaker — изоляция от TradingBot
    позволяет сбрасывать счётчик в тестах без перестройки всего бота.
    """

    def __init__(
        self,
        api: "BybitAPI",
        trade_history: "TradeHistory",
        telegram: "TelegramNotifier",
        portfolio_manager: "PortfolioManager",
        set_running: Callable[[bool], None],
        redis: "Optional[RedisClient]" = None,
        online_learner: "Optional[OnlineLearner]" = None,
    ) -> None:
        self._api = api
        self._trade_history = trade_history
        self._telegram = telegram
        self._portfolio_manager = portfolio_manager
        # Callback instead of a direct TradingBot reference so the circuit
        # breaker can halt trading without a circular import.
        self._set_running = set_running
        self._redis = redis
        self._online_learner = online_learner
        # Load persisted counter so circuit breaker survives bot restarts.
        self._consecutive_losses: int = self._load_cb_state()
        # State for dynamic exits — updated each cycle via update_market_state().
        self._current_signals: dict = {}
        self._current_market_ctx: dict = {}
        self._current_regime: str = "unknown"
        # Счётчик подтверждений для regime_flip/funding: {symbol: count}
        # При N=3 подряд — escalate to force_close
        self._exit_confirm_counts: Dict[str, int] = {}
        self._EXIT_CONFIRM_THRESHOLD = 3
        # Keeps strong references to fire-and-forget tasks so GC can't destroy them.
        self._background_tasks: set = set()

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
                    try:
                        exit_action, reason = self._check_dynamic_exit(sym, pos)
                        if exit_action == "force_close":
                            # Экстремальные условия — закрываем сразу
                            self._exit_confirm_counts.pop(sym, None)
                            logger.warning(
                                "Dynamic exit (force) %s [%s]: %s",
                                sym, pos.get("side"), reason,
                            )
                            await self._force_close_at_market(
                                sym, pos, price, monitored, lock, reason
                            )
                            continue
                        elif exit_action == "tighten_sl":
                            count = self._exit_confirm_counts.get(sym, 0) + 1
                            self._exit_confirm_counts[sym] = count
                            if count >= self._EXIT_CONFIRM_THRESHOLD:
                                # Режим стабильно против позиции — закрываем
                                self._exit_confirm_counts.pop(sym, None)
                                logger.warning(
                                    "Dynamic exit (confirmed x%d) %s [%s]: %s",
                                    count, sym, pos.get("side"), reason,
                                )
                                await self._force_close_at_market(
                                    sym, pos, price, monitored, lock, reason
                                )
                                continue
                            else:
                                # Первые N-1 циклов — только подтягиваем SL
                                logger.warning(
                                    "Dynamic exit (tighten SL %d/%d) %s [%s]: %s",
                                    count, self._EXIT_CONFIRM_THRESHOLD,
                                    sym, pos.get("side"), reason,
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

    # ── Dynamic exit state ────────────────────────────────────────────────────

    def update_market_state(
        self,
        signals: dict,
        market_ctx: dict,
        regime: str,
    ) -> None:
        """Обновляет рыночное состояние для динамических выходов.

        Вызывается trading_bot каждый цикл после генерации рекомендаций.
        """
        self._current_signals = signals
        self._current_market_ctx = market_ctx
        self._current_regime = regime

    def _check_dynamic_exit(self, sym: str, record: PositionRecord) -> tuple:
        """Проверяет условия досрочного выхода из позиции.

        :return: (action: str | None, reason: str)
            action="force_close" — закрыть по рынку немедленно (сильный сигнал)
            action="tighten_sl"  — подтянуть SL ближе (дать шанс на отскок)
            action=None          — ничего не делать
        """
        if not self._current_signals and not self._current_market_ctx:
            return None, ""

        side = record.get("side", "buy")
        ctx = self._current_market_ctx.get(sym, {})
        signal = self._current_signals.get(sym, {})

        # 1. Разворот сигнала с высокой уверенностью → закрыть сразу
        sig_action = signal.get("action", "")
        sig_conf = float(signal.get("confidence", 0.0))
        if sig_action and sig_conf >= 0.80:
            if side == "buy" and sig_action == "sell":
                return "force_close", f"signal_reversal SELL conf={sig_conf:.2f}"
            if side == "sell" and sig_action == "buy":
                return "force_close", f"signal_reversal BUY conf={sig_conf:.2f}"

        # 2. Смена режима против позиции → подтянуть SL (цена может отскочить)
        regime = self._current_regime
        if side == "buy" and regime == "trending_down":
            return "tighten_sl", "regime_flip trending_down vs LONG"
        if side == "sell" and regime == "trending_up":
            return "tighten_sl", "regime_flip trending_up vs SHORT"

        # 3. Funding развернулся против позиции → подтянуть SL
        funding_signal = ctx.get("funding_signal", "neutral")
        if side == "sell" and funding_signal == "short_overheated":
            return "tighten_sl", "funding short_overheated — short squeeze risk"
        if side == "buy" and funding_signal == "long_overheated":
            return "tighten_sl", "funding long_overheated — long liquidation risk"

        # 4. Экстремальный Fear&Greed → закрыть сразу (фиксируем прибыль)
        fear_greed = int(ctx.get("fear_greed", 50))
        if side == "sell" and fear_greed <= 10:
            return "force_close", f"fear_greed={fear_greed} extreme_fear"
        if side == "buy" and fear_greed >= 92:
            return "force_close", f"fear_greed={fear_greed} extreme_greed"

        # 5. Ликвидационный каскад → закрыть сразу
        liq = ctx.get("liquidation_pressure", "neutral")
        if side == "buy" and liq == "long_liquidation":
            return "force_close", "long_liquidation cascade — exit LONG"
        if side == "sell" and liq == "short_squeeze":
            return "force_close", "short_squeeze cascade — exit SHORT"

        return None, ""

    # ── Emergency close ───────────────────────────────────────────────────────

    async def close_all_positions(
        self,
        monitored: Dict[str, PositionRecord],
        lock: asyncio.Lock,
        reason: str = "emergency",
    ) -> None:
        """Принудительно закрывает все открытые позиции по рыночной цене.

        Используется при срабатывании защитных механизмов (дневной лимит,
        hard drawdown halt, ATR spike). Не требует достижения SL/TP.
        """
        async with lock:
            snapshot = list(monitored.items())
        for sym, pos in snapshot:
            if pos is None:
                continue
            side = pos.get("side", "buy")
            qty = pos.get("qty", 0.0)
            if qty <= 0:
                continue
            close_side = "sell" if side == "buy" else "buy"
            logger.warning(
                "Emergency close %s qty=%.6f side=%s reason=%s",
                sym,
                qty,
                side,
                reason,
            )
            if not Config.PAPER_TRADING:
                for oid in filter(
                    None, [pos.get("exchange_sl_id"), pos.get("exchange_tp_id")]
                ):
                    try:
                        await self._api.cancel_order(sym, oid)
                    except Exception as _e:
                        logger.debug("Cancel order %s failed: %s", oid, _e)
                await self._api.create_order(
                    sym, "market", close_side, qty, lock_suffix="close"
                )
            async with lock:
                monitored.pop(sym, None)
            logger.info("Emergency closed %s [%s]", sym, reason)

    async def close_losers_tighten_winners(
        self,
        monitored: Dict[str, PositionRecord],
        lock: asyncio.Lock,
        current_prices: Dict[str, float],
        reason: str = "hard_drawdown",
    ) -> None:
        """Hard drawdown: закрывает убыточные позиции, у прибыльных SL на breakeven.

        Убыточные (current < entry для buy, current > entry для sell) —
        закрываем по рынку, они и есть причина просадки.
        Прибыльные — переносим SL на цену входа (breakeven): позиция не может
        уйти в минус, но возьмёт профит если рынок продолжит движение в плюс.
        """
        async with lock:
            snapshot = list(monitored.items())
        for sym, pos in snapshot:
            if pos is None:
                continue
            qty = pos.get("qty", 0.0)
            if qty <= 0:
                continue
            side = pos.get("side", "buy")
            entry = pos.get("entry", 0.0)
            current = current_prices.get(sym, 0.0)
            if current <= 0 or entry <= 0:
                continue
            close_side = "sell" if side == "buy" else "buy"
            is_losing = (side == "buy" and current < entry) or (
                side == "sell" and current > entry
            )
            if is_losing:
                logger.warning(
                    "Drawdown close loser %s [%s] entry=%.4f current=%.4f reason=%s",
                    sym,
                    side,
                    entry,
                    current,
                    reason,
                )
                if not Config.PAPER_TRADING:
                    for oid in filter(
                        None, [pos.get("exchange_sl_id"), pos.get("exchange_tp_id")]
                    ):
                        try:
                            await self._api.cancel_order(sym, oid)
                        except Exception as _e:
                            logger.debug("Cancel order %s failed: %s", oid, _e)
                    await self._api.create_order(
                        sym, "market", close_side, qty, lock_suffix="close"
                    )
                async with lock:
                    monitored.pop(sym, None)
            else:
                # Прибыльная позиция — переносим SL на breakeven
                tp_price = float(pos.get("take_profit") or 0)
                logger.info(
                    "Drawdown tighten winner %s [%s] entry=%.4f → SL=breakeven",
                    sym,
                    side,
                    entry,
                )
                if not Config.PAPER_TRADING:
                    for oid in filter(
                        None, [pos.get("exchange_sl_id"), pos.get("exchange_tp_id")]
                    ):
                        try:
                            await self._api.cancel_order(sym, oid)
                        except Exception as _e:
                            logger.debug("Cancel SL/TP %s failed: %s", oid, _e)
                    try:
                        new_sl_id, new_tp_id = await self._api.place_exchange_sl_tp(
                            sym, close_side, qty, entry, tp_price
                        )
                        async with lock:
                            if sym in monitored and monitored[sym] is not None:
                                monitored[sym]["stop_loss"] = entry
                                monitored[sym]["exchange_sl_id"] = new_sl_id
                                monitored[sym]["exchange_tp_id"] = new_tp_id
                    except Exception as exc:
                        logger.warning("Tighten SL for %s failed: %s", sym, exc)
                else:
                    async with lock:
                        if sym in monitored and monitored[sym] is not None:
                            monitored[sym]["stop_loss"] = entry
                await self._telegram.notify(
                    f"🔒 *{sym}* [{side}]: SL перенесён на breakeven"
                    f" ${entry:.4f} (hard drawdown protection)"
                )

    # ── Circuit-breaker persistence ───────────────────────────────────────────

    def _load_cb_state(self) -> int:
        """Загружает счётчик подряд идущих убытков из Redis (0 если недоступен)."""
        if self._redis is None:
            return 0
        state = self._redis.load_trading_state("circuit_breaker")
        return int((state or {}).get("consecutive_losses", 0))

    def _save_cb_state(self) -> None:
        """Сохраняет счётчик в Redis чтобы пережить рестарт бота."""
        if self._redis is not None:
            self._redis.save_trading_state(
                "circuit_breaker",
                {"consecutive_losses": self._consecutive_losses},
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    # ── Partial TP (УЛУЧШЕНИЕ 3) ─────────────────────────────────────────────

    async def _check_partial_tp(
        self,
        sym: str,
        pos: PositionRecord,
        price: float,
        monitored: Dict[str, PositionRecord],
        lock: asyncio.Lock,
    ) -> PositionRecord:
        """Частичная фиксация прибыли при достижении PARTIAL_TP_TRIGGER пути к TP.

        При срабатывании:
          - закрывается PARTIAL_TP_FRACTION позиции (paper: только обновление qty)
          - SL переносится на breakeven (entry price)
          - выставляется флаг partial_tp_triggered = True

        :return: Обновлённая запись позиции.
        """
        try:
            enabled = bool(Config.PARTIAL_TP_ENABLED)
            trigger = float(Config.PARTIAL_TP_TRIGGER)
            fraction = float(Config.PARTIAL_TP_FRACTION)
        except (TypeError, ValueError):
            return pos
        if not enabled:
            return pos
        if pos.get("partial_tp_triggered", False):
            return pos

        side = pos.get("side", "buy")
        entry = pos.get("entry", 0.0)
        tp = pos.get("take_profit", 0.0)
        qty = pos.get("qty", 0.0)

        if not tp or not entry or not qty:
            return pos

        # Вычисляем прогресс цены к TP
        if side == "buy":
            progress = (price - entry) / (tp - entry) if tp > entry else 0.0
        else:
            progress = (entry - price) / (entry - tp) if entry > tp else 0.0

        if progress < trigger:
            return pos

        partial_qty = qty * fraction
        new_qty = qty * (1.0 - fraction)

        close_side = "sell" if side == "buy" else "buy"
        new_sl_id: str | None = None
        new_tp_id: str | None = None
        if not Config.PAPER_TRADING:
            # Отменяем биржевые SL/TP перед частичным закрытием —
            # иначе биржа попытается закрыть полный объём при достижении TP.
            exchange_sl_id = pos.get("exchange_sl_id")
            exchange_tp_id = pos.get("exchange_tp_id")
            for oid, label in ((exchange_sl_id, "SL"), (exchange_tp_id, "TP")):
                if oid:
                    try:
                        await self._api.cancel_order(sym, oid)
                    except Exception as exc:
                        logger.warning(
                            "Partial TP: cancel %s %s failed: %s", label, oid, exc
                        )
            try:
                await self._api.create_order(
                    sym, "market", close_side, partial_qty, lock_suffix="partial_tp"
                )
            except Exception as exc:
                logger.error("Partial TP order failed for %s: %s", sym, exc)
                # SL/TP уже отменены — восстанавливаем защиту для полного объёма
                sl_orig = float(pos.get("stop_loss") or 0)
                tp_orig = float(pos.get("take_profit") or 0)
                if sl_orig or tp_orig:
                    try:
                        r_sl, r_tp = await self._api.place_exchange_sl_tp(
                            sym, close_side, qty, sl_orig, tp_orig
                        )
                        async with lock:
                            if sym in monitored and monitored[sym] is not None:
                                monitored[sym]["exchange_sl_id"] = r_sl
                                monitored[sym]["exchange_tp_id"] = r_tp
                    except Exception as re_exc:
                        logger.critical(
                            "Partial TP: restore SL/TP for %s failed: %s", sym, re_exc
                        )
                return pos
            # Выставляем новые SL (breakeven) и TP для оставшегося объёма
            tp_price = float(pos.get("take_profit") or 0)
            if entry or tp_price:
                try:
                    new_sl_id, new_tp_id = await self._api.place_exchange_sl_tp(
                        sym, close_side, new_qty, entry, tp_price
                    )
                except Exception as exc:
                    logger.warning(
                        "Partial TP: re-place SL/TP for %s failed: %s", sym, exc
                    )

        async with lock:
            if sym in monitored and monitored[sym] is not None:
                monitored[sym]["qty"] = new_qty
                monitored[sym]["stop_loss"] = entry  # breakeven
                monitored[sym]["partial_tp_triggered"] = True
                if not Config.PAPER_TRADING:
                    monitored[sym]["exchange_sl_id"] = new_sl_id
                    monitored[sym]["exchange_tp_id"] = new_tp_id

        logger.info(
            "Partial TP triggered for %s: closed %.1f%% (%.6f units),"
            " SL moved to breakeven %.4f",
            sym,
            fraction * 100,
            partial_qty,
            entry,
        )
        await self._telegram.notify(
            f"Partial TP *{sym}*: closed {fraction:.0%},"
            f" SL → breakeven ${entry:.4f}"
        )
        return {  # type: ignore[return-value]
            **pos,
            "qty": new_qty,
            "stop_loss": entry,
            "partial_tp_triggered": True,
        }

    def _apply_trailing_stop(
        self,
        sym: str,
        pos: PositionRecord,
        price: float,
    ) -> PositionRecord:
        """
        Подтягивает SL к текущей цене на шаг ATR × multiplier.

        ATR-based шаг адаптируется к волатильности: узкий на спокойном рынке
        (меньше преждевременных стопов), широкий на волатильном (меньше шумовых).

        Returns a new pos dict if changed, same pos object if unchanged.
        The caller must write the result to monitored under the lock.

        :param sym: Символ позиции.
        :param pos: Текущая запись позиции.
        :param price: Текущая рыночная цена.
        :return: Обновлённая запись позиции (новый объект если изменилась).
        """
        atr = pos.get("atr", 0.0)
        mult = Config.TRAILING_STOP_ATR_MULT
        if not (atr and atr > 0 and mult > 0):
            return pos

        side = pos.get("side", "buy")
        current_sl = pos.get("stop_loss", 0.0)

        if side == "buy":
            trail = price - atr * mult
            if trail > current_sl:
                return {**pos, "stop_loss": trail}  # type: ignore[return-value]
        else:
            trail = price + atr * mult
            if trail < pos.get("stop_loss", float("inf")):
                return {**pos, "stop_loss": trail}  # type: ignore[return-value]

        return pos

    async def _tighten_sl(
        self,
        sym: str,
        pos: PositionRecord,
        price: float,
        monitored: Dict[str, PositionRecord],
        lock: asyncio.Lock,
    ) -> PositionRecord:
        """Подтягивает SL к текущей цене на 1.5×ATR при смене режима.

        Не закрывает позицию сразу — даёт шанс на отскок, но ограничивает
        дальнейший убыток. Если новый SL хуже текущего — не трогаем.
        """
        side = pos.get("side", "buy")
        current_sl = pos.get("stop_loss", 0.0)
        atr = pos.get("atr") or 0.0
        step = atr * 1.5 if atr > 0 else price * 0.015  # fallback 1.5%

        if side == "buy":
            new_sl = price - step
            if current_sl and new_sl <= current_sl:
                return pos  # уже тише — не двигаем
        else:
            new_sl = price + step
            if current_sl and new_sl >= current_sl:
                return pos

        logger.info(
            "Tighten SL %s [%s]: %.4f → %.4f (price=%.4f atr=%.4f)",
            sym, side, current_sl, new_sl, price, atr,
        )
        async with lock:
            if sym in monitored and monitored[sym] is not None:
                monitored[sym] = {**monitored[sym], "stop_loss": new_sl}
        return {**pos, "stop_loss": new_sl}  # type: ignore[return-value]

    async def _force_close_at_market(
        self,
        sym: str,
        pos: PositionRecord,
        price: float,
        monitored: Dict[str, PositionRecord],
        lock: asyncio.Lock,
        reason: str = "dynamic_exit",
    ) -> None:
        """Принудительно закрывает позицию по рынку независимо от SL/TP.

        Используется при срабатывании dynamic exit (regime_flip, signal_reversal
        и т.д.) — когда нужно выйти немедленно, не дожидаясь уровней SL/TP.
        """
        side = pos.get("side", "buy")
        qty = pos.get("qty", 0.0)
        if not qty:
            async with lock:
                monitored.pop(sym, None)
            return

        close_side = "sell" if side == "buy" else "buy"
        logger.info(
            "Force-close %s [%s] qty=%.6f price=%.4f reason=%s",
            sym, side, qty, price, reason,
        )

        if not Config.PAPER_TRADING:
            sl_tp_ids = [pos.get("exchange_sl_id"), pos.get("exchange_tp_id")]
            for oid in filter(None, sl_tp_ids):
                try:
                    await self._api.cancel_order(sym, oid)
                except Exception as _e:
                    logger.debug("Cancel order %s failed: %s", oid, _e)
            close_order = await self._api.create_order(
                sym, "market", close_side, qty, lock_suffix="close"
            )
            if close_order is None:
                logger.error(
                    "Force-close order failed for %s — position left open", sym
                )
                return

        await self._portfolio_manager.update_portfolio(
            sym, close_side, qty, price
        )
        async with lock:
            monitored.pop(sym, None)

        entry = pos.get("entry", price)
        if entry:
            pnl_pct = (price - entry) / entry if side == "buy" else (entry - price) / entry
        else:
            pnl_pct = 0.0
        pnl_sign = "+" if pnl_pct >= 0 else ""

        if pos.get("snap") and self._online_learner:
            strategy = (pos.get("snap") or {}).get("strategy", "unknown")
            await self._online_learner.on_trade_closed(sym, side, pnl_pct, strategy)

        trade_id = pos.get("trade_id")
        if trade_id:
            exit_commission = qty * price * Config.COMMISSION_RATE
            try:
                await self._trade_history.record_close(
                    trade_id=trade_id,
                    exit_price=price,
                    commission=exit_commission,
                )
            except Exception as exc:
                logger.error("record_close for trade %s failed: %s", trade_id, exc)

        stats = await self._trade_history.get_summary()
        await self._telegram.notify(
            f"{'📈' if pnl_pct >= 0 else '📉'} Force-closed *{sym}* @ ${price:.4f}"
            f" ({pnl_sign}{pnl_pct*100:.2f}%) — {reason}\n"
            f"Win Rate: {stats['win_rate']:.0%}  Total PnL: ${stats['total_pnl']:+.2f}"
        )
        self._update_circuit_breaker(side, price, entry)

    async def _check_and_close(
        self,
        sym: str,
        pos: PositionRecord,
        price: float,
        monitored: Dict[str, PositionRecord],
        lock: asyncio.Lock,
    ) -> bool:
        """
        Закрывает позицию если цена пересекла SL или TP.

        :param sym: Символ позиции.
        :param pos: Запись позиции.
        :param price: Текущая цена.
        :param monitored: Словарь открытых позиций.
        :param lock: Lock для изменения monitored.
        :return: True если позиция была закрыта.
        """
        sl = pos.get("stop_loss", 0.0)
        tp = pos.get("take_profit", 0.0)
        side = pos.get("side", "buy")
        qty = pos.get("qty", 0.0)

        triggered = False
        if side == "buy":
            if sl and price <= sl:
                logger.warning("SL hit %s: price=%.4f <= sl=%.4f", sym, price, sl)
                triggered = True
            elif tp and price >= tp:
                logger.info("TP hit %s: price=%.4f >= tp=%.4f", sym, price, tp)
                triggered = True
        else:
            if sl and price >= sl:
                logger.warning("SL hit %s: price=%.4f >= sl=%.4f", sym, price, sl)
                triggered = True
            elif tp and price <= tp:
                logger.info("TP hit %s: price=%.4f <= tp=%.4f", sym, price, tp)
                triggered = True

        if not triggered:
            return False

        close_side = "sell" if side == "buy" else "buy"
        # Default exit commission — overridden with actual fee in live mode.
        exit_commission = qty * price * Config.COMMISSION_RATE
        if not Config.PAPER_TRADING:
            # Шаг 1: сначала отменяем биржевые условные ордера, чтобы избежать
            # двойного закрытия (race condition между SL биржи и программным закрытием).
            exchange_sl_id = pos.get("exchange_sl_id")
            exchange_tp_id = pos.get("exchange_tp_id")
            if exchange_sl_id:
                try:
                    await self._api.cancel_order(sym, exchange_sl_id)
                except Exception as _ce:
                    logger.warning(
                        "Cancel SL order %s failed: %s — proceeding with close",
                        exchange_sl_id,
                        _ce,
                    )
            if exchange_tp_id:
                try:
                    await self._api.cancel_order(sym, exchange_tp_id)
                except Exception as _ce:
                    logger.warning(
                        "Cancel TP order %s failed: %s — proceeding with close",
                        exchange_tp_id,
                        _ce,
                    )

            # Шаг 2: программное закрытие позиции (lock_suffix="close" не конкурирует
            # с order_open:{sym} при открытии новой позиции).
            close_order = await self._api.create_order(
                sym, "market", close_side, qty, lock_suffix="close"
            )

            if close_order is None:
                # Закрытие не удалось после всех повторов — восстанавливаем биржевую
                # защиту (SL/TP) и оставляем позицию в мониторе для повторной попытки.
                logger.critical(
                    "Программное закрытие %s не удалось — восстанавливаем"
                    " биржевые SL/TP, позиция остаётся в мониторе",
                    sym,
                )
                if sl or tp:
                    re_sl_id, re_tp_id = await self._api.place_exchange_sl_tp(
                        sym, close_side, qty, sl, tp
                    )
                    async with lock:
                        if sym in monitored and monitored[sym] is not None:
                            monitored[sym]["exchange_sl_id"] = re_sl_id
                            monitored[sym]["exchange_tp_id"] = re_tp_id
                return False

            # Используем реальную комиссию из ответа биржи (зависит от уровня аккаунта).
            fee_cost = float((close_order.get("fee") or {}).get("cost") or 0)
            if fee_cost > 0:
                exit_commission = fee_cost

        await self._portfolio_manager.update_portfolio(sym, close_side, qty, price)
        async with lock:
            monitored.pop(sym, None)
        logger.info("Position closed: %s at %.4f", sym, price)

        # Save experience for the next SAC retraining run
        entry = pos.get("entry", price)
        if side == "buy":
            pnl_pct = (price - entry) / entry if entry else 0.0
        else:
            pnl_pct = (entry - price) / entry if entry else 0.0

        if pos.get("snap"):
            from src.experience_buffer import save as _exp_save  # noqa: PLC0415

            _snap = pos["snap"]
            _side = side
            _entry = entry
            _price = price
            _profile = os.environ.get("SAC_PROFILE", "")
            _exp_path = (
                "data/experiences_altcoin.jsonl"
                if _profile == "altcoin"
                else "data/experiences.jsonl"
            )
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: _exp_save(
                    snap=_snap,
                    action=_side,
                    entry_price=_entry,
                    exit_price=_price,
                    path=_exp_path,
                ),
            )

        if self._online_learner:
            strategy = (pos.get("snap") or {}).get("strategy", "unknown")
            await self._online_learner.on_trade_closed(sym, side, pnl_pct, strategy)

        trade_id = pos.get("trade_id")
        if trade_id:
            try:
                await self._trade_history.record_close(
                    trade_id=trade_id,
                    exit_price=price,
                    commission=exit_commission,
                )
            except Exception as exc:
                # Позиция уже удалена из монитора и закрыта на бирже.
                # Логируем потерю записи для ручной сверки — повторная попытка
                # закрыть ту же позицию хуже, чем потеря одной строки в истории.
                logger.error(
                    "record_close для trade %s не удалась: %s"
                    " — нужна ручная сверка trade_history",
                    trade_id,
                    exc,
                )

        stats = await self._trade_history.get_summary()
        pnl_sign = "+" if pnl_pct >= 0 else ""
        try:
            bal_val = float(self._portfolio_manager.current_balance)
            bal_line = f"\n💰 Баланс: ${bal_val:,.2f}" if Config.PAPER_TRADING else ""
        except (TypeError, ValueError):
            bal_line = ""
        await self._telegram.notify(
            f"{'📈' if pnl_pct >= 0 else '📉'} Closed *{sym}* @ ${price:.4f}"
            f"  ({pnl_sign}{pnl_pct*100:.2f}%)\n"
            f"Trades: {stats['closed_trades']}  "
            f"Win Rate: {stats['win_rate']:.0%}  "
            f"Total PnL: ${stats['total_pnl']:+.2f}"
            f"{bal_line}"
        )

        self._update_circuit_breaker(side, price, pos.get("entry", price))
        return True

    def _update_circuit_breaker(
        self,
        side: str,
        exit_price: float,
        entry_price: float,
    ) -> None:
        """
        Останавливает торговлю после N последовательных убыточных сделок.

        Серия убытков (например, flash crash) сигнализирует о сломанном
        рыночном режиме. Случайные убытки вперемешку с прибылями — норма.

        :param side: Направление закрытой сделки ("buy" или "sell").
        :param exit_price: Цена закрытия.
        :param entry_price: Цена открытия.
        """
        cb = Config.CIRCUIT_BREAKER_LOSSES
        if cb <= 0:
            return

        is_loss = (side == "buy" and exit_price < entry_price) or (
            side == "sell" and exit_price > entry_price
        )
        if is_loss:
            self._consecutive_losses += 1
            logger.warning(
                "Circuit breaker: %d/%d consecutive losses",
                self._consecutive_losses,
                cb,
            )
            self._save_cb_state()
            if self._consecutive_losses >= cb:
                if Config.PAPER_TRADING:
                    msg = (
                        f"⚠️ Circuit breaker: {self._consecutive_losses}"
                        " убытков подряд. В paper режиме торговля продолжается."
                    )
                else:
                    msg = (
                        f"⛔ Circuit breaker: {self._consecutive_losses}"
                        " losses in a row. Trading halted automatically."
                    )
                    self._set_running(False)
                logger.critical(msg)
                _t = asyncio.create_task(self._telegram.notify(msg))
                self._background_tasks.add(_t)
                _t.add_done_callback(self._background_tasks.discard)
        else:
            self._consecutive_losses = 0
            self._save_cb_state()
