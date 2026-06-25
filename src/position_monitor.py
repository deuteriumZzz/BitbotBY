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
        # State for dynamic exits — updated each cycle via update_market_state().  # noqa: E501
        self._current_signals: dict = {}
        self._current_market_ctx: dict = {}
        self._current_regime: str = "unknown"
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
                        should_exit, reason = self._check_dynamic_exit(sym, pos)
                        if should_exit:
                            logger.warning(
                                "Dynamic exit %s [%s]: %s", sym, pos.get("side"), reason
                            )
                            await self._check_and_close(
                                sym, pos, price, monitored, lock
                            )
                            continue
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
        """Проверяет условия досрочного закрытия позиции.

        :return: (should_exit: bool, reason: str).
        """
        if not self._current_signals and not self._current_market_ctx:
            return False, ""

        side = record.get("side", "buy")
        ctx = self._current_market_ctx.get(sym, {})
        signal = self._current_signals.get(sym, {})

        # 1. Разворот сигнала с высокой уверенностью
        sig_action = signal.get("action", "")
        sig_conf = float(signal.get("confidence", 0.0))
        if sig_action and sig_conf >= 0.72:
            if side == "buy" and sig_action == "sell":
                return True, f"signal_reversal SELL conf={sig_conf:.2f}"
            if side == "sell" and sig_action == "buy":
                return True, f"signal_reversal BUY conf={sig_conf:.2f}"

        # 2. Смена режима против позиции
        regime = self._current_regime
        if side == "buy" and regime == "trending_down":
            return True, "regime_flip trending_down vs LONG"
        if side == "sell" and regime == "trending_up":
            return True, "regime_flip trending_up vs SHORT"

        # 3. Funding развернулся против позиции
        funding_signal = ctx.get("funding_signal", "neutral")
        if side == "sell" and funding_signal == "short_overheated":
            return True, "funding short_overheated — short squeeze risk"
        if side == "buy" and funding_signal == "long_overheated":
            return True, "funding long_overheated — long liquidation risk"

        # 4. Экстремальный Fear&Greed
        fear_greed = int(ctx.get("fear_greed", 50))
        if side == "sell" and fear_greed <= 10:
            return True, f"fear_greed={fear_greed} extreme_fear — take SHORT profit"
        if side == "buy" and fear_greed >= 92:
            return True, f"fear_greed={fear_greed} extreme_greed — take LONG profit"

        # 5. Ликвидационный каскад против позиции
        liq = ctx.get("liquidation_pressure", "neutral")
        if side == "buy" and liq == "long_liquidation":
            return True, "long_liquidation cascade — exit LONG"
        if side == "sell" and liq == "short_squeeze":
            return True, "short_squeeze cascade — exit SHORT"

        return False, ""

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
                f"data/experiences_{_profile}.jsonl"
                if _profile
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
