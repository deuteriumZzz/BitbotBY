"""
Команды из приложения (Telegram MiniApp / Desktop) → те же действия бота,
что выполняют кнопки Telegram. Исполняются в процессе бота через
`bot_bus.CommandConsumer`.

`bot` — TradingBot; обращаемся к тем же методам, что TelegramCommander
получает колбэками (`_handle_close_all`, `_handle_restart`,
`_handle_mode_switch`) и к RuntimeConfig.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
from typing import Any, Dict

from config import Config
from src.bot_bus import Handler, emit, error

logger = logging.getLogger(__name__)

PIN_MAX_FAILS = 3
PIN_LOCK_SEC = 60


class BotCommands:
    def __init__(self, bot: Any) -> None:
        self._bot = bot
        self._tasks: set = set()

    @property
    def _r(self) -> Any:
        return self._bot.redis.redis_client

    def handlers(self) -> Dict[str, Handler]:
        return {
            "pause": self.pause,
            "resume": self.resume,
            "panic": self.panic,
            "restart": self.restart,
            "close_position": self.close_position,
            "update_position": self.update_position,
            "signal_answer": self.signal_answer,
            "account_mode": self.account_mode,
        }

    async def pause(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self._bot._runtime_config.set_paused(True)
        emit(self._r, "cycle", {"state": "paused"})
        return {"paused": True}

    async def resume(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self._bot._runtime_config.set_paused(False)
        emit(self._r, "cycle", {"state": "scan"})
        return {"paused": False}

    async def panic(self, args: Dict[str, Any]) -> Dict[str, Any]:
        closed = await self._bot._handle_close_all()
        self._bot._runtime_config.set_paused(True)
        emit(self._r, "cycle", {"state": "paused"})
        await self._bot.telegram.notify(
            f"🛑 PANIC из приложения: закрыто позиций — {closed}, бот на паузе."
        )
        return {"closed": closed, "paused": True}

    async def restart(self, args: Dict[str, Any]) -> Dict[str, Any]:
        # _handle_restart ждёт 2 с и шлёт SIGTERM — к этому моменту результат
        # команды уже сохранён в Redis.
        task = asyncio.create_task(self._bot._handle_restart())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return {"restarting": True}

    async def _find(self, trade_id: str) -> "tuple[str, Dict[str, Any]] | None":
        bot = self._bot
        async with bot._monitored_lock:
            return next(
                (
                    (sym, pos)
                    for sym, pos in bot._monitored.items()
                    if pos is not None and str(pos.get("trade_id")) == trade_id
                ),
                None,
            )

    async def _price(self, sym: str) -> "float | None":
        return await self._bot.api.get_current_price(sym)

    async def close_position(self, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            fraction = float(args.get("fraction", 1))
        except (TypeError, ValueError):
            fraction = -1
        if not (fraction == 1 or 0.1 <= fraction <= 0.9):
            return error("bad_fraction", "Закрыть можно всю позицию или от 10% до 90%.")
        found = await self._find(str(args.get("trade_id", "")))
        if found is None:
            return error("not_found", "Позиция уже закрыта или не найдена.")
        sym, pos = found
        price = await self._price(sym)
        if not price:
            return error(
                "no_price", f"Не удалось получить цену {sym}. Попробуйте ещё раз."
            )
        bot = self._bot
        monitor = bot._position_monitor
        if fraction == 1:
            await monitor._force_close_at_market(
                sym,
                pos,
                price,
                bot._monitored,
                bot._monitored_lock,
                reason="manual_app",
            )
        else:
            updated, unprotected = await monitor.close_fraction(
                sym,
                pos,
                fraction,
                bot._monitored,
                bot._monitored_lock,
                label="partial_app",
                price=price,
            )
            if updated is None:
                return error(
                    "order_failed", "Биржа не приняла ордер. Позиция не изменилась."
                )
            emit(self._r, "position.updated", _public(sym, updated))
            if unprotected:
                bot._save_monitored_state()
                return error(
                    "unprotected",
                    "Часть закрыта, но SL/TP на остаток не выставились — "
                    "проверьте позицию на бирже.",
                )
        bot._save_monitored_state()
        return {"symbol": sym, "price": price, "fraction": fraction}

    async def update_position(self, args: Dict[str, Any]) -> Dict[str, Any]:
        found = await self._find(str(args.get("trade_id", "")))
        if found is None:
            return error("not_found", "Позиция уже закрыта или не найдена.")
        sym, pos = found
        bot = self._bot
        changed: Dict[str, Any] = {}

        if "sl" in args or "tp" in args:
            try:
                sl = float(args.get("sl", pos.get("stop_loss") or 0))
                tp = float(args.get("tp", pos.get("take_profit") or 0))
            except (TypeError, ValueError):
                return error("bad_levels", "SL и TP должны быть числами.")
            price = await self._price(sym)
            if not price:
                return error(
                    "no_price", f"Не удалось получить цену {sym}. Попробуйте ещё раз."
                )
            long = pos.get("side", "buy") == "buy"
            ok = (0 < sl < price < tp) if long else (0 < tp < price < sl)
            if not ok:
                side = "LONG" if long else "SHORT"
                where = "ниже" if long else "выше"
                return error(
                    "bad_levels",
                    f"Для {side} стоп должен быть {where} цены {price:g}, "
                    "а тейк — с другой стороны.",
                )
            try:
                pos = await bot._position_monitor.update_levels(
                    sym, pos, sl, tp, bot._monitored, bot._monitored_lock
                )
            except Exception:
                logger.exception("update_levels failed for %s", sym)
                return error(
                    "order_failed",
                    "Биржа не приняла новые SL/TP — старые остались на месте.",
                )
            changed.update(sl=sl, tp=tp)

        if "trailing" in args:
            trailing = bool(args["trailing"])
            async with bot._monitored_lock:
                if bot._monitored.get(sym) is not None:
                    bot._monitored[sym] = {**bot._monitored[sym], "trailing": trailing}
                    pos = bot._monitored[sym]
            changed["trailing"] = trailing

        if not changed:
            return error("bad_request", "Укажите sl, tp или trailing.")
        bot._save_monitored_state()
        emit(self._r, "position.updated", _public(sym, pos))
        return {"symbol": sym, **changed}

    async def signal_answer(self, args: Dict[str, Any]) -> Dict[str, Any]:
        confirmed = bool(args.get("confirm"))
        if not self._bot.telegram.resolve_pending(
            str(args.get("signal_id", "")), confirmed
        ):
            return error(
                "expired", "Предложение уже неактуально: ответ получен или время вышло."
            )
        return {"confirmed": confirmed}

    async def account_mode(self, args: Dict[str, Any]) -> Dict[str, Any]:
        paper = bool(args.get("paper"))
        if paper == Config.PAPER_TRADING:
            return {"paper": paper, "changed": False}
        if not paper:
            refused = self._check_pin(str(args.get("pin", "")))
            if refused:
                return refused
        self._bot._runtime_config.apply_paper_trading(paper)
        await self._bot._handle_mode_switch()
        await self._bot.telegram.notify(
            "💰 Из приложения включён LIVE: бот торгует реальными деньгами."
            if not paper
            else "📄 Из приложения включён PAPER: сделки снова виртуальные."
        )
        emit(self._r, "account.mode", {"paper": paper})
        return {"paper": paper, "changed": True}

    def _check_pin(self, provided: str) -> "Dict[str, Any] | None":
        """None — PIN верный. Иначе ответ об ошибке.

        Без заданного TRADING_MODE_PIN переход в Live из приложения запрещён:
        в Telegram без PIN есть отдельный экран подтверждения, здесь — нет.
        """
        pin = Config.TRADING_MODE_PIN or ""
        if not pin:
            return error(
                "pin_required",
                "Сначала задайте PIN для перехода в Live — без него переход "
                "из приложения запрещён.",
            )
        fails_key = "bot:pin_fails"
        try:
            fails = int(self._r.get(fails_key) or 0)
        except Exception:
            fails = 0
        if fails >= PIN_MAX_FAILS:
            return error("pin_locked", "Три неверных PIN подряд — подождите минуту.")
        if not hmac.compare_digest(provided.encode(), pin.encode()):
            try:
                self._r.incr(fails_key)
                self._r.expire(fails_key, PIN_LOCK_SEC)
            except Exception as e:
                logger.debug("PIN fail counter unavailable: %s", e)
            logger.warning("Wrong Live PIN from app")
            return error("wrong_pin", "Неверный PIN.")
        try:
            self._r.delete(fails_key)
        except Exception:
            pass
        return None


def _public(sym: str, pos: Dict[str, Any]) -> Dict[str, Any]:
    """Поля позиции для приложения — без служебных (snap, id ордеров)."""
    return {
        "symbol": sym,
        "trade_id": pos.get("trade_id"),
        "side": "long" if pos.get("side", "buy") == "buy" else "short",
        "entry": pos.get("entry"),
        "qty": pos.get("qty"),
        "sl": pos.get("stop_loss"),
        "tp": pos.get("take_profit"),
        "trailing": pos.get("trailing", True),
    }
