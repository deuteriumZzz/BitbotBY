"""
Telegram-бот для подтверждения сделок и отправки сигналов.

Порядок работы:
  1. Бот отправляет предложение сделки с inline-кнопками ✅/❌.
  2. Пользователь отвечает в течение CONFIRM_TIMEOUT секунд.
  3. Нет ответа → авто-исполнение (True).
  4. ❌ → пропустить сделку.

Требования: python-telegram-bot>=20.0 (async)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.ext import Application, CallbackQueryHandler, ContextTypes

    _TG_AVAILABLE = True
except ImportError:
    _TG_AVAILABLE = False
    logger.warning(
        "python-telegram-bot not installed. "
        "Run: pip install python-telegram-bot>=20.0"
    )

_CONFIRM_TIMEOUT = 60  # seconds


class TelegramNotifier:
    """
    Отправляет Telegram-уведомления и ожидает подтверждения сделки.

    Работает в деградированном режиме если токен отсутствует
    или библиотека не установлена.
    """

    def __init__(self, token: str, chat_id: str):
        self._token = token
        self._chat_id = chat_id
        self._app: Optional[Application] = None
        self._polling_task: Optional[asyncio.Task] = None
        self._pending: dict = {}  # message_id → asyncio.Event
        self._decisions: dict = {}  # message_id → bool
        self._enabled = bool(_TG_AVAILABLE and token and chat_id)

    async def start(self) -> None:
        """Инициализирует и запускает polling Telegram в фоновом режиме."""
        if not self._enabled:
            return
        self._app = Application.builder().token(self._token).build()
        self._app.add_handler(
            CallbackQueryHandler(self._handle_callback, pattern="^(confirm|reject)$")
        )
        await self._app.initialize()
        await self._app.start()
        self._polling_task = asyncio.create_task(self._app.updater.start_polling())
        logger.info("Telegram notifier started")

    async def stop(self) -> None:
        """Останавливает Telegram-приложение и отменяет polling."""
        if self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

    async def _handle_callback(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """
        Обрабатывает нажатие inline-кнопки подтверждения/отклонения сделки.

        Проверяет авторизацию пользователя по chat_id и устанавливает
        решение в словаре _decisions.
        """
        query = update.callback_query
        # C4: only the configured chat owner can approve/reject trades
        if str(query.from_user.id) != str(self._chat_id):
            await query.answer("Not authorized.")
            return
        await query.answer()
        mid = query.message.message_id
        data = query.data  # "confirm" or "reject"
        if mid in self._pending:
            self._decisions[mid] = data == "confirm"
            self._pending[mid].set()
        await query.edit_message_reply_markup(reply_markup=None)

    async def ask_confirm(
        self,
        rec: dict,
        live_win_rate: float = 0.5,
        live_trades: int = 0,
        live_ev: float = 0.0,
        bt_win_rate: float = 0.0,
        bt_trades: int = 0,
        bt_ev: float = 0.0,
        timeout: int = _CONFIRM_TIMEOUT,
        auto_execute: bool = True,
    ) -> bool:
        """
        Отправляет предложение сделки; возвращает True если
        подтверждено или истёк таймаут.

        Показывает win rate из бэктеста и лайв отдельно.
        Возвращает True немедленно если Telegram отключён.

        :param rec: Рекомендация с ключами symbol, action, confidence, entry, sl, tp.
        :param live_win_rate: Win rate по живым сделкам.
        :param live_trades: Количество живых сделок.
        :param live_ev: Expected value по живым сделкам.
        :param bt_win_rate: Win rate по бэктесту.
        :param bt_trades: Количество сделок в бэктесте.
        :param bt_ev: Expected value по бэктесту.
        :param timeout: Таймаут ожидания в секундах.
        :return: True если сделка подтверждена или автоподтверждена по таймауту.
        """
        if not self._enabled:
            return True

        symbol = rec.get("symbol", "?")
        action = rec.get("action", "?").upper()
        confidence = rec.get("confidence", 0.0)
        entry = rec.get("entry", 0.0)
        sl = rec.get("stop_loss", 0.0)
        tp = rec.get("take_profit", 0.0)
        strategy = rec.get("strategy", "?")

        if bt_trades > 0:
            bt_line = (
                f"Backtest:  "
                f"*{bt_win_rate:.0%}*  "
                f"({bt_trades} trades)  "
                f"EV: *{bt_ev*100:+.2f}%*"
            )
        else:
            bt_line = "Backtest: no data"

        if live_trades > 0:
            live_line = (
                f"Live:      "
                f"*{live_win_rate:.0%}*  "
                f"({live_trades} trades)  "
                f"EV: *{live_ev*100:+.2f}%*"
            )
        else:
            live_line = "Live: -- (no trades yet)"

        text = (
            f"*{symbol}*  --  {action}\n\n"
            f"Strategy: `{strategy}`\n"
            f"Entry: `${entry:.4f}`\n"
            f"SL: `${sl:.4f}`   "
            f"TP: `${tp:.4f}`\n\n"
            f"AI confidence: *{confidence:.0%}*\n\n"
            f"{bt_line}\n"
            f"{live_line}\n\n"
            f"⚡ Авто-исполнение через {timeout}с — нажми Skip чтобы пропустить"
        )
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "Trade",
                        callback_data="confirm",
                    ),
                    InlineKeyboardButton(
                        "Skip",
                        callback_data="reject",
                    ),
                ]
            ]
        )
        try:
            bot: Bot = self._app.bot
            msg = await bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode="Markdown",
                reply_markup=keyboard,
            )
            mid = msg.message_id
            event = asyncio.Event()
            self._pending[mid] = event
            self._decisions[mid] = True  # default: всегда авто-исполнение

            timed_out = False
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                timed_out = True
                logger.info(
                    "Telegram timeout for %s -> %s",
                    symbol,
                    "auto-execute" if auto_execute else "skip",
                )

            decision = self._decisions.pop(mid, auto_execute)
            self._pending.pop(mid, None)

            if timed_out:
                timeout_text = "⚡ *Авто-исполнено*"
                try:
                    await bot.edit_message_text(
                        chat_id=self._chat_id,
                        message_id=mid,
                        text=text + f"\n\n{timeout_text}",
                        parse_mode="Markdown",
                    )
                except Exception:
                    pass
            else:
                try:
                    await bot.edit_message_reply_markup(
                        chat_id=self._chat_id,
                        message_id=mid,
                        reply_markup=None,
                    )
                except Exception:
                    pass
            return decision

        except Exception as e:
            logger.error("Telegram error: %s", e)
            return False  # C3: fail-closed — skip trade on Telegram errors

    async def notify(
        self,
        text: str,
        reply_markup=None,
        parse_mode: str = "Markdown",
    ) -> None:
        """
        Отправляет текстовое уведомление в Telegram-чат.

        :param text: Текст уведомления.
        :param reply_markup: Опциональная InlineKeyboardMarkup.
        :param parse_mode: Режим разметки (по умолчанию MarkdownV2).
        """
        if not self._enabled:
            return
        try:
            await self._app.bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
            )
        except Exception as e:
            logger.error("Telegram notify error: %s", e)
