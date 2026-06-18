"""
Telegram bot for trade confirmations.

Flow:
  1. Bot sends trade proposal with /❌ inline buttons.
  2. User has CONFIRM_TIMEOUT seconds to respond.
  3. If no response → auto-execute (True).
  4. If ❌ → skip trade.

Requirements: python-telegram-bot>=20.0 (async)
"""

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
    Sends Telegram alerts and waits for trade confirmation.
    Gracefully degrades when token is missing or library
    is not installed.
    """

    def __init__(self, token: str, chat_id: str):
        self._token = token
        self._chat_id = chat_id
        self._app: Optional[object] = None
        self._polling_task: Optional[asyncio.Task] = None
        self._pending: dict = {}  # message_id → asyncio.Event
        self._decisions: dict = {}  # message_id → bool
        self._enabled = bool(_TG_AVAILABLE and token and chat_id)

    async def start(self) -> None:
        """Build and start telegram polling (background)."""
        if not self._enabled:
            return
        self._app = Application.builder().token(self._token).build()
        self._app.add_handler(CallbackQueryHandler(self._handle_callback))
        await self._app.initialize()
        await self._app.start()
        self._polling_task = asyncio.create_task(self._app.updater.start_polling())
        logger.info("Telegram notifier started")

    async def stop(self) -> None:
        """Stop the Telegram application."""
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
        query = update.callback_query
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
    ) -> bool:
        """
        Send trade proposal; return True if confirmed or timed out.
        Shows backtest and live win rates separately.
        Returns True immediately if Telegram is disabled.
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
            f"Auto-execute in {timeout}s"
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
            self._decisions[mid] = True  # default: confirm

            try:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.info(f"Telegram timeout for {symbol} " "-> auto-confirm")
                await bot.edit_message_reply_markup(
                    chat_id=self._chat_id,
                    message_id=mid,
                    reply_markup=None,
                )

            decision = self._decisions.pop(mid, True)
            self._pending.pop(mid, None)
            return decision

        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return True  # fail-open: execute trade

    async def notify(self, text: str) -> None:
        """Send a plain notification message."""
        if not self._enabled:
            return
        try:
            await self._app.bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode="Markdown",
            )
        except Exception as e:
            logger.error(f"Telegram notify error: {e}")
