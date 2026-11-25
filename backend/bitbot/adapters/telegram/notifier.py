"""
Telegram: короткие уведомления владельцу и вход в приложение (Mini App).

Управление ботом и все настройки — только в приложении (MiniApp / Desktop).
Здесь остаётся лёгкий слой поверх Telegram Bot API (прямые вызовы через httpx):

- `notify` / `edit_notify` / закрепление — сообщения о сделках, рисках, задачах.
  Под сообщением кнопка web_app «Открыть в приложении» на нужную вкладку —
  если задан https-адрес приложения (WEBAPP_URL): Telegram открывает Mini App
  только по https.
- `ask_confirm` — предложение сделки. Решение принимается в приложении
  (`resolve_pending`); нет ответа до таймаута — авто-исполнение или пропуск,
  как настроено.
- Входящие: только `/start` (кнопка приложения) и `/app` (одноразовая ссылка
  входа при запуске на своём компьютере), только от владельца. Остальное
  игнорируется.

Токен — часть URL Bot API: в логи попадают имя метода и описание ошибки,
а не адрес (плюс общая маскировка секретов в логгере).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Callable, Dict, Optional

import httpx

from bitbot.infra.logger import mask_secrets

logger = logging.getLogger(__name__)

_CONFIRM_TIMEOUT = 60  # секунд на ответ по сделке
_API = "https://api.telegram.org/bot{token}/{method}"
_HTTP_TIMEOUT = 15.0
_POLL_TIMEOUT = 25  # long polling getUpdates, сек
_POLL_RETRY_S = 5.0

# Вкладки приложения, на которые ведут кнопки (frontend/src/lib/types.ts: Tab).
SCREEN_HOME = "terminal"
SCREEN_PIPELINE = "bots"  # сигналы на подтверждение, риск, аварийное управление
SCREEN_JOBS = "learn"  # обучение и проверки
SCREEN_JOURNAL = "journal"


def webapp_url() -> str:
    """https-адрес приложения или пустая строка (локальный запуск без туннеля)."""
    url = os.getenv("WEBAPP_URL", "").strip()
    return url if url.startswith("https://") else ""


def app_button(screen: str, text: str = "Открыть в приложении") -> Optional[dict]:
    """Inline-клавиатура с одной кнопкой web_app; None — приложение недоступно по https."""
    base = webapp_url()
    if not base:
        return None
    sep = "&" if "?" in base else "?"
    url = f"{base}{sep}tab={screen}"
    return {"inline_keyboard": [[{"text": text, "web_app": {"url": url}}]]}


class TelegramNotifier:
    """Уведомления владельцу и вход в приложение; без токена — тихо ничего не делает."""

    def __init__(self, token: str, chat_id: str):
        self._token = token
        self._chat_id = str(chat_id or "")
        self._enabled = bool(token and self._chat_id)
        self._client: Optional[httpx.AsyncClient] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._offset = 0
        self._pending: Dict[int, asyncio.Event] = {}  # message_id → ждём ответа
        self._decisions: Dict[int, bool] = {}
        self._decided_by: Dict[int, str] = {}
        self._live_msg_ids: Dict[str, int] = {}  # key → message_id (edit_notify)
        # Публикация событий для приложения; задаёт TradingBot.
        self._emit: "Callable[[str, dict], None] | None" = None
        # Пускать ли уведомление вида kind (тумблер + тихие часы); задаёт TradingBot.
        self.gate: "Callable[[str], bool] | None" = None
        # Одноразовый токен входа для /app; задаёт TradingBot (нужен Redis).
        self.login_issuer: "Callable[[], str] | None" = None

    # ── жизненный цикл ────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Кнопка приложения в меню чата и приём /start, /app."""
        if not self._enabled:
            return
        self._client = httpx.AsyncClient(timeout=_HTTP_TIMEOUT)
        if webapp_url():
            await self._call(
                "setChatMenuButton",
                chat_id=self._chat_id,
                menu_button={
                    "type": "web_app",
                    "text": "Приложение",
                    "web_app": {"url": webapp_url()},
                },
            )
        self._poll_task = asyncio.create_task(self._poll_updates())
        logger.info("Telegram notifier started")

    async def stop(self) -> None:
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.aclose()
            self._client = None

    # ── Bot API ───────────────────────────────────────────────────────────────

    async def _call(
        self, method: str, http_timeout: float = _HTTP_TIMEOUT, **params: Any
    ) -> Optional[Any]:
        """Вызов Bot API; результат или None (ошибка уже в логе, без токена)."""
        if self._client is None:
            return None
        payload = {k: v for k, v in params.items() if v is not None}
        try:
            resp = await self._client.post(
                _API.format(token=self._token, method=method),
                json=payload,
                timeout=http_timeout,
            )
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            logger.warning("Telegram %s failed: %s", method, type(e).__name__)
            return None
        if not data.get("ok"):
            desc = str(data.get("description", ""))
            # Markdown не разобрался (символы из названий, ошибок) — шлём как есть.
            if "parse" in desc.lower() and payload.get("parse_mode"):
                payload.pop("parse_mode")
                return await self._call(method, http_timeout, **payload)
            logger.warning("Telegram %s: %s", method, mask_secrets(desc))
            return None
        return data.get("result")

    async def _send(
        self, text: str, parse_mode: Optional[str], markup: Optional[dict]
    ) -> Optional[int]:
        res = await self._call(
            "sendMessage",
            chat_id=self._chat_id,
            text=mask_secrets(text),
            parse_mode=parse_mode,
            reply_markup=markup,
        )
        return res.get("message_id") if isinstance(res, dict) else None

    async def _edit(
        self,
        message_id: int,
        text: str,
        parse_mode: Optional[str] = "Markdown",
        markup: Optional[dict] = None,
    ) -> bool:
        res = await self._call(
            "editMessageText",
            chat_id=self._chat_id,
            message_id=message_id,
            text=mask_secrets(text),
            parse_mode=parse_mode,
            reply_markup=markup,
        )
        return res is not None

    # ── уведомления ──────────────────────────────────────────────────────────

    async def notify(
        self,
        text: str,
        parse_mode: str = "Markdown",
        kind: str = "",
        screen: str = SCREEN_HOME,
    ) -> None:
        """
        Сообщение владельцу с кнопкой перехода на вкладку `screen` приложения.

        :param kind: Вид уведомления (position_open, new_signal…) — для тумблеров
            и тихих часов. Без вида — приходит всегда (например, «позиция без SL»).
        """
        if not self._enabled:
            return
        if kind and self.gate is not None:
            try:
                if not self.gate(kind):
                    return
            except Exception as e:  # не смогли спросить — лучше отправить
                logger.debug("notify gate failed for %s: %s", kind, e)
        await self._send(text, parse_mode, app_button(screen))

    async def edit_notify(
        self, key: str, text: str, parse_mode: str = "Markdown"
    ) -> None:
        """Повторяющийся алерт `key`: первый раз — новое сообщение, дальше — правка его же."""
        if not self._enabled:
            return
        markup = app_button(SCREEN_HOME)
        msg_id = self._live_msg_ids.get(key)
        if msg_id and await self._edit(msg_id, text, parse_mode, markup):
            return
        new_id = await self._send(text, parse_mode, markup)
        if new_id:
            self._live_msg_ids[key] = new_id
        else:
            self._live_msg_ids.pop(key, None)

    async def pin_live_msg(self, key: str) -> None:
        if not self._enabled or key not in self._live_msg_ids:
            return
        await self._call(
            "pinChatMessage",
            chat_id=self._chat_id,
            message_id=self._live_msg_ids[key],
            disable_notification=True,
        )

    async def unpin_live_msg(self, key: str, resolved_text: str = "") -> None:
        """Перевести закреплённый алерт в «решено» и открепить."""
        if not self._enabled:
            return
        msg_id = self._live_msg_ids.pop(key, None)
        if not msg_id:
            return
        if resolved_text:
            await self._edit(msg_id, resolved_text)
        await self._call("unpinChatMessage", chat_id=self._chat_id, message_id=msg_id)

    # ── предложение сделки ────────────────────────────────────────────────────

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
        Предложение сделки владельцу; решение — в приложении.

        True — подтверждено или нет ответа при auto_execute. Без Telegram —
        True сразу (как раньше). Сбой отправки — False: сделку пропускаем.
        """
        if not self._enabled:
            return True
        symbol = rec.get("symbol", "?")
        action = str(rec.get("action", "?")).upper()
        confidence = rec.get("confidence", 0.0)
        entry = rec.get("entry", 0.0)
        sl = rec.get("stop_loss", 0.0)
        tp = rec.get("take_profit", 0.0)
        strategy = rec.get("strategy", "?")
        bt_line = (
            f"Backtest: *{bt_win_rate:.0%}* ({bt_trades} trades) "
            f"EV: *{bt_ev * 100:+.2f}%*"
            if bt_trades > 0
            else "Backtest: no data"
        )
        live_line = (
            f"Live: *{live_win_rate:.0%}* ({live_trades} trades) "
            f"EV: *{live_ev * 100:+.2f}%*"
            if live_trades > 0
            else "Live: -- (no trades yet)"
        )
        text = (
            f"*{symbol}*  --  {action}\n\n"
            f"Strategy: `{strategy}`\n"
            f"Entry: `${entry:.4f}`\n"
            f"SL: `${sl:.4f}`   TP: `${tp:.4f}`\n\n"
            f"AI confidence: *{confidence:.0%}*\n\n"
            f"{bt_line}\n{live_line}\n\n"
            + (
                f"⏱ {timeout}с на решение в приложении, иначе сделка будет пропущена"
                if not auto_execute
                else f"⚡ Авто-исполнение через {timeout}с — пропустить можно в приложении"
            )
        )
        mid = await self._send(
            text, "Markdown", app_button(SCREEN_PIPELINE, "Решить в приложении")
        )
        if not mid:
            return False  # fail-closed: не смогли спросить — не входим
        event = asyncio.Event()
        self._pending[mid] = event
        signal_id = f"sig{mid}"
        self._publish(
            "signal.pending",
            {
                "id": signal_id,
                "symbol": symbol,
                "side": "long" if action == "BUY" else "short",
                "confidence": confidence,
                "strategy": strategy,
                "reasoning": str(rec.get("reasoning") or "")[:500],
                "entry": entry,
                "stop_loss": sl,
                "take_profit": tp,
                "expires_at": time.time() + timeout,
                # По таймауту: True — исполнить, False — пропустить.
                "on_timeout_execute": auto_execute,
                "backtest": {
                    "win_rate": bt_win_rate,
                    "trades": bt_trades,
                    "ev": bt_ev,
                },
                "live": {
                    "win_rate": live_win_rate,
                    "trades": live_trades,
                    "ev": live_ev,
                },
            },
        )
        timed_out = False
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            timed_out = True
            logger.info(
                "Confirm timeout for %s -> %s",
                symbol,
                "auto-execute" if auto_execute else "skip",
            )
        decision = self._decisions.pop(mid, auto_execute)
        self._pending.pop(mid, None)
        by = self._decided_by.pop(mid, "timeout")
        self._publish(
            "signal.resolved",
            {"id": signal_id, "symbol": symbol, "confirmed": decision, "by": by},
        )
        if timed_out:
            footer = "⚡ *Авто-исполнено*" if decision else "⏭ *Нет ответа — пропущено*"
        else:
            footer = (
                "✅ *Принято в приложении*"
                if decision
                else "❌ *Пропущено в приложении*"
            )
        await self._edit(mid, text + f"\n\n{footer}")  # без кнопки: решение принято
        return decision

    def resolve_pending(self, signal_id: str, confirmed: bool) -> bool:
        """Ответ на предложение сделки из приложения. False — уже не ждём ответа."""
        try:
            mid = int(str(signal_id).removeprefix("sig"))
        except ValueError:
            return False
        event = self._pending.get(mid)
        if event is None or event.is_set():
            return False
        self._decisions[mid] = confirmed
        self._decided_by[mid] = "app"
        event.set()
        return True

    def set_event_emitter(self, emit: "Callable[[str, dict], None]") -> None:
        self._emit = emit

    def _publish(self, type_: str, data: dict) -> None:
        if self._emit is not None:
            try:
                self._emit(type_, data)
            except Exception as e:
                logger.debug("event %s not published: %s", type_, e)

    # ── входящие: /start и /app ───────────────────────────────────────────────

    async def _poll_updates(self) -> None:
        """Long polling getUpdates; 409 (бота опрашивает другая программа) — ждём."""
        while True:
            try:
                updates = await self._call(
                    "getUpdates",
                    http_timeout=_POLL_TIMEOUT + 10,
                    offset=self._offset or None,
                    timeout=_POLL_TIMEOUT,
                    allowed_updates=["message"],
                )
                if updates is None:
                    await asyncio.sleep(_POLL_RETRY_S)
                    continue
                for upd in updates:
                    self._offset = int(upd.get("update_id", 0)) + 1
                    await self._handle_message(upd.get("message") or {})
            except asyncio.CancelledError:
                raise
            except Exception as e:  # приём сообщений не должен ронять бота
                logger.warning("Telegram polling error: %s", type(e).__name__)
                await asyncio.sleep(_POLL_RETRY_S)

    async def _handle_message(self, msg: Dict[str, Any]) -> None:
        if str((msg.get("chat") or {}).get("id", "")) != self._chat_id:
            return  # не владелец — молчим
        text = str(msg.get("text") or "").strip().split("@")[0]
        if text.startswith("/start"):
            await self._send(self._welcome_text(), None, app_button(SCREEN_HOME))
        elif text.startswith("/app"):
            await self._send(self._login_link_text(), "Markdown", None)

    @staticmethod
    def _welcome_text() -> str:
        if webapp_url():
            return (
                "BitbotBY: сюда приходят уведомления о сделках и рисках. "
                "Управление и настройки — в приложении: кнопка ниже или "
                "«Приложение» в меню."
            )
        return (
            "BitbotBY: сюда приходят уведомления о сделках и рисках. "
            "Приложение открывается на компьютере с ботом (ярлык BitbotBY); "
            "для телефона включите доступ: ./bitbot phone on. "
            "Ссылка входа на компьютере — команда /app."
        )

    def _login_link_text(self) -> str:
        port = os.getenv("APP_PORT", "")
        if not port.isdigit() or self.login_issuer is None:
            return (
                "Ссылка входа работает только при запуске бота на своём компьютере "
                "(ярлык BitbotBY или ./bitbot start)."
            )
        from bitbot.config import login_link  # noqa: PLC0415

        token = self.login_issuer()
        return (
            "🔐 Ссылка входа в приложение — откройте её на компьютере, где запущен "
            "бот (нажмите, чтобы скопировать):\n\n"
            f"`http://localhost:{port}/app/#local={token}`\n\n"
            f"Работает {login_link.TTL // 60} минут и только один раз. "
            "Если вы её не запрашивали — просто не открывайте."
        )
