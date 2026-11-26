"""Живые цены открытых позиций для приложения: публичный WebSocket Bybit, без ключей.

Только для отображения. Решения (стопы, трейлинг, выход) бот принимает по своим ценам
(PositionMonitor, раз в 5 с) — этот поток торговлю не трогает. Дашборд подписывается на
`tickers.*` монет из открытых позиций и не чаще раза в секунду на монету публикует событие
`price` в общий поток событий — его получают все открытые окна приложения.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Awaitable, Callable, Dict, Optional, Set, Tuple

from bitbot.api import data
from bitbot.storage import bot_bus

logger = logging.getLogger(__name__)

PUBLIC_LINEAR = "wss://stream.bybit.com/v5/public/linear"
THROTTLE_S = 1.0
POLL_S = 5.0  # как часто сверять подписки с открытыми позициями (и слать ping, Bybit ждёт его раз в 20 с)


def to_bybit(symbol: str) -> str:
    """'BNB/USDT' → 'BNBUSDT'."""
    return symbol.split(":")[0].replace("/", "").upper()


def parse_ticker(raw: str) -> Optional[Tuple[str, float]]:
    """('BNBUSDT', 552.1) из сообщения tickers; None — не цена (ответ на подписку, pong, без lastPrice)."""
    try:
        msg = json.loads(raw)
        topic = str(msg.get("topic", ""))
        price = (msg.get("data") or {}).get("lastPrice")
        if not topic.startswith("tickers.") or price in (None, ""):
            return None
        return topic.split(".", 1)[1], float(price)
    except (ValueError, TypeError, AttributeError):
        return None


def _open_symbols() -> Dict[str, str]:
    """{'BNBUSDT': 'BNB/USDT'} для открытых позиций."""
    return {
        to_bybit(p["symbol"]): p["symbol"]
        for p in data.get_open_positions()
        if p.get("symbol")
    }


async def _connect() -> Any:
    import websockets  # noqa: PLC0415

    return await websockets.connect(PUBLIC_LINEAR, ping_interval=None, max_queue=256)


def _publish(symbol: str, price: float) -> None:
    bot_bus.emit(data.get_redis(), "price", {"symbol": symbol, "price": price})


async def run(
    connect: Callable[[], Awaitable[Any]] = _connect,
    symbols: Callable[[], Dict[str, str]] = _open_symbols,
    publish: Callable[[str, float], None] = _publish,
) -> None:
    """Держит соединение и подписки; при обрыве переподключается с паузой до минуты."""
    attempt = 0
    while True:
        try:
            ws = await connect()
            attempt = 0
            subscribed: Set[str] = set()
            names: Dict[str, str] = {}
            sent_at: Dict[str, float] = {}
            next_poll = 0.0
            while True:
                if time.monotonic() >= next_poll:
                    next_poll = time.monotonic() + POLL_S
                    names = symbols()
                    want = set(names)
                    if subscribed - want:
                        await ws.send(
                            json.dumps(
                                {
                                    "op": "unsubscribe",
                                    "args": [f"tickers.{s}" for s in subscribed - want],
                                }
                            )
                        )
                    if want - subscribed:
                        await ws.send(
                            json.dumps(
                                {
                                    "op": "subscribe",
                                    "args": [f"tickers.{s}" for s in want - subscribed],
                                }
                            )
                        )
                    subscribed = want
                    await ws.send('{"op":"ping"}')
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                tick = parse_ticker(raw)
                if tick is None or tick[0] not in names:
                    continue
                now = time.monotonic()
                if now - sent_at.get(tick[0], 0.0) >= THROTTLE_S:
                    sent_at[tick[0]] = now
                    publish(names[tick[0]], tick[1])
        except asyncio.CancelledError:
            raise
        except Exception as e:  # сеть, биржа, Redis — для отображения цен не критично
            logger.debug("live prices: %s", e)
            await asyncio.sleep(min(60, 2**attempt))
            attempt += 1
