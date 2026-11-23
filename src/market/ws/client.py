"""Публичный поток Bybit v5 (Phase 4b): маршрутизация сообщений и устойчивое соединение.

Клиент — `websockets` (уже в requirements): протокол Bybit v5 — тонкий JSON
(`subscribe`, `ping`), а существенная логика (целостность стакана, задержка,
переподключение) наша в любом случае; ccxt.pro скрывает `u` и биржевые метки
времени, `pybit` — лишняя зависимость. Выбор — предложение, фиксируется в
ADR-003 и требует подтверждения владельца.

Shadow: поток ничего не решает и не пишет в торговые таблицы. CLI:
    python -m src.market.ws.client --symbols BTC/USDT,ETH/USDT --seconds 60
собирает метрики публичного потока и печатает их JSON (публичные данные, без ключей).
URL, топики и глубины — по памяти из спецификации; сверить с докой (§14).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from src.market.ws.book import RESYNC, OrderBook
from src.market.ws.candles import CandleAssembler
from src.market.ws.health import StreamHealth

logger = logging.getLogger(__name__)

PUBLIC_LINEAR = "wss://stream.bybit.com/v5/public/linear"
PUBLIC_LINEAR_TESTNET = "wss://stream-testnet.bybit.com/v5/public/linear"
POLL_S = 1.0  # максимальная задержка реакции на остановку
PING_EVERY_S = 20.0
STALE_AFTER_S = 30.0  # нет сообщений дольше — соединение считается мёртвым
BACKOFF_BASE_S, BACKOFF_MAX_S = 1.0, 60.0
STABLE_AFTER_S = 60.0  # столько живое соединение сбрасывает backoff

_INTERVALS = {"1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "4h": "240", "1d": "D"}
_INTERVAL_MS = {"1": 60_000, "3": 180_000, "5": 300_000, "15": 900_000, "30": 1_800_000,
                "60": 3_600_000, "240": 14_400_000, "D": 86_400_000}


def to_bybit(symbol: str) -> str:
    """'BTC/USDT' или 'BTC/USDT:USDT' → 'BTCUSDT'."""
    return symbol.split(":")[0].replace("/", "")


def topics(symbols: List[str], timeframe: str = "15m", depth: int = 50) -> List[str]:
    iv = _INTERVALS[timeframe]
    out: List[str] = []
    for s in map(to_bybit, symbols):
        out += [f"kline.{iv}.{s}", f"orderbook.{depth}.{s}", f"tickers.{s}", f"publicTrade.{s}"]
    return out


class MarketStream:
    """Состояние всех публичных потоков и разбор сообщений. Без сети и без часов."""

    def __init__(self, symbols: List[str], timeframe: str = "15m", depth: int = 50,
                 health: Optional[StreamHealth] = None) -> None:
        self.depth, self.timeframe = depth, timeframe
        iv = _INTERVALS[timeframe]
        self.health = health or StreamHealth()
        self.books = {to_bybit(s): OrderBook(to_bybit(s), depth) for s in symbols}
        self.candles = {to_bybit(s): CandleAssembler(_INTERVAL_MS[iv]) for s in symbols}
        self.tickers: Dict[str, Dict[str, Any]] = {}
        self.trades: Dict[str, int] = {}

    def on_reconnect(self) -> None:
        self.health.on_reconnect()
        for b in self.books.values():
            b.reset()  # после обрыва прежние дельты бесполезны

    def handle(self, raw: str) -> List[Tuple[str, str]]:
        """Разбирает сообщение; возвращает действия [("resubscribe", topic), ...]."""
        try:
            msg = json.loads(raw)
        except ValueError:
            return []
        topic = msg.get("topic") if isinstance(msg, dict) else None
        if not topic:
            return []  # подтверждения подписки, pong
        kind = topic.split(".", 1)[0]
        sym = topic.rsplit(".", 1)[-1]
        stream = f"{kind}.{sym}"
        self.health.on_message(stream, msg.get("ts"))
        data = msg.get("data")
        actions: List[Tuple[str, str]] = []
        if kind == "orderbook" and sym in self.books and isinstance(data, dict):
            if self.books[sym].apply(msg.get("type", ""), data) == RESYNC:
                self.health.on_gap(stream)
                actions.append(("resubscribe", topic))  # новая подписка даёт свежий snapshot
        elif kind == "kline" and sym in self.candles and isinstance(data, list):
            for k in data:
                self.candles[sym].on_kline(k)
        elif kind == "tickers" and isinstance(data, dict):
            self.tickers.setdefault(sym, {}).update({k: v for k, v in data.items() if v not in ("", None)})
        elif kind == "publicTrade" and isinstance(data, list):
            self.trades[sym] = self.trades.get(sym, 0) + len(data)
        return actions


Connect = Callable[[], Awaitable[Any]]  # возвращает объект с send/recv/close


def default_connect(url: str = PUBLIC_LINEAR) -> Connect:
    async def _c() -> Any:
        import websockets  # noqa: PLC0415

        return await websockets.connect(url, ping_interval=None, max_queue=1024)  # ping — наш

    return _c


def backoff_delay(attempt: int, rng: Callable[[], float] = random.random) -> float:
    """Экспоненциальная пауза с полным jitter: U(0, min(max, base·2^n))."""
    return rng() * min(BACKOFF_MAX_S, BACKOFF_BASE_S * (2 ** attempt))


async def run_stream(
    connect: Connect,
    stream: MarketStream,
    sub_topics: List[str],
    stop: asyncio.Event,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    mono: Callable[[], float] = time.monotonic,
    stale_after: float = STALE_AFTER_S,
    ping_every: float = PING_EVERY_S,
    rng: Callable[[], float] = random.random,
) -> None:
    """Держит соединение: подписка, ping, обнаружение тишины, reconnect с backoff+jitter."""
    attempt = 0
    first = True
    while not stop.is_set():
        ws = None
        started = mono()
        try:
            ws = await connect()
            if not first:
                stream.on_reconnect()
            first = False
            await ws.send(json.dumps({"op": "subscribe", "args": sub_topics}))
            last_ping = last_rx = mono()
            while not stop.is_set():
                try:
                    # короткий опрос: сигнал остановки обрабатывается не позже POLL_S
                    raw = await asyncio.wait_for(
                        ws.recv(), timeout=min(ping_every, stale_after, POLL_S)
                    )
                except asyncio.TimeoutError:
                    if mono() - last_rx > stale_after:
                        raise ConnectionError("поток молчит дольше stale_after")
                    raw = None
                if raw is not None:
                    last_rx = mono()
                    for action, topic in stream.handle(raw):
                        if action == "resubscribe":
                            await ws.send(json.dumps({"op": "unsubscribe", "args": [topic]}))
                            await ws.send(json.dumps({"op": "subscribe", "args": [topic]}))
                if mono() - last_ping >= ping_every:
                    await ws.send(json.dumps({"op": "ping"}))
                    last_ping = mono()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — любой сбой соединения → переподключение
            logger.warning("WS: сбой %s: %s", type(exc).__name__, exc)
        finally:
            if ws is not None:
                try:
                    await ws.close()
                except Exception:  # noqa: BLE001
                    pass
        if stop.is_set():
            break
        attempt = 0 if mono() - started >= STABLE_AFTER_S else attempt + 1
        await sleep(backoff_delay(attempt, rng))


async def shadow(symbols: List[str], seconds: float, timeframe: str = "15m", testnet: bool = False) -> Dict[str, Any]:
    stream = MarketStream(symbols, timeframe)
    stop = asyncio.Event()
    url = PUBLIC_LINEAR_TESTNET if testnet else PUBLIC_LINEAR
    task = asyncio.create_task(run_stream(default_connect(url), stream, topics(symbols, timeframe), stop))
    await asyncio.sleep(seconds)
    stop.set()
    try:
        await asyncio.wait_for(task, 5)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        task.cancel()
    return {
        "health": stream.health.snapshot(),
        "books": {s: {"ready": b.ready, "gaps": b.gaps, "crossed": b.crossed, "snapshots": b.snapshots,
                      "spread_bps": b.spread_bps()} for s, b in stream.books.items()},
        "confirmed_candles": {s: len(c.confirmed) for s, c in stream.candles.items()},
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Shadow-замер публичного WS Bybit (без ключей)")
    p.add_argument("--symbols", default="BTC/USDT")
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--testnet", action="store_true")
    a = p.parse_args(argv)
    res = asyncio.run(shadow(a.symbols.split(","), a.seconds, a.timeframe, a.testnet))
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
