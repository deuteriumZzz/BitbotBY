"""L2-стакан по потоку Bybit v5: snapshot + delta с проверкой целостности.

Правила (ПРЕДПОЛОЖЕНИЯ из спецификации, сверить с актуальной докой, §14):
  * `snapshot` заменяет книгу целиком; `u == 1` — рестарт сервиса, тоже snapshot;
  * `delta` применяется только к готовой книге и только если `u == last_u + 1`;
  * размер `"0"` удаляет уровень;
  * `seq` непрерывным не считается и для целостности не используется;
  * книга с пересечением (лучшая покупка ≥ лучшей продажи) недостоверна.
Любое нарушение → книга снимается с готовности (`ready=False`), вызывающий код
обязан пересоздать подписку/взять REST-снимок; торговать парой по этой книге до
ресинка нельзя.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

OK = "ok"
RESYNC = "resync_needed"
IGNORED = "ignored"


class OrderBook:
    def __init__(self, symbol: str, depth: int = 50) -> None:
        self.symbol, self.depth = symbol, depth
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.last_u: Optional[int] = None
        self.ready = False
        self.gaps = 0
        self.crossed = 0
        self.snapshots = 0

    def reset(self) -> None:
        """Сброс при переподключении: книга не готова до нового snapshot."""
        self.bids.clear()
        self.asks.clear()
        self.last_u, self.ready = None, False

    @staticmethod
    def _apply(side: Dict[float, float], levels: List[List[str]]) -> None:
        for price, size in levels:
            p, s = float(price), float(size)
            if s == 0:
                side.pop(p, None)
            else:
                side[p] = s

    def _trim(self) -> None:
        if len(self.bids) > self.depth * 2:
            for p in sorted(self.bids)[: len(self.bids) - self.depth]:
                del self.bids[p]
        if len(self.asks) > self.depth * 2:
            for p in sorted(self.asks, reverse=True)[: len(self.asks) - self.depth]:
                del self.asks[p]

    def apply(self, msg_type: str, data: Dict[str, Any]) -> str:
        u = data.get("u")
        if msg_type == "snapshot" or u == 1:
            self.bids.clear()
            self.asks.clear()
            self._apply(self.bids, data.get("b", []))
            self._apply(self.asks, data.get("a", []))
            self.last_u, self.ready = u, True
            self.snapshots += 1
        elif msg_type == "delta":
            if not self.ready:
                return IGNORED  # дельта до snapshot — ждём ресинка
            if u is None or self.last_u is None or u != self.last_u + 1:
                self.ready = False
                self.gaps += 1
                return RESYNC
            self._apply(self.bids, data.get("b", []))
            self._apply(self.asks, data.get("a", []))
            self.last_u = u
        else:
            return IGNORED
        self._trim()
        if self.bids and self.asks and max(self.bids) >= min(self.asks):
            self.ready = False
            self.crossed += 1
            return RESYNC
        return OK

    # ── чтение ────────────────────────────────────────────────────────────────
    @property
    def best_bid(self) -> Optional[float]:
        return max(self.bids) if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return min(self.asks) if self.asks else None

    def spread_bps(self) -> Optional[float]:
        b, a = self.best_bid, self.best_ask
        return None if b is None or a is None else (a - b) / ((a + b) / 2) * 1e4

    def imbalance(self, levels: int = 10) -> Optional[float]:
        """(Σbid − Σask) / (Σbid + Σask) по лучшим `levels` уровням; None, если книга не готова."""
        if not self.ready or not self.bids or not self.asks:
            return None
        bid = sum(self.bids[p] for p in sorted(self.bids, reverse=True)[:levels])
        ask = sum(self.asks[p] for p in sorted(self.asks)[:levels])
        return (bid - ask) / (bid + ask) if bid + ask else None
