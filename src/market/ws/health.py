"""Метрики потоков (Phase 4b): задержка, пропуски, переподключения, «тихие» потоки.

Задержка = время прихода − биржевое время − оценка смещения часов. Смещение
оценивается нижним квантилем (5%) наблюдений «приход − биржевой ts» по скользящему
окну: предполагается, что минимальная задержка сети близка к нулю, поэтому
показанная `lag_ms` — задержка относительно лучших наблюдавшихся условий, а не
абсолютная. Для абсолютной нужна синхронизация часов (NTP/биржевое время).
Часы передаются снаружи — тесты детерминированы.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, Optional

WINDOW = 2000


def _q(values: list, q: float) -> Optional[float]:
    if not values:
        return None
    v = sorted(values)
    return round(v[min(len(v) - 1, int(q * len(v)))], 2)


class StreamHealth:
    def __init__(
        self,
        wall_ms: Callable[[], float] = lambda: time.time() * 1000,
        mono: Callable[[], float] = time.monotonic,
    ) -> None:
        self._wall, self._mono = wall_ms, mono
        self._raw: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=WINDOW))
        self.messages: Dict[str, int] = defaultdict(int)
        self.gaps: Dict[str, int] = defaultdict(int)
        self.resyncs: Dict[str, int] = defaultdict(int)
        self.reconnects = 0
        self._last: Dict[str, float] = {}

    def on_message(self, stream: str, exchange_ts_ms: Optional[float]) -> None:
        self.messages[stream] += 1
        self._last[stream] = self._mono()
        if exchange_ts_ms:
            self._raw[stream].append(self._wall() - float(exchange_ts_ms))

    def on_gap(self, stream: str) -> None:
        self.gaps[stream] += 1
        self.resyncs[stream] += 1

    def on_reconnect(self) -> None:
        self.reconnects += 1

    def seconds_since_last(self, stream: str) -> Optional[float]:
        return None if stream not in self._last else self._mono() - self._last[stream]

    def silent_streams(self, max_age_s: float) -> list:
        """Потоки, которые что-то присылали, но замолчали (тихий сбой)."""
        now = self._mono()
        return sorted(s for s, t in self._last.items() if now - t > max_age_s)

    def lag(self, stream: str) -> Dict[str, Optional[float]]:
        raw = list(self._raw[stream])
        if not raw:
            return {"p50": None, "p99": None, "offset_ms": None}
        offset = _q(raw, 0.05)
        lags = [r - offset for r in raw]
        return {"p50": _q(lags, 0.5), "p99": _q(lags, 0.99), "offset_ms": offset}

    def snapshot(self) -> Dict[str, Any]:
        streams = sorted(set(self.messages) | set(self.gaps))
        return {
            "reconnects": self.reconnects,
            "streams": {
                s: {
                    "messages": self.messages[s],
                    "gaps": self.gaps[s],
                    "gap_rate": (
                        round(self.gaps[s] / self.messages[s], 5)
                        if self.messages[s]
                        else None
                    ),
                    "resyncs": self.resyncs[s],
                    "lag_ms": self.lag(s),
                    "seconds_since_last": self.seconds_since_last(s),
                }
                for s in streams
            },
        }
