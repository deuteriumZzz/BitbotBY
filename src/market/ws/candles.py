"""Сборка свечей из kline-потока и сверка с REST (Phase 4b).

В фичи идут только ПОДТВЕРЖДЁННЫЕ свечи (`confirm=true`); формирующаяся свеча
хранится отдельно и в историю не попадает. Дубликаты по времени старта
отбрасываются, события старше последней подтверждённой — тоже (считаются).
Пропуск между подряд подтверждёнными свечами — `gaps`: нужен REST-бэкфилл.
Формат вывода как у ccxt: [ts_ms, open, high, low, close, volume].
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class CandleAssembler:
    def __init__(self, interval_ms: int, keep: int = 1000) -> None:
        self.interval_ms, self.keep = interval_ms, keep
        self.confirmed: List[List[float]] = []
        self.forming: Optional[List[float]] = None
        self.duplicates = 0
        self.out_of_order = 0
        self.gaps = 0

    @staticmethod
    def _row(k: Dict[str, Any]) -> List[float]:
        return [int(k["start"]), float(k["open"]), float(k["high"]), float(k["low"]),
                float(k["close"]), float(k["volume"])]

    def on_kline(self, k: Dict[str, Any]) -> bool:
        """True, если принята новая подтверждённая свеча."""
        row = self._row(k)
        if not k.get("confirm"):
            self.forming = row
            return False
        last = self.confirmed[-1][0] if self.confirmed else None
        if last is not None:
            if row[0] == last:
                self.duplicates += 1
                return False
            if row[0] < last:
                self.out_of_order += 1
                return False
            if row[0] - last != self.interval_ms:
                self.gaps += 1
        self.confirmed.append(row)
        del self.confirmed[: -self.keep]
        self.forming = None
        return True

    def ohlcv(self) -> List[List[float]]:
        return list(self.confirmed)


def compare_with_rest(
    ws: List[List[float]],
    rest: List[List[float]],
    price_tol: float = 1e-6,
    volume_tol: float = 1e-3,
) -> Dict[str, Any]:
    """Расхождение WS и REST по общим свечам (допуски — предложения). Нужно для отката на REST."""
    rest_by = {int(r[0]): r for r in rest}
    mismatched, missing, max_rel = [], [], 0.0
    compared = 0
    for w in ws:
        r = rest_by.get(int(w[0]))
        if r is None:
            missing.append(int(w[0]))
            continue
        compared += 1
        bad = False
        for i, tol in ((1, price_tol), (2, price_tol), (3, price_tol), (4, price_tol), (5, volume_tol)):
            base = abs(r[i]) or 1.0
            rel = abs(w[i] - r[i]) / base
            max_rel = max(max_rel, rel)
            bad = bad or rel > tol
        if bad:
            mismatched.append(int(w[0]))
    ws_starts = {int(w[0]) for w in ws}
    missing_in_ws = sorted(
        t for t in rest_by if t not in ws_starts and ws and ws[0][0] <= t <= ws[-1][0]
    )
    return {
        "compared": compared,
        "mismatched": mismatched,
        "missing_in_rest": missing,
        "missing_in_ws": missing_in_ws,
        "max_rel_diff": max_rel,
        "ok": not mismatched and not missing_in_ws,
    }
