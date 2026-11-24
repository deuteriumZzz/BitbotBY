"""RAG по Decision Ledger (Phase 10, MASTER_SPEC §13.1): «что было в похожих случаях».

Хранилище — сам журнал (JSONL) + индекс BM25 в памяти: векторная БД (Qdrant) и
эмбеддинги не нужны, пока документов тысячи, а не миллионы; переход — по ADR.
Документ = одно решение (`signals.decisions[i]`) + итог исполнения + ИСХОД сделки
(`outcome`, пишется при закрытии позиции). Содержимое RAG — НЕДОВЕРЕННЫЕ данные
(`wrap_untrusted`): тексты могли попасть в журнал из LLM и новостей.

Это описательная память, а не доказательство: «в 7 из 10 похожих случаев был плюс»
не означает преимущества (малые выборки, общий режим рынка, пересечение случаев).
Сбой RAG не должен влиять на решение: `safe_context` возвращает пустой блок
(политика `rag_failure` из `backend/bitbot/ai/failure_policy.py`).
В промпт AI не подключён — это изменит вход модели; включается отдельным решением.
CLI:  python -m bitbot.ai.ledger_rag "rsi перепродан BTC" [--symbol BTC/USDT] [--k 5]
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

from bitbot.ai.injection_guard import sanitize, wrap_untrusted
from bitbot.domain.decision.explain import explain_decision
from bitbot.research.signal_intelligence_report import load_records

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if len(t) > 1]


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._tf: List[Counter] = []
        self._len: List[int] = []
        self._df: Counter = Counter()

    def add(self, text: str) -> int:
        toks = tokenize(text)
        tf = Counter(toks)
        self._tf.append(tf)
        self._len.append(len(toks))
        self._df.update(tf.keys())
        return len(self._tf) - 1

    def scores(self, query: str) -> List[float]:
        n = len(self._tf)
        if not n:
            return []
        avg = sum(self._len) / n or 1.0
        out = [0.0] * n
        for term in set(tokenize(query)):
            df = self._df.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self._tf):
                f = tf.get(term, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * self._len[i] / avg)
                    out[i] += idf * f * (self.k1 + 1) / norm
        return out


def _decision_text(
    d: Dict[str, Any], ex: Optional[Dict[str, Any]], oc: Optional[Dict[str, Any]]
) -> str:
    parts = [explain_decision(d)]
    if ex:
        parts.append(
            "исполнено"
            if ex.get("opened")
            else f"не исполнено {ex.get('reject') or ''}"
        )
    if oc:
        parts.append(
            f"исход {oc.get('reason') or ''} {'прибыль' if (oc.get('pnl') or 0) > 0 else 'убыток'}"
        )
    return " ".join(parts)


class LedgerRAG:
    def __init__(self) -> None:
        self._idx = BM25()
        self.docs: List[Dict[str, Any]] = []

    @classmethod
    def from_records(cls, records: Iterable[Dict[str, Any]]) -> "LedgerRAG":
        recs = list(records)
        ex_by = {
            r["decision_id"]: r
            for r in recs
            if r.get("kind") == "execution" and r.get("decision_id")
        }
        oc_by = {
            r["decision_id"]: r
            for r in recs
            if r.get("kind") == "outcome" and r.get("decision_id")
        }
        rag = cls()
        for r in recs:
            if r.get("kind") != "signals":
                continue
            for d in r.get("decisions", []):
                did = d.get("decision_id")
                if not did:
                    continue
                ex, oc = ex_by.get(did), oc_by.get(did)
                rag.docs.append(
                    {
                        "decision_id": did,
                        "ts": r.get("ts"),
                        "symbol": d.get("symbol"),
                        "status": d.get("status"),
                        "strategy": d.get("strategy"),
                        "regime": d.get("regime"),
                        "confidence": d.get("confidence"),
                        "opened": bool(ex and ex.get("opened")),
                        "outcome": oc,
                    }
                )
                rag._idx.add(_decision_text(d, ex, oc))
        return rag

    @classmethod
    def from_ledger(cls, path: str = "data/decision_ledger.jsonl") -> "LedgerRAG":
        return cls.from_records(load_records(path))

    def __len__(self) -> int:
        return len(self.docs)

    def search(
        self,
        query: str,
        k: int = 5,
        *,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        only_with_outcome: bool = False,
    ) -> List[Dict[str, Any]]:
        scores = self._idx.scores(query)

        def keep(d: Dict[str, Any]) -> bool:
            return (
                (symbol is None or d["symbol"] == symbol)
                and (status is None or d["status"] == status)
                and (not only_with_outcome or d["outcome"] is not None)
            )

        ranked = sorted(
            ((s, i) for i, s in enumerate(scores) if s > 0 and keep(self.docs[i])),
            reverse=True,
        )
        return [{**self.docs[i], "score": round(s, 4)} for s, i in ranked[:k]]

    def similar_outcomes(self, query: str, k: int = 20, **kw: Any) -> Dict[str, Any]:
        """Агрегат по исходам похожих случаев (описательный, малые выборки)."""
        hits = self.search(query, k, only_with_outcome=True, **kw)
        pnls = [
            float(h["outcome"].get("pnl"))
            for h in hits
            if h["outcome"].get("pnl") is not None
        ]
        return {
            "n_similar": len(hits),
            "n_with_pnl": len(pnls),
            "win_rate": (
                round(sum(p > 0 for p in pnls) / len(pnls), 3) if pnls else None
            ),
            "mean_pnl": round(sum(pnls) / len(pnls), 4) if pnls else None,
            "caveat": "описательная статистика на малой выборке, не доказательство преимущества",
        }

    def context_block(self, query: str, k: int = 3, **kw: Any) -> str:
        """Блок для промпта: недоверенные данные с цитированием decision_id."""
        hits = self.search(query, k, **kw)
        if not hits:
            return ""
        lines = []
        for h in hits:
            oc = h["outcome"]
            res = (
                f"исход {sanitize(str(oc.get('reason')), 20)} pnl={oc.get('pnl')}"
                if oc
                else ("открыта" if h["opened"] else "не открыта")
            )
            lines.append(
                f"[{sanitize(str(h['decision_id']), 24)}] {sanitize(str(h['symbol']), 20)} "
                f"{sanitize(str(h['status']), 12)} стратегия {sanitize(str(h['strategy']), 30)} "
                f"режим {sanitize(str(h['regime']), 16)}: {res}"
            )
        return wrap_untrusted("decision_ledger", lines, 240)


def safe_context(
    rag: Optional[LedgerRAG], query: str, k: int = 3, **kw: Any
) -> Dict[str, Any]:
    """Контекст или пустота при сбое: `rag_failure` → решение без RAG, с пометкой."""
    try:
        if rag is None:
            raise RuntimeError("RAG не инициализирован")
        return {"context": rag.context_block(query, k, **kw), "rag_failure": False}
    except Exception as exc:  # noqa: BLE001 — сбой памяти не должен менять решение
        return {
            "context": "",
            "rag_failure": True,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(
            'Использование: python -m bitbot.ai.ledger_rag "запрос" [--symbol X] [--k N]'
        )
        return 2
    opts: Dict[str, str] = {}
    words: List[str] = []
    while args:
        a = args.pop(0)
        if a in ("--symbol", "--k", "--ledger") and args:
            opts[a[2:]] = args.pop(0)
        else:
            words.append(a)
    rag = LedgerRAG.from_ledger(opts.get("ledger", "data/decision_ledger.jsonl"))
    q = " ".join(words)
    print(f"Документов в индексе: {len(rag)}")
    for h in rag.search(q, int(opts.get("k", 5)), symbol=opts.get("symbol")):
        oc = h["outcome"]
        print(
            f"  {h['score']:.2f} {h['decision_id']} {h['symbol']} {h['status']} {h['strategy']} "
            f"{'исход ' + str(oc.get('pnl')) if oc else 'без исхода'}"
        )
    print(rag.similar_outcomes(q, symbol=opts.get("symbol")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
