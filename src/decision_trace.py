"""След решения: что бот сделал на каждом шаге конвейера.

Ничего не решает и не меняет — только слушает события, которые бот и так
публикует (radar.stage, ai.status, cycle, guard), и собирает из них:

- итог последнего цикла по узлам конвейера (Сканер → … → Сопровождение) —
  его показывает вкладка «Бот» (Redis bot:last_cycle, событие trace.cycle);
- шаги решения по монете — их сохраняют вместе с открытой сделкой
  («Почему открыта» в терминале и журнале).

Шаг — {"node", "ok", "text"}: ok=True — узел пропустил дальше (▲),
False — остановил (▼), None — пропущен или не участвует (–).
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from src.ai_providers import BY_ID
from src.experience_buffer import LOCAL_REASONING_PREFIX

logger = logging.getLogger(__name__)

Step = Dict[str, Any]

NODES = ("scan", "strat", "ai", "sac", "risk", "order", "track")

_AI_FAIL = {
    "rate_limit": "лимит",
    "no_balance": "нет денег",
    "bad_key": "ключ не принят",
    "bad_model": "нет такой модели",
    "bad_format": "сломал формат ответа",
    "bad_request": "отклонил запрос",
    "error": "нет связи",
}


def takes_part(mode: str, node: str) -> bool:
    """Как в SignalCombiner: local — без AI и SAC, ai — без SAC, dqn — без AI."""
    if node == "ai":
        return mode in ("ai", "hybrid")
    if node == "sac":
        return mode in ("dqn", "hybrid")
    return True


def _step(node: str, ok: Optional[bool], text: str) -> Step:
    return {"node": node, "ok": ok, "text": text[:200]}


def _name(provider: str) -> str:
    p = BY_ID.get(provider)
    return p.name if p else provider


def _plural(n: int, one: str, few: str, many: str) -> str:
    if 11 <= n % 100 <= 14:
        return many
    if n % 10 == 1:
        return one
    if 2 <= n % 10 <= 4:
        return few
    return many


def _coins(syms: List[str], sides: Dict[str, str]) -> str:
    shown = [
        f"{s.split('/')[0]} {'▲' if sides.get(s) == 'long' else '▼'}" for s in syms
    ]
    return ", ".join(shown[:3]) + ("…" if len(shown) > 3 else "")


class DecisionTrace:
    """Слушатель событий одного торгового цикла.

    :param mode: Текущий MODE бота (local/ai/dqn/hybrid), спрашивается в начале
        цикла.
    :param store: Куда отдать итог цикла (Redis + событие); None — никуда.
    """

    def __init__(
        self,
        mode: Callable[[], str],
        store: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self._mode_fn = mode
        self._store = store
        self._reset("idle")

    def _reset(self, state: str) -> None:
        try:
            self._mode = str(self._mode_fn())
        except Exception:  # бот ещё не до конца собран — режим по умолчанию
            self._mode = "ai"
        self._state = state
        self._at = time.time()
        self._scanned = 0
        self._candidates: Set[str] = set()
        self._ai: List[Tuple[str, str]] = []
        self._signals: Dict[str, Dict[str, Any]] = {}
        self._rejected: Dict[str, str] = {}
        self._opened: Dict[str, str] = {}
        self._guard = ""
        self._used = 0
        self._max = 0

    # ── События ───────────────────────────────────────────────────────────────

    def on_event(self, type_: str, data: Dict[str, Any]) -> None:
        """Принимает событие бота. Никогда не бросает: след не мешает торговле."""
        try:
            self._on_event(type_, data or {})
        except Exception as e:
            logger.debug("decision trace: %s skipped: %s", type_, e)

    def _on_event(self, type_: str, d: Dict[str, Any]) -> None:
        if type_ == "radar.stage":
            stage = d.get("stage")
            sym = str(d.get("symbol") or "")
            if stage == "scan":
                self._reset("scan")
                self._scanned = len(d.get("symbols") or [])
            elif stage == "prescore":
                self._candidates = {str(s) for s in d.get("symbols") or []}
            elif stage == "signal" and sym:
                self._signals[sym] = d
            elif stage == "rejected" and sym:
                self._rejected[sym] = str(d.get("reason") or "")
            elif stage == "open" and sym:
                self._opened[sym] = str(self._signals.get(sym, {}).get("side") or "")
        elif type_ == "ai.status" and d.get("provider"):
            self._ai.append((str(d["provider"]), str(d.get("status") or "")))
        elif type_ == "cycle":
            state = str(d.get("state") or "")
            if state == "paused":
                self._reset("paused")
                self._save()
            elif state in ("scan", "monitor"):
                self._used = int(d.get("used") or 0)
                self._max = int(d.get("max") or 0)
                if state == "monitor":
                    self._state = "monitor"
                    self._save()
        elif type_ == "guard" and d.get("active"):
            self._guard = str(d.get("detail") or d.get("kind") or "защита")
            self._save()

    def end_cycle(self) -> None:
        """Конец цикла: отдать итог."""
        self._save()

    def _save(self) -> None:
        if self._store is None:
            return
        try:
            self._store(self.summary())
        except Exception as e:
            logger.debug("decision trace: summary not stored: %s", e)

    # ── Итог цикла ────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        return {
            "at": self._at,
            "mode": self._mode,
            "state": self._state,
            "nodes": self._nodes(),
        }

    def _source(self, sig: Dict[str, Any]) -> str:
        """Кто дал сигнал: local, context, sac, hybrid или ai."""
        strat = str(sig.get("strategy") or "")
        if strat.startswith("hybrid("):
            return "hybrid"
        if strat == "sac":
            return "sac"
        if strat == "context_signal":
            return "context"
        if self._mode == "local" or str(sig.get("reasoning") or "").startswith(
            LOCAL_REASONING_PREFIX
        ):
            return "local"
        return "ai"

    def _ai_answer(self) -> Tuple[Optional[str], List[str]]:
        """Кто из провайдеров ответил и кто до него не смог."""
        ok = [p for p, s in self._ai if s == "ok"]
        failed = [
            f"{_name(p)} — {_AI_FAIL.get(s, s)}" for p, s in self._ai if s != "ok"
        ]
        return (_name(ok[-1]) if ok else None), failed

    def _nodes(self) -> Dict[str, Step]:
        if self._state == "idle":
            return {}
        if self._state == "paused":
            n = {
                node: _step(node, None, "пауза — новые входы остановлены")
                for node in NODES
            }
            n["track"] = _step("track", None, "пауза: открытые позиции сопровождаются")
            return n
        m = self._mode
        sides = {s: str(sig.get("side") or "") for s, sig in self._signals.items()}
        by_src: Dict[str, List[str]] = {}
        for sym, sig in self._signals.items():
            by_src.setdefault(self._source(sig), []).append(sym)
        cand = len(self._candidates)
        scanned = (
            f"{self._scanned} {_plural(self._scanned, 'монета', 'монеты', 'монет')}"
        )
        n = {
            "scan": _step(
                "scan",
                True if self._scanned else None,
                f"{scanned} · {cand} в кандидаты" if cand else scanned,
            )
        }
        skipped = ""
        if self._state == "monitor":
            skipped = "пропущено: все слоты заняты"
        elif self._guard and not self._signals and not cand:
            skipped = f"пропущено: {self._guard}"

        local = by_src.get("local", []) + by_src.get("context", [])
        if skipped:
            n["strat"] = _step("strat", None, skipped)
        elif local:
            label = "сигналы" if m == "local" else "запасной путь"
            n["strat"] = _step("strat", True, f"{label}: {_coins(local, sides)}")
        elif m == "local":
            n["strat"] = _step("strat", None, "сигналов нет")
        elif takes_part(m, "ai"):
            n["strat"] = _step("strat", None, "подсказка для AI")
        else:
            n["strat"] = _step("strat", None, "только запасной путь")

        if not takes_part(m, "ai"):
            n["ai"] = _step("ai", None, "не участвует")
        elif skipped:
            n["ai"] = _step("ai", None, skipped)
        else:
            who, failed = self._ai_answer()
            got = len(by_src.get("ai", [])) + len(by_src.get("hybrid", []))
            signals = f"{got} {_plural(got, 'сигнал', 'сигнала', 'сигналов')}"
            if who:
                after = f" после сбоя: {', '.join(failed)}" if failed else ""
                n["ai"] = _step("ai", True, f"ответил {who}{after} · {signals}")
            elif failed:
                n["ai"] = _step("ai", False, f"не ответил никто: {', '.join(failed)}")
            else:
                n["ai"] = _step(
                    "ai",
                    None,
                    "ответа нет" if cand else "не спрашивали — нет кандидатов",
                )

        if not takes_part(m, "sac"):
            n["sac"] = _step("sac", None, "не участвует")
        elif skipped:
            n["sac"] = _step("sac", None, skipped)
        else:
            got = len(by_src.get("sac", [])) + len(by_src.get("hybrid", []))
            n["sac"] = (
                _step(
                    "sac",
                    True,
                    f"{got} {_plural(got, 'сигнал', 'сигнала', 'сигналов')}",
                )
                if got
                else _step("sac", None, "сигналов нет")
            )

        n["risk"] = self._risk()
        n["order"] = (
            _step("order", True, f"открыта: {_coins(list(self._opened), self._opened)}")
            if self._opened
            else _step("order", None, "сделок нет")
        )
        held = self._used + len(self._opened)
        n["track"] = (
            _step(
                "track",
                True,
                f"сопровождает {held} {_plural(held, 'позицию', 'позиции', 'позиций')}",
            )
            if held
            else _step("track", None, "позиций нет")
        )
        return n

    def _risk(self) -> Step:
        if self._state == "monitor":
            return _step("risk", False, f"все слоты заняты ({self._used}/{self._max})")
        if self._guard:
            return _step("risk", False, self._guard)
        if not self._signals:
            return _step("risk", None, "нечего проверять")
        rej = len(self._rejected)
        reason = Counter(self._rejected.values()).most_common(1)[0][0] if rej else ""
        if self._opened:
            tail = f" · отсеяно {rej}: {reason}" if rej else ""
            return _step("risk", True, f"прошли {len(self._opened)}{tail}")
        if rej:
            return _step("risk", False, f"отсеяно {rej}: {reason}")
        return _step("risk", None, "до проверки не дошли")

    # ── Шаги по сделке ────────────────────────────────────────────────────────

    def opened(self, sym: str, pos: Dict[str, Any]) -> List[Step]:
        """Шаги решения по только что открытой сделке; заодно обновляет итог."""
        side = "long" if pos.get("side", "buy") == "buy" else "short"
        self._opened[sym] = side
        sig = self._signals.get(sym) or {}
        src = self._source(sig) if sig else ("local" if self._mode == "local" else "ai")
        m = self._mode
        conf = sig.get("confidence", pos.get("confidence"))
        conf_txt = f" · {conf:.2f}" if isinstance(conf, (int, float)) else ""
        strat = str(sig.get("strategy") or pos.get("strategy") or "")
        word = "LONG" if side == "long" else "SHORT"

        if self._scanned:
            pre = " · прошла предфильтр" if sym in self._candidates else ""
            steps = [_step("scan", True, f"в выборке из {self._scanned} монет{pre}")]
        else:
            steps = [_step("scan", None, "без данных о скане")]

        if src == "context":
            steps.append(_step("strat", True, "сигнал по контексту рынка"))
        elif src == "local":
            steps.append(_step("strat", True, f"стратегия «{strat}»"))
        elif takes_part(m, "ai") and src in ("ai", "hybrid"):
            steps.append(
                _step("strat", None, f"подсказка: AI отнёс сигнал к «{strat}»")
            )
        else:
            steps.append(_step("strat", None, "не участвовали"))

        if not takes_part(m, "ai"):
            steps.append(_step("ai", None, "не участвует"))
        elif src in ("ai", "hybrid"):
            who, failed = self._ai_answer()
            after = f" (после сбоя: {', '.join(failed)})" if failed else ""
            steps.append(_step("ai", True, f"{who or 'AI'}: {word}{conf_txt}{after}"))
        elif src == "sac":
            steps.append(_step("ai", False, "AI промолчал — решил SAC"))
        elif src == "context":
            steps.append(_step("ai", None, "сигнал без AI — по контексту рынка"))
        else:
            steps.append(
                _step("ai", False, "AI не дал сигнала — сработал запасной путь")
            )

        if not takes_part(m, "sac"):
            steps.append(_step("sac", None, "не участвует"))
        elif src in ("sac", "hybrid"):
            steps.append(_step("sac", True, str(sig.get("reasoning") or "сигнал SAC")))
        else:
            steps.append(_step("sac", None, "сигнала не дал"))

        steps.append(_step("risk", True, "проверки пройдены"))
        entry = pos.get("entry")
        price = f" по {entry:g}" if isinstance(entry, (int, float)) else ""
        steps.append(_step("order", True, f"открыта {word}{price}"))
        self._save()
        return steps
