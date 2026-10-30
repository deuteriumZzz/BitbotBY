"""След решения: собирается из событий бота и ничего не меняет в торговле."""

import json

from src.ai.experience_buffer import LOCAL_REASONING_PREFIX
from src.core.decision_trace import DecisionTrace, takes_part
from src.core.trade_history import TradeHistory
from src.dashboard.data import _steps


def _ai_cycle(t: DecisionTrace) -> None:
    for type_, data in [
        (
            "radar.stage",
            {"stage": "scan", "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"]},
        ),
        ("cycle", {"state": "scan", "used": 1, "max": 3}),
        ("radar.stage", {"stage": "prescore", "symbols": ["BTC/USDT", "ETH/USDT"]}),
        ("ai.status", {"provider": "groq", "status": "rate_limit"}),
        ("ai.status", {"provider": "gemini", "status": "ok"}),
        (
            "radar.stage",
            {
                "stage": "signal",
                "symbol": "BTC/USDT",
                "side": "long",
                "confidence": 0.71,
                "strategy": "breakout",
                "reasoning": "пробой с объёмом",
            },
        ),
        (
            "radar.stage",
            {
                "stage": "signal",
                "symbol": "ETH/USDT",
                "side": "short",
                "strategy": "amd",
            },
        ),
        ("radar.stage", {"stage": "guards", "symbol": "BTC/USDT"}),
        ("radar.stage", {"stage": "open", "symbol": "BTC/USDT"}),
        ("radar.stage", {"stage": "guards", "symbol": "ETH/USDT"}),
        (
            "radar.stage",
            {"stage": "rejected", "symbol": "ETH/USDT", "reason": "сильная корреляция"},
        ),
    ]:
        t.on_event(type_, data)


def test_ai_cycle_summary_per_node():
    saved: list = []
    t = DecisionTrace(mode=lambda: "ai", store=saved.append)
    _ai_cycle(t)
    t.end_cycle()
    n = saved[-1]["nodes"]
    assert n["scan"] == {"node": "scan", "ok": True, "text": "3 монеты · 2 в кандидаты"}
    assert n["strat"]["ok"] is None and n["strat"]["text"] == "подсказка для AI"
    assert n["ai"]["ok"] is True
    assert "Gemini" in n["ai"]["text"] and "Groq — лимит" in n["ai"]["text"]
    assert n["ai"]["text"].endswith("· 2 сигнала")
    assert n["sac"] == {"node": "sac", "ok": None, "text": "не участвует"}
    assert n["risk"] == {
        "node": "risk",
        "ok": True,
        "text": "прошли 1 · отсеяно 1: сильная корреляция",
    }
    assert n["order"] == {"node": "order", "ok": True, "text": "открыта: BTC ▲"}
    assert n["track"]["text"] == "сопровождает 2 позиции"


def test_steps_saved_with_opened_trade():
    t = DecisionTrace(mode=lambda: "ai")
    _ai_cycle(t)
    steps = t.opened("BTC/USDT", {"side": "buy", "entry": 67412.5})
    assert [s["node"] for s in steps] == ["scan", "strat", "ai", "sac", "risk", "order"]
    assert steps[0]["text"] == "в выборке из 3 монет · прошла предфильтр"
    assert steps[1]["text"] == "подсказка: AI отнёс сигнал к «breakout»"
    assert steps[2]["ok"] is True and steps[2]["text"].startswith("Gemini: LONG · 0.71")
    assert "после сбоя: Groq — лимит" in steps[2]["text"]
    assert steps[3] == {"node": "sac", "ok": None, "text": "не участвует"}
    assert steps[5] == {"node": "order", "ok": True, "text": "открыта LONG по 67412.5"}


def test_slots_full_cycle_says_why():
    saved: list = []
    t = DecisionTrace(mode=lambda: "ai", store=saved.append)
    t.on_event("radar.stage", {"stage": "scan", "symbols": ["BTC/USDT"]})
    t.on_event("cycle", {"state": "monitor", "used": 3, "max": 3})
    n = saved[-1]["nodes"]
    assert n["risk"] == {"node": "risk", "ok": False, "text": "все слоты заняты (3/3)"}
    assert n["ai"]["text"] == "пропущено: все слоты заняты"
    assert n["track"]["text"] == "сопровождает 3 позиции"


def test_local_fallback_is_named_honestly():
    t = DecisionTrace(mode=lambda: "ai")
    t.on_event("radar.stage", {"stage": "scan", "symbols": ["SOL/USDT"]})
    t.on_event(
        "radar.stage",
        {
            "stage": "signal",
            "symbol": "SOL/USDT",
            "side": "long",
            "strategy": "breakout",
            "reasoning": f"{LOCAL_REASONING_PREFIX} «breakout»: сигнал на покупку",
        },
    )
    assert t.summary()["nodes"]["strat"]["text"] == "запасной путь: SOL ▲"
    steps = t.opened("SOL/USDT", {"side": "buy", "entry": 150.0})
    assert steps[1] == {"node": "strat", "ok": True, "text": "стратегия «breakout»"}
    assert steps[2] == {
        "node": "ai",
        "ok": False,
        "text": "AI не дал сигнала — сработал запасной путь",
    }


def test_paused_and_broken_events_never_raise():
    def no_runtime_config() -> str:
        raise RuntimeError("runtime config is not ready")

    saved: list = []
    t = DecisionTrace(mode=no_runtime_config, store=saved.append)
    t.on_event("radar.stage", None)  # type: ignore[arg-type]
    t.on_event("cycle", {"state": "monitor", "used": "много"})
    t.on_event("cycle", {"state": "paused"})
    assert saved[-1]["state"] == "paused" and saved[-1]["mode"] == "ai"
    assert saved[-1]["nodes"]["order"]["ok"] is None
    assert not takes_part("local", "ai") and takes_part("hybrid", "sac")


async def test_trace_column_saved_and_read_back(tmp_path):
    th = TradeHistory(db_path=str(tmp_path / "trades.db"))
    tid = await th.record_open(
        symbol="BTC/USDT",
        strategy="breakout",
        action="buy",
        entry_price=1.0,
        quantity=1.0,
        confidence=0.7,
    )
    await th.set_trace(tid, [{"node": "ai", "ok": True, "text": "Gemini: LONG"}])
    row = th._conn.execute("SELECT trace FROM trades WHERE id = ?", (tid,)).fetchone()
    assert _steps(row[0]) == json.loads(row[0])
    assert _steps(row[0])[0]["text"] == "Gemini: LONG"
    assert _steps("") == [] and _steps("{сломано") == [] and _steps('{"a": 1}') == []
