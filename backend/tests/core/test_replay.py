"""Phase 4: воспроизведение решений комбинатора на записанных входах."""

from __future__ import annotations

import copy
import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from bitbot.domain.signals.combiner import SignalCombiner
from bitbot.research import replay as RP
from bitbot.storage.decision_ledger import DecisionLedger
from tests.core.test_reconciliation import make_bot
from tests.core.test_signal_combiner import make_rec, make_snapshot

# ── Подставные источники для «живого» прогона ────────────────────────────────


class _AI:
    unavailable = False

    def __init__(self, recs):
        self._recs = recs

    async def analyze(self, snapshots, balance):
        return copy.deepcopy(self._recs)


class _SAC:
    loaded = True

    def __init__(self, signals):
        self._s = signals

    def get_signal(self, snap, balance, position=0.0, entry_price=0.0):
        return dict(self._s.get(snap["symbol"], {"action": "hold", "confidence": 0.0}))


class _Fc:
    def __init__(self, by_marker):
        self._m = by_marker

    def predict_direction(self, close_prices, threshold_pct=0.002):
        return self._m.get(close_prices[0], "neutral")


def _rc(mode: str, min_conf: float = 0.4, chronos: bool = True) -> MagicMock:
    rc = MagicMock()
    rc.get_mode.return_value = mode
    rc.get_signal_confidence.return_value = min_conf
    rc.get_chronos_enabled.return_value = chronos
    rc.get_market_type.return_value = "linear"
    rc.get_market_profile.return_value = "bluechip"
    return rc


SYMS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
SNAPS = [make_snapshot(s, 100.0 + i, 2.0) for i, s in enumerate(SYMS)]
REGIMES = {
    "BTC/USDT": "trending_down",
    "ETH/USDT": "trending_up",
    "SOL/USDT": "ranging",
}
CTX = {
    "BTC/USDT": {"funding_signal": "long_overheated", "funding_rate": 0.002},
    "ETH/USDT": {"funding_signal": "short_overheated", "funding_rate": -0.002},
    "SOL/USDT": {},
}
MARKET_DATA = {
    "BTC/USDT": pd.DataFrame({"close": [1.5] * 5}),
    "ETH/USDT": pd.DataFrame({"close": [2.5] * 5}),
    "SOL/USDT": pd.DataFrame({"close": [3.5] * 5}),
}


async def _live_hybrid():
    ai_recs = [
        make_rec("BTC/USDT", "buy", 0.8),
        make_rec("ETH/USDT", "buy", 0.7),
        make_rec("SOL/USDT", "buy", 0.8),
    ]
    sac = {
        "BTC/USDT": {"action": "buy", "confidence": 0.75},
        "ETH/USDT": {"action": "buy", "confidence": 0.7},
        "SOL/USDT": {"action": "hold", "confidence": 0.0},  # SAC вето
    }
    combiner = SignalCombiner(
        ai=_AI(ai_recs),
        rc=_rc("hybrid"),
        sac=_SAC(sac),
        forecaster=_Fc({1.5: "down", 2.5: "up"}),
    )
    recs = await combiner.combine(
        copy.deepcopy(SNAPS),
        10_000.0,
        regime="ranging",
        regimes=REGIMES,
        market_context=copy.deepcopy(CTX),
        market_data=MARKET_DATA,
    )
    return combiner, recs


def _record(combiner, recs, ctx=CTX):
    inp = RP.build_replay_input(combiner.last_inputs, SNAPS, 10_000.0, ctx, recs)
    return json.loads(json.dumps(inp))  # как в журнале: через JSON


# ── Воспроизведение ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hybrid_decisions_replay_bit_for_bit_through_json() -> None:
    combiner, recs = await _live_hybrid()
    by_sym = {r["symbol"]: r for r in recs if r["strategy"] != "context_signal"}
    # сценарий нетривиален: Chronos против BTC, бонус и merge у ETH, вето SAC у SOL
    assert set(by_sym) == {"BTC/USDT", "ETH/USDT"} and "SOL/USDT" not in by_sym
    assert by_sym["BTC/USDT"]["chronos_signal"] == "down"
    assert by_sym["ETH/USDT"].get("context_boost") is True
    inp = _record(combiner, recs)
    assert [s["symbol"] for s in inp["snapshots"]] == ["BTC/USDT", "ETH/USDT"]
    assert await RP.replay_and_compare(inp) == []


@pytest.mark.asyncio
async def test_ai_mode_and_sac_only_mode_replay() -> None:
    ai_mode = SignalCombiner(
        ai=_AI([make_rec("ETH/USDT", "buy", 0.8)]),
        rc=_rc("ai", chronos=False),
        sac=_SAC({}),
    )
    recs = await ai_mode.combine(
        copy.deepcopy(SNAPS), 10_000.0, "ranging", REGIMES, copy.deepcopy(CTX)
    )
    assert recs
    assert await RP.replay_and_compare(_record(ai_mode, recs)) == []

    sac_mode = SignalCombiner(
        ai=_AI([]),
        rc=_rc("dqn", chronos=False),
        sac=_SAC({"BTC/USDT": {"action": "buy", "confidence": 0.9}}),
    )
    sac_recs = await sac_mode.combine(
        copy.deepcopy(SNAPS), 10_000.0, "ranging", REGIMES
    )
    assert [r["strategy"] for r in sac_recs] == ["sac"]
    # боевой вызов шёл без контекста рынка — и запись должна быть без него
    assert await RP.replay_and_compare(_record(sac_mode, sac_recs, None)) == []
    # а если подсунуть контекст, которого не было, replay это обнаружит
    assert await RP.replay_and_compare(_record(sac_mode, sac_recs, CTX)) != []


# ── Обнаружение расхождений ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tampered_inputs_are_detected() -> None:
    combiner, recs = await _live_hybrid()
    base = _record(combiner, recs)

    changed_ai = copy.deepcopy(base)
    changed_ai["ai_recs"][0]["confidence"] = 0.99
    assert any("confidence" in p for p in await RP.replay_and_compare(changed_ai))

    changed_expected = copy.deepcopy(base)
    changed_expected["expected"][0]["confidence"] += 0.01
    assert await RP.replay_and_compare(changed_expected) != []

    stricter = copy.deepcopy(base)
    stricter["settings"]["min_conf"] = 0.99
    problems = await RP.replay_and_compare(stricter)
    assert any("записано, но не воспроизведено" in p for p in problems)

    changed_chronos = copy.deepcopy(base)
    changed_chronos["chronos"]["BTC/USDT"] = "up"
    assert await RP.replay_and_compare(changed_chronos) != []


@pytest.mark.asyncio
async def test_unknown_schema_is_rejected() -> None:
    combiner, recs = await _live_hybrid()
    inp = _record(combiner, recs)
    inp["schema"] = 99
    with pytest.raises(ValueError):
        await RP.replay_combine(inp)


def test_build_returns_none_when_nothing_to_replay() -> None:
    ai_settings = {"settings": {"mode": "ai"}}
    assert RP.build_replay_input(ai_settings, SNAPS, 1.0, None, []) is None
    assert RP.build_replay_input({}, SNAPS, 1.0, None, [make_rec()]) is None
    bad = {"settings": {"incomplete": True}}
    assert RP.build_replay_input(bad, SNAPS, 1.0, None, [make_rec()]) is None


@pytest.mark.asyncio
async def test_capture_does_not_change_the_result() -> None:
    """Захват входов — только наблюдение: два прогона дают одинаковый результат."""
    _, first = await _live_hybrid()
    _, second = await _live_hybrid()
    assert [RP.compact_expected(r) for r in first] == [
        RP.compact_expected(r) for r in second
    ]


# ── Интеграция с ботом и CLI ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bot_writes_replay_input_that_replays_clean(tmp_path) -> None:
    bot = make_bot()
    path = tmp_path / "l.jsonl"
    bot.ledger = DecisionLedger(path=str(path), enabled=True)
    bot.ledger.begin_cycle(1)
    bot.combiner = SignalCombiner(
        ai=_AI([make_rec("ETH/USDT", "buy", 0.8)]),
        rc=_rc("ai", chronos=False),
        sac=_SAC({}),
    )
    bot._cycle.optimize_allocation = MagicMock(side_effect=lambda r, md: r)
    bot._runtime_config = _rc("ai", chronos=False)

    recs = await bot._generate_signals(
        copy.deepcopy(SNAPS), 10_000.0, REGIMES, {}, copy.deepcopy(CTX)
    )
    assert recs
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    kinds = [x["kind"] for x in lines]
    assert "replay_input" in kinds and "signals" in kinds
    rec = next(x for x in lines if x["kind"] == "replay_input")
    assert await RP.replay_and_compare(rec["input"]) == []


@pytest.mark.asyncio
async def test_cli_ok_on_clean_record(tmp_path, capsys) -> None:
    combiner, recs = await _live_hybrid()
    path = tmp_path / "l.jsonl"
    path.write_text(
        json.dumps(
            {
                "kind": "replay_input",
                "trace_id": "t-0",
                "input": _record(combiner, recs),
            }
        )
        + "\n"
    )
    # main() работает в приватном цикле — безопасно и из тела async-теста через поток
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(1) as ex:
        assert ex.submit(RP.main, [str(path)]).result() == 0
    assert "t-0: OK" in capsys.readouterr().out


def test_run_helper_uses_a_private_loop() -> None:
    """`_run` не должен подменять и закрывать текущий цикл потока (в отличие от
    asyncio.run, который обнуляет его и ломает последующий код)."""
    import asyncio

    seen = {}

    async def probe() -> int:
        seen["running"] = asyncio.get_running_loop()
        return 7

    assert RP._run(probe()) == 7
    assert seen["running"].is_closed()  # приватный цикл закрыт после работы


@pytest.mark.asyncio
async def test_cli_reports_divergence_and_missing_input(tmp_path) -> None:
    combiner, recs = await _live_hybrid()
    inp = _record(combiner, recs)
    inp["expected"][0]["confidence"] = 0.123
    path = tmp_path / "l.jsonl"
    path.write_text(
        json.dumps({"kind": "replay_input", "trace_id": "t-1", "input": inp}) + "\n"
    )
    # main() сам запускает цикл событий — вызываем из потока без активного цикла
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(1) as ex:
        assert ex.submit(RP.main, [str(path)]).result() == 1
        assert ex.submit(RP.main, [str(tmp_path / "none.jsonl")]).result() == 1


@pytest.mark.asyncio
async def test_bot_skips_replay_when_ledger_disabled_or_no_recs(tmp_path) -> None:
    bot = make_bot()
    bot.ledger = DecisionLedger(path=str(tmp_path / "l.jsonl"), enabled=False)
    bot._record_replay_input(SNAPS, 1.0, None, [make_rec()])
    bot.ledger = DecisionLedger(path=str(tmp_path / "l.jsonl"), enabled=True)
    bot._record_replay_input(SNAPS, 1.0, None, [])
    assert not (tmp_path / "l.jsonl").exists()
