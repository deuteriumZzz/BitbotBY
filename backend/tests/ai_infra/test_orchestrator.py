"""AI Orchestrator L1–L4: план по контексту и бюджету, выполнение с таймаутами и частичными сбоями."""

import asyncio

import pytest

from bitbot.ai.analyzer import PROMPT_VERSION
from bitbot.ai.orchestrator import (
    OrchContext,
    Plan,
    failures_never_confirm,
    plan,
    run_plan,
)


def ctx(**kw):
    base = dict(
        data_ok=True,
        n_independent_groups=3,
        has_conflict=False,
        meta_prob=0.8,
        ai_available=True,
        sac_ready=True,
        forecaster_ready=True,
    )
    base.update(kw)
    return OrchContext(**base)


def test_bad_data_is_a_safe_refusal():
    p = plan(ctx(data_ok=False))
    assert p.level == "L0" and p.stages == [] and p.reasons == ["insufficient_data"]


def test_l1_is_enough_when_decisive_independent_and_unconflicted():
    p = plan(ctx())
    assert (
        p.level == "L1"
        and p.stages == [["signal_intel", "meta_model"]]
        and "l1_sufficient" in p.reasons
    )
    assert plan(ctx(meta_prob=0.2)).level == "L1"  # уверенное «против» тоже решает


@pytest.mark.parametrize(
    "kw", [dict(meta_prob=0.55), dict(n_independent_groups=1), dict(has_conflict=True)]
)
def test_uncertainty_escalates(kw):
    assert plan(ctx(**kw)).level != "L1"


def test_full_escalation_with_conflict_reaches_l4_in_order():
    p = plan(ctx(meta_prob=0.5, has_conflict=True))
    assert p.level == "L4"
    assert p.stages == [
        ["signal_intel", "meta_model"],
        ["sac", "forecast"],
        ["llm", "rag"],
        ["recheck"],
    ]
    assert p.est_ms > 0


def test_no_meta_model_means_l1_has_only_signal_intel_and_llm_is_consulted():
    p = plan(ctx(meta_prob=None, n_independent_groups=1))
    assert p.stages[0] == ["signal_intel"] and p.level == "L3"


def test_unavailable_components_are_skipped_with_reasons():
    p = plan(
        ctx(meta_prob=0.5, sac_ready=False, forecaster_ready=False, ai_available=False)
    )
    assert p.level == "L1" and p.skipped == {
        "sac": "not_ready",
        "forecast": "not_ready",
        "llm": "ai_unavailable",
    }


def test_budget_limits_expensive_stages():
    p = plan(ctx(meta_prob=0.5, budget_ms=100))
    assert p.stages == [["signal_intel", "meta_model"]]
    assert p.skipped["forecast"] == "budget" and p.skipped["llm"] == "budget"


async def test_run_plan_runs_stage_in_parallel_and_collects_partial_failures():
    async def ok():
        return 1

    async def boom():
        raise RuntimeError("упал")

    async def slow():
        await asyncio.sleep(5)

    pl = Plan("L3", stages=[["signal_intel"], ["sac", "forecast", "llm"]])
    res = await run_plan(
        pl,
        {"signal_intel": ok, "sac": boom, "forecast": slow, "llm": ok},
        timeouts={"forecast": 0.05},
    )
    assert res["signal_intel"].ok and res["llm"].ok and res["llm"].value == 1
    assert (
        not res["sac"].ok
        and "RuntimeError" in res["sac"].error
        and res["sac"].policy == "no_vote"
    )
    assert (
        not res["forecast"].ok
        and res["forecast"].error == "timeout"
        and res["forecast"].policy == "neutral"
    )
    assert failures_never_confirm(res)


async def test_missing_runner_is_a_failure_not_a_crash_and_duplicates_run_once():
    calls = []

    async def once():
        calls.append(1)
        await asyncio.sleep(0.01)
        return "v"

    pl = Plan(
        "L2", stages=[["llm"], ["llm", "rag"]]
    )  # llm в двух стадиях — запрос не дублируется
    res = await run_plan(pl, {"llm": once})
    assert calls == [1] and res["llm"].ok
    assert res["rag"].error == "no_runner" and res["rag"].policy == "degrade"


async def test_cancellation_cancels_running_components():
    state = {"cancelled": False}

    async def forever():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    task = asyncio.create_task(run_plan(Plan("L2", stages=[["sac"]]), {"sac": forever}))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.05)
    assert state["cancelled"]


async def test_stop_after_skips_expensive_later_stages():
    ran = []

    def runner(name, value):
        async def f():
            ran.append(name)
            return value

        return f

    p = plan(ctx(meta_prob=0.5))  # L1 → L2 → L3
    runners = {
        n: runner(n, 1)
        for n in ("signal_intel", "meta_model", "sac", "forecast", "llm", "rag")
    }
    res = await run_plan(
        p, runners, stop_after=lambda r: "sac" in r
    )  # SAC уже ответил — LLM не нужен
    assert "llm" not in ran and "rag" not in ran and res["sac"].ok
    ran.clear()
    await run_plan(p, runners)  # без условия остановки — вся лестница
    assert "llm" in ran


def test_prompt_version_is_defined():
    assert PROMPT_VERSION.startswith("v") and len(PROMPT_VERSION) > 3
