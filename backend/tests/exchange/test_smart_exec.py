"""Умное исполнение: TWAP, Post-Only chase, частичные исполнения, резерв маржи."""

import threading

import pytest

from bitbot.adapters.exchange import smart_exec as se


# ── TWAP ──────────────────────────────────────────────────────────────────────
def test_twap_sums_to_total_with_remainder_in_last_slice():
    plan = se.twap_plan(1.0, 3, 60, step=0.001, min_qty=0.01)
    assert len(plan) == 3 and sum(s["qty"] for s in plan) == pytest.approx(1.0)
    assert plan[0]["qty"] == pytest.approx(0.333) and plan[-1]["qty"] == pytest.approx(
        0.334
    )
    assert [s["at_s"] for s in plan] == [0.0, 20.0, 40.0]


def test_twap_rounds_down_to_step_and_respects_minimum_order():
    assert sum(
        s["qty"] for s in se.twap_plan(1.0005, 2, 10, step=0.001, min_qty=0.01)
    ) == pytest.approx(1.0)
    p = se.twap_plan(0.03, 10, 10, step=0.001, min_qty=0.01)
    assert len(p) == 3 and all(s["qty"] >= 0.01 for s in p)  # не дробим ниже минимума
    one = se.twap_plan(0.05, 1, 0, step=0.01, min_qty=0.01)
    assert one == [{"at_s": 0.0, "qty": 0.05}]


def test_twap_validation():
    for args in (
        (0, 3, 10, 0.1, 0.1),
        (1, 0, 10, 0.1, 0.1),
        (1, 3, 10, 0, 0.1),
        (1, 3, -1, 0.1, 0.1),
    ):
        with pytest.raises(ValueError):
            se.twap_plan(*args)
    with pytest.raises(ValueError):
        se.twap_plan(0.005, 2, 10, step=0.001, min_qty=0.01)


# ── Post-Only chase ───────────────────────────────────────────────────────────
def test_chase_places_at_own_side_of_the_book_then_holds():
    st = se.ChaseState(started_at=0)
    assert se.chase_next("buy", st, 100.0, 100.1, 0.1, 1) == {
        "action": "place",
        "price": 100.0,
    }
    assert se.chase_next("buy", st, 100.0, 100.1, 0.1, 2)["action"] == "hold"
    st2 = se.ChaseState(started_at=0)
    assert (
        se.chase_next("sell", st2, 100.0, 100.1, 0.1, 1)["price"] == 100.1
    )  # продаём по ask


def test_chase_reprices_when_market_runs_away_and_falls_back_after_limit():
    st = se.ChaseState(started_at=0)
    se.chase_next("buy", st, 100.0, 100.1, 0.1, 1)
    r = se.chase_next("buy", st, 100.3, 100.4, 0.1, 2)  # bid ушёл вверх на 3 тика
    assert r == {"action": "reprice", "price": 100.3} and st.reprices == 1
    sell = se.ChaseState(started_at=0)
    se.chase_next("sell", sell, 100.0, 100.1, 0.1, 1)
    assert (
        se.chase_next("sell", sell, 99.7, 99.8, 0.1, 2)["action"] == "reprice"
    )  # для продажи — вниз
    prm = se.ChaseParams(max_reprices=1)
    assert (
        se.chase_next("buy", st, 100.6, 100.7, 0.1, 3, prm)["action"]
        == "market_fallback"
    )
    prm2 = se.ChaseParams(max_reprices=1, allow_market_fallback=False)
    assert se.chase_next("buy", st, 100.6, 100.7, 0.1, 3, prm2)["action"] == "give_up"


def test_chase_times_out_and_validates_side():
    st = se.ChaseState(started_at=0)
    se.chase_next("buy", st, 100.0, 100.1, 0.1, 1)
    assert (
        se.chase_next("buy", st, 100.0, 100.1, 0.1, 31)["action"] == "market_fallback"
    )
    with pytest.raises(ValueError):
        se.chase_next("long", st, 1, 2, 0.1, 0)


# ── Частичные исполнения ──────────────────────────────────────────────────────
def test_partial_fill_decisions_and_protection_always_equals_filled():
    d = se.partial_fill_decision
    assert d(10, 10, "FILLED", 5)["action"] == "done"
    assert d(10, 3, "PARTIALLY_FILLED", 10) == {
        "action": "wait",
        "protect_qty": 3,
        "cancel_remainder": False,
    }
    aged = d(10, 6, "PARTIALLY_FILLED", 61)
    assert aged["action"] == "accept_partial" and aged["cancel_remainder"]
    assert d(10, 2, "PARTIALLY_FILLED", 61)["action"] == "cancel_and_close_filled"
    assert d(10, 4, "CANCELED", 5)["action"] == "accept_partial"
    assert d(10, 0, "CANCELED", 5)["action"] == "nothing_filled"
    for filled in (0, 3, 6, 10):
        assert d(10, filled, "PARTIALLY_FILLED", 61)["protect_qty"] == filled
    with pytest.raises(ValueError):
        d(0, 1, "NEW", 1)


# ── Резерв маржи ──────────────────────────────────────────────────────────────
def test_inflight_margin_reserves_releases_and_is_idempotent():
    m = se.InflightMargin()
    assert m.reserve("o1", 3000, leverage=3) == 1000
    m.reserve("o1", 3000, leverage=3)  # повтор не удваивает
    assert m.reserved_total() == 1000 and m.available(5000) == 4000
    assert m.can_open(5000, 12_000, leverage=3) and not m.can_open(
        5000, 13_000, leverage=3
    )
    m.reserve("o2", 6000, leverage=2)
    assert m.available(5000) == 1000
    m.release("o1")
    m.release("missing")  # не падает
    assert m.reserved_total() == 3000 and m.available(1000) == 0.0  # не уходит в минус
    with pytest.raises(ValueError):
        m.reserve("x", -1)
    with pytest.raises(ValueError):
        m.reserve("x", 1, leverage=0)


def test_inflight_margin_is_thread_safe():
    m = se.InflightMargin()
    threads = [
        threading.Thread(target=lambda i=i: m.reserve(f"o{i}", 100.0))
        for i in range(50)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert m.reserved_total() == pytest.approx(5000.0)
