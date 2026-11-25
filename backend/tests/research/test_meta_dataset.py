"""Сборщик данных мета-модели: решение и исход сделки сходятся по decision_id."""

from bitbot.research import meta_dataset
from bitbot.storage.decision_ledger import DecisionLedger


def test_rows_join_decision_and_outcome_across_rotation(tmp_path) -> None:
    path = str(tmp_path / "l.jsonl")
    led = DecisionLedger(path=path, max_bytes=300, enabled=True)
    for i in range(6):
        led.write(
            "signals",
            mode="ai",
            decisions=[
                {"decision_id": f"d{i}", "mode": "ai", "outputs": [{"kind": "ai"}]}
            ],
        )
        led.write("cycle_pad", pad="p" * 200)  # ротация между решением и исходом
        led.write(
            "outcome", decision_id=f"d{i}", pnl=1.0 if i % 2 else -1.0, recorded=True
        )
    led.write("outcome", decision_id="orphan", pnl=2.0, recorded=True)  # нет решения
    led.write(
        "outcome", decision_id="d0x", pnl=None, recorded=False
    )  # не записана в БД
    led.write("outcome", pnl=1.0, recorded=True)  # без decision_id

    rows, stats = meta_dataset.build(path)
    assert [r["decision_id"] for r in rows] == [f"d{i}" for i in range(6)]
    assert [r["win"] for r in rows] == [False, True] * 3
    assert rows[0]["outputs"] == [{"kind": "ai"}]
    assert stats == {"unmatched": 2, "not_recorded": 1}

    st = meta_dataset.status(path)
    assert st["have"] == 6 and st["wins"] == 3 and st["need"] == meta_dataset.NEED


def test_status_on_empty_ledger(tmp_path) -> None:
    st = meta_dataset.status(str(tmp_path / "none.jsonl"))
    assert st["have"] == 0 and st["last"] is None and st["unmatched"] == 0
