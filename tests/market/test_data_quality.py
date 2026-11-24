"""Phase 2: Data Quality, snapshot_id, опциональная блокировка, спецификации контракта."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.market import data_quality as dq
from src.market.market_scanner import MarketScanner
from tests.core.test_reconciliation import make_bot
from tests.exchange.test_bybit_api_errors import _make_api

TF = "15m"
NOW = 1_800_000_000.0


def _df(n: int = 60, age_s: float = 100.0, step: int = 900, end: float = NOW):
    last = end - age_s
    idx = pd.to_datetime([last - step * (n - 1 - i) for i in range(n)], unit="s")
    close = np.linspace(100, 110, n)
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1000.0),
        },
        index=idx,
    )


def _check(df, **kw):
    return dq.check_ohlcv(df, TF, now=NOW, **kw)


# ── check_ohlcv ───────────────────────────────────────────────────────────────


def test_fresh_valid_candles() -> None:
    q = _check(_df(age_s=600))  # формирующаяся свеча — норма
    assert q.status == dq.VALID and q.reasons == () and q.n_rows == 60
    assert q.age_s == 600.0 and q.last_candle_time.endswith("+00:00")


def test_stale_when_last_candle_older_than_two_timeframes() -> None:
    q = _check(_df(age_s=3 * 900))
    assert q.status == dq.STALE and q.reasons[0].startswith("age>")


def _nan_last_close(d):
    d["close"] = d["close"].where(d.index != d.index[-1])


def _high_below_low(d):
    d["high"] = d["low"] - 1


def _close_above_high(d):
    d["close"] = d["high"] + 5


def _negative_volume(d):
    d["volume"] = -1.0


def _zero_open(d):
    d["open"] = 0.0


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (_nan_last_close, "nan_or_inf"),
        (_high_below_low, "high_below_low"),
        (_close_above_high, "close_outside_range"),
        (_negative_volume, "negative_volume"),
        (_zero_open, "non_positive_price"),
    ],
)
def test_invalid_ohlcv(mutate, reason) -> None:
    df = _df()
    mutate(df)
    q = _check(df)
    assert q.status == dq.INVALID and reason in q.reasons


def test_duplicate_and_unsorted_index_are_invalid() -> None:
    df = _df()
    dup = pd.concat([df, df.iloc[[-1]]])
    assert "duplicate_timestamps" in _check(dup).reasons
    shuffled = df.iloc[::-1]
    assert "index_not_monotonic" in _check(shuffled).reasons


def test_candle_from_the_future_is_invalid() -> None:
    q = _check(_df(age_s=-3 * 900))
    assert q.status == dq.INVALID and "candle_from_future" in q.reasons


def test_incomplete_and_unavailable() -> None:
    assert _check(_df(n=10)).status == dq.INCOMPLETE
    assert _check(_df().drop(columns=["volume"])).reasons == ("missing:volume",)
    assert _check(pd.DataFrame()).status == dq.UNAVAILABLE
    assert dq.check_ohlcv(None, TF).status == dq.UNAVAILABLE


def test_invalid_beats_stale() -> None:
    df = _df(age_s=10 * 900)
    df["high"] = df["low"] - 1
    assert _check(df).status == dq.INVALID


def test_millisecond_integer_index_is_supported() -> None:
    df = _df(age_s=100)
    # as_unit("ms"): в pandas 3 разрешение времени по умолчанию уже не наносекунды
    df.index = pd.Index(df.index.as_unit("ms").asi8)
    q = _check(df)
    assert q.status == dq.VALID and q.age_s == 100.0


def test_timeframe_seconds() -> None:
    assert dq.timeframe_seconds("15m") == 900
    assert dq.timeframe_seconds("4h") == 14400
    assert dq.timeframe_seconds("weird") is None


# ── check_context / snapshot_id ───────────────────────────────────────────────


def test_context_status_from_failed_sources() -> None:
    assert dq.check_context(None).status == dq.UNAVAILABLE
    all_failed = dq.check_context({"data_status": {"failed_sources": ["all"]}})
    assert all_failed.status == dq.UNAVAILABLE
    partial = dq.check_context({"data_status": {"failed_sources": ["etf", "oi"]}})
    assert partial.status == dq.INCOMPLETE
    assert partial.reasons == ("failed:etf", "failed:oi")
    assert dq.check_context({"data_status": {"failed_sources": []}}).status == dq.VALID
    assert dq.check_context({"funding_rate": 0}).status == dq.VALID


def test_snapshot_id_deterministic_and_input_sensitive() -> None:
    t = "2026-01-01T00:00:00+00:00"
    a = dq.make_snapshot_id("BTC/USDT", "15m", t, 100.5)
    assert a == dq.make_snapshot_id("BTC/USDT", "15m", t, 100.5)
    assert a != dq.make_snapshot_id(
        "BTC/USDT", "15m", "2026-01-01T00:15:00+00:00", 100.5
    )
    assert a != dq.make_snapshot_id("BTC/USDT", "15m", t, 100.6)
    assert a != dq.make_snapshot_id("ETH/USDT", "15m", t, 100.5)


# ── Сканер ────────────────────────────────────────────────────────────────────


def _scanner(df=None, error: Exception | None = None) -> MarketScanner:
    loader = MagicMock()
    if error is not None:
        loader.get_historical_data = AsyncMock(side_effect=error)
    else:
        loader.get_historical_data = AsyncMock(return_value=df)
    loader.calculate_technical_indicators = MagicMock(side_effect=lambda d: d)
    return MarketScanner(api=MagicMock(), data_loader=loader)


@pytest.mark.asyncio
async def test_scanner_records_quality_and_snapshot_carries_it() -> None:
    scanner = _scanner(_df(age_s=100, end=time.time()))
    res = await scanner._fetch_one("BTC/USDT", TF)
    assert res is not None
    assert scanner.quality["BTC/USDT"].status == dq.VALID
    scanner._timeframe = TF
    snap = scanner.build_snapshot("BTC/USDT", res[1], 0.0, [])
    assert snap["data_quality"]["status"] == dq.VALID
    assert len(snap["snapshot_id"]) == 12
    again = scanner.build_snapshot("BTC/USDT", res[1], 0.0, [])
    assert again["snapshot_id"] == snap["snapshot_id"]


@pytest.mark.asyncio
async def test_scanner_flags_short_history_and_errors() -> None:
    short = _scanner(_df(n=10, end=time.time()))
    assert await short._fetch_one("ETH/USDT", TF) is None
    assert short.quality["ETH/USDT"].status == dq.INCOMPLETE
    broken = _scanner(error=RuntimeError("boom"))
    assert await broken._fetch_one("SOL/USDT", TF) is None
    assert broken.quality["SOL/USDT"].status == dq.UNAVAILABLE
    broken.reset_quality()
    assert broken.quality == {}


# ── Бот: сводка и опциональная блокировка ─────────────────────────────────────


def _snap(symbol: str, status: str, reasons=()) -> dict:
    return {
        "symbol": symbol,
        "snapshot_id": "abc",
        "data_quality": {"status": status, "reasons": list(reasons)},
    }


def _bot_with_ledger(tmp_path):
    from src.core.decision_ledger import DecisionLedger

    bot = make_bot()
    bot.ledger = DecisionLedger(path=str(tmp_path / "l.jsonl"), enabled=True)
    return bot


def _lines(tmp_path):
    return [json.loads(x) for x in (tmp_path / "l.jsonl").read_text().splitlines()]


def test_gate_is_off_by_default_even_with_mocked_config(tmp_path) -> None:
    bot = _bot_with_ledger(tmp_path)
    snaps = [_snap("A/USDT", dq.VALID), _snap("B/USDT", dq.STALE, ["age"])]
    assert bot._quality_filter(snaps) == snaps  # реальный Config: флаг false
    with patch("src.core.trading_bot.Config") as cfg:
        cfg.DATA_QUALITY_GATE = MagicMock()  # truthy, но не True
        assert bot._quality_filter(snaps) == snaps


def test_gate_blocks_non_valid_symbols_when_enabled(tmp_path) -> None:
    bot = _bot_with_ledger(tmp_path)
    snaps = [_snap("A/USDT", dq.VALID), _snap("B/USDT", dq.STALE, ["age"])]
    with patch("src.core.trading_bot.Config") as cfg:
        cfg.DATA_QUALITY_GATE = True
        kept = bot._quality_filter(snaps)
    assert [s["symbol"] for s in kept] == ["A/USDT"]
    rec = _lines(tmp_path)[0]
    assert rec["kind"] == "data_quality_block"
    assert rec["blocked"][0]["symbol"] == "B/USDT"


def test_gate_keeps_snapshots_without_quality_info(tmp_path) -> None:
    bot = _bot_with_ledger(tmp_path)
    with patch("src.core.trading_bot.Config") as cfg:
        cfg.DATA_QUALITY_GATE = True
        assert bot._quality_filter([{"symbol": "X/USDT"}]) == [{"symbol": "X/USDT"}]


def test_record_data_quality_summarises_cycle(tmp_path) -> None:
    bot = _bot_with_ledger(tmp_path)
    bot.scanner.quality = {
        "A/USDT": dq.DataQuality(dq.VALID),
        "D/USDT": dq.DataQuality(dq.UNAVAILABLE, ("error:X",)),
    }
    bot._record_data_quality(
        [_snap("A/USDT", dq.VALID), _snap("B/USDT", dq.STALE, ["age"])]
    )
    rec = _lines(tmp_path)[0]
    assert rec["kind"] == "data_quality"
    assert rec["counts"] == {dq.VALID: 1, dq.STALE: 1}
    assert rec["issues"][0]["symbol"] == "B/USDT"
    assert rec["dropped"][0]["symbol"] == "D/USDT"


# ── Спецификации контракта и округление цен ───────────────────────────────────


def _api_with_market(market_type: str):
    api = _make_api()
    api._runtime_config = MagicMock()
    api._runtime_config.get_market_type.return_value = market_type
    api.exchange.markets = {
        "BTC/USDT:USDT": {
            "type": "swap",
            "precision": {"price": 0.1, "amount": 0.001},
            "limits": {"amount": {"min": 0.001}},
            "info": {"fundingInterval": "480"},
        }
    }
    api.exchange.price_to_precision = MagicMock(side_effect=lambda s, p: f"{p:.1f}")
    return api


def test_contract_spec_comes_from_loaded_markets() -> None:
    spec = _api_with_market("linear").get_contract_spec("BTC/USDT")
    assert spec == {
        "symbol": "BTC/USDT:USDT",
        "type": "swap",
        "tick_size": 0.1,
        "qty_step": 0.001,
        "min_qty": 0.001,
        "funding_interval_min": "480",
    }
    assert _api_with_market("spot").get_contract_spec("BTC/USDT") is None


def test_round_price_uses_exchange_precision_and_falls_back() -> None:
    api = _api_with_market("linear")
    assert api._round_price("BTC/USDT", 64123.456) == 64123.5
    assert api._round_price("ETH/USDT", 1.2345) == 1.2345  # нет рынка → как есть
    api.exchange.price_to_precision = MagicMock(side_effect=RuntimeError("x"))
    assert api._round_price("BTC/USDT", 64123.456) == 64123.456


@pytest.mark.asyncio
async def test_exchange_sl_tp_prices_are_rounded_to_tick() -> None:
    api = _api_with_market("linear")
    api.exchange.create_order.return_value = {"id": "1"}
    await api.place_exchange_sl_tp("BTC/USDT", "sell", 0.01, 63000.04, 66000.06)
    sl_call, tp_call = api.exchange.create_order.await_args_list
    assert sl_call.kwargs["params"]["stopPrice"] == 63000.0
    assert tp_call.args[4] == 66000.1
