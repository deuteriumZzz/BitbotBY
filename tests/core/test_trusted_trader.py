"""Сделки по постам доверенных Telegram-каналов (src/core/trusted_trader.py)."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core import trusted_trader as tt
from src.core.position_monitor import PositionMonitor
from src.core.trusted_trader import Signal, TrustedTrader, levels, parse_post
from src.core.types import is_trusted
from src.sentiment.telegram_sentiment import Post
from tests.core.test_order_executor import _make_cfg, _make_executor
from tests.infra.test_bot_bus import FakeRedis

MARKETS = {
    "WIF/USDT:USDT": {"base": "WIF", "quote": "USDT", "linear": True},
    "1000PEPE/USDT:USDT": {"base": "1000PEPE", "quote": "USDT", "linear": True},
    "BTC/USDT": {"base": "BTC", "quote": "USDT", "spot": True},
    "BTC/USDT:USDT": {"base": "BTC", "quote": "USDT", "linear": True},
    "SOL/USDT:USDT": {"base": "SOL", "quote": "USDT", "linear": True},
    "BONK/USDT:USDT": {"base": "BONK", "quote": "USDT", "linear": True},
    "OLD/USDT:USDT": {"base": "OLD", "quote": "USDT", "linear": True, "active": False},
    "ETH/USDC:USDC": {"base": "ETH", "quote": "USDC", "linear": True},
}
TRADABLE = tt.tradable_map(MARKETS, "linear")


def test_tradable_map_follows_market_type():
    assert TRADABLE == {
        "WIF": "WIF/USDT",
        "1000PEPE": "1000PEPE/USDT",
        "PEPE": "1000PEPE/USDT",
        "BTC": "BTC/USDT",
        "SOL": "SOL/USDT",
        "BONK": "BONK/USDT",
    }
    assert tt.tradable_map(MARKETS, "spot") == {"BTC": "BTC/USDT"}


# ── разбор постов ─────────────────────────────────────────────────────────────


def test_parse_typical_signal_post():
    post = (
        "🚀 #WIF/USDT LONG\n"
        "Entry: 2.30 - 2.35\n"
        "Targets: 2.50 / 2.80\n"
        "Stop: 2.10\n"
        "Leverage 10x"
    )
    assert parse_post(post, TRADABLE) == Signal(
        "WIF/USDT", "buy", 2.30, (2.10, False), (2.50, False)
    )


def test_parse_percent_levels_and_multiplied_contract():
    sig = parse_post("SHORT $PEPE now\nSL 5%\nTP 10%", TRADABLE)
    assert sig == Signal("1000PEPE/USDT", "sell", 0.0, (5.0, True), (10.0, True))


def test_parse_russian_post():
    sig = parse_post("Лонг по SOL, вход 145, стоп 139,5, цель 160", TRADABLE)
    assert sig == Signal("SOL/USDT", "buy", 145.0, (139.5, False), (160.0, False))


def test_parse_thousands_separator():
    sig = parse_post("BTC long 65,000 SL 63,500 TP 70,000", TRADABLE)
    assert isinstance(sig, Signal)
    assert sig.sl == (63500.0, False) and sig.tp == (70000.0, False)


def test_marked_coin_wins_over_a_mention():
    sig = parse_post("Buy $WIF here, BTC looks calm", TRADABLE)
    assert isinstance(sig, Signal) and sig.symbol == "WIF/USDT"


@pytest.mark.parametrize(
    "post,reason",
    [
        ("WIF TP1 hit ✅ +20%", "итогах"),
        ("Закрываем лонг по WIF", "итогах"),
        ("$WIF to the moon 🚀", "нет направления"),
        ("long or short WIF?", "и лонг, и шорт"),
        ("LONG $WIF and $BONK", "несколько монет: BONK, WIF"),
        ("LONG $NEWCOIN, early!", "NEWCOIN не торгуется на Bybit"),
        ("LONG $OLD", "OLD не торгуется на Bybit"),  # делистинг
        ("Long everything, market is green", "нет монеты"),
        ("my wifi says buy", "нет монеты"),  # подстрока — не монета
    ],
)
def test_unclear_posts_give_a_reason(post, reason):
    result = parse_post(post, TRADABLE)
    assert isinstance(result, str) and reason in result


def test_levels_default_and_from_post():
    buy = Signal("WIF/USDT", "buy", 0.0, (0.0, False), (0.0, False))
    sl, tp, why = levels(buy, 100.0, 0.03)
    assert (round(sl, 6), round(tp, 6), why) == (97.0, 106.0, "")
    short = Signal("WIF/USDT", "sell", 0.0, (5.0, True), (10.0, True))
    sl, tp, why = levels(short, 100.0, 0.03)
    assert (round(sl, 6), round(tp, 6), why) == (105.0, 90.0, "")


@pytest.mark.parametrize(
    "sl,tp,reason",
    [
        ((101.0, False), (0.0, False), "за стопом"),
        ((40.0, False), (0.0, False), "дальше 50%"),
        ((95.0, False), (99.0, False), "прошла цель"),
    ],
)
def test_levels_refuse_impossible_posts(sl, tp, reason):
    sig = Signal("WIF/USDT", "buy", 0.0, sl, tp)
    assert reason in levels(sig, 100.0, 0.03)[2]


# ── вход по посту ─────────────────────────────────────────────────────────────


class RC:
    def __init__(self, **hot):
        self.values = {
            "TELEGRAM_TRUSTED_TRADING": True,
            "TELEGRAM_TRUSTED_CHANNELS": "alpha,beta",
            "TELEGRAM_TRUSTED_RISK": 0.01,
            "TELEGRAM_TRUSTED_SLOTS": 2,
            "TELEGRAM_TRUSTED_SL": 0.05,
            "TELEGRAM_TRUSTED_DEDUP_MIN": 30,
            "TELEGRAM_TRUSTED_UNTRUST": 0,
            **hot,
        }
        self.paused = False

    def hot(self, key):
        return self.values[key]

    def set_hot(self, key, value):
        self.values[key] = value
        return True

    def is_paused(self):
        return self.paused

    def get_market_type(self):
        return "linear"


class Executor:
    """Открывает позицию, если не задана причина отказа."""

    def __init__(self, monitored, reject=""):
        self.monitored, self.reject = monitored, reject
        self.tops: list = []
        self.last_reject: dict = {}

    async def execute(self, top, market_data, balance):
        self.tops.append(top)
        if self.reject:
            self.last_reject[top["symbol"]] = self.reject
        else:
            self.monitored[top["symbol"]] = {"strategy": top["strategy"]}


def _trader(rc=None, monitored=None, reject="", blocked=""):
    monitored = {} if monitored is None else monitored
    events: list = []
    api = SimpleNamespace(
        exchange=SimpleNamespace(
            markets=MARKETS,
            fetch_ticker=AsyncMock(return_value={"ask": 100.0, "bid": 99.9}),
        )
    )
    history = MagicMock()
    history.recent_results = AsyncMock(return_value=[])
    telegram = MagicMock()
    telegram.notify = AsyncMock()
    trader = TrustedTrader(
        api=api,
        executor=Executor(monitored, reject),
        runtime_config=rc or RC(),
        trade_history=history,
        telegram=telegram,
        get_balance=AsyncMock(return_value=1000.0),
        get_monitored=lambda: monitored,
        blocked=lambda: blocked,
        emit=lambda t, d: events.append((t, d)),
        redis=FakeRedis(),
    )
    return trader, events


def _post(channel="alpha", text="LONG $WIF", age=1.0):
    return Post(time.time() - age, channel, text)


async def test_trusted_post_goes_straight_to_an_order():
    trader, events = _trader()
    await trader.on_post(_post(text="LONG $WIF"))
    (top,) = trader._executor.tops
    assert top["trusted"] and top["strategy"] == "tg:alpha"
    assert top["action"] == "buy" and top["entry"] == 100.0
    assert top["stop_loss"] == pytest.approx(95.0)  # стоп по умолчанию 5%
    assert top["take_profit"] == pytest.approx(110.0)
    # риск 1% от 1000 при стопе 5 пунктов → 2 монеты
    assert top["quantity"] == pytest.approx(2.0)
    assert [s["node"] for s in top["trace"]][0] == "scan"
    (kind, ev), *_ = events
    assert kind == "trusted.post" and ev["ok"] and ev["latency_s"] >= 0.9
    stored = json.loads(trader._r.get(tt.LAST_KEY))
    assert stored["channel"] == "alpha" and stored["symbol"] == "WIF/USDT"


async def test_untrusted_or_switched_off_does_nothing():
    trader, events = _trader(rc=RC(TELEGRAM_TRUSTED_TRADING=False))
    await trader.on_post(_post())
    trader2, _ = _trader()
    await trader2.on_post(_post(channel="random"))
    assert trader._executor.tops == [] and trader2._executor.tops == [] and not events


@pytest.mark.parametrize(
    "setup,reason",
    [
        ({"paused": True}, "пауз"),
        ({"blocked": "дневной лимит убытка"}, "дневной лимит"),
        ({"age": 600}, "с опозданием"),
        ({"text": "$WIF moon"}, "нет направления"),
        ({"slots_full": True}, "слоты для каналов заняты (2/2)"),
    ],
)
async def test_safeguards_stop_the_entry_with_a_reason(setup, reason):
    rc = RC()
    rc.paused = setup.get("paused", False)
    monitored = (
        {"A/USDT": {"strategy": "tg:x"}, "B/USDT": {"strategy": "tg:y"}}
        if setup.get("slots_full")
        else {"C/USDT": {"strategy": "rsi"}}  # обычная позиция слот не занимает
    )
    trader, events = _trader(
        rc=rc, monitored=monitored, blocked=setup.get("blocked", "")
    )
    await trader.on_post(
        _post(text=setup.get("text", "LONG $WIF"), age=setup.get("age", 1.0))
    )
    assert trader._executor.tops == []
    assert reason in events[-1][1]["reason"] and not events[-1][1]["ok"]


async def test_same_coin_from_several_channels_is_one_trade():
    trader, events = _trader()
    await trader.on_post(_post("alpha", "LONG $WIF"))
    await trader.on_post(_post("beta", "#WIF long, entry now"))
    assert len(trader._executor.tops) == 1
    assert "уже входили" in events[-1][1]["reason"]


async def test_rejected_order_reports_why_and_allows_a_retry():
    trader, events = _trader(reject="мало ликвидности или широкий спред")
    await trader.on_post(_post("alpha", "LONG $WIF"))
    assert events[-1][1]["reason"] == "мало ликвидности или широкий спред"
    trader._executor.reject = ""
    await trader.on_post(_post("beta", "LONG $WIF"))
    assert events[-1][1]["ok"]  # первый отказ не заблокировал монету


async def test_auto_untrust_after_losses_in_a_row():
    rc = RC(TELEGRAM_TRUSTED_UNTRUST=2)
    trader, events = _trader(rc=rc)
    trader._history.recent_results = AsyncMock(
        side_effect=lambda strategy, n, after: (
            [(12, -3.0), (11, -1.0)] if strategy == "tg:alpha" else [(9, 5.0)]
        )
    )
    await trader.review()
    assert rc.values["TELEGRAM_TRUSTED_CHANNELS"] == "beta"
    assert trader._r.hget(tt.UNTRUSTED_KEY, "alpha") == 12
    trader._telegram.notify.assert_awaited_once()
    assert "@alpha" in trader._telegram.notify.await_args.args[0]
    assert ("config.changed", {"keys": ["TELEGRAM_TRUSTED_CHANNELS"]}) in events


# ── исполнение и сопровождение ────────────────────────────────────────────────


_TRUSTED_TOP = {
    "symbol": "WIF/USDT",
    "action": "buy",
    "entry": 100.0,
    "stop_loss": 95.0,
    "take_profit": 110.0,
    "quantity": 2.0,
    "strategy": "tg:alpha",
    "trusted": True,
    "max_slots": 1,
    "trace": [{"node": "scan", "ok": True, "text": "пост в канале @alpha"}],
}


async def test_executor_lets_trusted_through_its_own_slots_only():
    monitored = {f"S{i}/USDT": {"strategy": "rsi"} for i in range(3)}
    executor, _ = _make_executor(monitored=monitored)
    executor._corr_filter.is_allowed = MagicMock(return_value=False)  # не помеха
    cfg = _make_cfg(paper=False, max_pos=3, max_corr=0.7)
    cfg.AUTO_EXECUTE = False  # подтверждения для поста не ждём
    rc = MagicMock()
    rc.get_trading_hours.return_value = "0-1"
    rc.is_trading_time.return_value = False  # вне часов торговли — тоже не помеха
    rc.get_max_positions.return_value = 3
    rc.get_market_type.return_value = "linear"
    rc.get_auto_execute.return_value = False
    rc.paper_guards_off.return_value = False
    executor._runtime_config = rc
    with patch("src.core.order_executor.Config", cfg), patch(
        "src.core.trade_history.get_backtest_stats", return_value={}
    ):
        await executor.execute(_TRUSTED_TOP, {}, 10_000.0)
        assert monitored["WIF/USDT"]["qty"] == 2.0
        assert monitored["WIF/USDT"]["trace"] == _TRUSTED_TOP["trace"]
        executor._telegram.ask_confirm.assert_not_called()
        # второй пост — свой слот занят, обычные три слота не в счёт
        await executor.execute({**_TRUSTED_TOP, "symbol": "BONK/USDT"}, {}, 1e4)
    assert "BONK/USDT" not in monitored
    assert executor.last_reject["BONK/USDT"] == "слоты для каналов заняты"


def test_dynamic_exits_never_close_a_trusted_position():
    monitor = PositionMonitor(
        api=MagicMock(),
        trade_history=MagicMock(),
        telegram=None,
        portfolio_manager=MagicMock(),
        set_running=lambda v: None,
    )
    monitor.update_market_state(
        {"WIF/USDT": {"action": "sell", "confidence": 0.95}},
        {"WIF/USDT": {"fear_greed": 95}},
        "trending_down",
    )
    trusted = {"side": "buy", "qty": 1.0, "entry": 1.0, "strategy": "tg:alpha"}
    normal = {**trusted, "strategy": "rsi"}
    assert is_trusted(trusted) and not is_trusted(normal) and not is_trusted(None)
    assert monitor._check_dynamic_exit("WIF/USDT", trusted) == (None, "")
    assert monitor._check_dynamic_exit("WIF/USDT", normal)[0] == "force_close"
