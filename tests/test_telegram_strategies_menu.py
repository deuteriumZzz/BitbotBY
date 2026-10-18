"""Меню «Стратегии» в Telegram: основные и экспериментальные раздельно, листание, рынок."""

from unittest.mock import MagicMock

import pytest

from src import telegram_commander as tc
from src.strategies import STRATEGY_REGISTRY, get_all_strategies

pytestmark = pytest.mark.skipif(
    not getattr(tc, "TELEGRAM_AVAILABLE", True), reason="нет python-telegram-bot"
)


def _commander(market="linear", mode="ai", disabled=()):
    c = object.__new__(tc.TelegramCommander)
    rows = [{**s, "enabled": s["name"] not in disabled} for s in get_all_strategies()]
    c._get_state = lambda: {"strategies": rows}
    c._rc = MagicMock()
    c._rc.get_market_type.return_value = market
    c._rc.get_mode.return_value = mode
    return c


def _callbacks(markup):
    return [b.callback_data for row in markup.inline_keyboard for b in row]


def test_main_screen_has_only_core_buttons_and_one_link_to_experimental():
    text, kb = _commander()._build_strategies()
    cbs = _callbacks(kb)
    experimental = {n for n, c in STRATEGY_REGISTRY.items() if c.experimental}
    strat = [c.split(":")[1] for c in cbs if c.startswith("strat:")]
    assert strat and not set(strat) & experimental  # ни одной экспериментальной кнопки
    assert "strat_exp:0" in cbs and "strat_reset" in cbs and len(cbs) < 40
    assert len(text) < 4096


def test_spot_hides_futures_only_strategies():
    futures_only = {n for n, c in STRATEGY_REGISTRY.items() if c.markets == "futures"}
    assert futures_only
    spot_text, spot_kb = _commander(market="spot")._build_strategies()
    fut_text, fut_kb = _commander(market="linear")._build_strategies()
    spot_names = {
        c.split(":")[1] for c in _callbacks(spot_kb) if c.startswith("strat:")
    }
    fut_names = {c.split(":")[1] for c in _callbacks(fut_kb) if c.startswith("strat:")}
    assert not spot_names & futures_only
    assert "Скрыто" in spot_text
    # экспериментальные «только фьючерсы» тоже не попадают на спот-страницы
    _, exp_kb = _commander(market="spot")._build_strategies(0)
    assert (
        not {c.split(":")[1] for c in _callbacks(exp_kb) if c.startswith("strat:")}
        & futures_only
    )
    assert fut_names >= spot_names


def test_experimental_pages_cover_everything_and_fit_telegram_limits():
    c = _commander()
    _, exp, _ = c._strategy_rows()
    pages = -(-len(exp) // tc.STRAT_PAGE)
    seen = []
    for page in range(pages):
        text, kb = c._build_strategies(page)
        cbs = _callbacks(kb)
        assert len(text) < 4096 and all(len(x.encode()) <= 64 for x in cbs)
        seen += [x.split(":")[1] for x in cbs if x.startswith("strat:")]
        assert ("strat_exp:%d" % (page + 1) in cbs) == (page < pages - 1)
        assert ("strat_exp:%d" % (page - 1) in cbs) == (page > 0)
        assert "strategies" in cbs  # «Основные»
    assert sorted(seen) == sorted(s["name"] for s in exp)


def test_toggle_buttons_from_experimental_page_keep_the_page():
    c = _commander()
    _, kb = c._build_strategies(1)
    toggles = [x for x in _callbacks(kb) if x.startswith("strat:")]
    assert toggles and all(x.endswith(":1") for x in toggles)


def test_out_of_range_page_is_clamped():
    text, kb = _commander()._build_strategies(999)
    assert "strat_exp:" in " ".join(_callbacks(kb)) or "strategies" in _callbacks(kb)
