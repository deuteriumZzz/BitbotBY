"""PAPER_GUARDS_OFF: защиты портфеля отключаются только в Paper, в Live игнорируется."""

from unittest.mock import MagicMock, patch

import pytest

from config import Config
from src import settings_registry as reg
from tests.test_order_executor_extra import _TOP_BUY, _make_cfg, _make_executor
from tests.test_runtime_config import rc_bundle  # noqa: F401


def test_setting_is_a_hot_bool_off_by_default():
    s = reg.BY_KEY["PAPER_GUARDS_OFF"]
    assert s.type == "bool" and s.default is False and s.hot
    assert s.group == "Защита портфеля"
    assert Config.PAPER_GUARDS_OFF is False


@pytest.mark.parametrize(
    "paper, stored, expected",
    [(True, b"1", True), (True, b"0", False), (False, b"1", False)],
)
def test_effective_only_in_paper(rc_bundle, paper, stored, expected):  # noqa: F811
    rc, redis_mock, cfg = rc_bundle
    cfg.PAPER_TRADING = paper
    cfg.PAPER_GUARDS_OFF = False
    redis_mock.redis_client.get.return_value = stored
    assert rc.paper_guards_off() is expected
    # сохранённый выбор виден как есть, но в Live не действует
    assert rc.get_paper_guards_off() is (stored == b"1")


def _rc(off: bool) -> MagicMock:
    rc = MagicMock()
    rc.paper_guards_off.return_value = off
    rc.get_max_positions.return_value = 5
    rc.get_market_type.return_value = "linear"
    rc.get_trading_hours.return_value = ""
    rc.get_auto_execute.return_value = True
    return rc


@pytest.mark.asyncio
@pytest.mark.parametrize("off, consulted", [(True, False), (False, True)])
async def test_executor_correlation_filter_follows_switch(off, consulted):
    executor, monitored, _ = _make_executor(runtime_config=_rc(off))
    monitored["ETH/USDT"] = {"qty": 0.5, "entry": 100.0}
    executor._corr_filter.is_allowed = MagicMock(return_value=False)
    with patch("src.order_executor.Config", _make_cfg(max_corr=0.7)):
        await executor.execute(_TOP_BUY, {}, 10_000.0)
    assert executor._corr_filter.is_allowed.called is consulted
