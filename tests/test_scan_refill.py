"""
Тесты добора кандидатов в TradingBot._scan_and_update_correlations.

Если scan_all() не смог загрузить данные для части кандидатов (молодой
листинг, делистинг, сбой API) — метод должен запросить у сканера ещё
кандидатов ниже по рейтингу и добрать до target_n, а не молча вернуть
меньше монет, чем запрошено.
"""

from unittest.mock import AsyncMock

import pandas as pd
import pytest

from tests.test_reconciliation import make_bot


def _df(rows: int = 60) -> pd.DataFrame:
    return pd.DataFrame({"close": [1.0] * rows})


class TestScanRefill:

    @pytest.mark.asyncio
    async def test_no_refill_kwargs_keeps_old_behavior(self):
        """refill_kwargs=None (дефолт) — недобор не восполняется (обратная совместимость)."""
        bot = make_bot()
        bot.scanner.scan_all = AsyncMock(
            return_value={"BTC/USDT": _df(), "ETH/USDT": _df()}
        )
        bot.scanner.get_top_symbols = AsyncMock()

        result = await bot._scan_and_update_correlations(
            ["BTC/USDT", "ETH/USDT", "FAKE/USDT"], target_n=3
        )

        assert len(result) == 2
        bot.scanner.get_top_symbols.assert_not_called()

    @pytest.mark.asyncio
    async def test_refill_tops_up_shortfall(self):
        """3 из 5 кандидатов не дали данных — добор должен найти 3 замены и набрать target_n=5."""
        bot = make_bot()
        bot.scanner.scan_all = AsyncMock(
            side_effect=[
                {"A/USDT": _df(), "B/USDT": _df()},  # первый проход: 2 из 5
                {"D/USDT": _df(), "E/USDT": _df(), "F/USDT": _df()},  # добор: 3
            ]
        )
        bot.scanner.get_top_symbols = AsyncMock(
            return_value=["D/USDT", "E/USDT", "F/USDT"]
        )

        result = await bot._scan_and_update_correlations(
            ["A/USDT", "B/USDT", "C/USDT", "X/USDT", "Y/USDT"],
            target_n=5,
            refill_kwargs={},
        )

        assert len(result) == 5
        bot.scanner.get_top_symbols.assert_called_once()
        call_kwargs = bot.scanner.get_top_symbols.call_args.kwargs
        assert call_kwargs["n"] == 3  # ровно недостающее количество

    @pytest.mark.asyncio
    async def test_refill_stops_when_pool_exhausted(self):
        """Если добор не находит новых кандидатов — не зацикливается, отдаёт что набрал."""
        bot = make_bot()
        bot.scanner.scan_all = AsyncMock(return_value={"A/USDT": _df()})
        bot.scanner.get_top_symbols = AsyncMock(return_value=[])  # пул исчерпан

        result = await bot._scan_and_update_correlations(
            ["A/USDT", "B/USDT"], target_n=5, refill_kwargs={}
        )

        assert len(result) == 1
        bot.scanner.get_top_symbols.assert_called_once()
