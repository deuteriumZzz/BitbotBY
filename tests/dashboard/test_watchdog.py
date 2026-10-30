"""Сторож «бот упал»: одна тревога на беду, «снова работает» — когда прошло."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from src.dashboard import watchdog
from tests.infra.test_bot_bus import FakeRedis


def test_problem():
    assert watchdog.problem(True, 10, 300) is None
    assert watchdog.problem(True, None, 300) is None
    assert "процесс" in watchdog.problem(False, 10, 300)
    assert "цикл" in watchdog.problem(True, 300 + watchdog.STALL_GRACE + 60, 300)


async def test_alert_once_then_recovery():
    r = FakeRedis()
    send = AsyncMock(return_value=True)
    with patch.object(watchdog.auth, "send_owner", send), patch.object(
        watchdog.data, "check_healthcheck", return_value=False
    ):
        await watchdog.check_once(r, started=0, now=100)
        await watchdog.check_once(r, started=0, now=160)
    assert send.await_count == 1
    assert "🛑" in send.await_args.args[0]

    with patch.object(watchdog.auth, "send_owner", send), patch.object(
        watchdog.data, "check_healthcheck", return_value=True
    ):
        await watchdog.check_once(r, started=0, now=220)
    assert send.await_count == 2
    assert "снова работает" in send.await_args.args[0]


async def test_stalled_cycle_and_switch_off():
    r = FakeRedis()
    now = 100_000.0
    r.set("bot:last_cycle", json.dumps({"at": now - 3 * 3600}))
    r.set("bot:alert:enabled:bot_stalled", "0")
    send = AsyncMock(return_value=True)
    with patch.object(watchdog.auth, "send_owner", send), patch.object(
        watchdog.data, "check_healthcheck", return_value=True
    ):
        # Панель только что запустилась — время цикла ещё не судим.
        await watchdog.check_once(r, started=now - 60, now=now)
        assert r.get(watchdog._ALERT_KEY) is None
        await watchdog.check_once(r, started=0, now=now)
    assert "цикл" in r.get(watchdog._ALERT_KEY)
    send.assert_not_awaited()  # тревога bot_stalled выключена
