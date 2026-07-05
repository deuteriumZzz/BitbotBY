"""Тесты для src/pumpfun_tracker.py."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.pumpfun_tracker import PumpfunTracker


class TestEnabled:
    def test_disabled_without_key(self):
        tracker = PumpfunTracker("")
        assert tracker.enabled is False

    def test_enabled_with_key(self):
        tracker = PumpfunTracker("some-key")
        assert tracker.enabled is True


class TestStartStop:
    def test_start_noop_without_key(self):
        tracker = PumpfunTracker("")
        tracker.start()
        assert tracker._task is None

    def test_start_creates_task_with_key(self):
        tracker = PumpfunTracker("some-key")
        with patch("src.pumpfun_tracker.asyncio.create_task") as mock_create_task:
            tracker.start()
            mock_create_task.assert_called_once()
            mock_create_task.call_args[0][0].close()
        assert tracker._running is True

    def test_start_is_idempotent(self):
        tracker = PumpfunTracker("some-key")
        with patch("src.pumpfun_tracker.asyncio.create_task") as mock_create:
            tracker.start()
            tracker.start()
            mock_create.assert_called_once()
            mock_create.call_args[0][0].close()

    def test_stop_cancels_task(self):
        tracker = PumpfunTracker("some-key")
        mock_task = MagicMock()
        tracker._task = mock_task
        tracker._running = True
        tracker.stop()
        mock_task.cancel.assert_called_once()
        assert tracker._task is None
        assert tracker._running is False

    def test_stop_without_task_is_safe(self):
        tracker = PumpfunTracker("some-key")
        tracker.stop()  # no-op, must not raise


class TestLaunchRatePerHour:
    def test_zero_when_no_events(self):
        tracker = PumpfunTracker("some-key")
        assert tracker.launch_rate_per_hour() == 0.0

    def test_counts_recent_events(self):
        tracker = PumpfunTracker("some-key")
        now = time.time()
        for i in range(10):
            tracker._timestamps.append(now - i)
        assert tracker.launch_rate_per_hour() == pytest.approx(10.0)

    def test_trims_events_older_than_window(self):
        tracker = PumpfunTracker("some-key")
        now = time.time()
        tracker._timestamps.append(now - 7200)  # 2h old — outside window
        tracker._timestamps.append(now - 60)  # 1min old — inside window
        assert tracker.launch_rate_per_hour() == pytest.approx(1.0)
        assert len(tracker._timestamps) == 1


class TestRun:
    @pytest.mark.asyncio
    async def test_records_timestamp_on_text_message(self):
        tracker = PumpfunTracker("some-key")
        tracker._running = True

        import aiohttp as _aiohttp

        msg = MagicMock()
        msg.type = _aiohttp.WSMsgType.TEXT

        async def _fake_aiter(_self):
            yield msg
            tracker._running = False

        ws_mock = MagicMock()
        ws_mock.send_json = AsyncMock()
        ws_mock.__aenter__ = AsyncMock(return_value=ws_mock)
        ws_mock.__aexit__ = AsyncMock(return_value=False)
        ws_mock.__aiter__ = _fake_aiter

        session_mock = MagicMock()
        session_mock.__aenter__ = AsyncMock(return_value=session_mock)
        session_mock.__aexit__ = AsyncMock(return_value=False)
        session_mock.ws_connect = MagicMock(return_value=ws_mock)

        with patch("aiohttp.ClientSession", return_value=session_mock):
            await tracker._run()

        assert len(tracker._timestamps) == 1

    @pytest.mark.asyncio
    async def test_reconnects_after_exception(self):
        tracker = PumpfunTracker("some-key")
        tracker._running = True

        call_count = 0

        def _side_effect(*_a, **_kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("boom")
            tracker._running = False
            session_mock = MagicMock()
            session_mock.__aenter__ = AsyncMock(return_value=session_mock)
            session_mock.__aexit__ = AsyncMock(return_value=False)

            async def _fake_aiter(_self):
                return
                yield  # pragma: no cover - makes this an async generator

            ws_mock = MagicMock()
            ws_mock.send_json = AsyncMock()
            ws_mock.__aenter__ = AsyncMock(return_value=ws_mock)
            ws_mock.__aexit__ = AsyncMock(return_value=False)
            ws_mock.__aiter__ = _fake_aiter
            session_mock.ws_connect = MagicMock(return_value=ws_mock)
            return session_mock

        with patch("aiohttp.ClientSession", side_effect=_side_effect):
            with patch(
                "src.pumpfun_tracker.asyncio.sleep", new=AsyncMock()
            ) as mock_sleep:
                await tracker._run()
                mock_sleep.assert_called()

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        tracker = PumpfunTracker("some-key")
        tracker._running = True

        with patch("aiohttp.ClientSession", side_effect=asyncio.CancelledError):
            with pytest.raises(asyncio.CancelledError):
                await tracker._run()
