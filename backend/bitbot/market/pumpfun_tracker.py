"""
Фоновый трекер темпа создания новых токенов на pump.fun через PumpPortal.

Слушает бесплатный WebSocket-фид событий создания токенов (subscribeNewToken)
и считает частоту запусков в скользящем часовом окне — сигнал для эвристики
meme-season в SeasonDetector. Без ключа — просто не запускается (enabled=False),
без ошибок и без деградации остального бота.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Deque, Optional

logger = logging.getLogger(__name__)

_WS_URL_TMPL = "wss://pumpportal.fun/api/data?api-key={key}"
_WINDOW_SECONDS = 3600.0  # скользящее окно для расчёта темпа — 1 час
_RECONNECT_DELAY_S = 10.0


class PumpfunTracker:
    """
    Использование:
        tracker = PumpfunTracker(Config.PUMPPORTAL_API_KEY)
        tracker.start()
        rate = tracker.launch_rate_per_hour()
        tracker.stop()
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._timestamps: Deque[float] = deque()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def start(self) -> None:
        """Запускает фоновое прослушивание. No-op если ключ не задан."""
        if not self.enabled or self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="pumpfun_tracker")
        logger.info("PumpfunTracker: запущен")

    def stop(self) -> None:
        """Останавливает фоновую задачу."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def launch_rate_per_hour(self) -> float:
        """Темп создания токенов (событий/час) по скользящему часовому окну."""
        self._trim(time.time())
        return len(self._timestamps) * (3600.0 / _WINDOW_SECONDS)

    def _trim(self, now: float) -> None:
        while self._timestamps and now - self._timestamps[0] > _WINDOW_SECONDS:
            self._timestamps.popleft()

    async def _run(self) -> None:
        import aiohttp

        url = _WS_URL_TMPL.format(key=self._api_key)
        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url, heartbeat=30) as ws:
                        await ws.send_json({"method": "subscribeNewToken"})
                        logger.info("PumpfunTracker: подключен, слушаю новые токены")
                        async for msg in ws:
                            if not self._running:
                                break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                self._timestamps.append(time.time())
                            elif msg.type in (
                                aiohttp.WSMsgType.ERROR,
                                aiohttp.WSMsgType.CLOSED,
                            ):
                                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("PumpfunTracker: соединение прервано: %s", exc)
            if self._running:
                await asyncio.sleep(_RECONNECT_DELAY_S)
