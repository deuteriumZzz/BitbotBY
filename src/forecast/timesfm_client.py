"""Клиент воркера TimesFM (Python 3.9, без torch): запуск, запросы, защита от сбоев.

Воркер запускается в `.venv-timesfm` с пониженным приоритетом и потолком потоков
(`src/infra/heavy_job.py`). Любой сбой (процесс упал, таймаут, ошибка модели) даёт
`None` — это «прогнозист недоступен»: по политике отказов (Phase 10) поправка нейтральна,
цикл не блокируется. После сбоя повторный запуск не раньше `cooldown_s`, чтобы не
получить шторм перезапусков. Из асинхронного кода вызывать через `asyncio.to_thread`.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

from src.infra.heavy_job import limit_env, lower_priority

logger = logging.getLogger(__name__)

DEFAULT_PYTHON = os.path.join(".venv-timesfm", "bin", "python")
DEFAULT_CMD = [DEFAULT_PYTHON, "-m", "src.forecast.timesfm_worker"]


class TimesFMWorker:
    def __init__(
        self,
        cmd: Optional[Sequence[str]] = None,
        start_timeout_s: float = 180.0,
        request_timeout_s: float = 60.0,
        cooldown_s: float = 60.0,
        threads: Optional[int] = None,
    ) -> None:
        self.cmd = list(cmd or DEFAULT_CMD)
        self.start_timeout_s, self.request_timeout_s = (
            start_timeout_s,
            request_timeout_s,
        )
        self.cooldown_s, self.threads = cooldown_s, threads
        self._proc: Optional[subprocess.Popen] = None
        self._out: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()
        self._next_id = 0
        self._failed_at = 0.0

    # ── процесс ───────────────────────────────────────────────────────────────
    def _reader(self, proc: subprocess.Popen, out: "queue.Queue[str]") -> None:
        assert proc.stdout
        for line in proc.stdout:
            out.put(line)
        out.put("")  # EOF: процесс завершился

    def _start(self) -> bool:
        if self._failed_at and time.monotonic() - self._failed_at < self.cooldown_s:
            return False
        env = limit_env({**os.environ, "PYTHONPATH": os.getcwd()}, self.threads)
        try:
            self._proc = subprocess.Popen(
                self.cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                env=env,
                preexec_fn=lower_priority(),
            )
        except OSError as exc:
            logger.warning("TimesFM: не удалось запустить воркер: %s", exc)
            self._failed_at = time.monotonic()
            return False
        self._out = queue.Queue()
        threading.Thread(
            target=self._reader, args=(self._proc, self._out), daemon=True
        ).start()
        try:
            ready = json.loads(self._out.get(timeout=self.start_timeout_s) or "{}")
        except (queue.Empty, ValueError):
            ready = {}
        if not ready.get("ready"):
            logger.warning("TimesFM: воркер не сообщил о готовности")
            self._kill()
            return False
        return True

    def _kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass
        self._proc = None
        self._failed_at = time.monotonic()

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def close(self) -> None:
        with self._lock:
            if self._proc is not None:
                try:
                    if self._proc.stdin:
                        self._proc.stdin.close()
                    self._proc.wait(timeout=5)
                except Exception:  # noqa: BLE001
                    self._proc.kill()
            self._proc = None

    # ── запросы ───────────────────────────────────────────────────────────────
    def forecast(
        self, series: Sequence[float], horizon: int
    ) -> Optional[Dict[str, List[float]]]:
        """{"median","mean","lo","hi"} или None, если прогнозист недоступен/ответил ошибкой."""
        with self._lock:
            if not self.alive() and not self._start():
                return None
            self._next_id += 1
            rid = self._next_id
            try:
                assert self._proc and self._proc.stdin
                payload = {
                    "id": rid,
                    "series": [float(v) for v in series],
                    "horizon": int(horizon),
                }
                self._proc.stdin.write(json.dumps(payload) + "\n")
                self._proc.stdin.flush()
                line = self._out.get(timeout=self.request_timeout_s)
            except (queue.Empty, OSError):
                logger.warning(
                    "TimesFM: таймаут или обрыв — воркер перезапустится после паузы"
                )
                self._kill()
                return None
            if not line:  # процесс умер
                self._kill()
                return None
            try:
                resp: Dict[str, Any] = json.loads(line)
            except ValueError:
                self._kill()
                return None
            if resp.get("id") != rid or "error" in resp:
                if "error" in resp:
                    logger.warning("TimesFM: %s", resp["error"])
                return None
            return {k: resp[k] for k in ("median", "mean", "lo", "hi")}

    def as_forecaster(self) -> Any:
        """Адаптер к `src.reliability.forecast_eval`: f(history, horizon) → median/lo/hi.

        Недоступный прогнозист не подменяется — исключение, чтобы сравнение не
        приняло «нет ответа» за ответ.
        """
        import numpy as np  # noqa: PLC0415

        def f(history: Any, h: int) -> Dict[str, Any]:
            r = self.forecast(list(history), h)
            if r is None:
                raise RuntimeError("TimesFM недоступен")
            return {k: np.asarray(r[k]) for k in ("median", "lo", "hi")}

        return f


class TimesFMForecaster:
    """Замена `chronos_analyzer` для `SignalCombiner` (`FORECASTER=timesfm`): тот же `predict_direction`.

    Прогноз на 1 шаг: медиана против последней цены; порог — как у Chronos (передаёт комбинатор).
    Недоступен/ошибка/мало истории → "neutral" (политика forecaster_unavailable: поправка нейтральна).
    Измерение 2026-11-23 (AUDIT_REPORT §10p): точечный прогноз не лучше naive — включать осознанно.
    """

    def __init__(self, worker: Optional[TimesFMWorker] = None) -> None:
        self.worker = worker or TimesFMWorker()

    def predict_direction(
        self, close_prices: Sequence[float], threshold_pct: float = 0.002
    ) -> str:
        if len(close_prices) < 10 or not close_prices[-1]:
            return "neutral"
        r = self.worker.forecast(list(close_prices[-512:]), 1)
        if r is None:
            return "neutral"
        diff = (float(r["median"][0]) - close_prices[-1]) / close_prices[-1]
        if diff > threshold_pct:
            return "up"
        if diff < -threshold_pct:
            return "down"
        return "neutral"

    def close(self) -> None:
        self.worker.close()
