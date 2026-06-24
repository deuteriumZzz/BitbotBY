"""
Онлайн-обучение SAC-модели в процессе торговли.

Три режима (выбирается через ONLINE_LEARNING_MODE в .env):

  disabled  — автообучение выключено. Модель не меняется пока не запустить
              «make train» вручную.

  online    — gradient steps сразу после каждой закрытой сделки (Режим A).
              ⚠ РИСКОВАННО: 3–5 стопов подряд могут испортить модель и
              заставить бота «выключиться» из торговли. Только для экспериментов.
              Рекомендуется: ONLINE_LEARNING_GRADIENT_STEPS ≤ 50.

  periodic  — полное переобучение в фоновом потоке каждые N закрытых сделок
              (Режим B). ✅ РЕКОМЕНДУЕТСЯ для продакшна.
              Новая модель горячо подменяет старую без остановки бота.
              ONLINE_LEARNING_TRIGGER=50 означает переобучение раз в 50 сделок.

  hybrid    — periodic + динамические веса стратегий в реальном времени
              (Режим C). Быстрый фидбек: если ema_crossover даёт стопы —
              его confidence понижается сразу, не дожидаясь переобучения.
              Рекомендуется: запускать только после 200+ накопленных сделок.

Порядок перехода (рекомендация):
  1. paper trading 2–4 недели → накопить 100+ сделок
  2. ONLINE_LEARNING_MODE=periodic
  3. После 500+ сделок → ONLINE_LEARNING_MODE=hybrid
  4. ONLINE_LEARNING_MODE=online — только для исследований, не для реального капитала
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import defaultdict, deque
from typing import Any, Dict

from config import Config

logger = logging.getLogger(__name__)

_WEIGHT_KEY_PREFIX = "strategy_weight:"
_WEIGHT_WINDOW = 20  # последних N сделок для расчёта веса стратегии


class OnlineLearner:
    """
    Управляет автоматическим обучением SAC-модели по мере накопления сделок.

    Использование:
        learner = OnlineLearner(redis_client=bot.redis)
        # После закрытия позиции:
        await learner.on_trade_closed(symbol, side, pnl_pct, strategy="ema_crossover")
    """

    def __init__(self, redis_client: Any = None) -> None:
        self._mode = Config.ONLINE_LEARNING_MODE
        self._trigger = Config.ONLINE_LEARNING_TRIGGER
        self._gradient_steps = Config.ONLINE_LEARNING_GRADIENT_STEPS
        self._redis = redis_client

        self._closed_count = 0
        self._is_training = False
        self._training_lock = asyncio.Lock()
        self._background_tasks: set = set()

        # Для режима hybrid: скользящее окно результатов по каждой стратегии
        self._strategy_results: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=_WEIGHT_WINDOW)
        )

        if self._mode == "disabled":
            logger.info("OnlineLearner: отключён (ONLINE_LEARNING_MODE=disabled)")
        elif self._mode == "online":
            logger.warning(
                "OnlineLearner: режим ONLINE — gradient steps после каждой сделки. "
                "ВНИМАНИЕ: нестабильно при серии убытков!"
            )
        elif self._mode == "periodic":
            logger.info(
                "OnlineLearner: режим PERIODIC — переобучение каждые %d сделок",
                self._trigger,
            )
        elif self._mode == "hybrid":
            logger.info(
                "OnlineLearner: режим HYBRID — periodic + динамические веса стратегий"
            )
        else:
            logger.warning(
                "OnlineLearner: неизвестный режим %r — используется 'periodic'",
                self._mode,
            )
            self._mode = "periodic"

    async def on_trade_closed(
        self,
        symbol: str,
        side: str,
        pnl_pct: float,
        strategy: str = "unknown",
    ) -> None:
        """
        Вызывается PositionMonitor после закрытия каждой позиции.

        :param symbol:   Торговая пара, напр. 'BTC/USDT'.
        :param side:     'buy' или 'sell'.
        :param pnl_pct:  Доходность сделки (-0.05 = −5%, 0.03 = +3%).
        :param strategy: Источник сигнала для режима hybrid.
        """
        if self._mode == "disabled":
            return

        self._closed_count += 1
        is_win = pnl_pct > 0
        logger.debug(
            "OnlineLearner: сделка #%d %s %s pnl=%.2f%% (%s)",
            self._closed_count,
            symbol,
            side,
            pnl_pct * 100,
            "WIN" if is_win else "LOSS",
        )

        if self._mode == "online":
            await self._online_update()

        elif self._mode == "periodic":
            await self._periodic_retrain_if_needed()

        elif self._mode == "hybrid":
            self._record_strategy_result(strategy, is_win)
            await self._sync_strategy_weights()
            await self._periodic_retrain_if_needed()

    def get_strategy_weight(self, strategy: str) -> float:
        """
        Возвращает множитель confidence для стратегии (только в режиме hybrid).

        Диапазон: 0.5 (плохая стратегия) — 1.5 (хорошая).
        Возвращает 1.0 если режим не hybrid или данных < 5 сделок.

        :param strategy: Название стратегии.
        :return: Множитель для confidence сигнала.
        """
        if self._mode != "hybrid":
            return 1.0

        if self._redis:
            try:
                val = self._redis.redis_client.get(f"{_WEIGHT_KEY_PREFIX}{strategy}")
                if val is not None:
                    return float(val)
            except Exception:
                pass

        return self._compute_weight(strategy)

    # ─────────────────────────── private ────────────────────────────────────

    async def _online_update(self) -> None:
        """
        Режим A: немедленные gradient steps после каждой сделки.

        ⚠ Загружает модель, добавляет последние опыты и делает
        ONLINE_LEARNING_GRADIENT_STEPS шагов градиента.
        Пропускается если модель не найдена или идёт другое обучение.
        """
        if self._is_training:
            logger.debug("OnlineLearner [online]: обучение уже идёт — пропускаем")
            return
        if not os.path.exists(Config.SAC_MODEL_PATH):
            logger.debug(
                "OnlineLearner [online]: модель не найдена (%s) — пропускаем",
                Config.SAC_MODEL_PATH,
            )
            return

        async with self._training_lock:
            self._is_training = True
            try:
                loop = asyncio.get_running_loop()
                steps = self._gradient_steps
                await loop.run_in_executor(None, self._run_online_update, steps)
            finally:
                self._is_training = False

    async def _periodic_retrain_if_needed(self) -> None:
        """
        Режим B: полное переобучение каждые ONLINE_LEARNING_TRIGGER сделок.

        Запускается в фоновом потоке — не блокирует торговый цикл.
        Новая модель горячо подменяет старую через атомарный os.replace().
        """
        if self._closed_count % self._trigger != 0:
            return
        if self._is_training:
            logger.info("OnlineLearner [periodic]: переобучение уже идёт — пропускаем")
            return

        logger.info(
            "OnlineLearner [periodic]: триггер на %d сделках — запускаем переобучение",
            self._closed_count,
        )

        # Fire-and-forget: не блокируем вызывающую корутину (position monitor)
        self._is_training = True
        task = asyncio.create_task(self._background_retrain())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _get_train_top_n(self) -> int:
        """Читает train_top_n из Redis — учитывает изменения из Telegram."""
        try:
            from src.runtime_config import RuntimeConfig

            rc = RuntimeConfig(redis_client=self._redis)
            return rc.get_train_top_n()
        except Exception:
            return int(os.environ.get("TRAIN_TOP_N", "20"))

    async def _background_retrain(self) -> None:
        """Запускает _run_full_retrain в thread executor, не блокируя event loop."""
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._run_full_retrain)
        finally:
            self._is_training = False

    def _run_online_update(self, gradient_steps: int) -> None:
        """Синхронный: загружает модель и делает N gradient steps из experience buffer."""  # noqa: E501
        try:
            from stable_baselines3 import SAC  # type: ignore[import]

            from reinforcement_learning.train_sac import (
                _finetune_on_experiences,
                _load_norm_stats,
            )

            model = SAC.load(Config.SAC_MODEL_PATH)
            norm_stats = _load_norm_stats(Config.SAC_MODEL_PATH)
            _finetune_on_experiences(model, norm_stats)
            tmp = Config.SAC_MODEL_PATH + ".online_tmp"
            model.save(tmp)
            os.replace(tmp, Config.SAC_MODEL_PATH)
            logger.info(
                "OnlineLearner [online]: gradient steps выполнено, модель сохранена"
            )
        except ImportError:
            logger.warning(
                "OnlineLearner [online]: stable_baselines3 не установлен — пропускаем"
            )
        except Exception as exc:
            logger.error("OnlineLearner [online] ошибка: %s", exc)

    def _run_full_retrain(self) -> None:
        """Синхронный: полный ретрейн через subprocess с горячей подменой модели."""
        import subprocess
        import sys

        try:
            tmp_path = Config.SAC_MODEL_PATH + ".new"
            env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                "HOME": os.environ.get("HOME", ""),
                "SAC_MODEL_PATH": tmp_path,
                "TRAIN_TOP_N": str(self._get_train_top_n()),
                "TRAIN_MIN_CANDLES": os.environ.get("TRAIN_MIN_CANDLES", "2880"),
                "TOTAL_TIMESTEPS": os.environ.get("TOTAL_TIMESTEPS", "50000"),
                "EXPERIENCES_PATH": os.environ.get(
                    "EXPERIENCES_PATH", "data/experiences.jsonl"
                ),
                "TELEGRAM_BOT_TOKEN": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
                "TELEGRAM_CHAT_ID": os.environ.get("TELEGRAM_CHAT_ID", ""),
                "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO"),
            }

            subprocess.run(
                [sys.executable, "reinforcement_learning/train_sac.py"],
                env=env,
                check=True,
                timeout=7200,  # 2 часа максимум
            )

            # Атомарная подмена: старая → .bak, новая → основной путь
            bak_path = Config.SAC_MODEL_PATH + ".bak"
            if os.path.exists(Config.SAC_MODEL_PATH):
                os.replace(Config.SAC_MODEL_PATH, bak_path)
            os.replace(tmp_path, Config.SAC_MODEL_PATH)

            logger.info(
                "OnlineLearner [periodic]: модель обновлена → %s",
                Config.SAC_MODEL_PATH,
            )
        except Exception as exc:
            logger.error("OnlineLearner [periodic] ошибка переобучения: %s", exc)

    def _record_strategy_result(self, strategy: str, is_win: bool) -> None:
        """Режим C: фиксирует результат сделки для стратегии."""
        self._strategy_results[strategy].append(1 if is_win else 0)

    def _compute_weight(self, strategy: str) -> float:
        """Вычисляет вес стратегии по скользящему win rate последних N сделок."""
        results = self._strategy_results.get(strategy)
        if not results or len(results) < 5:
            return 1.0  # нет данных — нейтральный вес

        win_rate = sum(results) / len(results)
        # 0% WR → 0.5 (штраф), 50% WR → 1.0 (нейтрально), 100% WR → 1.5 (бонус)
        return round(min(max(0.5 + win_rate, 0.5), 1.5), 3)

    async def _sync_strategy_weights(self) -> None:
        """Записывает актуальные веса стратегий в Redis."""
        if not self._redis:
            return
        for strategy, results in self._strategy_results.items():
            if len(results) < 5:
                continue
            weight = self._compute_weight(strategy)
            try:
                self._redis.redis_client.setex(
                    f"{_WEIGHT_KEY_PREFIX}{strategy}",
                    3600,
                    str(weight),
                )
            except Exception as exc:
                logger.debug("OnlineLearner: не удалось записать вес в Redis: %s", exc)
