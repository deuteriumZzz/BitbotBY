"""
Управление рисками: размер позиции по Келли (Half-Kelly) и дневной лимит потерь.

Остальные проверки перед ордером (слоты, корреляция, ликвидность, SL/TP,
нагрузка портфеля, просадка, плечо) — в OrderExecutor.execute.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from config import Config

_DAY_REF_TTL = 3 * 24 * 3600  # ссылка на баланс дня живёт дольше самих суток


class RiskManager:
    """
    Менеджер рисков для торгового бота: параметры риска, размер позиции по
    Келли и дневной лимит потерь.
    """

    def __init__(
        self,
        initial_balance: float,
        risk_per_trade: float = 0.02,
        redis_client: Any = None,
    ) -> None:
        """
        Инициализирует менеджер рисков.

        :param initial_balance: Начальный баланс счёта.
        :param risk_per_trade: Доля баланса на риск в сделке
            (по умолчанию 0.02 = 2%).
        :param redis_client: RedisClient для сохранения баланса начала дня:
            без него рестарт посреди дня обнулял бы дневной лимит потерь.
        """
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.max_positions: int = Config.MAX_POSITIONS
        self.drawdown_scale_enabled: bool = Config.DRAWDOWN_SCALE_ENABLED
        self.logger = logging.getLogger(__name__)
        # Daily-loss tracking: reset reference balance each UTC day so the
        # limit applies to intraday losses, not total losses since bot start.
        self._day_start_balance: float = initial_balance
        self._last_reset_date = datetime.utcnow().date()
        self._redis = redis_client
        # True — баланс начала дня восстановлен из Redis (рестарт в те же сутки UTC).
        self.day_reference_restored: bool = False
        self._restore_day_reference()

    # ── Daily reference persistence ───────────────────────────────────────────

    def _ref_key(self) -> str:
        mode = "paper" if Config.PAPER_TRADING else "live"
        return f"bot:risk:daily_ref:{mode}"

    def _raw_redis(self) -> Any:
        rc = self._redis
        if rc is None or not getattr(rc, "_available", True):
            return None
        return getattr(rc, "redis_client", None)

    def _restore_day_reference(self) -> None:
        """Восстанавливает баланс начала дня, если он сохранён за те же сутки UTC."""
        raw = self._raw_redis()
        if raw is None:
            return
        try:
            data = raw.get(self._ref_key())
            if not data:
                return
            if isinstance(data, bytes):
                data = data.decode()
            ref = json.loads(data)
            if ref.get("date") == self._last_reset_date.isoformat():
                self._day_start_balance = float(ref["balance"])
                self.day_reference_restored = True
                self.logger.info(
                    "Daily loss reference restored: $%.2f (%s)",
                    self._day_start_balance,
                    ref["date"],
                )
        except Exception as exc:
            self.logger.warning("Daily loss reference restore failed: %s", exc)

    def _save_day_reference(self) -> None:
        raw = self._raw_redis()
        if raw is None:
            return
        try:
            raw.setex(
                self._ref_key(),
                _DAY_REF_TTL,
                json.dumps(
                    {
                        "date": self._last_reset_date.isoformat(),
                        "balance": self._day_start_balance,
                    }
                ),
            )
        except Exception as exc:
            self.logger.warning("Daily loss reference save failed: %s", exc)

    def set_day_reference(self, balance: float) -> None:
        """Задаёт баланс начала дня (после синхронизации с биржей) и сохраняет его."""
        self._day_start_balance = balance
        self._save_day_reference()

    def calculate_kelly_size(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        win_rate: float,
        current_balance: float,
    ) -> float:
        """
        Рассчитывает размер позиции по критерию Келли (Half-Kelly).

        Kelly fraction = (p·b − q) / b, где b = reward/risk, p = win_rate.
        Используется Half-Kelly (×0.5) и кэп 20% баланса для защиты от
        overbetting при зашумлённых оценках win_rate.

        :param entry_price: Цена входа.
        :param stop_loss: Цена стоп-лосса.
        :param take_profit: Цена тейк-профита.
        :param win_rate: Доля прибыльных сделок [0, 1].
        :param current_balance: Текущий баланс USDT.
        :return: Размер позиции в единицах актива (0 если Kelly ≤ 0).
        """
        if entry_price <= 0 or stop_loss <= 0 or take_profit <= 0:
            return 0.0
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        if risk <= 0:
            return 0.0

        b = reward / risk  # reward-to-risk ratio
        if b == 0:
            # TP совпадает с entry — деление на ноль; сделка смысла не имеет.
            return 0.0
        p = max(0.0, min(1.0, win_rate))
        q = 1.0 - p

        kelly = (p * b - q) / b
        # Half-Kelly, capped at 20% to limit overbetting from noisy win_rate
        fraction = max(0.0, min(kelly * 0.5, 0.20))
        usdt = current_balance * fraction
        return usdt / entry_price

    def check_daily_loss_limit(self, current_balance: float) -> bool:
        """
        Проверяет, не превышен ли дневной лимит потерь.

        Лимит сбрасывается в UTC-полночь: `_day_start_balance` обновляется
        до текущего баланса, чтобы ограничивать только внутридневные потери,
        а не совокупные потери с момента запуска бота.

        :param current_balance: Текущий баланс счёта.
        :return: True если лимит НЕ превышен (торговля разрешена),
            False если бот должен прекратить торговлю на сегодня.
        """
        today = datetime.utcnow().date()
        if today != self._last_reset_date:
            self._day_start_balance = current_balance
            self._last_reset_date = today
            self._save_day_reference()
            self.logger.info(
                "Daily loss limit reset for %s; reference balance $%.2f",
                today,
                current_balance,
            )

        loss = self._day_start_balance - current_balance
        limit = self._day_start_balance * Config.DAILY_LOSS_LIMIT
        if loss >= limit:
            self.logger.warning(
                "Достигнут дневной лимит потерь: $%.2f >= $%.2f. "
                "Торговля остановлена.",
                loss,
                limit,
            )
            return False
        return True
