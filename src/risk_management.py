"""
Управление рисками: расчёт позиций, стоп-лосса, тейк-профита и валидация сигналов.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict

from config import Config
from src.redis_client import RedisClient


class RiskManager:
    """
    Менеджер рисков для торгового бота.

    Рассчитывает размер позиции, стоп-лосс, тейк-профит и валидирует
    торговые сигналы на основе параметров риска. Сохраняет расчёты
    в Redis для аудита.
    """

    def __init__(
        self,
        initial_balance: float,
        risk_per_trade: float = 0.02,
    ) -> None:
        """
        Инициализирует менеджер рисков.

        :param initial_balance: Начальный баланс счёта.
        :param risk_per_trade: Доля баланса на риск в сделке
            (по умолчанию 0.02 = 2%).
        """
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.redis = RedisClient()
        self.logger = logging.getLogger(__name__)

    async def calculate_position_size(
        self,
        current_balance: float,
        entry_price: float,
        stop_loss: float,
    ) -> float:
        """
        Рассчитывает размер позиции по правилам управления рисками.

        Размер = (баланс × риск) / |entry - stop_loss|.
        Результат сохраняется в Redis под ключом "risk_calculation".

        :param current_balance: Текущий баланс счёта.
        :param entry_price: Цена входа в позицию.
        :param stop_loss: Цена стоп-лосса.
        :return: Рассчитанный размер позиции (0 если разница цен = 0).
        """
        risk_amount = current_balance * self.risk_per_trade
        price_difference = abs(entry_price - stop_loss)

        if price_difference == 0:
            return 0

        position_size = risk_amount / price_difference

        risk_data = {
            "timestamp": datetime.now().isoformat(),
            "current_balance": current_balance,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "position_size": position_size,
            "risk_amount": risk_amount,
        }
        self.redis.save_trading_state("risk_calculation", risk_data)
        return position_size

    async def calculate_stop_loss(
        self, entry_price: float, signal: Dict[str, Any]
    ) -> float:
        """
        Рассчитывает цену стоп-лосса для сигнала.

        Если сигнал содержит ATR — использует ATR-based SL (1.5× ATR).
        Иначе применяет STOP_LOSS_PERCENT из Config.

        :param entry_price: Цена входа.
        :param signal: Сигнал с ключами "action" и опционально "atr".
        :return: Цена стоп-лосса.
        """
        atr = signal.get("atr", None)
        if atr and atr > 0:
            if signal["action"] == "buy":
                return entry_price - 1.5 * atr
            if signal["action"] == "sell":
                return entry_price + 1.5 * atr
            return entry_price

        pct = Config.STOP_LOSS_PERCENT
        if signal["action"] == "buy":
            return entry_price * (1 - pct)
        if signal["action"] == "sell":
            return entry_price * (1 + pct)
        return entry_price

    async def calculate_take_profit(
        self, entry_price: float, signal: Dict[str, Any]
    ) -> float:
        """
        Рассчитывает цену тейк-профита с соотношением риск/прибыль 1:2.

        :param entry_price: Цена входа.
        :param signal: Сигнал с ключом "action".
        :return: Цена тейк-профита.
        """
        rr = 2.0  # соотношение риск/прибыль 1:2

        if signal["action"] == "buy":
            stop_loss = await self.calculate_stop_loss(entry_price, signal)
            risk = entry_price - stop_loss
            return entry_price + risk * rr

        if signal["action"] == "sell":
            stop_loss = await self.calculate_stop_loss(entry_price, signal)
            risk = stop_loss - entry_price
            return entry_price - risk * rr

        return entry_price

    async def validate_signal(self, signal: Dict[str, Any], market_data: Any) -> bool:
        """
        Валидирует торговый сигнал по правилам риска.

        Проверяет: сигнал не "hold", уверенность >= MIN_SIGNAL_CONFIDENCE.

        :param signal: Словарь с ключами "action" и "confidence".
        :param market_data: Рыночные данные (не используется).
        :return: True если сигнал проходит проверку, False иначе.
        """
        if signal["action"] == "hold":
            return False

        if signal.get("confidence", 0) < Config.MIN_SIGNAL_CONFIDENCE:
            self.logger.warning("Уверенность сигнала слишком низкая")
            return False

        return True

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
        оverbetting при зашумлённых оценках win_rate.

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

        :param current_balance: Текущий баланс счёта.
        :return: True если лимит НЕ превышен (торговля разрешена),
            False если бот должен прекратить торговлю на сегодня.
        """
        loss = self.initial_balance - current_balance
        limit = self.initial_balance * Config.DAILY_LOSS_LIMIT
        if loss >= limit:
            self.logger.warning(
                f"Достигнут дневной лимит потерь: "
                f"${loss:.2f} >= ${limit:.2f}. "
                "Торговля остановлена."
            )
            return False
        return True
