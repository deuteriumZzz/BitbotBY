"""
Управление портфелем: отслеживание баланса, позиций и ребалансировка.

Менеджер портфеля отслеживает баланс USDT, открытые позиции,
рассчитывает стоимость портфеля и выполняет ребалансировку.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict

from config import Config
from src.redis_client import RedisClient


class PortfolioManager:
    """
    Менеджер портфеля криптовалютного трейдинга.

    Отвечает за отслеживание баланса, позиций, обновление портфеля
    на основе торговых операций, расчёт стоимости и ребалансировку.
    Использует Redis для сохранения состояния между запусками.
    """

    def __init__(self, initial_balance: float) -> None:
        """
        Инициализирует менеджер портфеля.

        :param initial_balance: Начальный баланс в USDT.
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions: Dict[str, float] = {}
        self.commission_rate: float = Config.COMMISSION_RATE
        self.total_commissions: float = 0.0
        self.redis = RedisClient()
        self.logger = logging.getLogger(__name__)

    async def update_portfolio(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
    ) -> bool:
        """
        Обновляет портфель на основе торговой операции.

        Для покупки проверяет достаточность баланса, для продажи —
        наличие позиции. Сохраняет новое состояние в Redis.

        :param symbol: Символ актива (например, "BTC").
        :param action: Действие ("buy" или "sell").
        :param quantity: Количество актива.
        :param price: Цена актива.
        :return: True если операция выполнена, False иначе.
        """
        try:
            if action == "buy":
                commission = quantity * price * self.commission_rate
                cost = quantity * price + commission
                if cost <= self.current_balance:
                    self.current_balance -= cost
                    self.total_commissions += commission
                    self.positions[symbol] = self.positions.get(symbol, 0) + quantity
                    await self._save_portfolio_state()
                    return True

            elif action == "sell":
                if symbol in self.positions and self.positions[symbol] >= quantity:
                    commission = quantity * price * self.commission_rate
                    revenue = quantity * price - commission
                    self.current_balance += revenue
                    self.total_commissions += commission
                    self.positions[symbol] -= quantity
                    if self.positions[symbol] <= 0:
                        del self.positions[symbol]
                    await self._save_portfolio_state()
                    return True

            return False

        except Exception as e:
            self.logger.error("Ошибка обновления портфеля: %s", e, exc_info=True)
            return False

    async def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Рассчитывает общую стоимость портфеля.

        Суммирует текущий баланс и стоимость всех открытых позиций.
        Позиции без цены в словаре игнорируются.

        :param current_prices: Текущие цены активов по символу.
        :return: Общая стоимость портфеля в USDT.
        """
        total_value = self.current_balance
        for symbol, quantity in self.positions.items():
            if symbol in current_prices:
                total_value += quantity * current_prices[symbol]
        return total_value

    async def _save_portfolio_state(self) -> None:
        """
        Сохраняет текущее состояние портфеля в Redis.

        Записывает timestamp, баланс, позиции и накопленные комиссии
        под ключом "portfolio_state".
        """
        state = {
            "timestamp": datetime.now().isoformat(),
            "balance": self.current_balance,
            "positions": self.positions,
            "total_commissions": self.total_commissions,
        }
        self.redis.save_trading_state("portfolio_state", state)

    def get_positions(self) -> Dict[str, float]:
        """
        Возвращает копию текущих открытых позиций.

        :return: Словарь {символ: количество}.
        """
        return self.positions.copy()

    async def get_position_size(self, symbol: str) -> float:
        """
        Возвращает текущий размер позиции для символа.

        :param symbol: Символ актива.
        :return: Количество актива в позиции (0.0 если позиции нет).
        """
        return self.positions.get(symbol, 0.0)

    async def rebalance_portfolio(
        self,
        target_allocations: Dict[str, float],
        current_prices: Dict[str, float],
    ) -> None:
        """
        Ребалансирует портфель к целевым аллокациям.

        Рассчитывает отклонение каждой позиции от цели и выполняет
        покупку или продажу для выравнивания. Аллокации должны
        суммироваться к 1.0 (100%).

        :param target_allocations: Целевые доли по символам (сумма = 1.0).
        :param current_prices: Текущие цены активов.
        """
        total_value = await self.get_portfolio_value(current_prices)

        for symbol, target_alloc in target_allocations.items():
            price = current_prices.get(symbol)
            if price is None or price <= 0:
                continue

            target_value = total_value * target_alloc
            position_size = await self.get_position_size(symbol)
            current_value = position_size * price

            if current_value < target_value:
                buy_value = target_value - current_value
                quantity = buy_value / price
                await self.update_portfolio(symbol, "buy", quantity, price)
            elif current_value > target_value:
                sell_value = current_value - target_value
                quantity = sell_value / price
                await self.update_portfolio(symbol, "sell", quantity, price)
