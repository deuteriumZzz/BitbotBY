"""
Среда Gymnasium для симуляции крипто-торговли с SAC (непрерывные действия).
"""

from __future__ import annotations

# Зона нечувствительности: |action| <= HOLD_ZONE → HOLD
import os as _os
from collections import deque
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

HOLD_ZONE = float(_os.getenv("SAC_HOLD_ZONE", "0.3"))
# Комиссия за сделку (синхронизирована с COMMISSION_RATE из .env)
COMMISSION = float(_os.getenv("COMMISSION_RATE", "0.001"))
# Размер вектора наблюдения: 11 рыночных + 3 портфельных
OBS_DIM = 14


class TradingEnv(gym.Env):
    """
    Среда для симуляции торговли на исторических OHLCV-данных.

    Пространство действий — непрерывное Box(-1, 1):
        action > HOLD_ZONE  → BUY  (fraction of balance)
        action < -HOLD_ZONE → SELL (fraction of position)
        else                → HOLD

    Пространство наблюдений Box(-inf, inf, shape=(14,)):
        [open, high, low, close, volume, rsi, macd, macd_signal,
         bb_upper, bb_middle, bb_lower, balance, position, current_value]
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
    ) -> None:
        """
        Инициализирует торговую среду.

        :param data: DataFrame с OHLCV и индикаторами.
        :param initial_balance: Начальный баланс (по умолчанию 10000.0).
        """
        super(TradingEnv, self).__init__()

        self.data = data.copy()
        self.initial_balance = initial_balance

        # Инициализация портфельного состояния до построения obs
        self.balance = initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.current_step = 0
        self.done = False
        self.current_value = initial_balance
        self.total_commission = 0.0

        # Peak value for drawdown tracking; rolling returns for Sharpe
        self.peak_value: float = initial_balance
        self._returns: deque = deque(maxlen=50)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

    def _get_observation(self) -> np.ndarray:
        """
        Возвращает вектор наблюдения из текущего шага данных.

        :return: float32-массив shape=(OBS_DIM,).
        """
        if self.current_step >= len(self.data):
            return np.zeros(OBS_DIM, dtype=np.float32)

        row = self.data.iloc[self.current_step]
        price = float(row["close"])
        return np.array(
            [
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                price,
                float(row["volume"]),
                float(row.get("rsi", 50)),
                float(row.get("macd", 0)),
                float(row.get("macd_signal", 0)),
                float(row.get("bb_upper", price * 1.02)),
                float(row.get("bb_middle", price)),
                float(row.get("bb_lower", price * 0.98)),
                self.balance,
                self.position,
                self.current_value,
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        seed: Any = None,
        options: Any = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Сбрасывает среду в начальное состояние.

        :param seed: Зерно генератора случайных чисел (Gymnasium API).
        :param options: Дополнительные параметры (Gymnasium API).
        :return: Кортеж (наблюдение, info).
        """
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.current_step = 0
        self.done = False
        self.current_value = self.initial_balance
        self.total_commission = 0.0
        self.peak_value = self.initial_balance
        self._returns.clear()
        return self._get_observation(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Выполняет один шаг среды.

        :param action: float32-массив shape=(1,), диапазон [-1, 1].
            > HOLD_ZONE  → BUY  (пропорционально свободному балансу)
            < -HOLD_ZONE → SELL (пропорционально текущей позиции)
            else         → HOLD
        :return: (obs, reward, terminated, truncated, info).
        """
        if self.done:
            return self._get_observation(), 0.0, True, False, {}

        a = float(action[0])
        current_price = float(self.data.iloc[self.current_step]["close"])
        prev_value = self.current_value

        if a > HOLD_ZONE and self.balance > 0:
            fraction = min(1.0, (a - HOLD_ZONE) / (1.0 - HOLD_ZONE))
            spend = self.balance * fraction
            bought = spend / current_price
            commission = spend * COMMISSION
            self.balance -= spend + commission
            self.position += bought
            self.entry_price = current_price
            self.total_commission += commission

        elif a < -HOLD_ZONE and self.position > 0:
            fraction = min(1.0, (abs(a) - HOLD_ZONE) / (1.0 - HOLD_ZONE))
            sell_qty = self.position * fraction
            revenue = sell_qty * current_price
            commission = revenue * COMMISSION
            self.balance += revenue - commission
            self.position -= sell_qty
            self.total_commission += commission
            if self.position < 1e-8:
                self.position = 0.0
                self.entry_price = 0.0

        self.current_value = self.balance + self.position * current_price

        # Log return in percent (±0.1% move → ±0.1 reward unit)
        log_ret = 100.0 * float(np.log(self.current_value / max(prev_value, 1e-8)))
        self._returns.append(log_ret)

        # Rolling Sharpe bonus (only once ≥10 returns accumulated)
        if len(self._returns) >= 10:
            mu = float(np.mean(self._returns))
            sd = float(np.std(self._returns)) + 1e-8
            sharpe_bonus = 0.1 * (mu / sd)
        else:
            sharpe_bonus = 0.0

        # Drawdown penalty (% of initial balance, factor 0.01)
        self.peak_value = max(self.peak_value, self.current_value)
        drawdown_pct = (
            100.0
            * max(0.0, self.peak_value - self.current_value)
            / self.initial_balance
        )
        reward = log_ret + sharpe_bonus - 0.01 * drawdown_pct

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        info: Dict[str, Any] = {
            "step": self.current_step,
            "balance": self.balance,
            "position": self.position,
            "value": self.current_value,
            "price": current_price,
        }
        return (
            self._get_observation(),
            reward,
            self.done,
            False,
            info,
        )

    def render(self, mode: str = "human") -> None:
        """
        Выводит текущее состояние среды в консоль.

        :param mode: Режим рендеринга (по умолчанию "human").
        """
        price = self.data.iloc[self.current_step]["close"]
        print(
            f"Step: {self.current_step}, "
            f"Price: {price:.2f}, "
            f"Balance: {self.balance:.2f}, "
            f"Position: {self.position:.4f}, "
            f"Value: {self.current_value:.2f}"
        )
