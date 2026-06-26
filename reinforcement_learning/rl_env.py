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
# Размер вектора наблюдения: 11 рыночных + 3 портфельных + 7 контекстных
OBS_DIM = 21


class TradingEnv(gym.Env):
    """
    Среда для симуляции торговли на исторических OHLCV-данных.

    Пространство действий — непрерывное Box(-1, 1):
        action > HOLD_ZONE  → BUY  (fraction of balance)
        action < -HOLD_ZONE → SELL (fraction of position)
        else                → HOLD

    Пространство наблюдений Box(-inf, inf, shape=(21,)):
        [open_rel, high_rel, low_rel, in_position, vol_norm, rsi_norm,
         macd_norm, macd_sig_norm, bb_upper_rel, bb_mid_rel, bb_lower_rel,
         balance_norm, pos_value_norm, val_norm,
         funding_rate_norm, ob_imbalance, pcr_norm, fear_greed_norm,
         iv_skew_norm, basis_norm, google_trends_norm]

    Все ценовые фичи нормализованы относительно close — BTC ($90k)
    и SOL ($150) дают одинаковый масштаб градиентов нейросети.
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

        # Пиковое значение для отслеживания просадки; скользящие доходности для Sharpe
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
        current_price = price  # same value, alias for pos_value_norm

        # Новые фичи (индексы 14-20) — если колонка отсутствует → нейтральный дефолт
        funding_raw = float(row.get("funding_rate", 0.0))
        ob_imb = float(row.get("ob_imbalance", 0.0))
        pcr_raw = float(row.get("pcr", 1.0))
        fg_raw = float(row.get("fear_greed", 50.0))
        iv_raw = float(row.get("iv_skew", 0.0))
        basis_raw = float(row.get("basis_pct", 0.0))
        gt_raw = float(row.get("google_trends", 50.0))

        new_features = [
            funding_raw * 1000.0,  # [14] funding_rate_norm
            float(np.clip(ob_imb, -1.0, 1.0)),  # [15] ob_imbalance
            float(np.clip(pcr_raw / 3.0, 0.0, 1.0)),  # [16] pcr_norm
            fg_raw / 100.0,  # [17] fear_greed_norm
            float(np.clip(iv_raw / 20.0, -1.0, 1.0)),  # [18] iv_skew_norm
            float(np.clip(basis_raw / 5.0, -1.0, 1.0)),  # [19] basis_norm
            gt_raw / 100.0,  # [20] google_trends_norm
        ]

        # log1p compresses volume from millions → ~0-1 range
        vol_norm = float(np.log1p(max(float(row["volume"]), 0.0)) / 15.0)

        # Normalize portfolio by initial_balance so all values ≈ 0-2
        balance_norm = self.balance / self.initial_balance
        pos_value_norm = (self.position * current_price) / self.initial_balance
        val_norm = self.current_value / self.initial_balance

        p = price if price > 0 else 1.0  # защита от деления на 0
        open_rel = float(row["open"]) / p - 1.0
        high_rel = float(row["high"]) / p - 1.0
        low_rel = float(row["low"]) / p - 1.0
        macd_norm = float(row.get("macd", 0)) / p
        macd_sig_norm = float(row.get("macd_signal", 0)) / p
        bb_upper_rel = float(row.get("bb_upper", price * 1.02)) / p - 1.0
        bb_mid_rel = float(row.get("bb_middle", price)) / p - 1.0
        bb_lower_rel = float(row.get("bb_lower", price * 0.98)) / p - 1.0
        in_position = 1.0 if self.position > 0 else 0.0

        return np.array(
            [
                open_rel,       # [0] откр. относительно close
                high_rel,       # [1] макс. относительно close
                low_rel,        # [2] мин. относительно close
                in_position,    # [3] есть позиция (0/1)
                vol_norm,
                float(row.get("rsi", 50)) / 100.0,
                macd_norm,      # [6] MACD / close
                macd_sig_norm,  # [7] MACD signal / close
                bb_upper_rel,   # [8] верхняя полоса / close - 1
                bb_mid_rel,     # [9] средняя полоса / close - 1
                bb_lower_rel,   # [10] нижняя полоса / close - 1
                balance_norm,
                pos_value_norm,
                val_norm,
                *new_features,
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
        # Observation was built from close[current_step] — that's what the agent "saw".
        # Execute at open[current_step + 1] to avoid look-ahead bias.
        exec_step = min(self.current_step + 1, len(self.data) - 1)
        exec_price = float(self.data.iloc[exec_step]["open"])
        # current_price tracks the mark-to-market value (last close the agent observed).
        current_price = float(self.data.iloc[self.current_step]["close"])
        prev_value = self.current_value

        if a > HOLD_ZONE and self.balance > 0:
            fraction = min(1.0, (a - HOLD_ZONE) / (1.0 - HOLD_ZONE))
            spend = self.balance * fraction
            bought = spend / exec_price
            commission = spend * COMMISSION
            self.balance -= spend + commission
            self.position += bought
            self.entry_price = exec_price
            self.total_commission += commission

        elif a < -HOLD_ZONE and self.position > 0:
            fraction = min(1.0, (abs(a) - HOLD_ZONE) / (1.0 - HOLD_ZONE))
            sell_qty = self.position * fraction
            revenue = sell_qty * exec_price
            commission = revenue * COMMISSION
            self.balance += revenue - commission
            self.position -= sell_qty
            self.total_commission += commission
            if self.position < 1e-8:
                self.position = 0.0
                self.entry_price = 0.0

        self.current_value = self.balance + self.position * current_price

        # Логарифмическая доходность в процентах (±0.1% движение → ±0.1 единица награды)
        log_ret = 100.0 * float(np.log(self.current_value / max(prev_value, 1e-8)))
        self._returns.append(log_ret)

        # Скользящий бонус Sharpe (только после накопления ≥10 доходностей)
        if len(self._returns) >= 10:
            mu = float(np.mean(self._returns))
            sd = float(np.std(self._returns)) + 1e-8
            sharpe_bonus = 0.1 * (mu / sd)
        else:
            sharpe_bonus = 0.0

        # Штраф за просадку (% от начального баланса, коэффициент 0.01)
        self.peak_value = max(self.peak_value, self.current_value)
        drawdown_pct = (
            100.0
            * max(0.0, self.peak_value - self.current_value)
            / self.initial_balance
        )
        reward = float(np.clip(log_ret + sharpe_bonus - 0.01 * drawdown_pct, -5.0, 5.0))

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
