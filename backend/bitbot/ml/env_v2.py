"""Среда SAC v2: экспозиция-цель, шорт, издержки.

ПРЕДЛОЖЕНИЕ, не принятое владельцем решение: действие a ∈ [-1, 1] — целевая доля
капитала в позиции (минус = шорт); исполнение по open СЛЕДУЮЩЕГО бара (решение
принято на close бара t — заявка не может исполниться по цене, которую
наблюдатель на тот момент не знал); издержки = комиссия + проскальзывание от
оборота + funding на удерживаемую позицию; награда — лог-изменение чистого
капитала. Параметры издержек по умолчанию — оценки, их нужно откалибровать по
реальным исполнениям (`fee`, `slippage` в лог-записях execution) до обучения.

Прежняя `rl_env.TradingEnv` остаётся для совместимости со старой моделью:
long-only, без funding и проскальзывания, 0.1% комиссии.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

DEFAULT_FEE = 0.00055  # Bybit taker, линейные контракты (оценка)
DEFAULT_SLIPPAGE = 0.0003  # оценка, калибруется по факту исполнения


class ExposureSim:
    """Ядро симуляции без зависимостей: один шаг = «решили на close t → open t+1 → close t+1»."""

    def __init__(
        self,
        open_: np.ndarray,
        close: np.ndarray,
        funding: Optional[np.ndarray] = None,
        *,
        initial: float = 10_000.0,
        fee: float = DEFAULT_FEE,
        slippage: float = DEFAULT_SLIPPAGE,
        max_exposure: float = 1.0,
    ) -> None:
        if len(open_) != len(close) or len(open_) < 3:
            raise ValueError("нужны open и close одинаковой длины ≥ 3")
        self.o, self.c = np.asarray(open_, float), np.asarray(close, float)
        self.f = (
            np.zeros_like(self.c) if funding is None else np.asarray(funding, float)
        )
        self.fee, self.slip, self.max_exp = fee, slippage, max_exposure
        self.initial = initial
        self.reset()

    def reset(self) -> None:
        self.t = 0  # индекс бара, на close которого принято решение
        self.cash = self.initial
        self.units = 0.0
        self.costs = 0.0  # накопленные комиссии+проскальзывание
        self.funding_paid = 0.0
        self.equity = self.initial

    @property
    def done(self) -> bool:
        return self.t >= len(self.c) - 1 or self.equity <= 0

    @property
    def exposure(self) -> float:
        return 0.0 if self.equity <= 0 else self.units * self.c[self.t] / self.equity

    def step(self, target: float) -> float:
        """Применяет целевую экспозицию; возвращает лог-доходность чистого капитала за бар."""
        if self.done:
            raise RuntimeError("эпизод закончен")
        target = float(np.clip(target, -self.max_exp, self.max_exp))
        prev = self.equity
        nxt = self.t + 1
        eq_open = self.cash + self.units * self.o[nxt]
        new_units = target * eq_open / self.o[nxt]
        trade = new_units - self.units
        cost = abs(trade) * self.o[nxt] * (self.fee + self.slip)
        self.cash -= trade * self.o[nxt] + cost
        self.units = new_units
        self.costs += cost
        pay = self.units * self.c[nxt] * self.f[nxt]  # лонг платит при funding > 0
        self.cash -= pay
        self.funding_paid += pay
        self.equity = self.cash + self.units * self.c[nxt]
        self.t = nxt
        if self.equity <= 0:
            self.equity = 0.0
            return float(np.log(1e-9 / prev))  # банкротство: наказание конечное
        return float(np.log(self.equity / prev))


def simulate(
    open_: np.ndarray, close: np.ndarray, targets: np.ndarray, funding=None, **kw: Any
) -> Dict[str, Any]:
    """Прогон политики-последовательности `targets[t]` (решение на close t). Для бэктеста."""
    sim = ExposureSim(open_, close, funding, **kw)
    equity = [sim.equity]
    for tgt in targets[: len(close) - 1]:
        if sim.done:
            break
        sim.step(tgt)
        equity.append(sim.equity)
    return {
        "equity": np.asarray(equity),
        "costs": sim.costs,
        "funding_paid": sim.funding_paid,
    }


try:  # Gymnasium нужен только для обучения; симулятор выше работает без него
    import gymnasium as gym
    from gymnasium import spaces

    class TradingEnvV2(gym.Env):
        """Обёртка над `ExposureSim`. `features` — причинные признаки: строка t известна на close t."""

        def __init__(self, features: np.ndarray, open_, close, funding=None, **kw: Any):
            super().__init__()
            self.features = np.asarray(features, np.float32)
            self.sim = ExposureSim(open_, close, funding, **kw)
            self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
            n = self.features.shape[1] + 2  # + экспозиция, накопленная доходность
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=(n,), dtype=np.float32
            )

        def _obs(self) -> np.ndarray:
            s = self.sim
            extra = [s.exposure, s.equity / s.initial - 1.0]
            return np.concatenate([self.features[s.t], extra]).astype(np.float32)

        def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
            super().reset(seed=seed)
            self.sim.reset()
            return self._obs(), {}

        def step(self, action):
            reward = self.sim.step(float(np.asarray(action).ravel()[0]))
            terminated = self.sim.equity <= 0  # банкротство — настоящий конец
            truncated = (not terminated) and self.sim.done  # конец данных — усечение
            return (
                self._obs(),
                reward,
                terminated,
                truncated,
                {"equity": self.sim.equity},
            )

except ImportError:  # pragma: no cover
    TradingEnvV2 = None  # type: ignore[assignment,misc]
