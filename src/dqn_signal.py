import logging
import os
from typing import Any, Dict

import numpy as np
import torch

from config import Config
from reinforcement_learning.rl_agent import RLAgent

logger = logging.getLogger(__name__)

# 7 рыночных + 4 портфельных = 11 (совпадает с RLAgent.get_state)
_STATE_SIZE = 11
_ACTION_SIZE = 3  # 0=hold, 1=buy, 2=sell
_ACTION_MAP = {0: "hold", 1: "buy", 2: "sell"}


class DQNSignal:
    """
    Инференс DQN-модели для одного снэпшота.

    Конвертирует snapshot → вектор состояния → Q-значения →
    {action, confidence}. Загружает модель из DQN_MODEL_PATH.
    При отсутствии файла весов возвращает hold с conf=0.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.agent = RLAgent(_STATE_SIZE, _ACTION_SIZE)
        self.loaded = False
        self._try_load()

    def _try_load(self) -> None:
        path = Config.DQN_MODEL_PATH
        if not path or not os.path.exists(path):
            self.logger.warning(
                f"DQN weights not found at '{path}'. "
                "Run: python reinforcement_learning/train_dqn.py"
            )
            return
        try:
            self.agent.load_model(path)
            self.agent.epsilon = 0.0  # greedy inference
            self.loaded = True
            self.logger.info(f"DQN loaded from {path}")
        except Exception as e:
            self.logger.warning(f"DQN load failed: {e}")

    def _to_state(
        self, snap: Dict[str, Any], balance: float
    ) -> np.ndarray:
        """
        Конвертирует снэпшот MarketScanner в вектор состояния.

        Формат совпадает с RLAgent.get_state() — 11 признаков:
        7 рыночных (price, volume, rsi, macd, bb_upper, bb_lower,
        volatility) + 4 портфельных.
        """
        ind = snap.get("indicators", {})
        price = snap.get("price", 0.0)
        bb_w = ind.get("bb_width", 0.04)

        bb_upper = price * (1 + bb_w / 2)
        bb_lower = price * (1 - bb_w / 2)
        macd_val = (
            1.0 if ind.get("macd") == "bullish" else -1.0
        )

        market = {
            "price": price,
            "volume": snap.get("volume_ratio", 1.0),
            "rsi": ind.get("rsi", 50.0),
            "macd": macd_val,
            "bollinger_upper": bb_upper,
            "bollinger_lower": bb_lower,
            "volatility": (
                snap.get("atr", price * 0.02) / price
                if price > 0 else 0.0
            ),
        }
        portfolio = {
            "total_value": balance,
            "total_unrealized_pnl": 0.0,
            "concentration_risk": 0.1,
            "cash_balance": balance,
        }
        return self.agent.get_state(market, portfolio)

    def get_signal(
        self,
        snap: Dict[str, Any],
        balance: float,
    ) -> Dict[str, Any]:
        """
        Greedy инференс DQN для одного снэпшота.

        Выбирает действие с max Q-значением; confidence — softmax.

        :param snap: Снэпшот из MarketScanner.build_snapshot().
        :param balance: Баланс USDT.
        :return: {action, confidence, source="dqn"}
        """
        default = {
            "action": "hold",
            "confidence": 0.0,
            "source": "dqn",
        }
        if not self.loaded:
            return default

        try:
            state = self._to_state(snap, balance)
            t = (
                torch.FloatTensor(state)
                .unsqueeze(0)
                .to(self.agent.device)
            )
            with torch.no_grad():
                q = self.agent.policy_net(t).squeeze()
                probs = torch.softmax(q, dim=0)
                idx = int(q.argmax().item())
                conf = float(probs[idx].item())

            return {
                "symbol": snap.get("symbol", ""),
                "action": _ACTION_MAP[idx],
                "confidence": round(conf, 3),
                "source": "dqn",
            }
        except Exception as e:
            self.logger.error(f"DQN inference error: {e}")
            return default
