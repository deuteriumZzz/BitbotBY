import logging
from typing import Any, Dict, List

from config import Config
from src.ai_analyzer import AIAnalyzer
from src.dqn_signal import DQNSignal

logger = logging.getLogger(__name__)


class SignalCombiner:
    """
    Объединяет сигналы DQN и AI согласно MODE.

    MODE=local   → [] (trading_bot использует local fallback)
    MODE=dqn     → только DQN, без Claude API
    MODE=ai      → только Claude API, без DQN
    MODE=hybrid  → оба должны согласиться; расхождение → hold

    hybrid-логика:
    - Оба buy/sell → combined conf = DQN_WEIGHT*DQN
                                   + AI_WEIGHT*AI
    - Расходятся → пропуск
    - AI молчит, DQN conf >= DQN_SOLO_CONFIDENCE → доверяем DQN
    """

    def __init__(self, ai: AIAnalyzer):
        self.ai = ai
        self.dqn = DQNSignal()
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"SignalCombiner mode={Config.MODE}")

    async def combine(
        self,
        snapshots: List[Dict[str, Any]],
        balance: float,
    ) -> List[Dict[str, Any]]:
        """
        Возвращает рекомендации согласно MODE.

        :param snapshots: Снэпшоты из MarketScanner.
        :param balance: Баланс USDT.
        :return: Список рекомендаций.
        """
        mode = Config.MODE

        if mode == "local":
            return []

        if mode == "ai":
            return await self.ai.analyze(snapshots, balance)

        if mode == "dqn":
            return self._dqn_only(snapshots, balance)

        if mode == "hybrid":
            return await self._hybrid(snapshots, balance)

        self.logger.warning(
            f"Unknown MODE='{mode}', fallback to 'ai'"
        )
        return await self.ai.analyze(snapshots, balance)

    @staticmethod
    def _sl_tp(
        price: float, atr: float, action: str
    ):
        """Стоп-лосс и тейк-профит через ATR."""
        if action == "buy":
            sl = price - 1.5 * atr
            tp = price + 3.0 * atr
        else:
            sl = price + 1.5 * atr
            tp = price - 3.0 * atr
        return round(sl, 6), round(tp, 6)

    def _dqn_only(
        self,
        snapshots: List[Dict],
        balance: float,
    ) -> List[Dict]:
        results = []
        min_conf = Config.MIN_SIGNAL_CONFIDENCE
        for snap in snapshots:
            sig = self.dqn.get_signal(snap, balance)
            if (
                sig["action"] == "hold"
                or sig["confidence"] < min_conf
            ):
                continue
            price = snap.get("price", 0)
            atr = snap.get("atr", price * 0.02)
            sl, tp = self._sl_tp(price, atr, sig["action"])
            results.append({
                "symbol": snap["symbol"],
                "action": sig["action"],
                "strategy": "dqn",
                "confidence": sig["confidence"],
                "entry": price,
                "stop_loss": sl,
                "take_profit": tp,
                "reasoning": (
                    f"DQN Q-conf {sig['confidence']:.0%}"
                ),
            })
        return results

    async def _hybrid(
        self,
        snapshots: List[Dict],
        balance: float,
    ) -> List[Dict]:
        """
        Гибридный режим: согласие DQN + AI.

        Взвешенный confidence: 40% DQN + 60% AI.
        """
        ai_recs = await self.ai.analyze(snapshots, balance)
        ai_map: Dict[str, Dict] = {
            r["symbol"]: r for r in ai_recs
        }

        results = []
        for snap in snapshots:
            sym = snap["symbol"]
            dqn = self.dqn.get_signal(snap, balance)
            ai = ai_map.get(sym)

            d_action = dqn["action"]
            d_conf = dqn["confidence"]

            if d_action == "hold":
                continue

            # AI молчит — принимаем DQN только при высоком conf
            if ai is None:
                if d_conf >= Config.DQN_SOLO_CONFIDENCE:
                    price = snap.get("price", 0)
                    atr = snap.get("atr", price * 0.02)
                    sl, tp = self._sl_tp(price, atr, d_action)
                    results.append({
                        "symbol": sym,
                        "action": d_action,
                        "strategy": "dqn",
                        "confidence": d_conf,
                        "entry": price,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "reasoning": (
                            f"DQN {d_conf:.0%}, AI silent"
                        ),
                    })
                continue

            a_action = ai.get("action", "hold")

            if d_action == a_action:
                a_conf = ai.get("confidence", 0)
                combined = round(
                    d_conf * Config.DQN_WEIGHT
                    + a_conf * Config.AI_WEIGHT,
                    3,
                )
                if combined < Config.MIN_SIGNAL_CONFIDENCE:
                    continue
                rec = dict(ai)
                rec["confidence"] = combined
                base_strat = ai.get("strategy", "ai")
                rec["strategy"] = (
                    f"hybrid({base_strat}+dqn)"
                )
                reason = ai.get("reasoning", "").strip()
                rec["reasoning"] = (
                    f"{reason} [DQN {d_conf:.0%}]"
                    if reason
                    else f"AI+DQN agree, DQN {d_conf:.0%}"
                )
                results.append(rec)
            else:
                self.logger.debug(
                    f"{sym}: AI={a_action} "
                    f"vs DQN={d_action} → hold"
                )

        return results
