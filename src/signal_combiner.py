from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from config import Config
from src.ai_analyzer import AIAnalyzer
from src.dqn_signal import SACSignal

logger = logging.getLogger(__name__)


_REGIME_WEIGHTS: dict[str, tuple[float, float]] = {
    # (sac_weight, ai_weight) per detected market regime
    "trending_up": (0.5, 0.5),    # momentum favours SAC
    "trending_down": (0.3, 0.7),  # uncertainty → trust AI more
    "ranging": (0.4, 0.6),        # default balance
    "unknown": (0.4, 0.6),
}


class SignalCombiner:
    """
    Объединяет сигналы SAC и AI согласно MODE.

    MODE=local   → [] (trading_bot использует local fallback)
    MODE=dqn     → только SAC, без Claude API
    MODE=ai      → только Claude API, без SAC
    MODE=hybrid  → оба должны согласиться; расхождение → hold

    hybrid-логика:
    - Оба buy/sell → combined conf = w_sac*SAC + w_ai*AI (режим-зависимо)
    - Расходятся → пропуск
    - AI молчит, SAC conf >= 0.80 → доверяем SAC
    """

    _W_SAC = 0.4
    _W_AI = 0.6
    _SAC_SOLO_MIN = 0.80

    def __init__(self, ai: AIAnalyzer):
        self.ai = ai
        self.sac = SACSignal()
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"SignalCombiner mode={Config.MODE}")

    async def combine(
        self,
        snapshots: List[Dict[str, Any]],
        balance: float,
        regime: str = "unknown",
        regimes: Dict[str, str] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Возвращает рекомендации согласно MODE.

        :param snapshots: Снэпшоты из MarketScanner.
        :param balance: Баланс USDT.
        :param regime: Fallback-режим (используется если regimes=None).
        :param regimes: Per-symbol режимы {"BTC/USDT": "trending_up", ...}.
        :return: Список рекомендаций.
        """
        mode = Config.MODE

        if mode == "local":
            return []

        if mode == "ai":
            return await self.ai.analyze(snapshots, balance)

        if mode == "dqn":
            return self._sac_only(snapshots, balance)

        if mode == "hybrid":
            return await self._hybrid(snapshots, balance, regime, regimes)

        self.logger.warning(f"Unknown MODE='{mode}', fallback to 'ai'")
        return await self.ai.analyze(snapshots, balance)

    @staticmethod
    def _sl_tp(price: float, atr: float, action: str) -> Tuple[float, float]:
        """
        Рассчитывает стоп-лосс и тейк-профит через ATR (1.5x / 3.0x).

        :param price: Текущая цена актива.
        :param atr: Средний истинный диапазон (ATR).
        :param action: Направление сделки ("buy" или "sell").
        :return: Кортеж (stop_loss, take_profit).
        """
        if action == "buy":
            sl = price - 1.5 * atr
            tp = price + 3.0 * atr
        else:
            sl = price + 1.5 * atr
            tp = price - 3.0 * atr
        return round(sl, 6), round(tp, 6)

    def _sac_only(
        self,
        snapshots: List[Dict[str, Any]],
        balance: float,
    ) -> List[Dict[str, Any]]:
        """
        Формирует рекомендации только на основе SAC-сигналов.

        :param snapshots: Снэпшоты из MarketScanner.
        :param balance: Баланс USDT.
        :return: Список рекомендаций с action, confidence, entry, sl/tp.
        """
        results: List[Dict[str, Any]] = []
        min_conf = Config.MIN_SIGNAL_CONFIDENCE
        for snap in snapshots:
            sig = self.sac.get_signal(snap, balance)
            if sig["action"] == "hold" or sig["confidence"] < min_conf:
                continue
            price = snap.get("price", 0)
            atr = snap.get("atr", price * 0.02)
            sl, tp = self._sl_tp(price, atr, sig["action"])
            results.append(
                {
                    "symbol": snap["symbol"],
                    "action": sig["action"],
                    "strategy": "sac",
                    "confidence": sig["confidence"],
                    "entry": price,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "reasoning": f"SAC conf {sig['confidence']:.0%}",
                }
            )
        return results

    async def _hybrid(
        self,
        snapshots: List[Dict],
        balance: float,
        regime: str = "unknown",
        regimes: Dict[str, str] | None = None,
    ) -> List[Dict]:
        """
        Гибридный режим: согласие SAC + AI с режим-зависимыми весами.

        Веса определяются per-symbol режимом из regimes (если передан),
        иначе используется общий fallback-режим.
        trending_up → SAC 50%/AI 50%, trending_down → SAC 30%/AI 70%,
        ranging / unknown → SAC 40%/AI 60%.
        """
        ai_recs = await self.ai.analyze(snapshots, balance)
        ai_map: Dict[str, Dict] = {r["symbol"]: r for r in ai_recs}

        results = []
        for snap in snapshots:
            sym = snap["symbol"]
            sym_regime = (regimes or {}).get(sym, regime)
            w_sac, w_ai = _REGIME_WEIGHTS.get(sym_regime, (self._W_SAC, self._W_AI))

            sac_sig = self.sac.get_signal(snap, balance)
            ai = ai_map.get(sym)

            d_action = sac_sig["action"]
            d_conf = sac_sig["confidence"]

            if d_action == "hold":
                continue

            # AI молчит — принимаем SAC только при высоком conf
            if ai is None:
                if d_conf >= self._SAC_SOLO_MIN:
                    price = snap.get("price", 0)
                    atr = snap.get("atr", price * 0.02)
                    sl, tp = self._sl_tp(price, atr, d_action)
                    results.append(
                        {
                            "symbol": sym,
                            "action": d_action,
                            "strategy": "sac",
                            "confidence": d_conf,
                            "entry": price,
                            "stop_loss": sl,
                            "take_profit": tp,
                            "reasoning": f"SAC {d_conf:.0%}, AI silent",
                        }
                    )
                continue

            a_action = ai.get("action", "hold")

            if d_action == a_action:
                a_conf = ai.get("confidence", 0)
                combined = round(d_conf * w_sac + a_conf * w_ai, 3)
                if combined < Config.MIN_SIGNAL_CONFIDENCE:
                    continue
                rec = dict(ai)
                rec["confidence"] = combined
                base_strat = ai.get("strategy", "ai")
                rec["strategy"] = f"hybrid({base_strat}+sac)"
                reason = ai.get("reasoning", "").strip()
                rec["reasoning"] = (
                    f"{reason} [SAC {d_conf:.0%}, regime={sym_regime}]"
                    if reason
                    else f"AI+SAC agree, SAC {d_conf:.0%}, regime={sym_regime}"
                )
                results.append(rec)
            else:
                self.logger.debug(
                    "%s: AI=%s vs SAC=%s → hold", sym, a_action, d_action
                )

        return results
