from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

import anthropic

from config import Config
from src.strategies import get_all_strategies

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert crypto trading analyst with deep knowledge of "
    "technical analysis, market sentiment, and risk management. "
    "You analyze multiple coins simultaneously and return structured "
    "JSON recommendations. Be precise, data-driven, and conservative "
    "with risk. Always respond with valid JSON only."
)


class AIAnalyzer:
    """
    Анализирует рынок через Claude API.

    Принимает снэпшоты нескольких монет (цена, индикаторы, новости,
    история), отправляет один batch-запрос в Claude, возвращает список
    торговых рекомендаций с reasoning на русском.
    """

    def __init__(self):
        self.enabled = bool(Config.ANTHROPIC_API_KEY)
        if self.enabled:
            self.client = anthropic.AsyncAnthropic(api_key=Config.ANTHROPIC_API_KEY)
        self.model = Config.AI_MODEL
        self.strategies = get_all_strategies()
        self.logger = logging.getLogger(__name__)

    def _build_strategy_list(self) -> str:
        """
        Формирует текстовый список доступных стратегий для промпта.

        :return: Многострочная строка с описанием каждой стратегии.
        """
        lines = []
        for s in self.strategies:
            desc = s["description"][:60]
            lines.append(
                f"  {s['name']}: {desc}... "
                f"[risk={s['risk_level']}, "
                f"market={s['market_type']}]"
            )
        return "\n".join(lines)

    def _build_prompt(
        self,
        snapshots: List[Dict[str, Any]],
        balance: float,
    ) -> str:
        """
        Строит промпт для Claude API на основе снэпшотов рынка.

        :param snapshots: Список снэпшотов из MarketScanner.
        :param balance: Баланс пользователя в USDT.
        :return: Готовый текст промпта.
        """
        strategy_block = self._build_strategy_list()
        data_block = json.dumps(snapshots, indent=2, ensure_ascii=False)
        min_conf = Config.MIN_SIGNAL_CONFIDENCE
        return (
            f"Analyze the following crypto market data and provide "
            f"trading recommendations.\n\n"
            f"USER BALANCE: ${balance:.2f} USDT\n\n"
            f"AVAILABLE STRATEGIES:\n{strategy_block}\n\n"
            f"MARKET DATA (real-time snapshots):\n{data_block}\n\n"
            f"RULES:\n"
            f"- Only include symbols with confidence >= {min_conf}\n"
            f"- news_sentiment > 0.3 boosts buy; < -0.3 boosts sell\n"
            f"- Match strategy to market_type\n"
            f"- stop_loss: 2-8% from entry; take_profit: 2x risk min\n"
            f"- Omit symbols with no clear signal\n"
            f"- Reasoning in Russian, 1-2 sentences max\n\n"
            f"Return ONLY a valid JSON array (no markdown):\n"
            f'[{{"symbol":"SOL/USDT","action":"buy","strategy":'
            f'"rsi_momentum","confidence":0.84,"entry":148.50,'
            f'"stop_loss":140.00,"take_profit":165.00,'
            f'"reasoning":"RSI перепродан (28)."}}]'
        )

    async def analyze(
        self,
        snapshots: List[Dict[str, Any]],
        balance: float,
    ) -> List[Dict[str, Any]]:
        """
        Отправляет снэпшоты в Claude API, возвращает рекомендации.

        При отсутствии ключа или ошибке — пустой список,
        trading_bot автоматически переключается на локальные стратегии.

        :param snapshots: Список снэпшотов из MarketScanner.
        :param balance: Баланс пользователя в USDT.
        :return: Отфильтрованный список рекомендаций.
        """
        if not self.enabled or not snapshots:
            return []

        prompt = self._build_prompt(snapshots, balance)

        try:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()

            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.split("```")[0]

            recs = json.loads(raw.strip())
            if not isinstance(recs, list):
                recs = [recs]

            required_keys = ("symbol", "action", "strategy", "confidence")
            valid = [
                r
                for r in recs
                if (
                    isinstance(r, dict)
                    and all(k in r for k in required_keys)
                    and r["confidence"] >= Config.MIN_SIGNAL_CONFIDENCE
                    and r["action"] in ("buy", "sell", "hold")
                )
            ]

            self.logger.info(
                f"AI: {len(valid)} signals " f"from {len(snapshots)} symbols"
            )
            return valid

        except json.JSONDecodeError as e:
            self.logger.error(f"AI JSON parse error: {e}")
            return []
        except anthropic.APIError as e:
            self.logger.error(f"Anthropic API error: {e}")
            return []
        except Exception as e:
            self.logger.error(f"AI analysis failed: {e}")
            return []

    def recommend_strategy_local(self, snapshot: Dict[str, Any]) -> Tuple[str, float]:
        """
        Локальный выбор стратегии без AI (fallback).

        :param snapshot: Снэпшот из MarketScanner.build_snapshot().
        :return: (strategy_name, confidence)
        """
        ind = snapshot.get("indicators", {})
        rsi = ind.get("rsi", 50)
        bb_width = ind.get("bb_width", 0.04)
        vol_ratio = snapshot.get("volume_ratio", 1.0)
        trend = ind.get("trend", "sideways")
        macd_dir = ind.get("macd", "bearish")
        bb_pos = ind.get("bb_position", "middle")

        if bb_width > 0.08 and vol_ratio > 2.0:
            return "breakout", 0.72

        if bb_pos == "near_lower" and rsi < 35:
            return "bollinger_bands", 0.75
        if bb_pos == "near_upper" and rsi > 65:
            return "bollinger_bands", 0.75

        if trend == "uptrend" and macd_dir == "bullish":
            return "swing_trading" if rsi < 60 else "trend_following", 0.70

        if trend == "downtrend" and macd_dir == "bearish":
            return "swing_trading" if rsi > 40 else "trend_following", 0.70

        if rsi < 30:
            return "rsi_momentum", 0.78
        if rsi > 70:
            return "rsi_momentum", 0.78

        return "ema_crossover", 0.60
