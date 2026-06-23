from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

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

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def _resolve_provider() -> str:
    """
    Определяет активного AI-провайдера согласно Config.AI_PROVIDER.

    auto      → Anthropic → OpenAI → DeepSeek → Groq → none
    anthropic → Claude
    deepseek  → DeepSeek
    openai    → ChatGPT
    groq      → Groq
    """
    p = Config.AI_PROVIDER
    if p == "anthropic":
        return "anthropic" if Config.ANTHROPIC_API_KEY else "none"
    if p == "deepseek":
        return "deepseek" if Config.DEEPSEEK_API_KEY else "none"
    if p == "openai":
        return "openai" if Config.OPENAI_API_KEY else "none"
    if p == "groq":
        return "groq" if getattr(Config, "GROQ_API_KEY", "") else "none"
    # auto: prefer paid flagship, then cheaper options
    if Config.ANTHROPIC_API_KEY:
        return "anthropic"
    if Config.OPENAI_API_KEY:
        return "openai"
    if Config.DEEPSEEK_API_KEY:
        return "deepseek"
    if getattr(Config, "GROQ_API_KEY", ""):
        return "groq"
    return "none"


class AIAnalyzer:
    """
    Анализирует рынок через Claude, DeepSeek или OpenAI (ChatGPT).

    Провайдер выбирается через AI_PROVIDER (.env):
      auto      → Claude → DeepSeek → OpenAI (первый найденный ключ)
      anthropic → только Claude
      deepseek  → только DeepSeek
      openai    → только ChatGPT

    Принимает снэпшоты нескольких монет, отправляет один batch-запрос,
    возвращает список торговых рекомендаций с reasoning на русском.
    """

    # HTTP status codes treated as billing/quota exhaustion → try next provider
    _BILLING_CODES: frozenset = frozenset({402, 429})

    def __init__(self, runtime_config: "Any | None" = None) -> None:
        self._provider = _resolve_provider()
        self._runtime_config = runtime_config
        self.enabled = self._provider != "none"
        self.strategies = get_all_strategies()
        self.logger = logging.getLogger(__name__)

        # Initialise clients for ALL providers that have keys (for fallback)
        self._anthropic = None
        self._deepseek = None
        self._openai = None
        self._groq = None

        if Config.ANTHROPIC_API_KEY:
            import anthropic

            self._anthropic = anthropic.AsyncAnthropic(api_key=Config.ANTHROPIC_API_KEY)
            self._anthropic_model = Config.AI_MODEL

        if Config.DEEPSEEK_API_KEY:
            from openai import AsyncOpenAI

            self._deepseek = AsyncOpenAI(
                api_key=Config.DEEPSEEK_API_KEY,
                base_url=_DEEPSEEK_BASE_URL,
            )
            self._deepseek_model = Config.DEEPSEEK_MODEL

        if Config.OPENAI_API_KEY:
            from openai import AsyncOpenAI

            self._openai = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
            self._openai_model = Config.OPENAI_MODEL

        if getattr(Config, "GROQ_API_KEY", ""):
            from openai import AsyncOpenAI

            self._groq = AsyncOpenAI(
                api_key=Config.GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1",
            )
            self._groq_model = getattr(Config, "GROQ_MODEL", "llama-3.3-70b-versatile")

        # Ordered list of available providers (primary first)
        _all = ["anthropic", "openai", "deepseek", "groq"]
        available = [p for p in _all if getattr(self, f"_{p}") is not None]
        # Put the configured primary provider first
        if self._provider in available:
            available.remove(self._provider)
            available.insert(0, self._provider)
        self._provider_order: List[str] = available

        if self._provider_order:
            self.logger.info(
                "AIAnalyzer → %s (fallback: %s)",
                self._provider_order[0],
                self._provider_order[1:] or "none",
            )
        else:
            self.logger.warning(
                "AIAnalyzer: нет ключей AI — бот использует локальные стратегии"
            )

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

    def _build_prompt(self, snapshots: List[Dict[str, Any]], balance: float) -> str:
        """
        Строит промпт для AI API на основе снэпшотов рынка.

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

    def _parse_response(self, raw: str) -> List[Dict[str, Any]]:
        """Парсит JSON-ответ AI, удаляет markdown-блоки если есть."""
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.split("```")[0]
        recs = json.loads(raw.strip())
        if not isinstance(recs, list):
            recs = [recs]
        required = ("symbol", "action", "strategy", "confidence")
        valid = []
        for r in recs:
            if not isinstance(r, dict):
                continue
            if not all(k in r for k in required):
                continue
            if r.get("action") not in ("buy", "sell", "hold"):
                continue
            try:
                conf = float(r["confidence"])
            except (TypeError, ValueError):
                continue
            if not (0.0 <= conf <= 1.0):
                continue
            if conf < Config.MIN_SIGNAL_CONFIDENCE:
                continue
            r["confidence"] = conf
            valid.append(r)
        return valid

    async def _call_anthropic(self, prompt: str) -> str:
        """Вызов Claude (Anthropic API)."""
        message = await self._anthropic.messages.create(
            model=self._anthropic_model,
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        block = message.content[0]
        return block.text if hasattr(block, "text") else ""

    async def _call_deepseek(self, prompt: str) -> str:
        """Вызов DeepSeek (OpenAI-совместимый API)."""
        response = await self._deepseek.chat.completions.create(
            model=self._deepseek_model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def _call_openai(self, prompt: str) -> str:
        """Вызов ChatGPT (OpenAI API)."""
        response = await self._openai.chat.completions.create(
            model=self._openai_model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def _call_groq(self, prompt: str) -> str:
        """Вызов Groq (Llama 3.3 70B через OpenAI-совместимый API)."""
        response = await self._groq.chat.completions.create(
            model=self._groq_model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def analyze(
        self,
        snapshots: List[Dict[str, Any]],
        balance: float,
    ) -> List[Dict[str, Any]]:
        """
        Отправляет снэпшоты в AI (Claude или DeepSeek), возвращает рекомендации.

        При отсутствии ключа или ошибке — пустой список,
        trading_bot автоматически переключается на локальные стратегии.

        :param snapshots: Список снэпшотов из MarketScanner.
        :param balance: Баланс пользователя в USDT.
        :return: Отфильтрованный список рекомендаций.
        """
        if not self.enabled or not snapshots:
            return []

        prompt = self._build_prompt(snapshots, balance)

        # Если провайдер переключён из Telegram — ставим его первым
        provider_order = list(self._provider_order)
        _rc = getattr(self, "_runtime_config", None)
        if _rc is not None:
            try:
                rt_provider = self._runtime_config.get_ai_provider()
                if (
                    rt_provider != "auto"
                    and rt_provider in provider_order
                    and provider_order[0] != rt_provider
                ):
                    provider_order.remove(rt_provider)
                    provider_order.insert(0, rt_provider)
            except Exception:
                pass

        for provider in provider_order:
            try:
                if provider == "anthropic":
                    raw = await self._call_anthropic(prompt)
                elif provider == "deepseek":
                    raw = await self._call_deepseek(prompt)
                elif provider == "groq":
                    raw = await self._call_groq(prompt)
                else:
                    raw = await self._call_openai(prompt)

                valid = self._parse_response(raw)
                self.logger.info(
                    "AI [%s]: %d signals from %d symbols",
                    provider,
                    len(valid),
                    len(snapshots),
                )
                return valid

            except json.JSONDecodeError as e:
                self.logger.warning(
                    "AI [%s] JSON parse error — trying next: %s", provider, e
                )
                continue
            except Exception as e:
                status = getattr(e, "status_code", None)
                if status == 400:
                    self.logger.error("AI [%s] bad request (400): %s", provider, e)
                    return []
                # Любая другая ошибка (429, 402, 500, таймаут) — пробуем следующего
                self.logger.warning(
                    "AI [%s] error (HTTP %s) — trying next provider: %s",
                    provider,
                    status or "?",
                    e,
                )
                continue

        self.logger.error("All AI providers exhausted")
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
