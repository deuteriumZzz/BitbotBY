from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from config import Config
from src import ai_providers as catalog
from src.strategies import STRATEGY_REGISTRY, create_strategy, get_all_strategies

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert crypto trading analyst with deep knowledge of "
    "technical analysis, market sentiment, and risk management. "
    "You analyze multiple coins simultaneously and return structured "
    "JSON recommendations. Be precise, data-driven, and conservative "
    "with risk. Always respond with valid JSON only."
)


def _env(name: str) -> str:
    """Значение настройки: Config (туда же попадают ключи из приложения) или env."""
    if not name:
        return ""
    value = getattr(Config, name, None)
    if isinstance(value, str) and value:
        return value
    return os.getenv(name, "")


def _retry_after(message: str) -> Optional[float]:
    """Срок из текста ошибки 429: «retry in 1m30s», «try again in 12.5s»,
    «Retry after 10 seconds» (LLM7)."""
    m = re.search(
        r"(?:try again|retry) in (\d+)m(\d+(?:\.\d+)?)?s?|"
        r"(?:try again|retry) (?:in|after) (\d+(?:\.\d+)?) ?s",
        message,
        re.IGNORECASE,
    )
    if not m:
        return None
    if m.group(1):
        return int(m.group(1)) * 60 + float(m.group(2) or 0)
    return float(m.group(3))


def _classify(exc: Exception) -> Tuple[str, float]:
    """(статус провайдера, пауза в секундах) по ошибке вызова."""
    code = getattr(exc, "status_code", None)
    if code == 429:
        wait = _retry_after(str(exc))
        return "rate_limit", (wait + 5) if wait else catalog.COOLDOWN_RATE_LIMIT
    if code == 402:
        return "no_balance", catalog.COOLDOWN_NO_BALANCE
    if code in (401, 403):
        return "bad_key", catalog.COOLDOWN_BAD_KEY
    if code == 404:
        return "bad_model", catalog.COOLDOWN_BAD_MODEL
    if code == 400:
        return "bad_request", 0.0
    return "error", 0.0


class AIAnalyzer:
    """
    Анализирует рынок через цепочку AI-провайдеров (каталог — src/ai_providers.py).

    Порядок, режим (только бесплатные / сначала бесплатные / только платные),
    модели и включённые провайдеры задаются в приложении и действуют со
    следующего цикла. AI_PROVIDER (/provider в Telegram) ставит выбранного
    провайдера первым. Ошибка или лимит — сразу следующий провайдер; после
    лимита, отказа ключа, «нет денег» или неизвестной модели провайдер
    пропускается до конца паузы.

    Принимает снэпшоты нескольких монет, отправляет один batch-запрос,
    возвращает список торговых рекомендаций с reasoning на русском.
    """

    def __init__(
        self,
        runtime_config: "Any | None" = None,
        telegram: "Any | None" = None,
    ) -> None:
        self._runtime_config = runtime_config
        self._telegram = telegram
        self.strategies = get_all_strategies()
        self.logger = logging.getLogger(__name__)
        # Кулдаун уведомлений: не спамим одним и тем же сообщением
        self._notified_at: Dict[str, float] = {}
        self._NOTIFY_COOLDOWN = 3600.0  # раз в час максимум
        # Пауза провайдера после лимита или ошибки ключа — до этого времени пропуск
        self._provider_retry_after: Dict[str, float] = {}
        self._clients: Dict[str, Any] = {}
        # События для приложения (ai.status); TradingBot подставляет свой emit
        self._emit: Optional[Callable[[str, Dict[str, Any]], None]] = None

        self._provider_order: List[str] = [pid for pid, _ in self._chain()]
        self.enabled = bool(self._provider_order)
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

    # ── Цепочка провайдеров ───────────────────────────────────────────────────

    def _settings(self) -> Dict[str, Any]:
        rc = self._runtime_config
        if rc is None:
            return {}
        try:
            return dict(rc.get_ai_chain())
        except Exception:
            return {}

    def _chain(self) -> List[Tuple[str, str]]:
        """[(провайдер, модель)] по текущим настройкам — пересчёт каждый цикл."""
        settings = self._settings()
        keyless = set(settings.get("keyless", []))
        available = [
            p.id
            for p in catalog.PROVIDERS
            if (_env(p.key_env) or (p.key_optional and p.id in keyless))
            and (p.kind != "openai" or p.url(_env))
        ]
        env_models = {
            p.id: _env(p.model_env)
            for p in catalog.PROVIDERS
            if p.model_env and _env(p.model_env)
        }
        pinned = "auto"
        if self._runtime_config is not None:
            try:
                pinned = self._runtime_config.get_ai_provider()
            except Exception:
                pass
        elif isinstance(getattr(Config, "AI_PROVIDER", None), str):
            pinned = Config.AI_PROVIDER
        return catalog.chain(settings, available, env_models, pinned)

    def _client(self, provider: str) -> Any:
        """Клиент провайдера — создаётся при первом обращении."""
        if provider not in self._clients:
            p = catalog.BY_ID[provider]
            key = _env(p.key_env)
            if p.kind == "anthropic":
                import anthropic

                self._clients[provider] = anthropic.AsyncAnthropic(
                    api_key=key, max_retries=0, timeout=60.0
                )
            else:
                from openai import AsyncOpenAI

                self._clients[provider] = AsyncOpenAI(
                    api_key=key
                    or "unused",  # LLM7 и Ollama без ключа — как в их документации
                    base_url=p.url(_env),
                    max_retries=0,  # без авто-ретраев — 429 сразу идёт в фолбэк
                    timeout=120.0 if provider == "ollama" else 30.0,
                )
        return self._clients[provider]

    def _build_strategy_list(self) -> str:
        """
        Формирует текстовый список доступных стратегий для промпта.

        Исключает стратегии, отключённые через Telegram
        (RuntimeConfig.get_disabled_strategies()) — без этого AI продолжал
        предлагать "отключенные" стратегии, потому что список строился
        один раз при инициализации и никогда не сверялся с Redis.

        :return: Многострочная строка с описанием каждой стратегии.
        """
        _rc = getattr(self, "_runtime_config", None)
        disabled = _rc.get_disabled_strategies() if _rc is not None else set()
        lines = []
        for s in self.strategies:
            if s["name"] in disabled:
                continue
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
        slim = [
            {
                "symbol": s.get("symbol"),
                "price": s.get("price"),
                "atr": s.get("atr"),
                "changes": s.get("changes"),
                "volume_ratio": s.get("volume_ratio"),
                "indicators": s.get("indicators"),
                "levels": s.get("levels"),
                "news_sentiment": s.get("news_sentiment"),
                "top_headlines": (s.get("top_headlines") or [])[:2],
            }
            for s in snapshots
        ]
        data_block = json.dumps(slim, ensure_ascii=False)
        _rc = getattr(self, "_runtime_config", None)
        _market_type = _rc.get_market_type() if _rc is not None else Config.MARKET_TYPE
        _market_rule = (
            "MARKET TYPE: SPOT — no margin, no shorting. 'sell' is valid ONLY as "
            "an exit for an existing long, never as a new entry — do not "
            "recommend 'sell' for a symbol you're not already long on.\n"
            if _market_type == "spot"
            else "MARKET TYPE: FUTURES (linear perpetual) — both 'buy' (long) "
            "and 'sell' (short) are valid new entries.\n"
        )
        min_conf = (
            _rc.get_signal_confidence(Config.PAPER_TRADING)
            if _rc is not None
            else (
                Config.MIN_SIGNAL_CONFIDENCE_PAPER
                if Config.PAPER_TRADING
                else Config.MIN_SIGNAL_CONFIDENCE
            )
        )
        return (
            f"Analyze the following crypto market data and provide "
            f"trading recommendations.\n\n"
            f"USER BALANCE: ${balance:.2f} USDT\n\n"
            f"{_market_rule}\n"
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
            _rc2 = getattr(self, "_runtime_config", None)
            _min_conf = (
                _rc2.get_signal_confidence(Config.PAPER_TRADING)
                if _rc2 is not None
                else (
                    Config.MIN_SIGNAL_CONFIDENCE_PAPER
                    if Config.PAPER_TRADING
                    else Config.MIN_SIGNAL_CONFIDENCE
                )
            )
            if conf < _min_conf:
                continue
            r["confidence"] = conf
            valid.append(r)
        return valid

    def _record_usage(self, provider: str, total_tokens: "int | None") -> None:
        """Копит счётчик токенов за сутки для панели статусов в Telegram."""
        if not total_tokens:
            return
        rc = getattr(self, "_runtime_config", None)
        if rc is None:
            return
        try:
            rc.record_token_usage(provider, total_tokens)
        except Exception:
            pass

    async def _call(self, provider: str, model: str, prompt: str) -> str:
        """Один запрос к провайдеру; ответ — текст модели."""
        client = self._client(provider)
        if catalog.BY_ID[provider].kind == "anthropic":
            message = await client.messages.create(
                model=model,
                max_tokens=4096,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            usage = getattr(message, "usage", None)
            if usage is not None:
                self._record_usage(
                    provider,
                    getattr(usage, "input_tokens", 0)
                    + getattr(usage, "output_tokens", 0),
                )
            block = message.content[0]
            return block.text if hasattr(block, "text") else ""
        response = await client.chat.completions.create(
            model=model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        usage = getattr(response, "usage", None)
        self._record_usage(provider, getattr(usage, "total_tokens", None))
        return response.choices[0].message.content or ""

    def _mark(self, provider: str, model: str, status: str, until: float = 0.0) -> None:
        """Статус провайдера: в Redis (Telegram, приложение) и событием ai.status."""
        rc = self._runtime_config
        if rc is not None:
            try:
                rc.set_provider_status(provider, status)
                rc.set_provider_cooldown(provider, until)
                rc.record_ai_call(provider)
                if status == "ok":
                    rc.set_last_ai_provider(provider)
            except Exception:
                pass
        if self._emit is not None:
            try:
                self._emit(
                    "ai.status",
                    {
                        "provider": provider,
                        "model": model,
                        "status": status,
                        "until": until or None,
                    },
                )
            except Exception:
                pass

    async def refresh_deepseek_balance(self) -> None:
        """Запрашивает актуальный баланс DeepSeek и кэширует в RuntimeConfig.

        Вызывается периодически (раз в 4ч из _season_check_loop) — баланс
        pay-per-use меняется медленно, частый опрос не нужен.
        """
        key = _env("DEEPSEEK_API_KEY")
        rc = getattr(self, "_runtime_config", None)
        if not key or rc is None:
            return
        try:
            import aiohttp

            headers = {"Authorization": f"Bearer {key}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.deepseek.com/user/balance",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        return
                    data = await resp.json()
            infos = data.get("balance_infos") or []
            usd = next(
                (b for b in infos if b.get("currency") == "USD"),
                infos[0] if infos else None,
            )
            if usd:
                rc.set_deepseek_balance(float(usd.get("total_balance", 0.0)))
        except Exception as exc:
            self.logger.debug("refresh_deepseek_balance failed: %s", exc)

    async def analyze(
        self,
        snapshots: List[Dict[str, Any]],
        balance: float,
    ) -> List[Dict[str, Any]]:
        """
        Отправляет снэпшоты по цепочке провайдеров, возвращает рекомендации.

        Нет доступных провайдеров или все отказали — пустой список,
        trading_bot автоматически переключается на локальные стратегии.

        :param snapshots: Список снэпшотов из MarketScanner.
        :param balance: Баланс пользователя в USDT.
        :return: Отфильтрованный список рекомендаций.
        """
        if not snapshots:
            return []
        chain = self._chain()
        self.enabled = bool(chain)
        if not chain:
            return []

        prompt = self._build_prompt(snapshots, balance)
        now = time.time()
        queue = list(chain)
        while queue:
            provider, model = queue.pop(0)
            wait = self._provider_retry_after.get(provider, 0) - now
            if wait > 0:
                self.logger.debug(
                    "Provider %s on cooldown for %ds — skipping", provider, wait
                )
                continue
            try:
                raw = await self._call(provider, model, prompt)
                valid = self._parse_response(raw)
            except json.JSONDecodeError as e:
                self.logger.warning(
                    "AI [%s] JSON parse error — trying next: %s", provider, e
                )
                self._mark(provider, model, "bad_format")
                continue
            except Exception as e:
                status, pause = _classify(e)
                default = catalog.BY_ID[provider].default_model
                if status == "bad_model" and model != default:
                    # Модели из .env или приложения у провайдера больше нет —
                    # сразу пробуем рекомендованную, провайдер не на паузу.
                    queue.insert(0, (provider, default))
                    pause = 0.0
                self.logger.warning(
                    "AI [%s/%s] %s (HTTP %s) — trying next provider: %s",
                    provider,
                    model,
                    status,
                    getattr(e, "status_code", None) or "?",
                    e,
                )
                until = time.time() + pause if pause else 0.0
                if until:
                    self._provider_retry_after[provider] = until
                self._mark(provider, model, status, until)
                if status not in ("error", "bad_request"):
                    await self._notify_provider_failure(provider, status, model)
                continue
            self.logger.info(
                "AI [%s/%s]: %d signals from %d symbols",
                provider,
                model,
                len(valid),
                len(snapshots),
            )
            self._mark(provider, model, "ok")
            return valid

        self.logger.error("All AI providers exhausted")
        if self._runtime_config is not None:
            try:
                self._runtime_config.set_last_ai_provider("local")
            except Exception:
                pass
        await self._notify_all_exhausted()
        return []

    async def _notify_provider_failure(
        self, provider: str, status: str, model: str = ""
    ) -> None:
        """Уведомляет в Telegram о сбое провайдера — не чаще раза в час."""
        from datetime import datetime, timezone

        if getattr(self, "_telegram", None) is None:
            return
        key = f"fail:{provider}"
        now = time.time()
        _notified = getattr(self, "_notified_at", {})
        if now - _notified.get(key, 0) < getattr(self, "_NOTIFY_COOLDOWN", 3600.0):
            return
        _notified[key] = now

        # Обратный отсчёт до полуночи UTC — когда сбрасываются дневные лимиты
        utc_now = datetime.now(timezone.utc)
        secs_to_reset = (
            86400 - utc_now.hour * 3600 - utc_now.minute * 60 - utc_now.second
        )
        h, m = divmod(secs_to_reset // 60, 60)
        reason = {
            "rate_limit": f"лимит запросов исчерпан (429). Сброс через {h}ч {m}мин.",
            "no_balance": "на счёте нет денег (402).",
            "bad_key": "ключ не принят (401/403) — проверьте его: Настройки → AI.",
            "bad_model": f"модель `{model}` не найдена — выберите другую: "
            "Настройки → AI.",
        }.get(status)
        if reason is None:
            return
        name = catalog.BY_ID[provider].name if provider in catalog.BY_ID else provider
        try:
            await self._telegram.notify(
                f"⚠️ *{name}*: {reason}\nПереключаюсь на следующего провайдера."
            )
        except Exception:
            pass

    async def _notify_all_exhausted(self) -> None:
        """Уведомляет когда все провайдеры недоступны — не чаще раза в час."""
        from datetime import datetime, timezone

        if getattr(self, "_telegram", None) is None:
            return
        key = "fail:all"
        now = time.time()
        _notified = getattr(self, "_notified_at", {})
        if now - _notified.get(key, 0) < getattr(self, "_NOTIFY_COOLDOWN", 3600.0):
            return
        _notified[key] = now

        utc_now = datetime.now(timezone.utc)
        secs_to_reset = (
            86400 - utc_now.hour * 3600 - utc_now.minute * 60 - utc_now.second
        )
        h, m = divmod(secs_to_reset // 60, 60)
        try:
            await self._telegram.notify(
                f"🔴 *Все AI провайдеры недоступны.*\n"
                f"Работаю на локальных стратегиях — качество сигналов снижено.\n"
                f"Провайдеры восстановятся через {h}ч {m}мин (полночь UTC)."
            )
        except Exception:
            pass

    def recommend_strategy_local(
        self, data: pd.DataFrame, exclude: set | None = None
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Локальный выбор стратегии без AI (fallback) — реальный перебор реестра.

        Запускает generate_signal() каждой невыключенной стратегии из
        STRATEGY_REGISTRY на настоящих OHLCV-данных и берёт buy/sell-сигнал
        с максимальной confidence. Раньше выбор шёл по захардкоженной
        индикаторной эвристике на 7 фиксированных имён — 8 из 16
        зарегистрированных стратегий (macd_crossover, scalping, мем-профиль,
        profile-варианты volume_spike) были физически недостижимы в этом
        фолбэке, сколько бы их ни включали через Telegram-чекбоксы. Каждая
        стратегия сама защищена от нехватки данных (см. BaseStrategy._last
        default), поэтому лишний try/except здесь не нужен.

        :param data: DataFrame с OHLCV + индикаторами для одного символа.
        :param exclude: Дополнительные имена для исключения (например,
            трендовые стратегии в боковом режиме) — поверх Telegram-списка
            отключённых.
        :return: (strategy_name, confidence, signal_dict); ("hold", 0.0, {})
            если ни одна включённая стратегия не дала buy/sell.
        """
        _rc = getattr(self, "_runtime_config", None)
        disabled = _rc.get_disabled_strategies() if _rc is not None else set()
        disabled = disabled | (exclude or set())

        best_name = "hold"
        best_conf = 0.0
        best_sig: Dict[str, Any] = {}
        for name in STRATEGY_REGISTRY:
            if name in disabled:
                continue
            try:
                sig = create_strategy(name).generate_signal(data)
            except Exception:
                # Одна упавшая стратегия не должна ронять весь турнир —
                # раньше эта изоляция была в trading_bot.py (try/except
                # вокруг create_strategy().generate_signal()), но при
                # переходе на turnir здесь её не перенесли: без неё
                # исключение долетает до generate_signals() и обрывает
                # сигналы для ВСЕХ символов цикла, а не только для этого.
                logger.exception(
                    "recommend_strategy_local: %s упала на данных, пропуск", name
                )
                continue
            if sig.get("action") not in ("buy", "sell"):
                continue
            conf = sig.get("confidence", 0.0)
            if conf > best_conf:
                best_name, best_conf, best_sig = name, conf, sig
        return best_name, best_conf, best_sig
