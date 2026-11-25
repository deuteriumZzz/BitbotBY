"""Каталог AI-провайдеров для торгового анализа: адреса, ключи, модели, цепочка.

Почти все провайдеры совместимы с API OpenAI — бот зовёт их одной функцией
(backend/bitbot/ai/analyzer.py). Новый провайдер — новая запись в PROVIDERS: его ключ
появится в приложении (backend/bitbot/config/secrets.py), а сам он — в цепочке, в разделе
«AI» приложения и в меню /provider в Telegram.

Модуль без зависимостей от Config и Redis — чистые данные и правила цепочки.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Model:
    id: str
    free: bool
    # Крупная модель: держит JSON-формат и годится для торговых решений.
    # Мелкие (3–9B) чаще ломают формат — в цепочке они работают, но в UI
    # помечены, и рекомендованными не бывают.
    strong: bool = True
    recommended: bool = False


@dataclass(frozen=True)
class Provider:
    id: str
    name: str
    models: Tuple[Model, ...]
    base_url: str = ""  # "{}" — сюда подставляется значение url_env
    key_env: str = ""
    model_env: str = ""  # старая переменная выбора модели из окружения
    kind: str = "openai"  # openai | anthropic
    free_tier: bool = True  # у провайдера есть бесплатный доступ
    key_optional: bool = False  # работает и без ключа
    url_env: str = ""  # адрес или его часть задаёт пользователь
    url_label: str = ""
    note: str = ""
    signup: str = ""
    check_url: str = ""  # проверка ключа; по умолчанию {адрес}/models
    verified: str = ""  # дата сверки ID моделей живым запросом

    @property
    def default_model(self) -> str:
        return next((m.id for m in self.models if m.recommended), self.models[0].id)

    def model(self, model_id: str) -> Optional[Model]:
        return next((m for m in self.models if m.id == model_id), None)

    def is_free(self, model_id: str) -> bool:
        m = self.model(model_id)
        if m is not None:
            return m.free
        if self.id == "openrouter":
            return model_id.endswith(":free")
        return self.free_tier

    def url(self, env: Callable[[str], str]) -> str:
        """Адрес API с учётом настроек пользователя; "" — провайдер не настроен."""
        value = env(self.url_env).strip() if self.url_env else ""
        if "{}" in self.base_url:
            return self.base_url.format(value) if value else ""
        return (value or self.base_url).rstrip("/")


def _m(id_: str, free: bool = True, strong: bool = True, rec: bool = False) -> Model:
    return Model(id_, free, strong, rec)


# Порядок = порядок цепочки по умолчанию: щедрые бесплатные с сильными
# моделями впереди, слабые и маленькие пулы дальше, платные последними.
# ponytail: ID моделей меняются быстрее, чем стоит сверять руками. verified —
# сверено живым запросом к /models; у остальных — известные ID без сверки,
# в приложении есть кнопка «Загрузить список моделей» с ключом пользователя.
PROVIDERS: Tuple[Provider, ...] = (
    Provider(
        "groq",
        "Groq",
        (
            _m("openai/gpt-oss-120b", rec=True),
            _m("meta-llama/llama-4-maverick-17b-128e-instruct"),
            _m("openai/gpt-oss-20b"),
        ),
        "https://api.groq.com/openai/v1",
        "GROQ_API_KEY",
        "GROQ_MODEL",
        note="Бесплатно, самый быстрый ответ.",
        signup="https://console.groq.com/keys",
    ),
    Provider(
        "gemini",
        "Gemini",
        (
            _m("gemini-flash-latest", rec=True),
            _m("gemini-3.6-flash"),
            _m("gemini-3.5-flash-lite"),
            _m("gemini-2.5-flash"),
            _m("gemini-pro-latest", free=False),
        ),
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        note="Бесплатный дневной лимит Google AI Studio.",
        signup="https://aistudio.google.com/apikey",
        check_url="https://generativelanguage.googleapis.com/v1beta/openai/models",
        verified="2026-10-17",
    ),
    Provider(
        "deepseek",
        "DeepSeek",
        (_m("deepseek-flash", False, rec=True), _m("deepseek-v4-pro", False)),
        "https://api.deepseek.com",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_MODEL",
        free_tier=False,
        note="Самый дешёвый из платных.",
        signup="https://platform.deepseek.com/api_keys",
        verified="2026-10-17",
    ),
    Provider(
        "openai",
        "OpenAI",
        (
            _m("gpt-5-mini", False, rec=True),
            _m("gpt-5", False),
            _m("gpt-4o-mini", False),
        ),
        "https://api.openai.com/v1",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        free_tier=False,
        note="Модели ChatGPT, платно.",
        signup="https://platform.openai.com/api-keys",
    ),
    Provider(
        "anthropic",
        "Claude",
        (
            _m("claude-haiku-5-5", False, rec=True),
            _m("claude-sonnet-5-5", False),
            _m("claude-opus-5-5", False),
        ),
        key_env="ANTHROPIC_API_KEY",
        model_env="AI_MODEL",
        kind="anthropic",
        free_tier=False,
        note="Лучшее качество анализа, платно.",
        signup="https://console.anthropic.com/settings/keys",
        check_url="https://api.anthropic.com/v1/models",
    ),
)

BY_ID: Dict[str, Provider] = {p.id: p for p in PROVIDERS}

# Режимы цепочки: только бесплатные (по умолчанию — бот никогда сам не начнёт
# тратить деньги), бесплатные впереди платных, только платные.
# Пауза для провайдера после ошибки, секунд: лимит без срока в ответе,
# нет денег, ключ не принят, модели нет.
COOLDOWN_RATE_LIMIT = 7 * 60
COOLDOWN_NO_BALANCE = 6 * 3600
COOLDOWN_BAD_KEY = 3600
COOLDOWN_BAD_MODEL = 10 * 60


def clean_chain(raw: Any) -> Dict[str, Any]:
    """Настройки AI из приложения: только выбор модели у провайдера; ValueError — что не так."""
    if not isinstance(raw, dict):
        raise ValueError("Ожидается объект с настройками AI.")
    models = raw.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("models: объект «провайдер → модель».")
    for pid, model in models.items():
        if pid not in BY_ID:
            raise ValueError(f"Нет провайдера {pid}.")
        if not isinstance(model, str) or not 0 < len(model.strip()) <= 200:
            raise ValueError(f"Модель для {pid} — непустая строка.")
    return {"models": {pid: m.strip() for pid, m in models.items()}}


def chain(
    settings: Dict[str, Any],
    available: List[str],
    env_models: Optional[Dict[str, str]] = None,
    pinned: str = "auto",
) -> List[Tuple[str, str]]:
    """Порядок опроса: [(провайдер, модель)] — решает бот, а не пользователь.

    Провайдеры с ключом в порядке каталога: сначала бесплатные, потом платные (платный
    зовётся, только если бесплатные не ответили). Модель: выбранная в приложении («Профи»)
    → из окружения → рекомендованная каталогом. pinned (AI_PROVIDER в окружении) — первым.
    Прежние настройки цепочки (режим, порядок, «один провайдер», резерв) не учитываются.
    """
    models = {**(env_models or {}), **settings.get("models", {})}
    out = [
        (p.id, models.get(p.id) or p.default_model)
        for p in PROVIDERS
        if p.id in available
    ]
    out.sort(key=lambda c: not BY_ID[c[0]].is_free(c[1]))  # стабильно
    return [c for c in out if c[0] == pinned] + [c for c in out if c[0] != pinned]
