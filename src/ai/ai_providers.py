"""Каталог AI-провайдеров для торгового анализа: адреса, ключи, модели, цепочка.

Почти все провайдеры совместимы с API OpenAI — бот зовёт их одной функцией
(src/ai/ai_analyzer.py). Новый провайдер — новая запись в PROVIDERS: его ключ
появится в приложении (src/settings/secrets_store.py), а сам он — в цепочке, в разделе
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
    model_env: str = ""  # старая переменная выбора модели из .env
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
        "cerebras",
        "Cerebras",
        (
            _m("gpt-oss-120b", rec=True),
            _m("llama-3.3-70b"),
            _m("qwen-3-235b-a22b-instruct-2507"),
        ),
        "https://api.cerebras.ai/v1",
        "CEREBRAS_API_KEY",
        note="Бесплатно около миллиона токенов в день, очень быстро.",
        signup="https://cloud.cerebras.ai",
    ),
    Provider(
        "openrouter",
        "OpenRouter",
        (
            _m("nvidia/nemotron-3-super-120b-a12b:free", rec=True),
            _m("google/gemma-4-31b-it:free"),
            _m("nvidia/nemotron-3-ultra-550b-a55b:free"),
            _m("google/gemma-4-26b-a4b-it:free"),
        ),
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
        note="Бесплатные модели с суффиксом «:free», свой дневной лимит.",
        signup="https://openrouter.ai/keys",
        check_url="https://openrouter.ai/api/v1/auth/key",
        verified="2026-10-17",
    ),
    Provider(
        "nvidia",
        "NVIDIA NIM",
        (
            _m("deepseek-ai/deepseek-v4.1-flash", rec=True),
            _m("moonshotai/kimi-k2.6"),
            _m("z-ai/glm-5.3"),
            _m("nvidia/nemotron-3-super-120b-a12b"),
            _m("mistralai/mistral-large"),
        ),
        "https://integrate.api.nvidia.com/v1",
        "NVIDIA_API_KEY",
        "NVIDIA_MODEL",
        note="Бесплатные кредиты build.nvidia.com, много открытых моделей.",
        signup="https://build.nvidia.com",
        verified="2026-10-17",
    ),
    Provider(
        "mistral",
        "Mistral",
        (
            _m("mistral-small-latest", rec=True),
            _m("ministral-8b-latest", strong=False),
        ),
        "https://api.mistral.ai/v1",
        "MISTRAL_API_KEY",
        note="Бесплатный тариф Experiment с ограничением запросов в секунду.",
        signup="https://console.mistral.ai/api-keys",
    ),
    Provider(
        "github",
        "GitHub Models",
        (
            _m("openai/gpt-4.1-mini", rec=True),
            _m("openai/gpt-4.1"),
            _m("meta/Llama-4-Maverick-17B-128E-Instruct-FP8"),
            _m("deepseek/DeepSeek-V3-0324"),
        ),
        "https://models.github.ai/inference",
        "GITHUB_MODELS_TOKEN",
        note="Бесплатно с токеном GitHub (право models:read), дневной лимит.",
        signup="https://github.com/settings/personal-access-tokens",
    ),
    Provider(
        "cloudflare",
        "Cloudflare Workers AI",
        (
            _m("@cf/openai/gpt-oss-120b", rec=True),
            _m("@cf/google/gemma-4-26b-a4b-it"),
            _m("@cf/meta/llama-3.1-8b-instruct-fp8", strong=False),
        ),
        "https://api.cloudflare.com/client/v4/accounts/{}/ai/v1",
        "CLOUDFLARE_API_TOKEN",
        url_env="CLOUDFLARE_ACCOUNT_ID",
        url_label="Account ID",
        note="10 000 бесплатных «нейронов» в день. Нужны токен и Account ID.",
        signup="https://dash.cloudflare.com/profile/api-tokens",
        check_url="https://api.cloudflare.com/client/v4/user/tokens/verify",
    ),
    Provider(
        "huggingface",
        "Hugging Face",
        (
            _m("openai/gpt-oss-120b", rec=True),
            _m("deepseek-ai/DeepSeek-V4.1-Flash"),
            _m("zai-org/GLM-5.3-Flash"),
            _m("Qwen/Qwen3-235B-A22B-Instruct-2507"),
        ),
        "https://router.huggingface.co/v1",
        "HF_TOKEN",
        note="Небольшие бесплатные кредиты каждый месяц.",
        signup="https://huggingface.co/settings/tokens",
        verified="2026-10-17",
    ),
    Provider(
        "ollama_cloud",
        "Ollama Cloud",
        (
            _m("gpt-oss:120b", rec=True),
            _m("deepseek-v4-flash"),
            _m("gpt-oss:20b"),
        ),
        "https://ollama.com/v1",
        "OLLAMA_API_KEY",
        note="Модели на серверах Ollama, бесплатный лимит с аккаунтом.",
        signup="https://ollama.com/settings/keys",
    ),
    Provider(
        "cohere",
        "Cohere",
        (
            _m("command-a-03-2025", rec=True),
            _m("command-r7b-12-2024", strong=False),
        ),
        "https://api.cohere.ai/compatibility/v1",
        "COHERE_API_KEY",
        note="Пробный ключ: 1000 запросов в месяц.",
        signup="https://dashboard.cohere.com/api-keys",
        check_url="https://api.cohere.com/v1/models",
    ),
    Provider(
        "llm7",
        "LLM7",
        (
            _m("GLM-5.3-Flash", rec=True),
            _m("gpt-oss:20b"),
            _m("DeepSeek-V4.1-Flash"),
            _m("gemma4:31b"),
        ),
        "https://api.llm7.io/v1",
        "LLM7_API_KEY",
        key_optional=True,
        note="Работает без ключа (GLM-5.3-Flash, gpt-oss:20b); "
        "бесплатный токен открывает остальные модели.",
        signup="https://token.llm7.io",
        verified="2026-10-17",
    ),
    Provider(
        "ollama",
        "Ollama на компьютере",
        (
            _m("gpt-oss:20b", rec=True),
            _m("qwen3:8b", strong=False),
            _m("llama3.1", strong=False),
        ),
        "http://localhost:11434/v1",
        key_optional=True,
        url_env="OLLAMA_BASE_URL",
        url_label="Адрес Ollama",
        note="Модели на вашем компьютере: без лимитов, но нужна мощная машина. "
        "Модель сначала скачайте: ollama pull gpt-oss:20b.",
        signup="https://ollama.com/download",
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
MODES = ("free", "auto", "paid")
DEFAULT_MODE = "free"

# Пауза для провайдера после ошибки, секунд: лимит без срока в ответе,
# нет денег, ключ не принят, модели нет.
COOLDOWN_RATE_LIMIT = 7 * 60
COOLDOWN_NO_BALANCE = 6 * 3600
COOLDOWN_BAD_KEY = 3600
COOLDOWN_BAD_MODEL = 10 * 60


def clean_chain(raw: Any) -> Dict[str, Any]:
    """Проверяет настройки цепочки из приложения; ValueError — что не так."""
    if not isinstance(raw, dict):
        raise ValueError("Ожидается объект с настройками цепочки.")
    out: Dict[str, Any] = {}
    if "mode" in raw:
        if raw["mode"] not in MODES:
            raise ValueError("Режим — free, auto или paid.")
        out["mode"] = raw["mode"]
    if "single" in raw:
        if not isinstance(raw["single"], bool):
            raise ValueError("single — true или false.")
        out["single"] = raw["single"]
    # keyless — провайдеры без ключа (LLM7, своя Ollama), которые пользователь
    # включил сам: без его согласия данные не уходят новому сервису.
    for key in ("order", "disabled", "keyless"):
        if key in raw:
            items = raw[key]
            if not isinstance(items, list) or any(i not in BY_ID for i in items):
                raise ValueError(f"{key}: список известных провайдеров.")
            if key == "keyless" and any(not BY_ID[i].key_optional for i in items):
                raise ValueError("Без ключа работают только LLM7 и Ollama.")
            out[key] = list(dict.fromkeys(items))
    if "models" in raw:
        models = raw["models"]
        if not isinstance(models, dict):
            raise ValueError("models: объект «провайдер → модель».")
        for pid, model in models.items():
            if pid not in BY_ID:
                raise ValueError(f"Нет провайдера {pid}.")
            if not isinstance(model, str) or not 0 < len(model.strip()) <= 200:
                raise ValueError(f"Модель для {pid} — непустая строка.")
        out["models"] = {pid: m.strip() for pid, m in models.items()}
    return out


def chain(
    settings: Dict[str, Any],
    available: List[str],
    env_models: Optional[Dict[str, str]] = None,
    pinned: str = "auto",
) -> List[Tuple[str, str]]:
    """Порядок опроса: [(провайдер, модель)].

    available — провайдеры с ключом (или без ключа, если он не нужен).
    Модель: выбранная в приложении → из .env → рекомендованная каталогом.
    pinned (AI_PROVIDER, /provider в Telegram) идёт первым при любом режиме:
    это явный выбор пользователя.

    single — «один провайдер»: решения всегда принимает одна модель (закреплённая,
    иначе первая в цепочке), без запасных. Недоступна — цикл идёт без AI, на
    локальных стратегиях, а не на другой модели.
    """
    order = [p for p in settings.get("order", []) if p in BY_ID]
    ids = order + [p.id for p in PROVIDERS if p.id not in order]
    disabled = set(settings.get("disabled", []))
    models = {**(env_models or {}), **settings.get("models", {})}
    mode = settings.get("mode", DEFAULT_MODE)
    out: List[Tuple[str, str]] = []
    for pid in ids:
        if pid not in available or (pid in disabled and pid != pinned):
            continue
        model = models.get(pid) or BY_ID[pid].default_model
        free = BY_ID[pid].is_free(model)
        if pid != pinned and (
            (mode == "free" and not free) or (mode == "paid" and free)
        ):
            continue
        out.append((pid, model))
    if mode == "auto":
        out.sort(key=lambda c: not BY_ID[c[0]].is_free(c[1]))  # стабильно
    head = [c for c in out if c[0] == pinned]
    if settings.get("single"):
        return head if pinned != "auto" else out[:1]
    return head + [c for c in out if c[0] != pinned]
