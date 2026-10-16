"""
Зашифрованное хранилище ключей API, которые пользователь вводит в приложении.

- Файл `data/secrets.enc.json`, каждое значение зашифровано Fernet.
  Мастер-ключ — env `BITBOT_SECRETS_KEY` (вне файла и вне Redis); без него
  хранилище выключено. Сгенерировать:
  python -c "from cryptography.fernet import Fernet; \
  print(Fernet.generate_key().decode())"
- Наружу (API) значения не отдаются никогда: только «задан» и последние 4
  символа, а у паролей и PIN — даже их нет.
- Бот при старте подставляет сохранённые значения в Config и os.environ
  (`apply_secrets`): ключ из приложения важнее ключа из .env.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src import ai_providers

logger = logging.getLogger(__name__)
audit = logging.getLogger("audit")

# httpx пишет полный URL запроса на INFO, а у Telegram токен — часть URL.
logging.getLogger("httpx").setLevel(logging.WARNING)

DEFAULT_PATH = os.path.join("data", "secrets.enc.json")
MAX_VALUE_LEN = 4096


@dataclass(frozen=True)
class Field:
    env: str
    label: str
    secret: bool = True
    optional: bool = False
    min_len: int = 16
    show_last4: bool = True


@dataclass(frozen=True)
class Service:
    id: str
    name: str
    group: str
    fields: Tuple[Field, ...]
    model_env: Optional[str] = None
    validator: Optional[str] = None


def _key(env: str, label: str = "API Key", **kw: Any) -> Field:
    return Field(env, label, **kw)


def _ai_service(p: ai_providers.Provider) -> Service:
    """Карточка ключа AI-провайдера — из каталога src/ai_providers.py.

    Модель выбирается в разделе «AI» приложения, поэтому поля модели здесь нет.
    """
    fields: List[Field] = []
    if p.key_env:
        fields.append(_key(p.key_env, optional=p.key_optional))
    if p.url_env:
        fields.append(
            Field(
                p.url_env,
                p.url_label,
                secret=False,
                optional=p.key_optional,  # у Ollama есть адрес по умолчанию
                min_len=8,
            )
        )
    return Service(p.id, p.name, "AI-модели", tuple(fields), validator=p.id)


_AI_SERVICES: Tuple[Service, ...] = tuple(
    _ai_service(p) for p in ai_providers.PROVIDERS
)


SERVICES: Tuple[Service, ...] = (
    Service(
        "bybit",
        "Bybit",
        "Биржа",
        (_key("BYBIT_API_KEY"), _key("BYBIT_API_SECRET", "API Secret")),
        validator="bybit",
    ),
    *_AI_SERVICES,
    Service(
        "telegram",
        "Telegram-бот",
        "Уведомления",
        (
            _key("TELEGRAM_BOT_TOKEN", "Токен бота", min_len=30),
            Field("TELEGRAM_CHAT_ID", "Ваш chat_id", secret=False, min_len=1),
        ),
        validator="telegram",
    ),
    Service(
        "news",
        "NewsAPI",
        "Данные и соцсети",
        (_key("NEWS_API_KEY"),),
        validator="newsapi",
    ),
    Service(
        "finnhub",
        "Finnhub",
        "Данные и соцсети",
        (_key("FINNHUB_API_KEY"),),
        validator="finnhub",
    ),
    Service("glassnode", "Glassnode", "Данные и соцсети", (_key("GLASSNODE_API_KEY"),)),
    Service(
        "reddit",
        "Reddit",
        "Данные и соцсети",
        (
            Field("REDDIT_CLIENT_ID", "Client ID", secret=False, min_len=1),
            _key("REDDIT_CLIENT_SECRET", "Client Secret", min_len=20),
            Field(
                "REDDIT_USER_AGENT",
                "User-Agent",
                secret=False,
                optional=True,
                min_len=1,
            ),
        ),
    ),
    Service(
        "twitter",
        "Twitter / X",
        "Данные и соцсети",
        (
            _key("TWITTER_BEARER_TOKEN", "Bearer Token", optional=True),
            _key("TWSCRAPE_ACCOUNTS", "Аккаунты twscrape", optional=True),
        ),
    ),
    Service(
        "telethon",
        "Памп-каналы Telegram",
        "Данные и соцсети",
        (
            Field("TELEGRAM_SENTIMENT_API_ID", "API ID", secret=False, min_len=1),
            _key("TELEGRAM_SENTIMENT_API_HASH", "API Hash", min_len=32),
            Field("TELEGRAM_SENTIMENT_PHONE", "Телефон", secret=False, min_len=8),
        ),
    ),
    Service(
        "pumpportal", "PumpPortal", "Данные и соцсети", (_key("PUMPPORTAL_API_KEY"),)
    ),
    Service(
        "server",
        "Пароли и PIN",
        "Сервер и доступ",
        (
            _key(
                "TRADING_MODE_PIN",
                "PIN для перехода в Live",
                optional=True,
                min_len=4,
                show_last4=False,
            ),
            _key(
                "REDIS_PASSWORD",
                "Пароль Redis",
                optional=True,
                min_len=12,
                show_last4=False,
            ),
            _key(
                "DASHBOARD_API_KEY",
                "API-ключ дашборда",
                optional=True,
                show_last4=False,
            ),
            _key(
                "ALERTMANAGER_WEBHOOK_SECRET",
                "Секрет вебхука Alertmanager",
                optional=True,
                show_last4=False,
            ),
        ),
    ),
)

BY_ID: Dict[str, Service] = {s.id: s for s in SERVICES}


class SecretsDisabled(RuntimeError):
    """BITBOT_SECRETS_KEY не задан — хранить ключи нечем."""


def _fernet() -> Any:
    key = os.getenv("BITBOT_SECRETS_KEY", "")
    if not key:
        raise SecretsDisabled
    from cryptography.fernet import Fernet  # noqa: PLC0415

    return Fernet(key.encode())


def enabled() -> bool:
    return bool(os.getenv("BITBOT_SECRETS_KEY"))


def _path() -> str:
    return os.getenv("SECRETS_PATH", DEFAULT_PATH)


def _load_raw() -> Dict[str, Any]:
    try:
        with open(_path(), encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return {"items": {}, "meta": {}}
    raw.setdefault("items", {})
    raw.setdefault("meta", {})
    return raw


def _save_raw(raw: Dict[str, Any]) -> None:
    path = _path()
    folder = os.path.dirname(path) or "."
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".secrets.")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_values() -> Dict[str, str]:
    """Расшифрованные значения {ENV: value}. Только для процесса бота и проверок."""
    f = _fernet()
    out: Dict[str, str] = {}
    for env, token in _load_raw()["items"].items():
        try:
            out[env] = f.decrypt(token.encode()).decode()
        except Exception:
            logger.error(
                "Secret %s cannot be decrypted — wrong BITBOT_SECRETS_KEY?", env
            )
    return out


def get_value(env: str) -> Optional[str]:
    """Значение из хранилища, иначе из окружения."""
    try:
        v = load_values().get(env)
    except SecretsDisabled:
        v = None
    return v or os.getenv(env) or None


def apply_secrets(config_cls: Any) -> List[str]:
    """При старте бота: значения из приложения → Config и os.environ."""
    try:
        values = load_values()
        models = _load_raw()["meta"].get("_models", {})
    except SecretsDisabled:
        return []
    except Exception as e:
        logger.error("Secrets store unreadable: %s", e)
        return []
    for env, value in {**values, **models}.items():
        os.environ[env] = value
        if hasattr(config_cls, env):
            setattr(config_cls, env, value)
    if values:
        logger.info("Applied %d API keys from the app", len(values))
    return sorted(values)


def public_view() -> List[Dict[str, Any]]:
    """Что можно показать в UI: без значений ключей."""
    raw = _load_raw()
    values = load_values()
    models = raw["meta"].get("_models", {})
    out = []
    for s in SERVICES:
        meta = raw["meta"].get(s.id, {})
        fields = []
        for fl in s.fields:
            v = values.get(fl.env)
            fields.append(
                {
                    "env": fl.env,
                    "label": fl.label,
                    "secret": fl.secret,
                    "optional": fl.optional,
                    "set": v is not None,
                    "last4": v[-4:] if v and fl.secret and fl.show_last4 else None,
                    "value": v if v is not None and not fl.secret else None,
                }
            )
        any_set = any(f["set"] for f in fields)
        out.append(
            {
                "service": s.id,
                "name": s.name,
                "group": s.group,
                "fields": fields,
                "model": (
                    {"env": s.model_env, "value": models.get(s.model_env)}
                    if s.model_env
                    else None
                ),
                "status": meta.get("status", "unchecked" if any_set else "unset"),
                "message": meta.get("message", ""),
                "checked_at": meta.get("checked_at"),
            }
        )
    return out


def check_input(service: Service, fields: Dict[str, Any]) -> Dict[str, str]:
    """Ошибки формы по полям (до сетевой проверки)."""
    errors: Dict[str, str] = {}
    by_env = {f.env: f for f in service.fields}
    for env, value in fields.items():
        fl = by_env.get(env)
        if fl is None:
            errors[env] = "Нет такого поля."
            continue
        if not isinstance(value, str) or not value.strip():
            errors[env] = "Пустое значение."
            continue
        value = value.strip()
        if len(value) > MAX_VALUE_LEN:
            errors[env] = "Слишком длинное значение."
        elif env in ("TELEGRAM_CHAT_ID", "TELEGRAM_SENTIMENT_API_ID"):
            if not value.lstrip("-").isdigit():
                errors[env] = f"{fl.label} — это число."
        elif env == "TELEGRAM_SENTIMENT_PHONE":
            if not (value.startswith("+") and value[1:].isdigit() and len(value) >= 9):
                errors[env] = "Телефон в формате +375291234567."
        elif len(value) < fl.min_len:
            errors[env] = f"«{fl.label}» слишком короткий — скопируйте его целиком."
    return errors


def save(
    service: Service,
    fields: Dict[str, str],
    model: Optional[str],
    status: str,
    message: str,
    source: str,
) -> None:
    f = _fernet()
    raw = _load_raw()
    for env, value in fields.items():
        raw["items"][env] = f.encrypt(value.strip().encode()).decode()
    if model and service.model_env:
        raw["meta"].setdefault("_models", {})[service.model_env] = model.strip()
    raw["meta"][service.id] = {
        "status": status,
        "message": message,
        "checked_at": int(time.time()),
    }
    _save_raw(raw)
    audit.info(
        "secret.saved service=%s fields=%s status=%s source=%s",
        service.id,
        ",".join(sorted(fields)),
        status,
        source,
    )


def delete(service: Service, source: str) -> None:
    raw = _load_raw()
    for fl in service.fields:
        raw["items"].pop(fl.env, None)
    raw["meta"].pop(service.id, None)
    _save_raw(raw)
    audit.info("secret.deleted service=%s source=%s", service.id, source)


# ── Проверка ключей у самих сервисов ──────────────────────────────────────────

Result = Tuple[str, str]  # (status: ok | warn | bad, message)
_NOT_SUPPORTED = "Сохранено. Проверка для этого сервиса пока не поддерживается."


def _http_result(code: int, ok_message: str) -> Result:
    if 200 <= code < 300:
        return "ok", ok_message
    if code in (401, 403):
        return "bad", f"Ключ не подходит ({code})."
    if code == 402:
        return "warn", "Ключ рабочий, но на счёте сервиса нет денег (402)."
    if code == 429:
        return "warn", "Ключ рабочий, но лимит запросов сейчас исчерпан (429)."
    return "warn", f"Сервис ответил {code} — ключ сохранён, проверим позже."


async def _check_http(client: Any, service_id: str, values: Dict[str, str]) -> Result:
    p = ai_providers.BY_ID.get(service_id)
    if p is not None:
        # Бесплатный запрос (список моделей или проверка токена) — лимит не тратит.
        key = values.get(p.key_env, "") if p.key_env else ""
        if p.kind == "anthropic":
            headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        else:
            headers = {"Authorization": f"Bearer {key}"} if key else {}
        url = p.check_url or p.url(lambda env: values.get(env, "")) + "/models"
        resp = await client.get(url, headers=headers)
        return _http_result(resp.status_code, "Ключ работает.")
    if service_id == "newsapi":
        resp = await client.get(
            "https://newsapi.org/v2/top-headlines/sources",
            headers={"X-Api-Key": values["NEWS_API_KEY"]},
        )
        return _http_result(resp.status_code, "Ключ работает.")
    if service_id == "finnhub":
        resp = await client.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": "AAPL"},
            headers={"X-Finnhub-Token": values["FINNHUB_API_KEY"]},
        )
        return _http_result(resp.status_code, "Ключ работает.")
    if service_id == "telegram":
        token = values["TELEGRAM_BOT_TOKEN"]
        resp = await client.get(f"https://api.telegram.org/bot{token}/getMe")
        if resp.status_code == 200:
            name = (resp.json().get("result") or {}).get("username", "")
            return "ok", f"Бот @{name} отвечает." if name else "Бот отвечает."
        return _http_result(resp.status_code, "")
    return "warn", _NOT_SUPPORTED


def bybit_permissions_result(info: Dict[str, Any]) -> Result:
    """Разбор ответа Bybit /v5/user/query-api."""
    perms = info.get("permissions") or {}
    if "Withdraw" in (perms.get("Wallet") or []):
        return (
            "bad",
            "У ключа включён вывод средств. Создайте ключ без права вывода — "
            "боту оно не нужно, а при утечке позволит украсть деньги.",
        )
    if str(info.get("readOnly")) == "1":
        return (
            "warn",
            "Ключ только для чтения: бот сможет смотреть баланс, но не торговать.",
        )
    ips = info.get("ips") or []
    note = "" if ips and ips != ["*"] else " Совет: привяжите ключ к IP сервера."
    return "ok", "Ключ работает: чтение ✓ торговля ✓ вывод средств выключен ✓." + note


async def _check_bybit(values: Dict[str, str], testnet: bool) -> Result:
    import ccxt.async_support as ccxt  # noqa: PLC0415

    ex = ccxt.bybit(
        {"apiKey": values["BYBIT_API_KEY"], "secret": values["BYBIT_API_SECRET"]}
    )
    if testnet:
        ex.set_sandbox_mode(True)
    try:
        resp = await ex.privateGetV5UserQueryApi()
        return bybit_permissions_result(resp.get("result") or {})
    except ccxt.AuthenticationError:
        return "bad", "Bybit не принял ключ: проверьте API Key и Secret."
    except ccxt.NetworkError:
        return "warn", "Bybit не ответил — ключ сохранён, проверим позже."
    finally:
        await ex.close()


async def validate(
    service: Service, values: Dict[str, str], client: Any = None, testnet: bool = False
) -> Result:
    """Проверка у сервиса. values — полный набор полей (новые + уже сохранённые)."""
    missing = [
        f.label for f in service.fields if not f.optional and f.env not in values
    ]
    if missing:
        return (
            "warn",
            f"Сохранено. Чтобы сервис заработал, добавьте: {', '.join(missing)}.",
        )
    if service.validator is None:
        return "warn", _NOT_SUPPORTED
    if service.validator == "bybit":
        return await _check_bybit(values, testnet)
    import httpx  # noqa: PLC0415

    own = client is None
    client = client or httpx.AsyncClient(timeout=8)
    try:
        return await _check_http(client, service.validator, values)
    except httpx.HTTPError:
        return "warn", "Сервис не ответил — ключ сохранён, проверим позже."
    finally:
        if own:
            await client.aclose()
