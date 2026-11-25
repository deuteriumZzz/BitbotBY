"""Значения настроек: приведение типа, переопределения из приложения, действующие значения, правка, история.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from bitbot.config.registry.model import OVERRIDES_KEY, Setting
from bitbot.config.registry.table import BY_KEY, SETTINGS

logger = logging.getLogger(__name__)


def coerce(s: Setting, raw: Any) -> Any:
    """Приводит значение к типу настройки. ValueError — с текстом для пользователя."""
    if s.type == "bool":
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str) and raw.lower() in ("true", "false", "1", "0"):
            return raw.lower() in ("true", "1")
        raise ValueError("Нужно true или false.")
    if s.type in ("int", "float", "pct"):
        if isinstance(raw, bool):
            raise ValueError("Нужно число.")
        try:
            num = float(raw)
        except (TypeError, ValueError):
            raise ValueError("Нужно число.") from None
        if s.type == "int":
            if num != int(num):
                raise ValueError("Нужно целое число.")
            num = int(num)
        if s.min is not None and num < s.min:
            raise ValueError(f"Не меньше {_fmt(s, s.min)}.")
        if s.max is not None and num > s.max:
            raise ValueError(f"Не больше {_fmt(s, s.max)}.")
        value: Any = num
    elif s.type == "choice":
        value = str(raw)
        if value not in {v for v, _ in s.options}:
            raise ValueError("Нет такого варианта.")
    else:
        value = str(raw).strip()
    if s.check:
        msg = s.check(value)
        if msg:
            raise ValueError(msg)
    return value


def _fmt(s: Setting, v: float) -> str:
    if s.type == "pct":
        return f"{v * 100:g}%"
    return f"{v:g}{(' ' + s.unit) if s.unit else ''}"


def effective_config(config_cls: Any, extra: Optional[Dict[str, Any]] = None) -> Any:
    """Экземпляр Config с действующими значениями класса (+ extra).

    Config — dataclass: значения по умолчанию его полей вычисляются при импорте,
    и `Config()` не видит того, что подставлено позже в атрибуты класса
    (переопределения из UI, ключи из хранилища, Paper/Live из Telegram).
    """
    cand = config_cls()
    if dataclasses.is_dataclass(cand):
        for f in dataclasses.fields(cand):
            setattr(cand, f.name, getattr(config_cls, f.name, getattr(cand, f.name)))
    for k, v in (extra or {}).items():
        setattr(cand, k, v)
    return cand


def load_overrides(r: Any) -> Dict[str, Any]:
    """Переопределения из UI (Redis-хэш). Битые записи пропускаются."""
    if r is None:
        return {}
    try:
        items = list((r.hgetall(OVERRIDES_KEY) or {}).items())
    except Exception as e:
        logger.debug("load_overrides failed: %s", e)
        return {}
    out: Dict[str, Any] = {}
    for k, v in items:
        k = k.decode() if isinstance(k, bytes) else str(k)
        v = v.decode() if isinstance(v, bytes) else v
        s = BY_KEY.get(k)
        if s is None or s.hot or s.readonly:
            continue
        try:
            out[k] = coerce(s, json.loads(v))
        except (ValueError, TypeError):
            logger.warning("Ignoring invalid config override %s=%r", k, v)
    return out


def _set_config(config_cls: Any, s: Setting, value: Any) -> None:
    # Только при старте. Часть настроек есть лишь в реестре (поля в Config нет) и
    # читается через os.getenv — в том числе дочерними процессами (обучение,
    # бэктест), которым окружение и передаёт значение. Режим бота на ходу
    # (профиль, рынок, paper/live) сюда не относится — см. runtime.child_env().
    os.environ[s.key] = str(value).lower() if isinstance(value, bool) else str(value)
    if hasattr(config_cls, s.config_attr):
        setattr(config_cls, s.config_attr, value)


def apply_overrides(config_cls: Any, r: Any) -> List[str]:
    """Применяет переопределения при старте бота, до Config.validate().

    Если с ними Config.validate() падает, а без них — нет, переопределения
    откатываются: неудачная настройка из UI не должна мешать боту стартовать.
    """
    overrides = load_overrides(r)
    if not overrides:
        return []
    saved = {
        k: (getattr(config_cls, BY_KEY[k].config_attr, None), os.environ.get(k))
        for k in overrides
    }
    for k, v in overrides.items():
        _set_config(config_cls, BY_KEY[k], v)
    try:
        effective_config(config_cls).validate()
    except ValueError as e:
        for k, (attr_val, env_val) in saved.items():
            if hasattr(config_cls, BY_KEY[k].config_attr):
                setattr(config_cls, BY_KEY[k].config_attr, attr_val)
            if env_val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = env_val
        try:
            effective_config(config_cls).validate()
        except ValueError:
            # Конфиг не проходит и без переопределений — дело не в них.
            for k, v in overrides.items():
                _set_config(config_cls, BY_KEY[k], v)
            return list(overrides)
        logger.error("UI config overrides rejected at startup (%s) — using .env", e)
        return []
    logger.info("Applied UI config overrides: %s", ", ".join(sorted(overrides)))
    return list(overrides)


def current_values(config_cls: Any, rc: Any, r: Any) -> Dict[str, Dict[str, Any]]:
    """{key: {value, source}}; source — live (hot) | ui | env | default."""
    overrides = load_overrides(r)
    out: Dict[str, Dict[str, Any]] = {}
    for s in SETTINGS:
        if s.hot and rc is not None:
            try:
                value = s.hot_get(rc)  # type: ignore[misc]
            except Exception:
                value = None
            if value is not None:
                # Действующее значение у бота (Redis или фолбэк на Config).
                out[s.key] = {"value": value, "source": "live"}
                continue
        if s.key in overrides:
            out[s.key] = {"value": overrides[s.key], "source": "ui"}
            continue
        if s.key in os.environ:
            try:
                out[s.key] = {"value": coerce(s, os.environ[s.key]), "source": "env"}
                continue
            except ValueError:
                pass
        value = getattr(config_cls, s.config_attr, s.default)
        out[s.key] = {"value": value, "source": "default"}
    return out


def validate_patch(
    config_cls: Any, patch: Dict[str, Any], current: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """(значения, ошибки по ключам). None в значении — вернуть к .env / умолчанию."""
    values: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    for k, raw in patch.items():
        s = BY_KEY.get(k)
        if s is None:
            errors[k] = "Нет такой настройки."
        elif s.readonly:
            errors[k] = "Меняется только в .env на сервере."
        elif raw is None:
            if s.hot:
                errors[k] = "Укажите значение."
            else:
                values[k] = None
        else:
            try:
                values[k] = coerce(s, raw)
            except ValueError as e:
                errors[k] = str(e)
    if errors:
        return values, errors

    merged = {k: v["value"] for k, v in current.items()}
    merged.update({k: v for k, v in values.items() if v is not None})
    if merged.get("LEVERAGE_MIN", 1) > merged.get("LEVERAGE_MAX", 20):
        errors["LEVERAGE_MIN"] = "Минимальное плечо больше максимального."
    if errors:
        return values, errors

    # Та же проверка, что бот делает при старте: Config.validate() на кандидате.
    # Наличие ключей (Bybit для Live, хотя бы один AI-ключ) — забота хранилища
    # ключей: у дашборда их в env может не быть, и тогда любой PATCH падал бы.
    from bitbot.config.secrets import SERVICES  # noqa: PLC0415

    secrets_set: Dict[str, Any] = {
        f.env: "configured-elsewhere"
        for svc in SERVICES
        for f in svc.fields
        if f.secret and hasattr(config_cls, f.env)
    }

    def problem(vals: Dict[str, Any]) -> Optional[str]:
        attrs = dict(secrets_set)
        attrs.update(
            {
                BY_KEY[k].config_attr: v
                for k, v in vals.items()
                if k in BY_KEY and hasattr(config_cls, BY_KEY[k].config_attr)
            }
        )
        try:
            effective_config(config_cls, attrs).validate()
        except ValueError as e:
            return str(e)
        return None

    # Отказываем только в том, что ломает сама правка. Если конфигурация не проходит
    # проверку и без неё (например, Hybrid без обученной SAC), иначе застревала бы
    # любая правка — даже тихих часов.
    after = problem(merged)
    if after and after != problem({k: v["value"] for k, v in current.items()}):
        errors["_config"] = after
    return values, errors


HISTORY_KEY = "bot:config_history"


HISTORY_MAX = 200


def record_history(
    r: Any, before: Dict[str, Dict[str, Any]], values: Dict[str, Any], by: str = "app"
) -> None:
    """Пишет в журнал, что и на что изменили из приложения.

    Значения настроек — не секреты (ключи API лежат отдельно и сюда не попадают).
    """
    changes = [
        {
            "key": k,
            "label": BY_KEY[k].label,
            "old": before.get(k, {}).get("value"),
            "new": v,
        }
        for k, v in values.items()
        if k in BY_KEY and before.get(k, {}).get("value") != v
    ]
    if not changes:
        return
    try:
        r.lpush(
            HISTORY_KEY,
            json.dumps({"ts": time.time(), "by": by, "changes": changes}),
        )
        r.ltrim(HISTORY_KEY, 0, HISTORY_MAX - 1)
    except Exception:
        logger.warning("config history not recorded", exc_info=True)


def load_history(r: Any, limit: int = 50) -> List[Dict[str, Any]]:
    """Последние изменения настроек, свежие сверху."""
    out: List[Dict[str, Any]] = []
    for raw in r.lrange(HISTORY_KEY, 0, max(0, limit - 1)) or []:
        try:
            out.append(json.loads(raw))
        except ValueError:
            continue
    return out


def apply_patch(rc: Any, r: Any, values: Dict[str, Any]) -> Dict[str, List[str]]:
    """Записывает уже проверенные значения. hot — сразу, остальные — в хэш."""
    applied_hot: List[str] = []
    pending: List[str] = []
    failed: List[str] = []
    for k, v in values.items():
        s = BY_KEY[k]
        if s.hot:
            ok = s.hot_set(rc, v)  # type: ignore[misc]
            (failed if ok is False else applied_hot).append(k)
        elif v is None:
            r.hdel(OVERRIDES_KEY, k)
            pending.append(k)
        else:
            r.hset(OVERRIDES_KEY, k, json.dumps(v))
            pending.append(k)
    return {"applied_hot": applied_hot, "pending_restart": pending, "failed": failed}
