"""API для Telegram MiniApp и Desktop: состояние, метрики, команды боту, WebSocket.

Контракт — docs/design/CONTRACT.md. Всё, что меняет бота (команды, настройки,
ключи), требует сессии из входа через Telegram (src/dashboard/auth.py) или
X-API-Key. Без настроенного входа управление выключено: дашборд без ключа
открыт любому, кто до него достучится, и PANIC не должен быть доступен так.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse
from starlette.requests import HTTPConnection

from config import Config
from src import ai_providers, bot_bus, secrets_store
from src import settings_registry as registry
from src.dashboard import auth, data
from src.runtime_config import RuntimeConfig


async def _owner_only(conn: HTTPConnection) -> None:
    """Данные бота видит только владелец, если вход через Telegram настроен.

    С DASHBOARD_API_KEY это уже проверил middleware. Без ключа и без входа
    дашборд открыт, как раньше (локальный запуск). /ws проверяет сессию сам.
    """
    path = conn.url.path
    if path.startswith("/api/auth/") or path == "/ws":
        return
    if os.getenv("DASHBOARD_API_KEY") or not auth.telegram_auth_configured():
        return
    if not auth.check_session(data.get_redis(), auth.bearer_token(conn.headers)):
        raise HTTPException(
            401,
            {"code": "unauthorized", "message": "Войдите через Telegram."},
        )


router = APIRouter(dependencies=[Depends(_owner_only)])

PERIODS: Dict[str, Optional[timedelta]] = {
    "24h": timedelta(days=1),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
    "all": None,
}
COMMAND_WAIT_SEC = 8.0
HEARTBEAT_SEC = 10.0
_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def _err(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": message}}, status_code=status
    )


def _redis_str(r: Any, key: str) -> Optional[str]:
    if r is None:
        return None
    try:
        return r.get(key)
    except Exception:
        return None


def build_state() -> Dict[str, Any]:
    portfolio = data.redis_get(data.portfolio_key()) or {}
    r = data.get_redis()
    positions = data.get_open_positions()
    try:
        max_pos = int(_redis_str(r, "bot:max_positions") or Config.MAX_POSITIONS)
    except ValueError:
        max_pos = Config.MAX_POSITIONS
    return {
        "bot": {
            "alive": data.check_healthcheck(),
            "paused": _redis_str(r, "bot:paused") == "1",
            "paper": data.paper_mode(),
            "mode": _redis_str(r, "bot:mode") or Config.MODE,
            "profile": _redis_str(r, "bot:market_profile"),
            "ai": _redis_str(r, "bot:last_ai_provider"),
        },
        "balance": portfolio.get("balance", Config.INITIAL_BALANCE),
        "slots": {"used": len(positions), "max": max_pos},
        "positions": positions,
        "timestamp": datetime.now().isoformat(),
    }


# ── Чтение ────────────────────────────────────────────────────────────────────


@router.get("/api/state")
async def get_state() -> JSONResponse:
    return JSONResponse(build_state())


@router.get("/api/positions")
async def get_positions() -> JSONResponse:
    return JSONResponse(data.get_open_positions())


@router.get("/api/metrics")
async def get_metrics(period: str = "30d") -> JSONResponse:
    if period not in PERIODS:
        return _err(400, "bad_period", "Период — один из: 24h, 7d, 30d, all.")
    delta = PERIODS[period]
    trades = data.get_closed_trades(datetime.now() - delta if delta else None)
    balance = float(
        (data.redis_get(data.portfolio_key()) or {}).get(
            "balance", Config.INITIAL_BALANCE
        )
    )
    # Баланс на начало периода = текущий минус результат периода.
    start = balance - sum(float(t["pnl"]) for t in trades)
    return JSONResponse({"period": period, **data.compute_metrics(trades, start)})


JOURNAL_LIMIT = 1000


@router.get("/api/journal")
async def get_journal(period: str = "30d") -> JSONResponse:
    if period not in PERIODS:
        return _err(400, "bad_period", "Период — один из: 24h, 7d, 30d, all.")
    delta = PERIODS[period]
    trades = data.get_journal(datetime.now() - delta if delta else None, JOURNAL_LIMIT)
    return JSONResponse({"period": period, "limit": JOURNAL_LIMIT, "trades": trades})


@router.get("/api/pnl/daily")
async def get_pnl_daily(days: int = 35) -> JSONResponse:
    days = max(1, min(days, 366))
    since = (datetime.now() - timedelta(days=days)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return JSONResponse(data.daily_pnl(data.get_closed_trades(since)))


@router.get("/api/strategies")
async def get_strategies() -> JSONResponse:
    from src.strategies import STRATEGY_REGISTRY  # noqa: PLC0415

    backtest: Dict[str, Dict[str, Any]] = {}
    bt_path = Path("data/backtest_results.json")
    if bt_path.exists():
        try:
            for row in json.loads(bt_path.read_text()).get("results", []):
                if isinstance(row, dict) and row.get("strategy"):
                    backtest[str(row["strategy"])] = {
                        "win_rate": float(row.get("win_rate", 0)),
                        "trades": int(row.get("total_trades", 0)),
                        "sharpe": float(row.get("sharpe_ratio", 0)),
                    }
        except (ValueError, TypeError):
            backtest = {}

    disabled: set = set()
    r = data.get_redis()
    if r is not None:
        try:
            disabled = set(r.smembers("bot:disabled_strategies") or [])
        except Exception:
            disabled = set()

    live: Dict[str, Dict[str, float]] = {}
    for t in data.get_closed_trades():
        s = live.setdefault(str(t["strategy"]), {"trades": 0, "wins": 0, "pnl": 0.0})
        s["trades"] += 1
        s["wins"] += 1 if float(t["pnl"]) > 0 else 0
        s["pnl"] += float(t["pnl"])

    out = []
    for name in STRATEGY_REGISTRY:
        lv = live.get(name)
        out.append(
            {
                "name": name,
                "enabled": name not in disabled,
                "backtest": backtest.get(name),
                "live": (
                    {
                        "trades": int(lv["trades"]),
                        "win_rate": round(lv["wins"] / lv["trades"], 4),
                        "ev": round(lv["pnl"] / lv["trades"], 2),
                    }
                    if lv
                    else None
                ),
            }
        )
    return JSONResponse(out)


# ── Команды боту ──────────────────────────────────────────────────────────────


async def _body(request: Request) -> Dict[str, Any]:
    try:
        body = await request.json()
    except Exception:
        return {}
    return body if isinstance(body, dict) else {}


def _control_denied(request: Request) -> Optional[JSONResponse]:
    """None — запрос может менять состояние бота; иначе готовый ответ-отказ.

    Пускаем по сессии из входа через Telegram или по X-API-Key. Если не
    настроено ни то, ни другое, управление выключено: дашборд без ключа
    открыт любому, кто до него достучится.
    """
    key = os.getenv("DASHBOARD_API_KEY", "")
    if key and hmac.compare_digest(
        request.headers.get("X-API-Key", "").encode(), key.encode()
    ):
        return None
    if auth.check_session(data.get_redis(), auth.bearer_token(request.headers)):
        return None
    if not key and not auth.telegram_auth_configured():
        return _err(
            403,
            "control_disabled",
            "Управление из приложения выключено: подключите Telegram-бота "
            "или задайте DASHBOARD_API_KEY на сервере.",
        )
    return _err(401, "unauthorized", "Войдите через Telegram, чтобы управлять ботом.")


def _source(request: Request) -> str:
    client = request.headers.get("X-Client", "app")
    return client if client in ("miniapp", "desktop", "web") else "app"


async def _command(
    request: Request, type_: str, args: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    denied = _control_denied(request)
    if denied is not None:
        return denied
    body = await _body(request)
    request_id = str(
        body.get("request_id") or request.headers.get("Idempotency-Key", "")
    )
    if not _REQUEST_ID.match(request_id):
        return _err(
            400,
            "request_id_required",
            "Нужен request_id — уникальный идентификатор команды (8–64 символа).",
        )
    r = data.get_redis()
    if r is None:
        return _err(503, "bot_unreachable", "Нет связи с ботом: Redis недоступен.")
    if not bot_bus.enqueue(r, type_, args, request_id, _source(request)):
        res = bot_bus.get_result(r, request_id)
        if res is None:
            return _err(409, "in_progress", "Эта команда уже выполняется.")
        return JSONResponse({"request_id": request_id, "status": "done", "result": res})

    deadline = time.monotonic() + COMMAND_WAIT_SEC
    while time.monotonic() < deadline:
        res = bot_bus.get_result(r, request_id)
        if res is not None:
            return JSONResponse(
                {"request_id": request_id, "status": "done", "result": res},
                status_code=200 if res.get("ok") else 422,
            )
        await asyncio.sleep(0.2)
    return JSONResponse({"request_id": request_id, "status": "queued"}, status_code=202)


@router.post("/api/bot/pause")
async def cmd_pause(request: Request) -> JSONResponse:
    return await _command(request, "pause")


@router.post("/api/bot/resume")
async def cmd_resume(request: Request) -> JSONResponse:
    return await _command(request, "resume")


@router.post("/api/bot/panic")
async def cmd_panic(request: Request) -> JSONResponse:
    return await _command(request, "panic")


@router.post("/api/bot/restart")
async def cmd_restart(request: Request) -> JSONResponse:
    return await _command(request, "restart")


@router.post("/api/positions/{trade_id}/close")
async def cmd_close_position(trade_id: str, request: Request) -> JSONResponse:
    body = await _body(request)
    return await _command(
        request,
        "close_position",
        {"trade_id": trade_id, "fraction": body.get("fraction", 1)},
    )


@router.patch("/api/positions/{trade_id}")
async def cmd_update_position(trade_id: str, request: Request) -> JSONResponse:
    body = await _body(request)
    args: Dict[str, Any] = {"trade_id": trade_id}
    args.update({k: body[k] for k in ("sl", "tp", "trailing") if k in body})
    if len(args) == 1:
        return _err(400, "bad_request", "Укажите sl, tp или trailing.")
    return await _command(request, "update_position", args)


@router.post("/api/signals/{signal_id}/confirm")
async def cmd_signal_confirm(signal_id: str, request: Request) -> JSONResponse:
    return await _command(
        request, "signal_answer", {"signal_id": signal_id, "confirm": True}
    )


@router.post("/api/signals/{signal_id}/reject")
async def cmd_signal_reject(signal_id: str, request: Request) -> JSONResponse:
    return await _command(
        request, "signal_answer", {"signal_id": signal_id, "confirm": False}
    )


@router.get("/api/signals/pending")
async def get_pending_signals() -> JSONResponse:
    r = data.get_redis()
    out = []
    if r is not None:
        try:
            for key in r.scan_iter("bot:pending_signal:*"):
                raw = r.get(key)
                if isinstance(raw, str) and raw:
                    out.append(json.loads(raw))
        except Exception:
            out = []
    return JSONResponse(out)


@router.post("/api/account/mode")
async def cmd_account_mode(request: Request) -> JSONResponse:
    body = await _body(request)
    if not isinstance(body.get("paper"), bool):
        return _err(400, "bad_request", "Укажите paper: true или false.")
    return await _command(
        request,
        "account_mode",
        {"paper": body["paper"], "pin": str(body.get("pin", ""))},
    )


@router.get("/api/commands/{request_id}")
async def get_command(request_id: str) -> JSONResponse:
    if not _REQUEST_ID.match(request_id):
        return _err(400, "bad_request_id", "Неверный request_id.")
    r = data.get_redis()
    res = bot_bus.get_result(r, request_id) if r is not None else None
    if res is None:
        return JSONResponse(
            {"request_id": request_id, "status": "queued"}, status_code=202
        )
    return JSONResponse({"request_id": request_id, "status": "done", "result": res})


# ── Настройки (реестр src/settings_registry.py) ───────────────────────────────


def _runtime_config(r: Any) -> Optional[RuntimeConfig]:
    # RuntimeConfig ждёт обёртку с .redis_client, как RedisClient бота.
    return RuntimeConfig(redis_client=SimpleNamespace(redis_client=r)) if r else None


@router.get("/api/config/schema")
async def get_config_schema() -> JSONResponse:
    return JSONResponse([s.public() for s in registry.SETTINGS])


@router.get("/api/config")
async def get_config() -> JSONResponse:
    r = data.get_redis()
    return JSONResponse(registry.current_values(Config, _runtime_config(r), r))


@router.patch("/api/config")
async def patch_config(request: Request) -> JSONResponse:
    denied = _control_denied(request)
    if denied is not None:
        return denied
    body = await _body(request)
    if not body:
        return _err(400, "bad_request", 'Передайте изменения: {"КЛЮЧ": значение}.')
    r = data.get_redis()
    if r is None:
        return _err(503, "bot_unreachable", "Нет связи с ботом: Redis недоступен.")
    rc = _runtime_config(r)
    values, errors = registry.validate_patch(
        Config, body, registry.current_values(Config, rc, r)
    )
    if errors:
        return JSONResponse(
            {
                "error": {
                    "code": "invalid",
                    "message": "Изменения не сохранены — исправьте ошибки.",
                    "fields": errors,
                }
            },
            status_code=422,
        )
    result = registry.apply_patch(rc, r, values)
    bot_bus.emit(r, "config.changed", {"keys": sorted(values)})
    return JSONResponse(result)


# ── Быстрые настройки бота (RuntimeConfig в Redis, бот читает каждый цикл) ────

_TICKER = re.compile(r"^[A-Z0-9]{2,15}(/USDT)?$")
_ALERT_LABELS = {
    "atr_spike": "Всплеск волатильности",
    "drawdown": "Просадка",
    "daily_limit": "Дневной лимит убытка",
    "bot_stalled": "Бот перестал работать",
    "new_signal": "Новые сигналы",
    "position_open": "Открытие позиции",
    "position_close": "Закрытие позиции",
}


def _rc_or_error(
    request: Request,
) -> "tuple[Any, Any, Optional[JSONResponse]]":
    denied = _control_denied(request)
    if denied is not None:
        return None, None, denied
    r = data.get_redis()
    if r is None:
        return (
            None,
            None,
            _err(503, "bot_unreachable", "Нет связи с ботом: Redis недоступен."),
        )
    return _runtime_config(r), r, None


@router.post("/api/profile")
async def set_profile(request: Request) -> JSONResponse:
    rc, r, denied = _rc_or_error(request)
    if denied is not None:
        return denied
    name = str((await _body(request)).get("name", ""))
    if not rc.apply_market_profile(name):
        return _err(400, "bad_profile", "Профиль — bluechip, altcoin или meme.")
    bot_bus.emit(r, "config.changed", {"keys": ["MARKET_PROFILE"], "profile": name})
    return JSONResponse({"profile": name})


@router.patch("/api/strategies/{name}")
async def set_strategy(name: str, request: Request) -> JSONResponse:
    from src.strategies import STRATEGY_REGISTRY  # noqa: PLC0415

    rc, r, denied = _rc_or_error(request)
    if denied is not None:
        return denied
    if name not in STRATEGY_REGISTRY:
        return _err(404, "unknown_strategy", "Нет такой стратегии.")
    enabled = (await _body(request)).get("enabled")
    if not isinstance(enabled, bool):
        return _err(400, "bad_request", "Укажите enabled: true или false.")
    if (name not in rc.get_disabled_strategies()) != enabled:
        rc.toggle_strategy(name)
    bot_bus.emit(r, "config.changed", {"keys": ["STRATEGIES"]})
    return JSONResponse({"name": name, "enabled": enabled})


@router.post("/api/strategies/reset")
async def reset_strategies(request: Request) -> JSONResponse:
    rc, r, denied = _rc_or_error(request)
    if denied is not None:
        return denied
    rc.reset_strategies()
    bot_bus.emit(r, "config.changed", {"keys": ["STRATEGIES"]})
    return JSONResponse({"ok": True})


@router.get("/api/symbols")
async def get_symbols() -> JSONResponse:
    rc = _runtime_config(data.get_redis())
    if rc is None:
        return JSONResponse({"forced": [], "excluded": []})
    return JSONResponse(
        {
            "forced": sorted(rc.get_forced_symbols()),
            "excluded": sorted(rc.get_excluded_symbols()),
        }
    )


@router.put("/api/symbols")
async def put_symbols(request: Request) -> JSONResponse:
    rc, r, denied = _rc_or_error(request)
    if denied is not None:
        return denied
    body = await _body(request)
    lists: Dict[str, set] = {}
    for key in ("forced", "excluded"):
        raw = body.get(key, [])
        if not isinstance(raw, list):
            return _err(400, "bad_request", f"{key} — список тикеров.")
        norm = set()
        for item in raw:
            t = str(item).strip().upper()
            if not _TICKER.match(t):
                return _err(400, "bad_ticker", f"Неверный тикер: {item}.")
            norm.add(t if "/" in t else f"{t}/USDT")
        lists[key] = norm
    if lists["forced"] & lists["excluded"]:
        return _err(400, "conflict", "Монета не может быть сразу в обоих списках.")
    for key, getter, add, remove in (
        (
            "forced",
            rc.get_forced_symbols,
            rc.add_forced_symbol,
            rc.remove_forced_symbol,
        ),
        (
            "excluded",
            rc.get_excluded_symbols,
            rc.add_excluded_symbol,
            rc.remove_excluded_symbol,
        ),
    ):
        current = getter()
        for sym in lists[key] - current:
            add(sym)
        for sym in current - lists[key]:
            remove(sym)
    bot_bus.emit(r, "config.changed", {"keys": ["SYMBOLS"]})
    return JSONResponse({k: sorted(v) for k, v in lists.items()})


@router.get("/api/alerts")
async def get_alerts() -> JSONResponse:
    rc = _runtime_config(data.get_redis())
    out = []
    for kind, label in _ALERT_LABELS.items():
        out.append(
            {
                "kind": kind,
                "label": label,
                "enabled": rc.get_alert_enabled(kind) if rc else True,
                "cooldown_min": rc.get_alert_cooldown_mins(kind) if rc else 0,
            }
        )
    return JSONResponse(out)


@router.patch("/api/alerts/{kind}")
async def set_alert(kind: str, request: Request) -> JSONResponse:
    rc, r, denied = _rc_or_error(request)
    if denied is not None:
        return denied
    if kind not in _ALERT_LABELS:
        return _err(404, "unknown_alert", "Нет такого уведомления.")
    body = await _body(request)
    if "enabled" in body:
        if not isinstance(body["enabled"], bool):
            return _err(400, "bad_request", "enabled — true или false.")
        rc.set_alert_enabled(kind, body["enabled"])
    if "cooldown_min" in body:
        try:
            minutes = int(body["cooldown_min"])
        except (TypeError, ValueError):
            minutes = -1
        if not 0 <= minutes <= 1440:
            return _err(400, "bad_cooldown", "Пауза — от 0 до 1440 минут.")
        rc.set_alert_cooldown_mins(kind, minutes)
    return JSONResponse(
        {
            "kind": kind,
            "enabled": rc.get_alert_enabled(kind),
            "cooldown_min": rc.get_alert_cooldown_mins(kind),
        }
    )


_TIMEFRAMES = {"1m", "3m", "5m", "15m", "1h", "4h", "1d"}


@router.get("/api/candles")
async def get_candles(symbol: str, tf: str = "15m", limit: int = 200) -> JSONResponse:
    """Свечи из кэша DataLoader (data/cache/{BASE}_USDT_{tf}.csv)."""
    base = symbol.upper().split("/")[0]
    if not re.fullmatch(r"[A-Z0-9]{2,15}", base) or tf not in _TIMEFRAMES:
        return _err(400, "bad_request", "Неверный символ или таймфрейм.")
    path = Path("data/cache") / f"{base}_USDT_{tf}.csv"
    if not path.exists():
        return JSONResponse([])
    limit = max(10, min(limit, 1000))
    rows = path.read_text().strip().splitlines()[1:][-limit:]
    out = []
    for line in rows:
        parts = line.split(",")
        try:
            ts, o, h, low, c = (float(x) for x in parts[1:6])
        except (ValueError, IndexError):
            continue
        out.append(
            {"time": int(ts // 1000), "open": o, "high": h, "low": low, "close": c}
        )
    return JSONResponse(out)


# ── AI-провайдеры (src/ai_providers.py) ───────────────────────────────────────


def _model_env(p: ai_providers.Provider) -> str:
    return os.getenv(p.model_env, "") if p.model_env else ""


def _secret(env: str) -> str:
    return (secrets_store.get_value(env) or "") if env else ""


def _ai_view(rc: Any) -> Dict[str, Any]:
    """Каталог, ключи, настройки цепочки и живые статусы — для раздела «AI».

    Значения ключей не отдаются — только есть ключ или нет.
    """
    settings = rc.get_ai_chain() if rc else {}
    pinned = rc.get_ai_provider() if rc else Config.AI_PROVIDER
    keyless = set(settings.get("keyless", []))
    models = settings.get("models", {})
    order = [p for p in settings.get("order", []) if p in ai_providers.BY_ID]
    order += [p.id for p in ai_providers.PROVIDERS if p.id not in order]
    now = time.time()
    available: List[str] = []
    providers = []
    for pid in order:
        p = ai_providers.BY_ID[pid]
        key_set = bool(_secret(p.key_env))
        url_ok = p.kind != "openai" or bool(p.url(_secret))
        usable = (key_set or (p.key_optional and pid in keyless)) and url_ok
        if usable:
            available.append(pid)
        model = models.get(pid) or _model_env(p) or p.default_model
        known = p.model(model)
        until = rc.get_provider_cooldown(pid) if rc else 0.0
        source = "app" if pid in models else "env" if _model_env(p) else "catalog"
        providers.append(
            {
                "id": pid,
                "name": p.name,
                "note": p.note,
                "signup": p.signup,
                "free_tier": p.free_tier,
                "key_optional": p.key_optional,
                "url_label": p.url_label if "{}" in p.base_url else "",
                "key_set": key_set,
                "url_ok": url_ok,
                "usable": usable,
                "keyless_on": pid in keyless,
                "disabled": pid in settings.get("disabled", []),
                "model": model,
                "model_source": source,
                "free": p.is_free(model),
                "strong": known.strong if known else True,
                "models": [
                    {
                        "id": m.id,
                        "free": m.free,
                        "strong": m.strong,
                        "recommended": m.recommended,
                    }
                    for m in p.models
                ],
                "verified": p.verified,
                "status": rc.get_provider_status(pid) if rc else "unknown",
                "cooldown_until": until if until > now else None,
                "calls_today": rc.get_ai_calls_today(pid) if rc else 0,
                "tokens_today": rc.get_token_usage_today(pid) if rc else 0,
            }
        )
    env_models = {p.id: _model_env(p) for p in ai_providers.PROVIDERS if _model_env(p)}
    chain = ai_providers.chain(settings, available, env_models, pinned)
    return {
        "mode": settings.get("mode", ai_providers.DEFAULT_MODE),
        "single": bool(settings.get("single", False)),
        "pinned": pinned,
        "last": rc.get_last_ai_provider() if rc else "",
        "providers": providers,
        "chain": [{"provider": pid, "model": m} for pid, m in chain],
    }


@router.get("/api/ai")
async def get_ai() -> JSONResponse:
    r = data.get_redis()
    return JSONResponse(_ai_view(_runtime_config(r) if r is not None else None))


@router.put("/api/ai/chain")
async def put_ai_chain(request: Request) -> JSONResponse:
    """Режим, порядок, включённые, модели, провайдеры без ключа — бот применит
    со следующего цикла."""
    rc, r, denied = _rc_or_error(request)
    if denied is not None:
        return denied
    try:
        rc.set_ai_chain(await _body(request))
    except ValueError as e:
        return _err(400, "bad_request", str(e))
    bot_bus.emit(r, "config.changed", {"keys": ["AI_CHAIN"]})
    return JSONResponse(_ai_view(rc))


@router.get("/api/ai/providers/{provider_id}/models")
async def get_ai_models(provider_id: str, request: Request) -> JSONResponse:
    """Живой список моделей провайдера по сохранённому ключу (лимит не тратит)."""
    denied = _control_denied(request)
    if denied is not None:
        return denied
    p = ai_providers.BY_ID.get(provider_id)
    if p is None:
        return _err(404, "unknown_provider", "Нет такого провайдера.")
    key = _secret(p.key_env)
    if not key and not p.key_optional:
        return _err(400, "no_key", "Сначала добавьте ключ.")
    if p.kind == "anthropic":
        url = "https://api.anthropic.com/v1/models"
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    else:
        base = p.url(_secret)
        if not base:
            return _err(400, "no_url", f"Укажите {p.url_label}.")
        url = base + "/models"
        headers = {"Authorization": f"Bearer {key}"} if key else {}
    import httpx  # noqa: PLC0415

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, headers=headers)
        payload = resp.json() if resp.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return _err(502, "provider_unreachable", f"{p.name} не ответил.")
    if payload is None:
        return _err(502, "provider_error", f"{p.name} ответил {resp.status_code}.")
    items = payload.get("data") if isinstance(payload, dict) else payload
    ids = sorted(
        {
            str(m["id"]).removeprefix("models/")
            for m in items or []
            if isinstance(m, dict) and m.get("id")
        }
    )
    return JSONResponse({"provider": p.id, "models": ids})


# ── Ключи API (src/secrets_store.py) ──────────────────────────────────────────


def _secrets_unavailable(request: Request) -> Optional[JSONResponse]:
    denied = _control_denied(request)
    if denied is not None:
        return denied
    if not secrets_store.enabled():
        return _err(
            503,
            "secrets_disabled",
            "Хранилище ключей выключено: задайте BITBOT_SECRETS_KEY на сервере.",
        )
    return None


def _secret_item(service_id: str) -> Dict[str, Any]:
    return next(i for i in secrets_store.public_view() if i["service"] == service_id)


@router.get("/api/secrets")
async def get_secrets(request: Request) -> JSONResponse:
    denied = _secrets_unavailable(request)
    if denied is not None:
        return denied
    return JSONResponse(secrets_store.public_view())


@router.put("/api/secrets/{service_id}")
async def put_secret(service_id: str, request: Request) -> JSONResponse:
    denied = _secrets_unavailable(request)
    if denied is not None:
        return denied
    service = secrets_store.BY_ID.get(service_id)
    if service is None:
        return _err(404, "unknown_service", "Нет такого сервиса.")
    body = await _body(request)
    fields = body.get("fields") or {}
    model = body.get("model")
    if not isinstance(fields, dict) or (not fields and not model):
        return _err(400, "bad_request", "Передайте fields: {ИМЯ: значение} или model.")
    if model is not None and (
        not service.model_env
        or not isinstance(model, str)
        or not model.strip()
        or len(model) > 200
    ):
        return _err(400, "bad_model", "Неверное имя модели.")
    errors = secrets_store.check_input(service, fields)
    if errors:
        return JSONResponse(
            {
                "error": {
                    "code": "invalid",
                    "message": "Проверьте поля.",
                    "fields": errors,
                }
            },
            status_code=422,
        )

    if fields:
        stored = secrets_store.load_values()
        values = {f.env: stored[f.env] for f in service.fields if f.env in stored}
        values.update({k: v.strip() for k, v in fields.items()})
        # Тестовая сеть Bybit: значение из приложения важнее .env.
        testnet = registry.load_overrides(data.get_redis()).get(
            "TESTNET", os.getenv("TESTNET", "").lower() == "true"
        )
        status, message = await secrets_store.validate(
            service, values, testnet=bool(testnet)
        )
        if status == "bad":
            return _err(422, "key_rejected", message)
    else:
        status, message = "ok", "Модель сохранена."

    secrets_store.save(service, fields, model, status, message, _source(request))
    bot_bus.emit(
        data.get_redis(), "secret.changed", {"service": service.id, "status": status}
    )
    # Бот читает ключи при старте — новый ключ заработает после перезапуска.
    return JSONResponse({**_secret_item(service.id), "restart_required": True})


@router.delete("/api/secrets/{service_id}")
async def delete_secret(service_id: str, request: Request) -> JSONResponse:
    denied = _secrets_unavailable(request)
    if denied is not None:
        return denied
    service = secrets_store.BY_ID.get(service_id)
    if service is None:
        return _err(404, "unknown_service", "Нет такого сервиса.")
    secrets_store.delete(service, _source(request))
    bot_bus.emit(
        data.get_redis(), "secret.changed", {"service": service.id, "status": "unset"}
    )
    return JSONResponse({**_secret_item(service.id), "restart_required": True})


# ── Вход: Telegram MiniApp и Desktop ──────────────────────────────────────────

_AUTH_ERRORS = {
    "bad_init_data": "Не удалось прочитать данные Telegram. "
    "Откройте приложение из бота.",
    "bad_signature": "Подпись Telegram не сходится. "
    "Откройте приложение из своего бота.",
    "expired": "Данные входа устарели. Закройте и снова откройте приложение.",
    "not_owner": "Это приложение управляет чужим ботом — вход только для владельца.",
}


def _session_response(r: Any, subject: str, source: str) -> JSONResponse:
    token = auth.create_session(r, subject, source)
    return JSONResponse({"token": token, "expires_in": auth.SESSION_TTL})


@router.post("/api/auth/telegram")
async def auth_telegram(request: Request) -> JSONResponse:
    token, owner = auth.owner_credentials()
    if not token or not owner:
        return _err(503, "auth_unconfigured", "На сервере не настроен Telegram-бот.")
    body = await _body(request)
    user, code = auth.verify_init_data(str(body.get("initData", "")), token, owner)
    if user is None:
        return _err(401, code, _AUTH_ERRORS[code])
    r = data.get_redis()
    if r is None:
        return _err(503, "bot_unreachable", "Нет связи с ботом: Redis недоступен.")
    return _session_response(r, str(user["id"]), "miniapp")


@router.post("/api/auth/desktop/request")
async def auth_desktop_request() -> JSONResponse:
    if not auth.telegram_auth_configured():
        return _err(503, "auth_unconfigured", "На сервере не настроен Telegram-бот.")
    r = data.get_redis()
    if r is None:
        return _err(503, "bot_unreachable", "Нет связи с ботом: Redis недоступен.")
    code = auth.issue_desktop_code(r)
    if code is None:
        return _err(
            429, "too_soon", "Код уже отправлен. Новый можно запросить через 30 секунд."
        )
    sent = await auth.send_owner(
        f"🔐 Код для входа в BitbotBY Desktop: {code}\n"
        "Действует 5 минут. Если вы его не запрашивали — просто проигнорируйте."
    )
    if not sent:
        return _err(502, "send_failed", "Не удалось отправить код в Telegram.")
    return JSONResponse({"sent": True, "expires_in": auth.CODE_TTL})


@router.post("/api/auth/desktop/verify")
async def auth_desktop_verify(request: Request) -> JSONResponse:
    r = data.get_redis()
    if r is None:
        return _err(503, "bot_unreachable", "Нет связи с ботом: Redis недоступен.")
    body = await _body(request)
    result = auth.check_desktop_code(r, str(body.get("code", "")))
    if result == "ok":
        _, owner = auth.owner_credentials()
        return _session_response(r, str(owner), "desktop")
    messages = {
        "expired": "Код истёк. Запросите новый.",
        "locked": "Слишком много неверных попыток. Запросите новый код.",
        "wrong": "Неверный код.",
    }
    return _err(401, f"code_{result}", messages[result])


@router.post("/api/auth/logout")
async def auth_logout(request: Request) -> JSONResponse:
    auth.drop_session(data.get_redis(), auth.bearer_token(request.headers))
    return JSONResponse({"ok": True})


# ── WebSocket ─────────────────────────────────────────────────────────────────


async def _ws_authorized(ws: WebSocket) -> bool:
    """Первое сообщение — {type:"auth", token}; токен не в URL, чтобы не попал в логи.

    token — сессия из входа через Telegram или DASHBOARD_API_KEY.
    """
    key = os.getenv("DASHBOARD_API_KEY", "")
    if not key and not auth.telegram_auth_configured():
        return True
    try:
        msg = await asyncio.wait_for(ws.receive_json(), timeout=5)
    except Exception:
        return False
    if not isinstance(msg, dict) or msg.get("type") != "auth":
        return False
    token = str(msg.get("token", ""))
    if key and hmac.compare_digest(token.encode(), key.encode()):
        return True
    return auth.check_session(data.get_redis(), token) is not None


@router.websocket("/ws")
async def ws_events(ws: WebSocket) -> None:
    await ws.accept()
    if not await _ws_authorized(ws):
        try:
            await ws.close(code=4401)
        except RuntimeError:
            pass  # клиент ушёл, не дождавшись проверки, — сокет уже закрыт
        return
    await ws.send_json({"type": "state", "ts": time.time(), "data": build_state()})

    import redis.asyncio as aioredis  # noqa: PLC0415

    ar = aioredis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        password=os.getenv("REDIS_PASSWORD") or None,
        decode_responses=True,
    )
    pubsub = ar.pubsub()

    async def pump() -> None:
        try:
            await pubsub.subscribe(bot_bus.EVENTS_CHANNEL)
        except Exception:
            await ws.send_json(
                {"type": "cycle", "ts": time.time(), "data": {"state": "offline"}}
            )
            return
        next_beat = 0.0
        while True:
            msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if msg and msg.get("type") == "message":
                await ws.send_text(msg["data"])
            if time.monotonic() >= next_beat:
                next_beat = time.monotonic() + HEARTBEAT_SEC
                await ws.send_json(
                    {
                        "type": "heartbeat",
                        "ts": time.time(),
                        "data": {"alive": data.check_healthcheck()},
                    }
                )

    async def drain() -> None:
        # Клиент ничего не обязан слать; ждём только отключения.
        while True:
            await ws.receive_text()

    tasks = [asyncio.create_task(pump()), asyncio.create_task(drain())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except WebSocketDisconnect:
        pass
    finally:
        for t in tasks:
            t.cancel()
        try:
            # aclose() есть в redis-py ≥ 5.0.1; у старых версий — close().
            await (getattr(pubsub, "aclose", None) or pubsub.close)()
            await (getattr(ar, "aclose", None) or ar.close)()
        except Exception:
            pass
