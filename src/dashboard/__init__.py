"""Приложение дашборда на FastAPI — импортируй `app` для монтирования
или запуска через uvicorn."""

from __future__ import annotations

import asyncio
import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.dashboard import auth as _auth
from src.dashboard import data as _data
from src.dashboard import watchdog as _watchdog
from src.dashboard.routers import api as _api
from src.dashboard.routers import control as _control
from src.dashboard.routers import frontend as _frontend
from src.dashboard.routers import ops as _ops


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    watchdog = asyncio.create_task(_watchdog.run())
    yield
    watchdog.cancel()


app = FastAPI(title="BitbotBY Dashboard", docs_url=None, lifespan=_lifespan)

# /webhook/alerts проверяет свой секрет (ALERTMANAGER_WEBHOOK_SECRET) сам.
_NO_AUTH_PATHS = {"/health", "/metrics", "/app", "/webhook/alerts"}
# Без ключа и без входа данные открыты — но только по адресу этого компьютера:
# страница чужого сайта через DNS-rebinding приходит со своим Host и не пройдёт.
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]"}
MAX_BODY_BYTES = 1024 * 1024
# Вход в приложение — до выдачи сессии; /app/ — статика фронтенда без данных,
# MiniApp и браузер загружают её без заголовков, а данные берут из /api с сессией.
_NO_AUTH_PREFIXES = ("/api/auth/", "/app/")

# Собранный фронтенд (cd webapp && npm run build). Нет сборки — нет и /app.
_WEBAPP_DIST = Path(__file__).resolve().parents[2] / "webapp" / "dist"


def _host_allowed(host_header: str) -> bool:
    extra = {
        h.strip().lower()
        for h in os.getenv("DASHBOARD_ALLOWED_HOSTS", "").split(",")
        if h.strip()
    }
    return _auth._host_only(host_header) in (_LOCAL_HOSTS | extra)


@app.middleware("http")
async def _api_key_middleware(request: Request, call_next):
    required = os.getenv("DASHBOARD_API_KEY", "")
    path = request.url.path
    exempt = path in _NO_AUTH_PATHS or path.startswith(_NO_AUTH_PREFIXES)
    if required and not exempt:
        provided = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(provided.encode(), required.encode()) and not (
            _auth.check_session(_data.get_redis(), _auth.bearer_token(request.headers))
        ):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
    elif (
        not required
        and not exempt
        and not _host_allowed(request.headers.get("host", ""))
        and not _auth.telegram_auth_configured()
    ):
        return JSONResponse({"error": "Forbidden host"}, status_code=403)
    return await call_next(request)


class _BodyLimit:
    """Тело запроса не больше MAX_BODY_BYTES: /api/auth/* открыты до входа."""

    def __init__(self, app) -> None:
        self.inner = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            return await self.inner(scope, receive, send)
        length = dict(scope.get("headers") or []).get(b"content-length")
        if length is not None and (not length.isdigit() or int(length) > MAX_BODY_BYTES):
            resp = JSONResponse({"error": "Payload too large"}, status_code=413)
            return await resp(scope, receive, send)
        seen = 0

        async def limited_receive():
            nonlocal seen
            msg = await receive()
            seen += len(msg.get("body", b"") or b"")
            if seen > MAX_BODY_BYTES:  # chunked без Content-Length
                raise ValueError("request body too large")
            return msg

        return await self.inner(scope, limited_receive, send)


# Десктоп (Tauri) открывает приложение со своего origin и ходит в /api с
# заголовком Authorization — браузерному движку нужен CORS. Добавлен после
# проверки ключа, поэтому стоит снаружи: отвечает на preflight (он приходит без
# ключа) и ставит заголовки и на ответы 401. Ключ или сессия по-прежнему нужны.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "tauri://localhost",  # macOS, Linux
        "http://tauri.localhost",  # Windows
        "https://tauri.localhost",
    ],
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-Client", "X-API-Key"],
    max_age=600,
)
app.add_middleware(_BodyLimit)

app.include_router(_api.router)
app.include_router(_control.router)
app.include_router(_ops.router)
app.include_router(_frontend.router)
if (_WEBAPP_DIST / "index.html").exists():
    app.mount("/app", StaticFiles(directory=_WEBAPP_DIST, html=True), name="webapp")
