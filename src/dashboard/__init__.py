"""Приложение дашборда на FastAPI — импортируй `app` для монтирования
или запуска через uvicorn."""

from __future__ import annotations

import hmac
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.dashboard import auth as _auth
from src.dashboard import data as _data
from src.dashboard.routers import api as _api
from src.dashboard.routers import control as _control
from src.dashboard.routers import frontend as _frontend
from src.dashboard.routers import ops as _ops

app = FastAPI(title="BitbotBY Dashboard", docs_url=None)

_NO_AUTH_PATHS = {"/health", "/metrics", "/app"}
# Вход в приложение — до выдачи сессии; /app/ — статика фронтенда без данных,
# MiniApp и браузер загружают её без заголовков, а данные берут из /api с сессией.
_NO_AUTH_PREFIXES = ("/api/auth/", "/app/")

# Собранный фронтенд (cd webapp && npm run build). Нет сборки — нет и /app.
_WEBAPP_DIST = Path(__file__).resolve().parents[2] / "webapp" / "dist"


@app.middleware("http")
async def _api_key_middleware(request: Request, call_next):
    required = os.getenv("DASHBOARD_API_KEY", "")
    path = request.url.path
    if (
        required
        and path not in _NO_AUTH_PATHS
        and not path.startswith(_NO_AUTH_PREFIXES)
    ):
        provided = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(provided.encode(), required.encode()) and not (
            _auth.check_session(_data.get_redis(), _auth.bearer_token(request.headers))
        ):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return await call_next(request)


app.include_router(_api.router)
app.include_router(_control.router)
app.include_router(_ops.router)
app.include_router(_frontend.router)
if (_WEBAPP_DIST / "index.html").exists():
    app.mount("/app", StaticFiles(directory=_WEBAPP_DIST, html=True), name="webapp")
