"""Приложение дашборда на FastAPI — импортируй `app` для монтирования
или запуска через uvicorn."""

from __future__ import annotations

import hmac
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.dashboard.routers import api as _api
from src.dashboard.routers import frontend as _frontend
from src.dashboard.routers import ops as _ops

app = FastAPI(title="BitbotBY Dashboard", docs_url=None)

_NO_AUTH_PATHS = {"/health", "/metrics"}


@app.middleware("http")
async def _api_key_middleware(request: Request, call_next):
    required = os.getenv("DASHBOARD_API_KEY", "")
    if required and request.url.path not in _NO_AUTH_PATHS:
        provided = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(provided.encode(), required.encode()):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return await call_next(request)


app.include_router(_api.router)
app.include_router(_ops.router)
app.include_router(_frontend.router)
