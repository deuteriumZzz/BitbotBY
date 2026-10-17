"""Маршрут фронтенда: корень ведёт в приложение, старый дашборд остаётся на /legacy."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse, Response

router = APIRouter()

_HTML = (Path(__file__).parent.parent / "index.html").read_text(encoding="utf-8")
_WEBAPP_INDEX = Path(__file__).resolve().parents[3] / "webapp" / "dist" / "index.html"


@router.get("/", response_class=HTMLResponse)
async def dashboard() -> Response:
    """Новое приложение, если оно собрано; иначе старая одностраничная панель."""
    if _WEBAPP_INDEX.exists():
        return RedirectResponse("/app/", status_code=302)
    return HTMLResponse(_HTML)


@router.get("/legacy", response_class=HTMLResponse)
async def legacy_dashboard() -> HTMLResponse:
    return HTMLResponse(_HTML)
