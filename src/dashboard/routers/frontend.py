"""Frontend route: serves the single-page HTML dashboard."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

_HTML = (Path(__file__).parent.parent / "index.html").read_text(encoding="utf-8")


@router.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    return HTMLResponse(_HTML)
