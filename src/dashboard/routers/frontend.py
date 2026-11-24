"""Маршрут фронтенда: корень ведёт в приложение (/app/).

Docker собирает приложение сам (стадия webapp в Dockerfile); при запуске без
Docker вместо него отдаётся короткая подсказка, как его собрать.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse, Response

router = APIRouter()

_WEBAPP_INDEX = Path(__file__).resolve().parents[3] / "webapp" / "dist" / "index.html"
_NOT_BUILT = (
    "<!doctype html><meta charset='utf-8'><title>BitbotBY</title>"
    "<p>Приложение не собрано. Выполните: "
    "<code>cd webapp &amp;&amp; npm ci &amp;&amp; npm run build</code>, "
    "затем перезапустите панель.</p>"
)


@router.get("/", response_class=HTMLResponse)
async def dashboard() -> Response:
    if _WEBAPP_INDEX.exists():
        return RedirectResponse("/app/", status_code=302)
    return HTMLResponse(_NOT_BUILT, status_code=503)
