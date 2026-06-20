"""REST API routes: /api/status, /api/trades, /api/backtest."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config import Config
from src.dashboard.data import check_healthcheck, get_stats, get_trades, redis_get

router = APIRouter()


@router.get("/api/status")
async def get_status() -> JSONResponse:
    portfolio = redis_get("portfolio_state") or {}
    alive = check_healthcheck()
    stats = get_stats()
    return JSONResponse(
        {
            "bot_running": alive,
            "mode": Config.MODE,
            "paper_trading": Config.PAPER_TRADING,
            "balance": portfolio.get("balance", Config.INITIAL_BALANCE),
            "positions": portfolio.get("positions", {}),
            "total_commissions": portfolio.get("total_commissions", 0),
            "timestamp": datetime.now().isoformat(),
            **stats,
        }
    )


@router.get("/api/trades")
async def get_trades_route() -> JSONResponse:
    return JSONResponse(get_trades(limit=100))


@router.get("/api/backtest")
async def get_backtest() -> JSONResponse:
    bt_path = Path("data/backtest_results.json")
    if not bt_path.exists():
        return JSONResponse({"results": [], "generated_at": None})
    try:
        data = json.loads(bt_path.read_text())
        raw_results = data.get("results", [])
        if not isinstance(raw_results, list):
            raise ValueError("results must be a list")
        safe_results = [
            {
                "strategy": str(r.get("strategy", "")),
                "win_rate": float(r.get("win_rate", 0)),
                "total_trades": int(r.get("total_trades", 0)),
                "expected_value": float(r.get("expected_value", 0)),
                "total_return_pct": float(r.get("total_return_pct", 0)),
                "max_drawdown_pct": float(r.get("max_drawdown_pct", 0)),
                "sharpe_ratio": float(r.get("sharpe_ratio", 0)),
            }
            for r in raw_results
            if isinstance(r, dict)
        ]
        return JSONResponse(
            {"results": safe_results, "generated_at": data.get("generated_at")}
        )
    except Exception:
        return JSONResponse({"results": []})
