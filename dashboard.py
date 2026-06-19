"""
BitbotBY Web Dashboard — runs on port 8080.
Serves live bot status, signals, and trade history.
"""

import asyncio
import hmac
import json
import logging
import os
import sqlite3
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest

sys.path.insert(0, os.path.dirname(__file__))

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BitbotBY Dashboard", docs_url=None)

# C1: Optional API-key auth — set DASHBOARD_API_KEY env var to enable.
# /health and /metrics are exempt (Docker healthcheck & Prometheus scrape).
_NO_AUTH_PATHS = {"/health", "/metrics"}


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    required = os.getenv("DASHBOARD_API_KEY", "")
    if required and request.url.path not in _NO_AUTH_PATHS:
        provided = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(provided.encode(), required.encode()):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return await call_next(request)


# ── Prometheus gauges ─────────────────────────────────
_g_running = Gauge("bitbot_running", "1 если бот жив (healthcheck < 120s)")
_g_balance = Gauge("bitbot_balance_usdt", "Свободный баланс USDT")
_g_pnl = Gauge("bitbot_total_pnl_usdt", "Накопленный PnL в USDT")
_g_win_rate = Gauge("bitbot_win_rate_percent", "Win rate закрытых сделок")
_g_trades = Gauge("bitbot_total_trades", "Всего сделок в базе")

# ── Redis helper ──────────────────────────────────────


def _get_redis():
    try:
        import redis as redis_lib

        r = redis_lib.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD") or None,
            decode_responses=True,
        )
        r.ping()
        return r
    except Exception:
        return None


def _redis_get(key: str) -> Optional[dict]:
    r = _get_redis()
    if not r:
        return None
    try:
        raw = r.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None


# ── SQLite helper ─────────────────────────────────────

_DB_PATH = os.path.join("data", "trades.db")


def _get_trades(limit: int = 50) -> List[dict]:
    if not os.path.exists(_DB_PATH):
        return []
    try:
        conn = sqlite3.connect(_DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT symbol, strategy, action,
                   entry_price, exit_price,
                   quantity, pnl, pnl_pct,
                   confidence, commission,
                   entry_time, exit_time, status
            FROM trades
            ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"DB error: {e}")
        return []


def _get_stats() -> dict:
    if not os.path.exists(_DB_PATH):
        return {}
    try:
        conn = sqlite3.connect(_DB_PATH)
        row = conn.execute(
            """
            SELECT
              COUNT(*) as total,
              SUM(CASE WHEN status='closed' THEN 1 ELSE 0 END)
                  as closed,
              SUM(CASE WHEN status='closed' AND pnl>0
                  THEN 1 ELSE 0 END) as wins,
              SUM(CASE WHEN status='closed'
                  THEN pnl ELSE 0 END) as total_pnl,
              SUM(commission) as total_comm
            FROM trades
            """
        ).fetchone()
        conn.close()
        total, closed, wins, pnl, comm = row
        return {
            "total_trades": total or 0,
            "closed_trades": closed or 0,
            "win_rate": (round(wins / closed * 100, 1) if closed else 0),
            "total_pnl": round(pnl or 0, 2),
            "total_commissions": round(comm or 0, 2),
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return {}


def _check_healthcheck() -> bool:
    """Bot is alive if healthcheck.txt updated < 120s ago."""
    hc = Path("data/healthcheck.txt")
    if not hc.exists():
        return False
    try:
        content = hc.read_text().strip()
        if len(content) > 50:  # L2: guard against oversized file
            return False
        ts = datetime.fromisoformat(content)
        age = (datetime.now() - ts).total_seconds()
        return age < 120
    except Exception:
        return False


# ── API endpoints ─────────────────────────────────────


@app.get("/api/status")
async def get_status():
    portfolio = _redis_get("portfolio_state") or {}
    alive = _check_healthcheck()
    stats = _get_stats()
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


@app.get("/api/trades")
async def get_trades():
    return JSONResponse(_get_trades(limit=100))


@app.get("/api/backtest")
async def get_backtest():
    bt_path = Path("data/backtest_results.json")
    if not bt_path.exists():
        return JSONResponse({"results": [], "generated_at": None})
    try:
        data = json.loads(bt_path.read_text())
        raw_results = data.get("results", [])
        if not isinstance(raw_results, list):
            raise ValueError("results must be a list")
        # M3: whitelist-validate each result row before serving
        safe_results = []
        for r in raw_results:
            if not isinstance(r, dict):
                continue
            safe_results.append(
                {
                    "strategy": str(r.get("strategy", "")),
                    "win_rate": float(r.get("win_rate", 0)),
                    "total_trades": int(r.get("total_trades", 0)),
                    "expected_value": float(r.get("expected_value", 0)),
                    "total_return_pct": float(r.get("total_return_pct", 0)),
                    "max_drawdown_pct": float(r.get("max_drawdown_pct", 0)),
                    "sharpe_ratio": float(r.get("sharpe_ratio", 0)),
                }
            )
        return JSONResponse(
            {"results": safe_results, "generated_at": data.get("generated_at")}
        )
    except Exception:
        return JSONResponse({"results": []})


@app.get("/metrics")
async def metrics():
    portfolio = _redis_get("portfolio_state") or {}
    stats = _get_stats()
    _g_running.set(1 if _check_healthcheck() else 0)
    _g_balance.set(portfolio.get("balance", Config.INITIAL_BALANCE))
    _g_pnl.set(stats.get("total_pnl", 0))
    _g_win_rate.set(stats.get("win_rate", 0))
    _g_trades.set(stats.get("total_trades", 0))
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _send_telegram_sync(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return
    try:
        data = urllib.parse.urlencode(
            {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        ).encode()
        urllib.request.urlopen(
            f"https://api.telegram.org/bot{token}/sendMessage", data, timeout=5
        )
    except Exception as e:
        logger.warning(f"Telegram alert send failed: {e}")


def _tg_escape(s: str) -> str:
    """Strip Telegram Markdown special chars from untrusted content."""
    return s.replace("*", "").replace("_", "").replace("`", "").replace("[", "")


@app.post("/webhook/alerts")
async def alert_webhook(request: Request):
    """Receives Alertmanager webhook and forwards to Telegram."""
    # H3: validate shared secret if configured
    expected = os.getenv("ALERTMANAGER_WEBHOOK_SECRET", "")
    if expected:
        provided = request.headers.get("X-Webhook-Secret", "")
        if not hmac.compare_digest(provided.encode(), expected.encode()):
            return JSONResponse({"error": "Forbidden"}, status_code=403)

    payload = await request.json()
    alerts = payload.get("alerts", [])
    if not alerts:
        return JSONResponse({"status": "no_alerts"})

    lines = []
    for alert in alerts:
        status = alert.get("status", "firing")
        # H3: escape all dynamic content before injecting into Telegram message
        name = _tg_escape(str(alert.get("labels", {}).get("alertname", "unknown")))
        severity = _tg_escape(str(alert.get("labels", {}).get("severity", "warning")))
        summary = _tg_escape(str(alert.get("annotations", {}).get("summary", "")))
        description = _tg_escape(
            str(alert.get("annotations", {}).get("description", ""))
        )

        icon = "\U0001f534" if status == "firing" else "✅"
        label = "CRITICAL" if severity == "critical" else "WARNING"
        msg = f"{icon} [{label}] {name}"
        if summary:
            msg += f"\n{summary}"
        if description:
            msg += f"\n{description}"
        lines.append(msg)

    text = "\n\n".join(lines)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _send_telegram_sync, text)
    return JSONResponse({"status": "ok", "forwarded": len(lines)})


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(_HTML)


# ── HTML Dashboard ────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BitbotBY Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif;
         background: #0f1117; color: #e2e8f0; min-height: 100vh; }
  .header { background: #1a1d2e; border-bottom: 1px solid #2d3748;
            padding: 16px 24px; display: flex;
            justify-content: space-between; align-items: center; }
  .header h1 { font-size: 1.4rem; color: #63b3ed; font-weight: 700; }
  .status-badge { padding: 4px 12px; border-radius: 20px;
                  font-size: 0.8rem; font-weight: 600; }
  .status-ok { background: #22543d; color: #68d391; }
  .status-err { background: #742a2a; color: #fc8181; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 16px; padding: 24px; }
  .card { background: #1a1d2e; border: 1px solid #2d3748;
          border-radius: 12px; padding: 20px; }
  .card-label { font-size: 0.75rem; color: #718096;
                text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
  .card-value { font-size: 1.8rem; font-weight: 700; color: #e2e8f0; }
  .card-value.green { color: #68d391; }
  .card-value.red { color: #fc8181; }
  .card-value.blue { color: #63b3ed; }
  .section { padding: 0 24px 24px; }
  .section h2 { font-size: 1rem; color: #a0aec0;
                margin-bottom: 16px; padding-bottom: 8px;
                border-bottom: 1px solid #2d3748; }
  table { width: 100%; border-collapse: collapse;
          background: #1a1d2e; border-radius: 12px;
          overflow: hidden; }
  th { background: #2d3748; padding: 10px 14px;
       text-align: left; font-size: 0.75rem;
       color: #a0aec0; text-transform: uppercase; }
  td { padding: 10px 14px; border-bottom: 1px solid #2d3748;
       font-size: 0.85rem; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #2d3748; }
  .badge { padding: 2px 8px; border-radius: 4px;
           font-size: 0.75rem; font-weight: 600; }
  .buy { background: #22543d; color: #68d391; }
  .sell { background: #742a2a; color: #fc8181; }
  .hold { background: #2d3748; color: #a0aec0; }
  .open { background: #2a4365; color: #63b3ed; }
  .closed { background: #1a202c; color: #718096; }
  .mode-badge { background: #2d3748; color: #63b3ed;
                padding: 2px 8px; border-radius: 4px;
                font-size: 0.8rem; margin-left: 8px; }
  .refresh-info { color: #4a5568; font-size: 0.75rem;
                  text-align: center; padding: 16px; }
  .pnl-pos { color: #68d391; }
  .pnl-neg { color: #fc8181; }
  .paper-banner { background: #2d3748; color: #f6e05e;
                  text-align: center; padding: 8px;
                  font-size: 0.85rem; border-bottom: 1px solid #4a5568; }
</style>
</head>
<body>

<div class="header">
  <h1>BitbotBY <span class="mode-badge" id="mode">...</span></h1>
  <span class="status-badge status-err" id="bot-status">Загрузка...</span>
</div>
<div class="paper-banner" id="paper-banner" style="display:none">
  PAPER TRADING — реальные ордера не исполняются
</div>

<div class="grid">
  <div class="card">
    <div class="card-label">Баланс USDT</div>
    <div class="card-value blue" id="balance">—</div>
  </div>
  <div class="card">
    <div class="card-label">PnL всего</div>
    <div class="card-value" id="pnl">—</div>
  </div>
  <div class="card">
    <div class="card-label">Win Rate</div>
    <div class="card-value" id="win-rate">—</div>
  </div>
  <div class="card">
    <div class="card-label">Сделок всего</div>
    <div class="card-value" id="total-trades">—</div>
  </div>
  <div class="card">
    <div class="card-label">Позиции</div>
    <div class="card-value" id="positions">—</div>
  </div>
  <div class="card">
    <div class="card-label">Комиссии</div>
    <div class="card-value" id="commissions">—</div>
  </div>
</div>

<div class="section">
  <h2>История сделок</h2>
  <table>
    <thead>
      <tr>
        <th>Монета</th><th>Действие</th><th>Стратегия</th>
        <th>Вход</th><th>Выход</th><th>PnL</th>
        <th>AI</th><th>Статус</th><th>Время</th>
      </tr>
    </thead>
    <tbody id="trades-body">
      <tr><td colspan="9" style="text-align:center;color:#4a5568">
        Загрузка...
      </td></tr>
    </tbody>
  </table>
</div>

<div class="section">
  <h2>Бэктест стратегий</h2>
  <table>
    <thead>
      <tr>
        <th>Стратегия</th><th>Win Rate</th><th>EV</th>
        <th>Доходность</th><th>Drawdown</th><th>Sharpe</th><th>Сделок</th>
      </tr>
    </thead>
    <tbody id="bt-body">
      <tr><td colspan="7" style="text-align:center;color:#4a5568">
        Запустите backtest.py для данных
      </td></tr>
    </tbody>
  </table>
</div>

<div class="refresh-info">Обновляется каждые 5 секунд</div>

<script>
function fmt(n, d=2) {
  if (n === null || n === undefined) return '—';
  return Number(n).toFixed(d);
}
function pnlClass(v) {
  return v > 0 ? 'pnl-pos' : v < 0 ? 'pnl-neg' : '';
}
function esc(s) {
  if (s == null) return '—';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

async function loadStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();

    const alive = d.bot_running;
    const badge = document.getElementById('bot-status');
    badge.textContent = alive ? 'Работает' : 'Остановлен';
    badge.className = 'status-badge ' + (alive ? 'status-ok' : 'status-err');

    document.getElementById('mode').textContent = d.mode || '—';
    document.getElementById('balance').textContent =
      '$' + fmt(d.balance);

    const pnl = d.total_pnl || 0;
    const pnlEl = document.getElementById('pnl');
    pnlEl.textContent = (pnl >= 0 ? '+' : '') + '$' + fmt(pnl);
    pnlEl.className = 'card-value ' + (pnl >= 0 ? 'green' : 'red');

    const wr = d.win_rate || 0;
    const wrEl = document.getElementById('win-rate');
    wrEl.textContent = wr + '%';
    wrEl.className = 'card-value ' + (wr >= 50 ? 'green' : 'red');

    document.getElementById('total-trades').textContent =
      d.closed_trades + ' / ' + d.total_trades;
    document.getElementById('positions').textContent =
      Object.keys(d.positions || {}).length;
    document.getElementById('commissions').textContent =
      '-$' + fmt(d.total_commissions);

    if (d.paper_trading) {
      document.getElementById('paper-banner').style.display = 'block';
    }
  } catch(e) {
    console.error('Status error:', e);
  }
}

async function loadTrades() {
  try {
    const r = await fetch('/api/trades');
    const trades = await r.json();
    const tbody = document.getElementById('trades-body');
    if (!trades.length) {
      tbody.innerHTML = '<tr><td colspan="9" style="text-align:center;color:#4a5568">Нет сделок</td></tr>';
      return;
    }
    tbody.innerHTML = trades.map(t => {
      const pnl = t.pnl;
      const pnlStr = pnl !== null
        ? '<span class="' + pnlClass(pnl) + '">' +
          (pnl >= 0 ? '+' : '') + '$' + fmt(pnl) + '</span>'
        : '—';
      const action = (t.action || '').toLowerCase();
      const statusCls = t.status === 'open' ? 'open' : 'closed';
      const entry_time = t.entry_time
        ? t.entry_time.substring(0, 16).replace('T', ' ')
        : '—';
      return '<tr>' +
        '<td>' + esc(t.symbol) + '</td>' +
        '<td><span class="badge ' + esc(action) + '">' +
          esc((t.action || '—').toUpperCase()) + '</span></td>' +
        '<td>' + esc(t.strategy) + '</td>' +
        '<td>$' + fmt(t.entry_price, 4) + '</td>' +
        '<td>' + (t.exit_price ? '$' + fmt(t.exit_price, 4) : '—') + '</td>' +
        '<td>' + pnlStr + '</td>' +
        '<td>' + fmt((t.confidence||0)*100, 0) + '%</td>' +
        '<td><span class="badge ' + esc(statusCls) + '">' +
          esc(t.status) + '</span></td>' +
        '<td>' + esc(entry_time) + '</td>' +
        '</tr>';
    }).join('');
  } catch(e) {
    console.error('Trades error:', e);
  }
}

async function loadBacktest() {
  try {
    const r = await fetch('/api/backtest');
    const d = await r.json();
    const results = d.results || [];
    const tbody = document.getElementById('bt-body');
    if (!results.length) return;
    results.sort((a, b) => (b.expected_value||0) - (a.expected_value||0));
    tbody.innerHTML = results.map((r, i) => {
      const ev = (r.expected_value||0)*100;
      const ret = (r.total_return_pct||0)*100;
      const dd = (r.max_drawdown_pct||0)*100;
      return '<tr>' +
        '<td>' + (i===0?'#1 ':'') + esc(r.strategy) + '</td>' +
        '<td>' + fmt((r.win_rate||0)*100, 1) + '%</td>' +
        '<td class="' + pnlClass(ev) + '">' +
          (ev>=0?'+':'') + fmt(ev, 2) + '%</td>' +
        '<td class="' + pnlClass(ret) + '">' +
          (ret>=0?'+':'') + fmt(ret, 1) + '%</td>' +
        '<td class="pnl-neg">-' + fmt(dd, 1) + '%</td>' +
        '<td>' + fmt(r.sharpe_ratio, 2) + '</td>' +
        '<td>' + (r.total_trades||0) + '</td>' +
        '</tr>';
    }).join('');
  } catch(e) { console.error('Backtest error:', e); }
}

async function refresh() {
  await Promise.all([loadStatus(), loadTrades(), loadBacktest()]);
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",  # L3: bind to localhost only; use a reverse proxy for external access
        port=8080,
        log_level="warning",
    )
