"""
backtest.py — Walk-forward backtesting for BitbotBY strategies.

Usage:
    python3 backtest.py

Env overrides:
    BT_MONTHS      months of history (default 6)
    BT_SYMBOL      symbol (default Config.SYMBOL)
    BT_TIMEFRAME   timeframe (default Config.TIMEFRAME)
    BT_MIN_CONF    min confidence (default Config.MIN_SIGNAL_CONFIDENCE)
"""

import asyncio
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src.alpha_tester import AlphaTester
from src.indicators import add_indicators
from src.market_impact import estimate_from_df as _ac_impact
from src.strategies import STRATEGY_REGISTRY, BaseStrategy

# ── Constants ──────────────────────────────────────────────────────────────

_FETCH_LIMIT = 200  # ccxt batch size (Bybit limit)
_CSV_CACHE_TTL = 86_400  # 24 h
_WARMUP = 50  # minimum candles before first signal
_ATR_MULT = 1.5  # stop-loss ATR multiplier
_RR = 2.0  # risk/reward ratio (1:2)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Dataclass ──────────────────────────────────────────────────────────────


@dataclass
class BacktestResult:
    """Per-strategy backtest metrics."""

    strategy: str
    total_trades: int
    win_rate: float  # 0.0–1.0
    avg_profit_pct: float  # avg winning trade % (positive)
    avg_loss_pct: float  # avg losing trade % (negative)
    expected_value: float  # wr*avg_profit + (1-wr)*avg_loss
    total_return_pct: float  # (final - initial) / initial
    max_drawdown_pct: float  # worst peak-to-trough %
    sharpe_ratio: float  # annualised, rf=0
    num_wins: int
    num_losses: int
    trades: list = field(default_factory=list, repr=False)
    trade_returns: list = field(default_factory=list, repr=False)


# ── CSV cache helpers (no Redis, no API key required) ─────────────────────


def _csv_cache_path(symbol: str, timeframe: str) -> str:
    safe = symbol.replace("/", "_").replace(":", "_")
    os.makedirs("data/cache", exist_ok=True)
    return f"data/cache/{safe}_{timeframe}.csv"


def _load_csv_cache(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    age = time.time() - os.path.getmtime(path)
    if age > _CSV_CACHE_TTL:
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        logger.info(f"CSV cache loaded: {path} ({len(df)} rows)")
        return df
    except Exception as e:
        logger.warning(f"CSV cache read failed: {e}")
        return None


def _save_csv_cache(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_csv(path)
    except Exception as e:
        logger.warning(f"CSV cache save failed: {e}")


# ── OHLCV fetch (ccxt, no auth needed for public data) ────────────────────


async def _fetch_batches(
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
) -> pd.DataFrame:
    """Download OHLCV via ccxt without an API key."""
    try:
        import ccxt.async_support as ccxt_async
    except ImportError:
        logger.error("ccxt not installed")
        return pd.DataFrame()

    exchange = ccxt_async.bybit({"enableRateLimit": True})
    all_frames: List[pd.DataFrame] = []
    current = since_ms

    try:
        while current < until_ms:
            try:
                raw = await exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=current,
                    limit=_FETCH_LIMIT,
                )
            except Exception as e:
                logger.error(f"fetch_ohlcv error: {e}")
                break

            if not raw:
                break

            df_batch = pd.DataFrame(
                raw,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ],
            )
            all_frames.append(df_batch)

            last_ts = int(raw[-1][0])
            if last_ts <= current:
                break
            current = last_ts + 1

            if len(raw) < _FETCH_LIMIT:
                break  # reached end of available data

    finally:
        await exchange.close()

    if not all_frames:
        return pd.DataFrame()

    result = (
        pd.concat(all_frames, ignore_index=True)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    logger.info(f"Fetched {len(result)} candles total")
    return result


async def load_data(
    symbol: str,
    timeframe: str,
    months: int,
) -> pd.DataFrame:
    """
    Load OHLCV data with CSV cache, then compute indicators.

    Falls back to cached CSV or historical_btc.csv if API
    is unavailable.
    """
    cache_path = _csv_cache_path(symbol, timeframe)
    cached = _load_csv_cache(cache_path)

    now_ms = int(datetime.now().timestamp() * 1000)
    since_ms = int((datetime.now() - timedelta(days=months * 30)).timestamp() * 1000)

    combined: pd.DataFrame

    if cached is not None and not cached.empty:
        last_ts = int(cached["timestamp"].max())
        logger.info("Cache hit — fetching incremental update " f"since {last_ts}")
        new_df = await _fetch_batches(
            symbol,
            timeframe,
            since_ms=last_ts + 1,
            until_ms=now_ms,
        )
        if not new_df.empty:
            combined = (
                pd.concat([cached, new_df], ignore_index=True)
                .drop_duplicates(subset=["timestamp"])
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
        else:
            combined = cached
    else:
        logger.info(
            f"No cache — fetching {months}m history " f"for {symbol} {timeframe}..."
        )
        combined = await _fetch_batches(
            symbol,
            timeframe,
            since_ms=since_ms,
            until_ms=now_ms,
        )
        if combined.empty:
            # last-resort: try historical_btc.csv in project root
            fallback = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "historical_btc.csv",
            )
            if os.path.exists(fallback):
                logger.warning(f"API unavailable — loading fallback " f"{fallback}")
                combined = pd.read_csv(fallback)
                combined.columns = [c.lower() for c in combined.columns]
                needed = {"open", "high", "low", "close", "volume"}
                if not needed.issubset(set(combined.columns)):
                    logger.error("Fallback CSV missing required columns")
                    return pd.DataFrame()
                if "timestamp" not in combined.columns:
                    combined["timestamp"] = range(len(combined))

    if combined.empty:
        logger.error("No data available — cannot run backtest")
        return pd.DataFrame()

    _save_csv_cache(combined, cache_path)
    logger.info(f"Total candles before indicators: {len(combined)}")

    # Compute technical indicators
    combined = add_indicators(combined)

    # Aliases required by strategies
    if "ema_12" in combined.columns:
        combined["ema_short"] = combined["ema_12"]
    if "ema_26" in combined.columns:
        combined["ema_long"] = combined["ema_26"]

    return combined


# ── Simulation helpers ────────────────────────────────────────────────────


def _sharpe(returns: List[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean_r = statistics.mean(returns)
    std_r = statistics.stdev(returns)
    if std_r == 0.0:
        return 0.0
    return (mean_r / std_r) * (252**0.5)


# ── Core walk-forward engine ──────────────────────────────────────────────


def run_strategy(
    name: str,
    strategy: BaseStrategy,
    df: pd.DataFrame,
    initial_balance: float,
    risk_per_trade: float,
    commission_rate: float,
    stop_loss_pct: float,
    min_confidence: float,
) -> BacktestResult:
    """
    Run a single strategy over df in walk-forward fashion.
    No look-ahead bias: signal is generated on df.iloc[:i+1].
    """
    balance = initial_balance
    peak_balance = initial_balance
    max_drawdown = 0.0

    open_trade: Optional[Dict[str, Any]] = None
    closed_trades: List[Dict[str, Any]] = []
    trade_returns: List[float] = []

    n = len(df)
    dot_interval = max(1000, n // 10)

    for i in range(_WARMUP, n):
        if i % dot_interval == 0:
            print(".", end="", flush=True)

        window = df.iloc[: i + 1]

        if open_trade is None:
            # ── Try to enter ──────────────────────────────────────
            try:
                signal = strategy.generate_signal(window)
            except Exception as exc:
                logger.warning(f"[{name}] signal error at i={i}: {exc}")
                continue

            action = signal.get("action", "hold")
            confidence = float(signal.get("confidence", 0.0))

            if action in ("buy", "sell") and confidence >= min_confidence:
                mid = float(window["close"].iloc[-1])
                # Kelly criterion after 10 closed trades (half-Kelly, capped 20%)
                n_closed = len(closed_trades)
                if n_closed >= 10:
                    wr = sum(1 for t in closed_trades if t["pnl"] > 0) / n_closed
                    b = _RR  # reward/risk ratio fixed at 2.0
                    kelly_f = max(0.0, min((wr * b - (1 - wr)) / b * 0.5, 0.20))
                    fraction = kelly_f if kelly_f > 0 else risk_per_trade
                else:
                    fraction = risk_per_trade
                position_usdt = balance * fraction
                # Almgren-Chriss adaptive impact: adapts to order size & volatility
                impact = _ac_impact(window, position_usdt)
                if action == "buy":
                    entry_price = mid * (1.0 + impact)
                else:
                    entry_price = mid * (1.0 - impact)
                atr_val = (
                    float(window["atr"].iloc[-1]) if "atr" in window.columns else 0.0
                )

                if atr_val > 0:
                    sl_dist = _ATR_MULT * atr_val
                else:
                    sl_dist = entry_price * stop_loss_pct

                if action == "buy":
                    stop_loss = entry_price - sl_dist
                    take_profit = entry_price + _RR * sl_dist
                    direction = 1
                else:
                    stop_loss = entry_price + sl_dist
                    take_profit = entry_price - _RR * sl_dist
                    direction = -1

                position_usdt = balance * risk_per_trade
                quantity = position_usdt / entry_price
                entry_comm = quantity * entry_price * commission_rate
                balance -= entry_comm

                open_trade = {
                    "action": action,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "direction": direction,
                    "quantity": quantity,
                    "position_usdt": position_usdt,
                    "entry_comm": entry_comm,
                    "entry_i": i,
                }

        else:
            # ── Manage open position ──────────────────────────────
            current_price = float(window["close"].iloc[-1])
            d = open_trade["direction"]
            sl = open_trade["stop_loss"]
            tp = open_trade["take_profit"]

            sl_hit = (d == 1 and current_price <= sl) or (
                d == -1 and current_price >= sl
            )
            tp_hit = (d == 1 and current_price >= tp) or (
                d == -1 and current_price <= tp
            )

            if sl_hit or tp_hit:
                exit_impact = _ac_impact(window, open_trade["position_usdt"])
                d = open_trade["direction"]
                if d == 1:  # long → sell at bid
                    exit_price = current_price * (1.0 - exit_impact)
                else:  # short → buy at ask
                    exit_price = current_price * (1.0 + exit_impact)
                qty = open_trade["quantity"]
                exit_comm = qty * exit_price * commission_rate
                total_comm = open_trade["entry_comm"] + exit_comm
                pnl = (exit_price - open_trade["entry_price"]) * qty * d - total_comm
                balance += qty * exit_price - exit_comm
                balance = max(balance, 0.0)

                peak_balance = max(peak_balance, balance)
                dd = (
                    (peak_balance - balance) / peak_balance if peak_balance > 0 else 0.0
                )
                max_drawdown = max(max_drawdown, dd)

                pos_usdt = open_trade["position_usdt"]
                ret = pnl / pos_usdt if pos_usdt > 0 else 0.0
                trade_returns.append(ret)

                closed_trades.append(
                    {
                        "entry_i": open_trade["entry_i"],
                        "exit_i": i,
                        "action": open_trade["action"],
                        "entry_price": open_trade["entry_price"],
                        "exit_price": exit_price,
                        "pnl": round(pnl, 6),
                        "return_pct": round(ret, 6),
                        "closed_by": "tp" if tp_hit else "sl",
                    }
                )
                open_trade = None

    # ── Force-close any remaining position ───────────────────────
    if open_trade is not None and n > 0:
        d = open_trade["direction"]
        mid_close = float(df["close"].iloc[-1])
        close_impact = _ac_impact(df.iloc[-50:], open_trade["position_usdt"])
        if d == 1:
            exit_price = mid_close * (1.0 - close_impact)
        else:
            exit_price = mid_close * (1.0 + close_impact)
        qty = open_trade["quantity"]
        d = open_trade["direction"]
        exit_comm = qty * exit_price * commission_rate
        total_comm = open_trade["entry_comm"] + exit_comm
        pnl = (exit_price - open_trade["entry_price"]) * qty * d - total_comm
        balance += qty * exit_price - exit_comm
        balance = max(balance, 0.0)

        peak_balance = max(peak_balance, balance)
        dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0.0
        max_drawdown = max(max_drawdown, dd)

        pos_usdt = open_trade["position_usdt"]
        ret = pnl / pos_usdt if pos_usdt > 0 else 0.0
        trade_returns.append(ret)

        closed_trades.append(
            {
                "entry_i": open_trade["entry_i"],
                "exit_i": n - 1,
                "action": open_trade["action"],
                "entry_price": open_trade["entry_price"],
                "exit_price": exit_price,
                "pnl": round(pnl, 6),
                "return_pct": round(ret, 6),
                "closed_by": "force",
            }
        )

    # ── Aggregate metrics ─────────────────────────────────────────
    total_trades = len(closed_trades)
    wins = [t for t in closed_trades if t["pnl"] > 0]
    losses = [t for t in closed_trades if t["pnl"] <= 0]

    win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
    avg_profit = statistics.mean([t["return_pct"] for t in wins]) if wins else 0.0
    avg_loss = statistics.mean([t["return_pct"] for t in losses]) if losses else 0.0
    ev = win_rate * avg_profit + (1 - win_rate) * avg_loss

    total_return = (
        (balance - initial_balance) / initial_balance if initial_balance > 0 else 0.0
    )
    sharpe = _sharpe(trade_returns)

    return BacktestResult(
        strategy=name,
        total_trades=total_trades,
        win_rate=win_rate,
        avg_profit_pct=avg_profit,
        avg_loss_pct=avg_loss,
        expected_value=ev,
        total_return_pct=total_return,
        max_drawdown_pct=max_drawdown,
        sharpe_ratio=sharpe,
        num_wins=len(wins),
        num_losses=len(losses),
        trades=closed_trades,
        trade_returns=trade_returns,
    )


# ── Report printer ────────────────────────────────────────────────────────


def print_report(
    results: List[BacktestResult],
    symbol: str,
    timeframe: str,
    months: int,
) -> None:
    sep = "=" * 68
    header = f"BACKTEST RESULTS — {symbol} {timeframe} " f"— {months} months"
    print(f"\n{sep}")
    print(header)
    print(sep)
    print(
        f"{'Rank':<5} {'Strategy':<20} {'Trades':>6}  "
        f"{'WinRate':>8}  {'EV':>7}  "
        f"{'Return':>7}  {'Drawdown':>9}  {'Sharpe':>7}"
    )
    print(
        f"{'----':<5} {'-' * 20:<20} {'------':>6}  "
        f"{'--------':>8}  {'------':>7}  "
        f"{'------':>7}  {'--------':>9}  {'------':>7}"
    )

    for rank, r in enumerate(results, 1):
        wr_str = f"{r.win_rate * 100:.1f}%"
        ev_str = f"{r.expected_value * 100:+.2f}%"
        ret_str = f"{r.total_return_pct * 100:+.1f}%"
        dd_str = f"{r.max_drawdown_pct * 100:.1f}%"
        sh_str = f"{r.sharpe_ratio:.2f}"
        print(
            f"{rank:>4}  {r.strategy:<20} {r.total_trades:>6}  "
            f"{wr_str:>8}  {ev_str:>7}  "
            f"{ret_str:>7}  {dd_str:>9}  {sh_str:>7}"
        )

    print(sep)
    if results:
        best = results[0]
        print(f"Best strategy: {best.strategy}  (highest EV)")
        print(f"Recommended MODE: ai with " f"DEFAULT_STRATEGY={best.strategy}")
    print(sep)


# ── JSON export ───────────────────────────────────────────────────────────


def save_json(
    results: List[BacktestResult],
    symbol: str,
    timeframe: str,
    months: int,
    output_path: str = "data/backtest_results.json",
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    summary_list = []
    for r in results:
        summary_list.append(
            {
                "strategy": r.strategy,
                "total_trades": r.total_trades,
                "win_rate": round(r.win_rate, 4),
                "avg_profit_pct": round(r.avg_profit_pct, 6),
                "avg_loss_pct": round(r.avg_loss_pct, 6),
                "expected_value": round(r.expected_value, 6),
                "total_return_pct": round(r.total_return_pct, 6),
                "max_drawdown_pct": round(r.max_drawdown_pct, 6),
                "sharpe_ratio": round(r.sharpe_ratio, 4),
                "num_wins": r.num_wins,
                "num_losses": r.num_losses,
            }
        )

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "symbol": symbol,
        "timeframe": timeframe,
        "months": months,
        "results": summary_list,
    }

    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")


# ── Entry point ───────────────────────────────────────────────────────────


async def main() -> None:
    # ── Read env overrides ────────────────────────────────────────
    months = int(os.getenv("BT_MONTHS", "6"))
    symbol = os.getenv("BT_SYMBOL", Config.SYMBOL)
    timeframe = os.getenv("BT_TIMEFRAME", Config.TIMEFRAME)
    min_conf = float(os.getenv("BT_MIN_CONF", str(Config.MIN_SIGNAL_CONFIDENCE)))

    print(f"[backtest] {symbol} {timeframe} x {months}m  " f"min_conf={min_conf}")

    # ── Load data ─────────────────────────────────────────────────
    df = await load_data(symbol, timeframe, months)
    if df.empty:
        print("ERROR: no data loaded — aborting")
        return

    print(f"[backtest] {len(df)} candles loaded")

    # ── Run all strategies ────────────────────────────────────────
    results: List[BacktestResult] = []
    for strat_name, strat_cls in STRATEGY_REGISTRY.items():
        print(f"\n[{strat_name}] ", end="", flush=True)
        strategy_inst = strat_cls()
        result = run_strategy(
            name=strat_name,
            strategy=strategy_inst,
            df=df,
            initial_balance=Config.INITIAL_BALANCE,
            risk_per_trade=Config.RISK_PER_TRADE,
            commission_rate=Config.COMMISSION_RATE,
            stop_loss_pct=Config.STOP_LOSS_PERCENT,
            min_confidence=min_conf,
        )
        results.append(result)
        print(
            f" done — {result.total_trades} trades, "
            f"EV={result.expected_value * 100:+.2f}%"
        )

    # ── Sort by expected_value descending ─────────────────────────
    results.sort(key=lambda r: r.expected_value, reverse=True)

    # ── Print report ──────────────────────────────────────────────
    print_report(results, symbol, timeframe, months)

    # ── Alpha significance tests ──────────────────────────────────
    tester = AlphaTester()
    sep = "=" * 68
    print(f"\n{sep}")
    print("ALPHA SIGNIFICANCE  (bootstrap Sharpe + Wilcoxon signed-rank)")
    print(sep)
    sig_count = 0
    for r in results:
        alpha = tester.test(r.trade_returns, name=r.strategy)
        print(alpha.summary())
        if alpha.is_significant:
            sig_count += 1
    print(sep)
    print(
        f"{sig_count}/{len(results)} strategies show statistically "
        f"significant alpha at p<{0.05:.0%}"
    )
    print(sep)

    # ── Save JSON ─────────────────────────────────────────────────
    save_json(results, symbol, timeframe, months)


if __name__ == "__main__":
    asyncio.run(main())
