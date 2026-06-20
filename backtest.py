"""
backtest.py — Walk-forward бэктестирование стратегий BitbotBY.

Использование:
    python3 backtest.py

Переменные окружения:
    BT_MONTHS      месяцев истории (по умолчанию 6)
    BT_SYMBOL      символ (по умолчанию Config.SYMBOL)
    BT_TIMEFRAME   таймфрейм (по умолчанию Config.TIMEFRAME)
    BT_MIN_CONF    мин. уверенность (по умолчанию Config.MIN_SIGNAL_CONFIDENCE)
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

# Разрешаем импорты из корня проекта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import STABLECOIN_BASES, Config
from src.alpha_tester import AlphaTester
from src.indicators import add_indicators
from src.market_impact import estimate_from_df as _ac_impact
from src.strategies import STRATEGY_REGISTRY, BaseStrategy

# ── Константы ──────────────────────────────────────────────────────────────

_FETCH_LIMIT = 200  # размер батча ccxt (лимит Bybit)
_CSV_CACHE_TTL = 86_400  # 24 ч
_WARMUP = 50  # минимум свечей до первого сигнала
_ATR_MULT = 1.5  # множитель ATR для стоп-лосса
_RR = 2.0  # соотношение риск/доход (1:2)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Датакласс ──────────────────────────────────────────────────────────────


@dataclass
class BacktestResult:
    """Метрики бэктеста для одной стратегии."""

    strategy: str
    total_trades: int
    win_rate: float  # 0.0–1.0
    avg_profit_pct: float  # средний % выигрышной сделки (положительный)
    avg_loss_pct: float  # средний % проигрышной сделки (отрицательный)
    expected_value: float  # wr*avg_profit + (1-wr)*avg_loss
    total_return_pct: float  # (итог - начало) / начало
    max_drawdown_pct: float  # наихудшая просадка от пика до минимума %
    sharpe_ratio: float  # годовой, rf=0
    num_wins: int
    num_losses: int
    trades: list = field(default_factory=list, repr=False)
    trade_returns: list = field(default_factory=list, repr=False)


# ── Вспомогательные функции CSV-кэша (без Redis, без API-ключа) ───────────


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
        logger.info(f"CSV-кэш загружен: {path} ({len(df)} строк)")
        return df
    except Exception as e:
        logger.warning(f"Ошибка чтения CSV-кэша: {e}")
        return None


def _save_csv_cache(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_csv(path)
    except Exception as e:
        logger.warning(f"Ошибка сохранения CSV-кэша: {e}")


# ── Топ-N символов по объёму за 24ч (публичный API, без авторизации) ─────


async def _fetch_top_symbols(n: int = 20) -> List[str]:
    """Получает топ-N символов /USDT по объёму за 24ч с Bybit (без API-ключа)."""
    try:
        import ccxt.async_support as ccxt_async

        exchange = ccxt_async.bybit(
            {
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            }
        )
        try:
            tickers = await exchange.fetch_tickers()
        finally:
            await exchange.close()

        usdt = {
            sym: t
            for sym, t in tickers.items()
            if sym.endswith("/USDT")
            and (t.get("quoteVolume") or 0) > 0
            and sym.split("/")[0] not in STABLECOIN_BASES
        }
        ranked = sorted(
            usdt.items(), key=lambda x: x[1].get("quoteVolume", 0), reverse=True
        )
        symbols = [sym for sym, _ in ranked[:n]]
        logger.info("Top %d symbols: %s...", n, ", ".join(symbols[:5]))
        return symbols
    except Exception as e:
        logger.error(
            "fetch_tickers завершился ошибкой: %s — возврат к Config.SYMBOLS", e
        )
        return Config.SYMBOLS[:n]


# ── Загрузка OHLCV (ccxt, публичные данные, авторизация не нужна) ─────────


async def _fetch_batches(
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
) -> pd.DataFrame:
    """Загружает OHLCV через ccxt без API-ключа (публичные данные биржи)."""
    try:
        import ccxt.async_support as ccxt_async
    except ImportError:
        logger.error("ccxt не установлен")
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
                logger.error(f"Ошибка fetch_ohlcv: {e}")
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

            if current >= until_ms:
                break  # достигли запрошенной конечной даты

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
    logger.info(f"Получено {len(result)} свечей всего")
    return result


async def load_data(
    symbol: str,
    timeframe: str,
    months: int,
) -> pd.DataFrame:
    """
    Загружает OHLCV-данные с CSV-кэшем, затем вычисляет индикаторы.

    Использует кэш или historical_btc.csv если API недоступен.
    """
    cache_path = _csv_cache_path(symbol, timeframe)
    cached = _load_csv_cache(cache_path)

    now_ms = int(datetime.now().timestamp() * 1000)
    since_ms = int((datetime.now() - timedelta(days=months * 30)).timestamp() * 1000)

    combined: pd.DataFrame

    if cached is not None and not cached.empty:
        last_ts = int(cached["timestamp"].max())
        logger.info("Кэш найден — получаем инкрементальное обновление " f"с {last_ts}")
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
            f"Кэш отсутствует — загружаем историю {months}м "
            f"для {symbol} {timeframe}..."
        )
        combined = await _fetch_batches(
            symbol,
            timeframe,
            since_ms=since_ms,
            until_ms=now_ms,
        )
        if combined.empty:
            # последний резерв: пробуем historical_btc.csv в корне проекта
            fallback = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "historical_btc.csv",
            )
            if os.path.exists(fallback):
                logger.warning(f"API недоступен — загружаем резерв " f"{fallback}")
                combined = pd.read_csv(fallback)
                combined.columns = [c.lower() for c in combined.columns]
                needed = {"open", "high", "low", "close", "volume"}
                if not needed.issubset(set(combined.columns)):
                    logger.error("В резервном CSV отсутствуют обязательные колонки")
                    return pd.DataFrame()
                if "timestamp" not in combined.columns:
                    combined["timestamp"] = range(len(combined))

    if combined.empty:
        logger.error("Нет данных — невозможно запустить бэктест")
        return pd.DataFrame()

    _save_csv_cache(combined, cache_path)
    logger.info(f"Всего свечей до расчёта индикаторов: {len(combined)}")

    # Вычисляем технические индикаторы
    combined = add_indicators(combined)

    # Псевдонимы, необходимые стратегиям
    if "ema_12" in combined.columns:
        combined["ema_short"] = combined["ema_12"]
    if "ema_26" in combined.columns:
        combined["ema_long"] = combined["ema_26"]

    return combined


# ── Вспомогательные функции симуляции ────────────────────────────────────


def _sharpe(returns: List[float]) -> float:
    """Коэффициент Sharpe на сделку (не годовой — периоды удержания разные)."""
    if len(returns) < 2:
        return 0.0
    mean_r = statistics.mean(returns)
    std_r = statistics.stdev(returns)
    if std_r == 0.0:
        return 0.0
    # НЕ умножаем на sqrt(252): этот множитель предполагает дневные доходности.
    # Здесь P&L% на сделку с переменным временем удержания — умножение на 252
    # завышает Sharpe ~в 15 раз для стратегий на 15м.
    return mean_r / std_r


# ── Основной движок walk-forward ──────────────────────────────────────────


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
    Прогоняет одну стратегию по df в режиме walk-forward.
    Без look-ahead: сигнал генерируется на df.iloc[:i+1].
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
            # ── Попытка войти в позицию ───────────────────────────
            try:
                signal = strategy.generate_signal(window)
            except Exception as exc:
                logger.warning(f"[{name}] signal error at i={i}: {exc}")
                continue

            action = signal.get("action", "hold")
            confidence = float(signal.get("confidence", 0.0))

            if action in ("buy", "sell") and confidence >= min_confidence:
                # Входим на OPEN следующей свечи, а не на CLOSE сигнальной —
                # исключаем look-ahead bias (сигнал срабатывает на закрытии бара;
                # ранний реалистичный вход — открытие бара i+1).
                if i + 1 >= n:
                    continue  # нет следующего бара; пропускаем сигнал в конце данных
                next_open = float(df.iloc[i + 1]["open"])
                # Критерий Келли после 10 закрытых сделок (half-Kelly, ограничен 20%)
                n_closed = len(closed_trades)
                if n_closed >= 10:
                    wr = sum(1 for t in closed_trades if t["pnl"] > 0) / n_closed
                    b = _RR  # reward/risk ratio fixed at 2.0
                    kelly_f = max(0.0, min((wr * b - (1 - wr)) / b * 0.5, 0.20))
                    fraction = kelly_f if kelly_f > 0 else risk_per_trade
                else:
                    fraction = risk_per_trade
                position_usdt = balance * fraction
                # Адаптивный market impact Almgren-Chriss:
                # зависит от размера ордера и волатильности
                impact = _ac_impact(window, position_usdt)
                if action == "buy":
                    entry_price = next_open * (1.0 + impact)
                else:
                    entry_price = next_open * (1.0 - impact)
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

                quantity = position_usdt / entry_price
                entry_comm = quantity * entry_price * commission_rate
                balance -= (
                    position_usdt + entry_comm
                )  # резервируем капитал + платим комиссию

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
            # ── Управление открытой позицией ──────────────────────
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
                if d == 1:  # лонг → продаём по bid
                    exit_price = current_price * (1.0 - exit_impact)
                else:  # шорт → покупаем по ask
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

    # ── Принудительное закрытие оставшейся позиции ───────────────
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

    # ── Агрегация метрик ─────────────────────────────────────────
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


# ── Агрегация по нескольким символам ─────────────────────────────────────


def aggregate_results(
    results_per_symbol: dict[str, List[BacktestResult]]
) -> List[BacktestResult]:
    """
    Объединяет BacktestResult по символам в один агрегированный результат на стратегию.

    Метрики:
      total_trades / wins / losses  → сумма по символам
      win_rate / avg_profit / avg_loss / EV → пересчёт из агрегированных счётчиков
      total_return_pct  → среднее по символам
      max_drawdown_pct  → наихудшее по символам
      sharpe_ratio      → пересчёт из объединённого списка trade_returns
    """
    strategy_buckets: dict[str, List[BacktestResult]] = {}
    for sym_results in results_per_symbol.values():
        for r in sym_results:
            strategy_buckets.setdefault(r.strategy, []).append(r)

    aggregated: List[BacktestResult] = []
    for strat, parts in strategy_buckets.items():
        total_trades = sum(p.total_trades for p in parts)
        num_wins = sum(p.num_wins for p in parts)
        num_losses = sum(p.num_losses for p in parts)
        win_rate = num_wins / total_trades if total_trades > 0 else 0.0

        all_wins = [t["return_pct"] for p in parts for t in p.trades if t["pnl"] > 0]
        all_losses = [t["return_pct"] for p in parts for t in p.trades if t["pnl"] <= 0]
        avg_profit = statistics.mean(all_wins) if all_wins else 0.0
        avg_loss = statistics.mean(all_losses) if all_losses else 0.0
        ev = win_rate * avg_profit + (1 - win_rate) * avg_loss

        avg_return = statistics.mean(p.total_return_pct for p in parts)
        max_dd = max(p.max_drawdown_pct for p in parts)

        combined_returns = [r for p in parts for r in p.trade_returns]
        sharpe = _sharpe(combined_returns)

        aggregated.append(
            BacktestResult(
                strategy=strat,
                total_trades=total_trades,
                win_rate=win_rate,
                avg_profit_pct=avg_profit,
                avg_loss_pct=avg_loss,
                expected_value=ev,
                total_return_pct=avg_return,
                max_drawdown_pct=max_dd,
                sharpe_ratio=sharpe,
                num_wins=num_wins,
                num_losses=num_losses,
                trade_returns=combined_returns,
            )
        )

    aggregated.sort(key=lambda r: r.expected_value, reverse=True)
    return aggregated


# ── Печать отчёта ────────────────────────────────────────────────────────


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
        print(f"Лучшая стратегия: {best.strategy}  (наибольший EV)")
        print(f"Рекомендуемый MODE: ai с " f"DEFAULT_STRATEGY={best.strategy}")
    print(sep)


# ── Экспорт JSON ─────────────────────────────────────────────────────────


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
        logger.info(f"Результаты сохранены в {output_path}")
    except Exception as e:
        logger.error(f"Ошибка сохранения JSON: {e}")


# ── Точка входа ──────────────────────────────────────────────────────────


async def main() -> None:
    # ── Чтение переопределений из окружения ───────────────────────
    months = int(os.getenv("BT_MONTHS", "6"))
    timeframe = os.getenv("BT_TIMEFRAME", Config.TIMEFRAME)
    min_conf = float(os.getenv("BT_MIN_CONF", str(Config.MIN_SIGNAL_CONFIDENCE)))

    # Определение символов:
    #   BT_SYMBOLS=A,B,C  → явный список
    #   BT_SYMBOL=A       → один явный символ
    #   (ничего)          → автоматически получить топ BT_TOP_N с Bybit по объёму за 24ч
    top_n = int(os.getenv("BT_TOP_N", str(Config.SCAN_TOP_N)))
    raw_syms = os.getenv("BT_SYMBOLS") or os.getenv("BT_SYMBOL")
    if raw_syms:
        symbols = [s.strip() for s in raw_syms.split(",") if s.strip()]
    else:
        print(f"[backtest] Auto-fetching top {top_n} symbols by 24h volume...")
        symbols = await _fetch_top_symbols(top_n)

    multi = len(symbols) > 1
    print(
        f"[backtest] {len(symbols)} symbol(s) | {timeframe} | "
        f"{months}m | min_conf={min_conf}"
    )
    print(f"[backtest] symbols: {', '.join(symbols)}")

    results_per_symbol: dict[str, List[BacktestResult]] = {}

    for symbol in symbols:
        print(f"\n{'─' * 50}")
        print(f"  {symbol}")
        print(f"{'─' * 50}")

        df = await load_data(symbol, timeframe, months)
        if df.empty:
            print(f"  ОШИБКА: нет данных для {symbol} — пропускаем")
            continue

        print(f"[backtest] {len(df)} свечей загружено")

        sym_results: List[BacktestResult] = []
        for strat_name, strat_cls in STRATEGY_REGISTRY.items():
            print(f"\n[{strat_name}] ", end="", flush=True)
            result = run_strategy(
                name=strat_name,
                strategy=strat_cls(),
                df=df,
                initial_balance=Config.INITIAL_BALANCE,
                risk_per_trade=Config.RISK_PER_TRADE,
                commission_rate=Config.COMMISSION_RATE,
                stop_loss_pct=Config.STOP_LOSS_PERCENT,
                min_confidence=min_conf,
            )
            sym_results.append(result)
            print(
                f" done — {result.total_trades} trades,"
                f" EV={result.expected_value * 100:+.2f}%"
            )

        sym_results.sort(key=lambda r: r.expected_value, reverse=True)
        results_per_symbol[symbol] = sym_results

        if not multi:
            print_report(sym_results, symbol, timeframe, months)

    if not results_per_symbol:
        print("ОШИБКА: нет результатов — прерываем")
        return

    # ── Сводная таблица по каждому символу (режим нескольких символов) ──────
    if multi:
        sep = "=" * 68
        for symbol, sym_results in results_per_symbol.items():
            print_report(sym_results, symbol, timeframe, months)

        # ── Агрегированный отчёт по всем символам ────────────────
        agg = aggregate_results(results_per_symbol)
        print(f"\n{sep}")
        print(f"AGGREGATED RESULTS — {len(symbols)} symbols — {timeframe} — {months}m")
        print(sep)
        print(
            f"{'Rank':<5} {'Strategy':<20} {'Trades':>6}  "
            f"{'WinRate':>8}  {'EV':>7}  {'AvgReturn':>9}  {'MaxDD':>7}  {'Sharpe':>7}"
        )
        print(
            f"{'----':<5} {'-'*20:<20} {'------':>6}  {'--------':>8}  {'------':>7}"
            f"  {'--------':>9}  {'------':>7}  {'------':>7}"
        )
        for rank, r in enumerate(agg, 1):
            print(
                f"{rank:>4}  {r.strategy:<20} {r.total_trades:>6}  "
                f"{r.win_rate*100:>7.1f}%  {r.expected_value*100:>+7.2f}%  "
                f"{r.total_return_pct*100:>+8.1f}%"
                f"  {r.max_drawdown_pct*100:>6.1f}%  {r.sharpe_ratio:>7.2f}"
            )
        print(sep)
        if agg:
            best = agg[0]
            print(
                f"Лучшая стратегия по всем символам: {best.strategy}"
                f"  (EV={best.expected_value*100:+.2f}%)"
            )
        print(sep)

        save_json(agg, ",".join(symbols), timeframe, months)
    else:
        save_json(results_per_symbol[symbols[0]], symbols[0], timeframe, months)

    # ── Тесты значимости альфы (агрегированные или по одному символу) ────────
    final_results = (
        aggregate_results(results_per_symbol)
        if multi
        else results_per_symbol[symbols[0]]
    )
    tester = AlphaTester()
    sep = "=" * 68
    print(f"\n{sep}")
    print("ЗНАЧИМОСТЬ АЛЬФЫ  (бутстрап Sharpe + знаковый ранговый тест Wilcoxon)")
    if multi:
        print("(на основе агрегированных доходностей сделок по всем символам)")
    print(sep)
    sig_count = 0
    for r in final_results:
        alpha = tester.test(r.trade_returns, name=r.strategy)
        print(alpha.summary())
        if alpha.is_significant:
            sig_count += 1
    print(sep)
    print(
        f"{sig_count}/{len(final_results)} стратегий показывают статистически "
        f"значимую альфу при p<{0.05:.0%}"
    )
    print(sep)


if __name__ == "__main__":
    asyncio.run(main())
