"""
backtest_live_divergence.py — п.5 роадмапа: "live vs backtest parity".

backtest.py и src/core/order_executor.py уже используют ОДНУ и ту же модель
рыночного импакта Almgren-Chriss (src/market_impact.estimate_from_df) для
entry_price — то есть в самой формуле импакта расхождения нет. То, что эта
метрика ловит — разница между ценой на закрытой свече, на которой сработал
сигнал (её "видит" бэктест), и реальной ценой исполнения в paper/live —
т.е. эффект ЗАДЕРЖКИ (Telegram-подтверждение, API round-trip, цикл
TRADING_INTERVAL), которую AC-модель не учитывает вообще (она зависит
только от размера ордера, не от времени).

Для каждой закрытой paper/live сделки:
  1. Ищем OHLCV-кэш символа (data/cache/{SYMBOL}_{TIMEFRAME}.csv). Пробуем
     оба таймфрейма (5m/15m) — в trades нет колонки timeframe, а профиль
     (и вместе с ним активный таймфрейм) мог поменяться между сделками.
  2. Берём последнюю свечу СТРОГО ДО entry_time — цену, которую увидел бы
     бэктест (он всегда торгует по уже закрытой свече).
  3. Прогоняем ту же estimate_from_df() с тем же notional (qty * entry) —
     получаем "backtest_implied_price".
  4. Сравниваем с реальным entry_price → % расхождения (entry_divergence_pct).
  5. Отдельно сравниваем реальную комиссию с Config.COMMISSION_RATE * notional
     (commission_divergence_pct) — независимая от AC-импакта проверка.

Ограничения (честно, а не спрятано): trades.entry_time пишется как
datetime.now() (локальное время процесса), а таймстемпы в кэше — UTC ms
(ccxt). Смещение вычисляется динамически как datetime.now()-utcnow() —
верно только если этот отчёт запускают на той же машине/таймзоне, что и
сам бот. Сделки по символам без кэша или без данных до entry_time
пропускаются и учитываются в счётчике skipped.

Использование:
    python3 -m src.research.backtest_live_divergence
    python3 -m src.research.backtest_live_divergence --db data/trades_live.db
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import Config
from src.market.market_impact import estimate_from_df

_TIMEFRAMES_TO_TRY = ("15m", "5m")
_AC_WINDOW = (
    50  # тот же размер окна, что backtest.py использует на закрытии (df.iloc[-50:])
)


@dataclass
class DivergenceRecord:
    symbol: str
    strategy: str
    entry_time: str
    entry_divergence_pct: float
    commission_divergence_pct: Optional[float]
    timeframe_used: str


def _cache_path(symbol: str, timeframe: str) -> str:
    safe = symbol.replace("/", "_").replace(":", "_")
    return os.path.join("data", "cache", f"{safe}_{timeframe}.csv")


def load_cache(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Читает OHLCV-кэш символа для заданного таймфрейма, если файл есть.

    :param symbol: Символ в формате "BTC/USDT".
    :param timeframe: "15m" или "5m".
    :return: DataFrame с колонками timestamp(ms)/open/high/low/close/volume,
        отсортированный по timestamp, либо None если файла нет/он пуст.
    """
    path = _cache_path(symbol, timeframe)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception:
        return None
    if "timestamp" not in df.columns or df.empty:
        return None
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df if not df.empty else None


def _local_to_utc_ms(entry_time_iso: str) -> Optional[int]:
    """Конвертирует локальный naive ISO timestamp (datetime.now()) в UTC ms.

    Смещение считается динамически (datetime.now() - datetime.utcnow()) —
    предполагает, что отчёт запускают в той же таймзоне, что и сам бот.
    """
    try:
        local_dt = datetime.fromisoformat(entry_time_iso)
    except ValueError:
        return None
    offset = datetime.now() - datetime.utcnow()
    utc_dt = local_dt - offset
    return int(utc_dt.timestamp() * 1000)


def find_reference(
    symbol: str, entry_time_iso: str
) -> Optional[Tuple[float, pd.DataFrame, str]]:
    """Находит последнюю свечу строго до entry_time среди доступных таймфреймов.

    Пробует оба поддерживаемых таймфрейма и выбирает ту свечу, что ближе по
    времени к entry_time (эвристика: раз в trades нет колонки timeframe,
    более близкая по времени свеча — более вероятный кандидат на то, какой
    таймфрейм был активен на момент сделки).

    :param symbol: Символ в формате "BTC/USDT".
    :param entry_time_iso: entry_time из trades (naive локальный ISO).
    :return: (close_price, window_df последних _AC_WINDOW свечей до неё,
        использованный таймфрейм) либо None, если данных нет ни для одного
        таймфрейма.
    """
    entry_utc_ms = _local_to_utc_ms(entry_time_iso)
    if entry_utc_ms is None:
        return None

    best: Optional[Tuple[float, pd.DataFrame, str, int]] = None
    for tf in _TIMEFRAMES_TO_TRY:
        df = load_cache(symbol, tf)
        if df is None:
            continue
        prior = df[df["timestamp"] < entry_utc_ms]
        if prior.empty:
            continue
        gap = entry_utc_ms - int(prior["timestamp"].iloc[-1])
        if best is None or gap < best[3]:
            window = prior.iloc[-_AC_WINDOW:]
            best = (float(prior["close"].iloc[-1]), window, tf, gap)

    if best is None:
        return None
    close_price, window, tf, _gap = best
    return close_price, window, tf


def load_closed_trades(db_path: str) -> List[Dict]:
    """Читает закрытые сделки с полями, нужными для сверки цены исполнения.

    :param db_path: Путь к trades_paper.db/trades_live.db.
    :return: Список словарей с symbol/strategy/action/entry_price/quantity/
        commission/entry_time; пусто, если файла или сделок нет.
    """
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT symbol, strategy, action, entry_price, quantity, "
            "commission, entry_time FROM trades "
            "WHERE status='closed' AND entry_price IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "symbol": symbol,
            "strategy": strategy,
            "action": action,
            "entry_price": float(entry_price),
            "quantity": float(quantity),
            "commission": float(commission or 0.0),
            "entry_time": entry_time,
        }
        for symbol, strategy, action, entry_price, quantity, commission, entry_time in rows
    ]


def compute_divergence(trade: Dict) -> Optional[DivergenceRecord]:
    """Считает расхождение реальной цены/комиссии исполнения с тем, что
    предсказала бы модель бэктеста (та же AC-функция, свеча до входа).

    :param trade: Одна запись из load_closed_trades().
    :return: DivergenceRecord либо None, если для символа/времени нет
        подходящих исторических данных.
    """
    ref = find_reference(trade["symbol"], trade["entry_time"])
    if ref is None:
        return None
    close_price, window, timeframe = ref

    notional = trade["quantity"] * trade["entry_price"]
    impact = estimate_from_df(window, notional, timeframe)
    implied_price = (
        close_price * (1.0 + impact)
        if trade["action"] == "buy"
        else close_price * (1.0 - impact)
    )
    if implied_price <= 0:
        return None
    entry_divergence_pct = (trade["entry_price"] - implied_price) / implied_price

    assumed_commission = notional * Config.COMMISSION_RATE
    commission_divergence_pct = (
        (trade["commission"] - assumed_commission) / assumed_commission
        if assumed_commission > 0
        else None
    )

    return DivergenceRecord(
        symbol=trade["symbol"],
        strategy=trade["strategy"],
        entry_time=trade["entry_time"],
        entry_divergence_pct=entry_divergence_pct,
        commission_divergence_pct=commission_divergence_pct,
        timeframe_used=timeframe,
    )


def build_report(records: List[DivergenceRecord], total_trades: int) -> str:
    """Строит текстовый отчёт по расхождениям цены/комиссии backtest vs live.

    :param records: Результаты compute_divergence() для сопоставленных сделок.
    :param total_trades: Общее число закрытых сделок (включая пропущенные).
    :return: Многострочный отчёт для печати в консоль.
    """
    sep = "=" * 68
    lines = [
        sep,
        "LIVE vs BACKTEST DIVERGENCE (п.5 роадмапа)",
        "(та же AC-модель импакта, свеча до входа = 'цена бэктеста')",
        sep,
    ]

    skipped = total_trades - len(records)
    lines.append(
        f"Сопоставлено {len(records)}/{total_trades} закрытых сделок "
        f"({skipped} пропущено — нет кэша/истории для символа)"
    )
    if not records:
        lines.append("Нет сопоставленных сделок — отчёт невозможен.")
        lines.append(sep)
        return "\n".join(lines)

    entry_devs = [r.entry_divergence_pct for r in records]
    mean_entry = sum(entry_devs) / len(entry_devs)
    mean_abs_entry = sum(abs(d) for d in entry_devs) / len(entry_devs)
    lines.append(sep)
    lines.append(
        f"Цена входа: среднее расхождение {mean_entry:+.3%}, "
        f"среднее |расхождение| {mean_abs_entry:.3%}"
    )
    if mean_abs_entry > 0.005:
        lines.append(
            "  ⚠ >0.5% — предположение бэктеста о цене входа систематически "
            "неточно, стоит перепроверить окно impact/задержку исполнения"
        )

    comm_devs = [
        r.commission_divergence_pct
        for r in records
        if r.commission_divergence_pct is not None
    ]
    if comm_devs:
        mean_comm = sum(comm_devs) / len(comm_devs)
        lines.append(
            f"Комиссия: среднее расхождение с Config.COMMISSION_RATE {mean_comm:+.3%}"
        )

    lines.append(sep)
    lines.append("ПО СИМВОЛАМ:")
    by_symbol: Dict[str, List[DivergenceRecord]] = {}
    for r in records:
        by_symbol.setdefault(r.symbol, []).append(r)
    for symbol, recs in by_symbol.items():
        devs = [r.entry_divergence_pct for r in recs]
        avg = sum(devs) / len(devs)
        lines.append(f"  {symbol}: {len(recs)} сделок, среднее расхождение {avg:+.3%}")
    lines.append(sep)
    return "\n".join(lines)


def _default_db_path() -> str:
    name = (
        "trades_paper.db"
        if os.getenv("PAPER_TRADING", "").lower() == "true"
        else "trades_live.db"
    )
    return os.path.join("data", name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default=_default_db_path(),
        help="Путь к SQLite базе сделок (по умолчанию — trades_paper.db "
        "если PAPER_TRADING=true, иначе trades_live.db)",
    )
    args = parser.parse_args()

    trades = load_closed_trades(args.db)
    if not trades:
        print(f"[backtest_live_divergence] Нет закрытых сделок в {args.db}")
        return

    records = [r for r in (compute_divergence(t) for t in trades) if r is not None]
    print(
        f"[backtest_live_divergence] {len(trades)} закрытых сделок, источник: {args.db}\n"
    )
    print(build_report(records, total_trades=len(trades)))


if __name__ == "__main__":
    main()
