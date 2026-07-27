"""
ai_signal_isolation_backtest.py — изолирует вклад AI (LLM)-сигнала от голых
TA-индикаторов.

Контекст (см. память project_senior_quant_roadmap, п.7): полный walk-forward
backtest.py показал, что все 8 базовых стратегий (EMA/RSI/MACD/Bollinger/
trend/mean-reversion) не имеют статистического edge на BTC/ETH/AVAX 15m.
Открытый вопрос — даёт ли AI/SAC-слой поверх этого edge, которого нет у
голых индикаторов. Этот скрипт отвечает только за AI (LLM)-часть.

SAC НЕ тестируется: models/ пуст, обученной модели нет (dqn_signal.py
молча остаётся в состоянии "не загружен"). Изолировать SAC можно будет
только после отдельной задачи по обучению модели.

Методология:
    - Та же симуляция сделок, что и run_strategy() в backtest.py: вход на
      OPEN следующего бара после сигнала (без look-ahead), управление
      позицией по SL/TP на каждом баре, асимметричная комиссия
      (тейкер на вход/SL, мейкер на TP), Almgren-Chriss market impact.
    - В отличие от run_strategy(), AI вызывается НЕ на каждом баре (это
      были бы десятки тысяч платных LLM-вызовов), а на N сэмплированных
      "циклах", равномерно распределённых по общему периоду — как и
      реальный бот, один batch-вызов на цикл анализирует сразу все
      символы, которые сейчас не в позиции.
    - SL/TP берутся из самого ответа AI (как в проде — order_executor.py
      просто читает top["stop_loss"]/top["take_profit"]), с ATR-фолбэком,
      если AI их не вернул или они противоречат направлению сделки.
    - Trade-level доходность (return_pct = pnl / position_usdt)
      самонормализована и не зависит от размера позиции — поэтому здесь
      используется простой Config.RISK_PER_TRADE без Kelly-бампа (Kelly
      влиял бы только на total_return/drawdown, не на Sharpe/значимость).

Ограничение метода: между сэмплированными точками сделки не отслеживаются
для входа (только управление уже открытой позицией) — упущенные входы
между чекпоинтами занижают число сделок относительно "полного" прогона,
это цена экономии на LLM-вызовах.

Переменные окружения:
    AIB_SYMBOLS    список символов через запятую (default BTC/USDT,ETH/USDT,AVAX/USDT)
    AIB_TIMEFRAME  таймфрейм ccxt (default 15m)
    AIB_MONTHS     месяцев истории (default 6) — совпадает с прошлым прогоном
                   базовых стратегий для сопоставимости результатов
    AIB_SAMPLES    число сэмплированных циклов вызова AI (default 150)

Использование:
    python3 ai_signal_isolation_backtest.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backtest import _ATR_MULT, _RR, _WARMUP, _sharpe, load_data
from config import Config
from src.ai_analyzer import AIAnalyzer
from src.alpha_tester import AlphaTester
from src.market_impact import estimate_from_df as _ac_impact
from src.market_scanner import MarketScanner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_SYMBOLS = "BTC/USDT,ETH/USDT,AVAX/USDT"


@dataclass
class _SymbolState:
    symbol: str
    df: pd.DataFrame
    next_idx: int
    open_trade: Optional[Dict[str, Any]] = None


@dataclass
class _Portfolio:
    balance: float
    peak_balance: float = 0.0
    max_drawdown: float = 0.0
    trade_returns: List[float] = field(default_factory=list)
    closed_trades: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.peak_balance = self.balance


def _resolve_exit(
    state: _SymbolState,
    i: int,
    portfolio: _Portfolio,
    commission_rate: float,
    maker_commission_rate: float,
    timeframe: str,
) -> None:
    """Проверяет SL/TP на баре i для открытой позиции state.open_trade и
    закрывает её при попадании — идентично управляющей ветке run_strategy()."""
    trade = state.open_trade
    assert trade is not None
    window = state.df.iloc[: i + 1]
    current_price = float(window["close"].iloc[-1])
    d = trade["direction"]
    sl, tp = trade["stop_loss"], trade["take_profit"]

    sl_hit = (d == 1 and current_price <= sl) or (d == -1 and current_price >= sl)
    tp_hit = (d == 1 and current_price >= tp) or (d == -1 and current_price <= tp)
    if not (sl_hit or tp_hit):
        return

    exit_impact = _ac_impact(window, trade["position_usdt"], timeframe=timeframe)
    exit_price = (
        current_price * (1.0 - exit_impact) if d == 1 else current_price * (1.0 + exit_impact)
    )
    qty = trade["quantity"]
    exit_rate = maker_commission_rate if tp_hit else commission_rate
    exit_comm = qty * exit_price * exit_rate
    total_comm = trade["entry_comm"] + exit_comm
    pnl = (exit_price - trade["entry_price"]) * qty * d - total_comm

    portfolio.balance += qty * exit_price - exit_comm
    portfolio.balance = max(portfolio.balance, 0.0)
    portfolio.peak_balance = max(portfolio.peak_balance, portfolio.balance)
    dd = (
        (portfolio.peak_balance - portfolio.balance) / portfolio.peak_balance
        if portfolio.peak_balance > 0
        else 0.0
    )
    portfolio.max_drawdown = max(portfolio.max_drawdown, dd)

    pos_usdt = trade["position_usdt"]
    ret = pnl / pos_usdt if pos_usdt > 0 else 0.0
    portfolio.trade_returns.append(ret)
    portfolio.closed_trades.append(
        {
            "symbol": state.symbol,
            "entry_i": trade["entry_i"],
            "exit_i": i,
            "action": trade["action"],
            "entry_price": trade["entry_price"],
            "exit_price": exit_price,
            "pnl": round(pnl, 6),
            "return_pct": round(ret, 6),
            "closed_by": "tp" if tp_hit else "sl",
        }
    )
    state.open_trade = None


def _advance_symbol(
    state: _SymbolState,
    target_idx: int,
    portfolio: _Portfolio,
    commission_rate: float,
    maker_commission_rate: float,
    timeframe: str,
) -> None:
    """Прогоняет управление открытой позицией по каждому бару от
    state.next_idx до target_idx включительно (без генерации новых сигналов —
    та часть только на сэмплированных чекпоинтах, см. main())."""
    for i in range(state.next_idx, target_idx + 1):
        if state.open_trade is not None:
            _resolve_exit(state, i, portfolio, commission_rate, maker_commission_rate, timeframe)
    state.next_idx = target_idx + 1


def _force_close_tail(
    state: _SymbolState,
    portfolio: _Portfolio,
    commission_rate: float,
    timeframe: str,
) -> None:
    """Принудительно закрывает висящую позицию в конце данных (как хвост
    run_strategy())."""
    if state.open_trade is None:
        return
    trade = state.open_trade
    d = trade["direction"]
    df = state.df
    mid_close = float(df["close"].iloc[-1])
    close_impact = _ac_impact(df.iloc[-50:], trade["position_usdt"], timeframe=timeframe)
    exit_price = mid_close * (1.0 - close_impact) if d == 1 else mid_close * (1.0 + close_impact)
    qty = trade["quantity"]
    exit_comm = qty * exit_price * commission_rate
    total_comm = trade["entry_comm"] + exit_comm
    pnl = (exit_price - trade["entry_price"]) * qty * d - total_comm

    portfolio.balance += qty * exit_price - exit_comm
    portfolio.balance = max(portfolio.balance, 0.0)
    portfolio.peak_balance = max(portfolio.peak_balance, portfolio.balance)
    dd = (
        (portfolio.peak_balance - portfolio.balance) / portfolio.peak_balance
        if portfolio.peak_balance > 0
        else 0.0
    )
    portfolio.max_drawdown = max(portfolio.max_drawdown, dd)

    pos_usdt = trade["position_usdt"]
    ret = pnl / pos_usdt if pos_usdt > 0 else 0.0
    portfolio.trade_returns.append(ret)
    portfolio.closed_trades.append(
        {
            "symbol": state.symbol,
            "entry_i": trade["entry_i"],
            "exit_i": len(df) - 1,
            "action": trade["action"],
            "entry_price": trade["entry_price"],
            "exit_price": exit_price,
            "pnl": round(pnl, 6),
            "return_pct": round(ret, 6),
            "closed_by": "force",
        }
    )
    state.open_trade = None


def _entry_sl_tp(
    action: str, entry_price: float, atr_val: float, rec_sl: float, rec_tp: float
) -> "tuple[float, float]":
    """AI-предложенные SL/TP, если валидны (по правильную сторону от входа),
    иначе ATR-фолбэк — идентичный тому, что использует run_strategy()."""
    if action == "buy" and rec_sl > 0 and rec_tp > 0 and rec_sl < entry_price < rec_tp:
        return rec_sl, rec_tp
    if action == "sell" and rec_sl > 0 and rec_tp > 0 and rec_tp < entry_price < rec_sl:
        return rec_sl, rec_tp

    sl_dist = _ATR_MULT * atr_val if atr_val > 0 else entry_price * 0.02
    if action == "buy":
        return entry_price - sl_dist, entry_price + _RR * sl_dist
    return entry_price + sl_dist, entry_price - _RR * sl_dist


async def _run_ai_round(
    ai: AIAnalyzer,
    scanner: MarketScanner,
    states: Dict[str, _SymbolState],
    checkpoint_idx: Dict[str, int],
    portfolio: _Portfolio,
    commission_rate: float,
    risk_per_trade: float,
    timeframe: str,
) -> int:
    """Один сэмплированный цикл: догоняет управление позициями до чекпоинта
    каждого символа, собирает снэпшоты для "плоских" символов, делает ОДИН
    batch-вызов AI (как в живом боте) и открывает сделки по ответам.

    :return: число символов, реально отправленных в этот AI-вызов (0, если
        батч пуст — значит вызов AI не потребовался).
    """
    snapshots = []
    eligible: Dict[str, int] = {}
    for symbol, idx in checkpoint_idx.items():
        state = states[symbol]
        if idx + 1 >= len(state.df):
            continue
        if state.open_trade is None:
            window = state.df.iloc[: idx + 1]
            snap = scanner.build_snapshot(symbol, window, news_sentiment=0.0, headlines=[])
            if snap:
                snapshots.append(snap)
                eligible[symbol] = idx

    if not snapshots:
        return 0

    recs = await ai.analyze(snapshots, portfolio.balance)
    rec_map = {r["symbol"]: r for r in recs if r.get("symbol") in eligible}

    for symbol, idx in eligible.items():
        rec = rec_map.get(symbol)
        if rec is None or rec.get("action") not in ("buy", "sell"):
            continue
        state = states[symbol]
        action = rec["action"]
        next_open = float(state.df.iloc[idx + 1]["open"])
        window = state.df.iloc[: idx + 1]
        atr_val = float(window["atr"].iloc[-1]) if "atr" in window.columns else 0.0

        impact = _ac_impact(window, portfolio.balance * risk_per_trade, timeframe=timeframe)
        entry_price = next_open * (1.0 + impact) if action == "buy" else next_open * (1.0 - impact)
        stop_loss, take_profit = _entry_sl_tp(
            action,
            entry_price,
            atr_val,
            float(rec.get("stop_loss") or 0.0),
            float(rec.get("take_profit") or 0.0),
        )
        direction = 1 if action == "buy" else -1

        position_usdt = portfolio.balance * risk_per_trade
        quantity = position_usdt / entry_price
        entry_comm = quantity * entry_price * commission_rate
        portfolio.balance -= position_usdt + entry_comm

        state.open_trade = {
            "action": action,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "direction": direction,
            "quantity": quantity,
            "position_usdt": position_usdt,
            "entry_comm": entry_comm,
            "entry_i": idx,
        }
        # Управление начинается с бара входа (idx+1) включительно — так же,
        # как run_strategy() начинает проверять SL/TP с того же бара, где
        # был исполнен вход.
        state.next_idx = idx + 1

    return len(snapshots)


def _build_checkpoints(states: Dict[str, _SymbolState], n_samples: int) -> List[Dict[str, int]]:
    """Строит список чекпоинтов: равномерно распределённые по времени точки
    у самого короткого по истории символа, спроецированные на бар-индексы
    остальных символов через ближайший бар "не позже" этой временной метки
    (без look-ahead)."""
    ref_symbol = min(states, key=lambda s: len(states[s].df))
    ref_df = states[ref_symbol].df
    lo, hi = _WARMUP, len(ref_df) - 2
    if hi <= lo:
        return []
    n = min(n_samples, hi - lo + 1)
    ref_positions = sorted(set(np.linspace(lo, hi, num=n, dtype=int).tolist()))

    checkpoints: List[Dict[str, int]] = []
    for pos in ref_positions:
        ts = ref_df.iloc[pos]["timestamp"]
        round_idx: Dict[str, int] = {}
        for symbol, state in states.items():
            arr = state.df["timestamp"].values
            j = int(np.searchsorted(arr, ts, side="right")) - 1
            if j >= _WARMUP and j + 1 < len(state.df):
                round_idx[symbol] = j
        if round_idx:
            checkpoints.append(round_idx)
    return checkpoints


async def run_isolation_backtest(
    symbols: List[str],
    timeframe: str,
    months: int,
    n_samples: int,
) -> Optional[_Portfolio]:
    print(f"[ai-isolation] Загрузка данных: {symbols} {timeframe} {months}м...")
    data: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        df = await load_data(symbol, timeframe, months)
        if df.empty or len(df) < _WARMUP + 2:
            print(f"[ai-isolation] Пропуск {symbol}: недостаточно данных")
            continue
        data[symbol] = df
    if not data:
        print("ОШИБКА: ни один символ не набрал достаточно данных")
        return None

    states = {s: _SymbolState(symbol=s, df=df, next_idx=_WARMUP) for s, df in data.items()}
    checkpoints = _build_checkpoints(states, n_samples)
    print(f"[ai-isolation] {len(checkpoints)} сэмплированных циклов вызова AI")
    if not checkpoints:
        print("ОШИБКА: недостаточно общей истории для построения чекпоинтов")
        return None

    ai = AIAnalyzer()
    if not ai.enabled:
        print("ОШИБКА: ни один AI-провайдер не сконфигурирован (см. .env)")
        return None
    scanner = MarketScanner(api=None, data_loader=None)  # build_snapshot() чист от self.api/data_loader
    portfolio = _Portfolio(balance=Config.INITIAL_BALANCE)
    commission_rate = Config.COMMISSION_RATE
    maker_commission_rate = 0.0002
    risk_per_trade = Config.RISK_PER_TRADE

    ai_calls = 0
    for round_n, checkpoint_idx in enumerate(checkpoints, 1):
        for symbol, idx in checkpoint_idx.items():
            _advance_symbol(
                states[symbol], idx, portfolio, commission_rate, maker_commission_rate, timeframe
            )
        n_sent = await _run_ai_round(
            ai,
            scanner,
            states,
            checkpoint_idx,
            portfolio,
            commission_rate,
            risk_per_trade,
            timeframe,
        )
        if n_sent:
            ai_calls += 1
        if round_n % 10 == 0 or round_n == len(checkpoints):
            print(
                f"[ai-isolation] цикл {round_n}/{len(checkpoints)} — "
                f"AI-вызовов: {ai_calls}, сделок закрыто: {len(portfolio.closed_trades)}"
            )

    # Довести управление позициями до конца данных + принудительно закрыть хвост
    for symbol, state in states.items():
        _advance_symbol(
            state, len(state.df) - 1, portfolio, commission_rate, maker_commission_rate, timeframe
        )
        _force_close_tail(state, portfolio, commission_rate, timeframe)

    return portfolio


def _print_report(portfolio: _Portfolio, symbols: List[str], timeframe: str, months: int) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"AI (LLM) SIGNAL ISOLATION — {', '.join(symbols)} — {timeframe} — {months}m")
    print(sep)
    total = len(portfolio.closed_trades)
    if total == 0:
        print("Сделок не открыто ни одной — AI ни разу не дал buy/sell выше порога уверенности")
        print(sep)
        return

    wins = [t for t in portfolio.closed_trades if t["pnl"] > 0]
    win_rate = len(wins) / total
    avg_ret = sum(t["return_pct"] for t in portfolio.closed_trades) / total
    sharpe = _sharpe(portfolio.trade_returns)
    total_return = (portfolio.balance - Config.INITIAL_BALANCE) / Config.INITIAL_BALANCE

    print(f"Всего сделок: {total} | WinRate: {win_rate*100:.1f}% | Sharpe: {sharpe:.2f}")
    print(f"Avg return/trade: {avg_ret*100:+.2f}% | Total return: {total_return*100:+.1f}%")
    print(f"Max drawdown: {portfolio.max_drawdown*100:.1f}%")
    print("По символам:")
    for symbol in symbols:
        sym_trades = [t for t in portfolio.closed_trades if t["symbol"] == symbol]
        if sym_trades:
            sym_wr = sum(1 for t in sym_trades if t["pnl"] > 0) / len(sym_trades)
            print(f"  {symbol}: {len(sym_trades)} сделок, winrate {sym_wr*100:.1f}%")
    print(sep)

    tester = AlphaTester()
    alpha = tester.test(portfolio.trade_returns, name="ai_llm_isolated")
    print("ЗНАЧИМОСТЬ АЛЬФЫ AI-СЛОЯ (бутстрап Sharpe + Wilcoxon)")
    print(sep)
    print(alpha.summary())
    print(sep)
    print(
        "Напоминание: базовые TA-стратегии на этих же символах/таймфрейме "
        "показали train Sharpe<=0 во всех окнах (см. project_senior_quant_roadmap "
        "п.7) — если AI тоже не значим, edge в этом слое пока не подтверждён."
    )
    print(sep)


def _save_json(portfolio: _Portfolio, symbols: List[str], timeframe: str, months: int) -> None:
    output_path = "data/ai_isolation_backtest_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "symbols": symbols,
                "timeframe": timeframe,
                "months": months,
                "total_trades": len(portfolio.closed_trades),
                "trade_returns": portfolio.trade_returns,
                "final_balance": round(portfolio.balance, 2),
                "max_drawdown_pct": round(portfolio.max_drawdown, 6),
                "trades": portfolio.closed_trades,
            },
            f,
            indent=2,
        )
    print(f"[ai-isolation] Результаты сохранены в {output_path}")


async def main() -> None:
    symbols = [s.strip() for s in os.getenv("AIB_SYMBOLS", _DEFAULT_SYMBOLS).split(",") if s.strip()]
    timeframe = os.getenv("AIB_TIMEFRAME", "15m")
    months = int(os.getenv("AIB_MONTHS", "6"))
    n_samples = int(os.getenv("AIB_SAMPLES", "150"))

    portfolio = await run_isolation_backtest(symbols, timeframe, months, n_samples)
    if portfolio is None:
        return
    _print_report(portfolio, symbols, timeframe, months)
    _save_json(portfolio, symbols, timeframe, months)


if __name__ == "__main__":
    asyncio.run(main())
