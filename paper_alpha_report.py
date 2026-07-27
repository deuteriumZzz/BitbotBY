"""
paper_alpha_report.py — статистическая значимость Sharpe на РЕАЛЬНЫХ
paper/live сделках (в отличие от backtest.py, который тестирует
гипотетические сделки на исторических данных).

Отвечает на вопрос "отличимо ли текущее живое исполнение стратегии от
шума", переиспользуя AlphaTester (src/alpha_tester.py) — тот же bootstrap
Sharpe + Wilcoxon + Bonferroni-поправку, что уже применяется в backtest.py,
только к pnl_pct из trades_paper.db/trades_live.db вместо симулированных
сделок.

Использование:
    python3 paper_alpha_report.py
    python3 paper_alpha_report.py --db data/trades_live.db
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from typing import Dict, List, Tuple

from src.alpha_tester import AlphaTester


def _default_db_path() -> str:
    name = "trades_paper.db" if os.getenv("PAPER_TRADING", "").lower() == "true" else "trades_live.db"
    return os.path.join("data", name)


def load_closed_returns(db_path: str) -> List[Tuple[str, float]]:
    """Читает (strategy, pnl_pct) всех закрытых сделок из SQLite.

    :param db_path: Путь к trades_paper.db или trades_live.db.
    :return: Список (strategy, pnl_pct); пусто, если файла нет или сделок нет.
    """
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT strategy, pnl_pct FROM trades "
            "WHERE status='closed' AND pnl_pct IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    return [(strategy, float(pnl_pct)) for strategy, pnl_pct in rows]


def group_returns_by_strategy(
    rows: List[Tuple[str, float]],
) -> Dict[str, List[float]]:
    """Группирует доходности сделок по стратегии.

    :param rows: Список (strategy, pnl_pct), например из load_closed_returns.
    :return: {strategy: [pnl_pct, ...]}, сохраняя порядок первого появления.
    """
    grouped: Dict[str, List[float]] = {}
    for strategy, pnl_pct in rows:
        grouped.setdefault(strategy, []).append(pnl_pct)
    return grouped


def build_report(returns_by_strategy: Dict[str, List[float]]) -> str:
    """Строит текстовый отчёт значимости альфы по каждой стратегии + сводно.

    Зеркалит логику Bonferroni-секции backtest.py:main() — тот же
    порог 0.05/N, где N = число проверяемых стратегий, чтобы не выдать
    ложноположительный результат от простого перебора нескольких стратегий
    на одной и той же короткой истории сделок.

    :param returns_by_strategy: {strategy: [pnl_pct, ...]}.
    :return: Многострочный отчёт для печати в консоль.
    """
    tester = AlphaTester()
    sep = "=" * 68
    lines = [
        sep,
        "ЗНАЧИМОСТЬ АЛЬФЫ НА РЕАЛЬНЫХ PAPER/LIVE СДЕЛКАХ",
        "(bootstrap Sharpe + Wilcoxon, та же методология что в backtest.py)",
        sep,
    ]

    n_tests = len(returns_by_strategy)
    if n_tests == 0:
        lines.append("Нет закрытых сделок — отчёт невозможен.")
        lines.append(sep)
        return "\n".join(lines)

    bonferroni_alpha = 0.05 / n_tests
    sig_count = 0
    sig_count_corrected = 0
    all_returns: List[float] = []
    for strategy, returns in returns_by_strategy.items():
        all_returns.extend(returns)
        result = tester.test(returns, name=strategy)
        lines.append(result.summary())
        if result.is_significant:
            sig_count += 1
        if result.sharpe_pvalue < bonferroni_alpha:
            sig_count_corrected += 1

    lines.append(sep)
    lines.append(
        f"{sig_count}/{n_tests} стратегий значимы при p<5% "
        "(БЕЗ поправки на множественное тестирование)"
    )
    lines.append(
        f"{sig_count_corrected}/{n_tests} проходят после Bonferroni-поправки "
        f"(p<{bonferroni_alpha:.4f}, скорректировано на {n_tests} тестов)"
    )
    if sig_count_corrected < sig_count:
        lines.append(
            f"  ⚠ {sig_count - sig_count_corrected} стратегия(й) значимы только "
            f"без поправки — вероятно, ложные срабатывания от перебора {n_tests} "
            "вариантов на одной и той же короткой истории, а не реальный edge"
        )
    lines.append(sep)

    lines.append("ПОРТФЕЛЬ ЦЕЛИКОМ (все стратегии вместе, без Bonferroni — один тест):")
    lines.append(tester.test(all_returns, name="ALL STRATEGIES").summary())
    lines.append(sep)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default=_default_db_path(),
        help="Путь к SQLite базе сделок (по умолчанию — trades_paper.db "
        "если PAPER_TRADING=true, иначе trades_live.db)",
    )
    args = parser.parse_args()

    rows = load_closed_returns(args.db)
    if not rows:
        print(f"[paper_alpha_report] Нет закрытых сделок в {args.db}")
        return

    grouped = group_returns_by_strategy(rows)
    print(f"[paper_alpha_report] {len(rows)} закрытых сделок, "
          f"{len(grouped)} стратегия(й), источник: {args.db}\n")
    print(build_report(grouped))


if __name__ == "__main__":
    main()
