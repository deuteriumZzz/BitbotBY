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
    name = (
        "trades_paper.db"
        if os.getenv("PAPER_TRADING", "").lower() == "true"
        else "trades_live.db"
    )
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


def load_closed_returns_by_profile(db_path: str) -> List[Tuple[str, float]]:
    """Читает (profile, pnl_pct) закрытых сделок, читая profile из context_flags.

    SignalCombiner тегирует активный рыночный профиль (bluechip/altcoin/meme)
    в context_flags["profile"] на момент входа (см. src/signal_combiner.py).
    Это даёт реализованный PnL/Sharpe за периоды активного профиля вместо
    голого счётчика срабатываний "Season check: ... signal=meme"
    (п.6 роадмапа). Сделки без тега (до появления этой инструментации, или
    если профиль не был определён) попадают в бакет "unknown" — не
    отбрасываются, чтобы было видно, сколько истории ещё не размечено.

    :param db_path: Путь к trades_paper.db или trades_live.db.
    :return: Список (profile, pnl_pct); пусто, если файла нет или сделок нет.
    """
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT context_flags, pnl_pct FROM trades "
            "WHERE status='closed' AND pnl_pct IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    result: List[Tuple[str, float]] = []
    for context_flags, pnl_pct in rows:
        profile = "unknown"
        if context_flags:
            try:
                flags = json.loads(context_flags)
                profile = flags.get("profile") or "unknown"
            except (json.JSONDecodeError, AttributeError):
                pass
        result.append((profile, float(pnl_pct)))
    return result


def load_closed_returns_by_market_type(db_path: str) -> List[Tuple[str, float]]:
    """Читает (market_type, pnl_pct) закрытых сделок из context_flags.

    SignalCombiner тегирует активный тип рынка (spot/linear) в
    context_flags["market_type"] на момент входа (см. src/signal_combiner.py)
    — та же логика, что и load_closed_returns_by_profile, но для spot/futures:
    механика сделки (плечо, шорт, funding, ликвидация) настолько разная, что
    смешивать их в одну win_rate/EV-выборку так же некорректно, как смешивать
    рыночные профили. Сделки без тега — в "unknown".

    :param db_path: Путь к trades_paper.db или trades_live.db.
    :return: Список (market_type, pnl_pct); пусто, если файла/сделок нет.
    """
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT context_flags, pnl_pct FROM trades "
            "WHERE status='closed' AND pnl_pct IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    result: List[Tuple[str, float]] = []
    for context_flags, pnl_pct in rows:
        market_type = "unknown"
        if context_flags:
            try:
                flags = json.loads(context_flags)
                market_type = flags.get("market_type") or "unknown"
            except (json.JSONDecodeError, AttributeError):
                pass
        result.append((market_type, float(pnl_pct)))
    return result


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


def group_returns_by_profile(rows: List[Tuple[str, float]]) -> Dict[str, List[float]]:
    """Группирует доходности сделок по рыночному профилю на момент входа.

    :param rows: Список (profile, pnl_pct), например из load_closed_returns_by_profile.
    :return: {profile: [pnl_pct, ...]}, сохраняя порядок первого появления.
    """
    grouped: Dict[str, List[float]] = {}
    for profile, pnl_pct in rows:
        grouped.setdefault(profile, []).append(pnl_pct)
    return grouped


def group_returns_by_market_type(
    rows: List[Tuple[str, float]],
) -> Dict[str, List[float]]:
    """Группирует доходности сделок по типу рынка (spot/linear) на момент входа.

    :param rows: Список (market_type, pnl_pct), напр. из load_closed_returns_by_market_type.
    :return: {market_type: [pnl_pct, ...]}, сохраняя порядок первого появления.
    """
    grouped: Dict[str, List[float]] = {}
    for market_type, pnl_pct in rows:
        grouped.setdefault(market_type, []).append(pnl_pct)
    return grouped


def _significance_section(
    returns_by_group: Dict[str, List[float]], group_noun: str
) -> List[str]:
    """Секция значимости: per-group AlphaTester.test() + Bonferroni-поправка.

    Общая логика для группировки по стратегии (build_report) и по рыночному
    профилю (build_profile_report) — идентична, отличается только
    существительным в строке счёта. Зеркалит Bonferroni-секцию
    backtest.py:main() — порог 0.05/N, где N = число групп, чтобы не выдать
    ложноположительный результат от простого перебора нескольких групп на
    одной и той же короткой истории сделок.

    :param returns_by_group: {group_name: [pnl_pct, ...]}, непусто.
    :param group_noun: Существительное для строк счёта ("стратегий"/"профилей").
    :return: Строки отчёта (без заголовка и без портфельной секции).
    """
    tester = AlphaTester()
    sep = "=" * 68
    lines: List[str] = []

    n_tests = len(returns_by_group)
    bonferroni_alpha = 0.05 / n_tests
    sig_count = 0
    sig_count_corrected = 0
    for name, returns in returns_by_group.items():
        result = tester.test(returns, name=name)
        lines.append(result.summary())
        if result.is_significant:
            sig_count += 1
        if result.sharpe_pvalue < bonferroni_alpha:
            sig_count_corrected += 1

    lines.append(sep)
    lines.append(
        f"{sig_count}/{n_tests} {group_noun} значимы при p<5% "
        "(БЕЗ поправки на множественное тестирование)"
    )
    lines.append(
        f"{sig_count_corrected}/{n_tests} проходят после Bonferroni-поправки "
        f"(p<{bonferroni_alpha:.4f}, скорректировано на {n_tests} тестов)"
    )
    if sig_count_corrected < sig_count:
        lines.append(
            f"  ⚠ {sig_count - sig_count_corrected} {group_noun} значимы только "
            f"без поправки — вероятно, ложные срабатывания от перебора {n_tests} "
            "вариантов на одной и той же короткой истории, а не реальный edge"
        )
    lines.append(sep)
    return lines


def build_report(returns_by_strategy: Dict[str, List[float]]) -> str:
    """Строит текстовый отчёт значимости альфы по каждой стратегии + сводно.

    :param returns_by_strategy: {strategy: [pnl_pct, ...]}.
    :return: Многострочный отчёт для печати в консоль.
    """
    sep = "=" * 68
    lines = [
        sep,
        "ЗНАЧИМОСТЬ АЛЬФЫ НА РЕАЛЬНЫХ PAPER/LIVE СДЕЛКАХ",
        "(bootstrap Sharpe + Wilcoxon, та же методология что в backtest.py)",
        sep,
    ]
    if not returns_by_strategy:
        lines.append("Нет закрытых сделок — отчёт невозможен.")
        lines.append(sep)
        return "\n".join(lines)

    lines.extend(_significance_section(returns_by_strategy, group_noun="стратегий"))
    all_returns = [r for rs in returns_by_strategy.values() for r in rs]
    lines.append("ПОРТФЕЛЬ ЦЕЛИКОМ (все стратегии вместе, без Bonferroni — один тест):")
    lines.append(AlphaTester().test(all_returns, name="ALL STRATEGIES").summary())
    lines.append(sep)
    return "\n".join(lines)


def build_profile_report(returns_by_profile: Dict[str, List[float]]) -> str:
    """Строит отчёт значимости альфы по рыночному профилю на момент входа.

    П.6 роадмапа: реализованный PnL/Sharpe (с доверительным интервалом) за
    периоды активного профиля bluechip/altcoin/meme — вместо голого счётчика
    срабатываний "Season check: index=... signal=meme" в trading_bot.py.

    :param returns_by_profile: {profile: [pnl_pct, ...]}.
    :return: Многострочный отчёт для печати в консоль.
    """
    sep = "=" * 68
    lines = [
        sep,
        "ЗНАЧИМОСТЬ АЛЬФЫ ПО РЫНОЧНОМУ ПРОФИЛЮ (bluechip/altcoin/meme)",
        "(та же методология, группировка по context_flags['profile'])",
        sep,
    ]
    if not returns_by_profile:
        lines.append("Нет закрытых сделок — отчёт невозможен.")
        lines.append(sep)
        return "\n".join(lines)

    lines.extend(_significance_section(returns_by_profile, group_noun="профилей"))
    all_returns = [r for rs in returns_by_profile.values() for r in rs]
    lines.append("ПОРТФЕЛЬ ЦЕЛИКОМ (все профили вместе, без Bonferroni — один тест):")
    lines.append(AlphaTester().test(all_returns, name="ALL PROFILES").summary())
    lines.append(sep)
    return "\n".join(lines)


def build_market_type_report(returns_by_market_type: Dict[str, List[float]]) -> str:
    """Строит отчёт значимости альфы по типу рынка (spot/futures) на момент входа.

    Та же методология, что build_profile_report — реализованный PnL/Sharpe
    (с доверительным интервалом) отдельно за периоды spot и linear, вместо
    одной смешанной выборки, где плечо/шорт/funding одного режима искажают
    статистику другого.

    :param returns_by_market_type: {market_type: [pnl_pct, ...]}.
    :return: Многострочный отчёт для печати в консоль.
    """
    sep = "=" * 68
    lines = [
        sep,
        "ЗНАЧИМОСТЬ АЛЬФЫ ПО ТИПУ РЫНКА (spot/futures)",
        "(та же методология, группировка по context_flags['market_type'])",
        sep,
    ]
    if not returns_by_market_type:
        lines.append("Нет закрытых сделок — отчёт невозможен.")
        lines.append(sep)
        return "\n".join(lines)

    lines.extend(
        _significance_section(returns_by_market_type, group_noun="типов рынка")
    )
    all_returns = [r for rs in returns_by_market_type.values() for r in rs]
    lines.append("ПОРТФЕЛЬ ЦЕЛИКОМ (spot+futures вместе, без Bonferroni — один тест):")
    lines.append(AlphaTester().test(all_returns, name="ALL MARKET TYPES").summary())
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
    print(
        f"[paper_alpha_report] {len(rows)} закрытых сделок, "
        f"{len(grouped)} стратегия(й), источник: {args.db}\n"
    )
    print(build_report(grouped))

    grouped_by_profile = group_returns_by_profile(
        load_closed_returns_by_profile(args.db)
    )
    print()
    print(build_profile_report(grouped_by_profile))

    grouped_by_market_type = group_returns_by_market_type(
        load_closed_returns_by_market_type(args.db)
    )
    print()
    print(build_market_type_report(grouped_by_market_type))


if __name__ == "__main__":
    main()
