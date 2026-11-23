"""Теневой Risk Gate (Phase 9, этап «тень»): что сказал бы единый гейт о фактическом входе.

Вызывается ПОСЛЕ того, как исполнитель уже принял решение, и ничего не меняет:
результат только пишется в Decision Ledger (`risk_gate_shadow`) и затем сравнивается
с тем, что сделала действующая защита. Позиции «до входа» берутся из снимка,
сделанного до `execute`, а входом считается фактически открытая позиция.

Ограничения входов (честно, чтобы тень не вводила в заблуждение):
  * бета к BTC и CVaR актива считаются по окну `CorrelationFilter` (≤ 50 свечей, лог-
    доходности за свечу, не за срок удержания); при нехватке данных берутся
    предложенные значения по умолчанию (beta=1.0, cvar=0.05) — это отмечено в
    `inputs.source`. Стресс-лимит поэтому почти не срабатывает: это нормально,
    пока нет динамического CVaR (MASTER_SPEC §11 п.18);
  * капитал для экспозиций = свободный баланс + залог открытых позиций (paper) или
    кэш equity (live); дневной убыток и пик берутся в системе отсчёта бота, поэтому в
    paper дневной убыток по такому капиталу может недосрабатывать.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.risk.metrics import beta as calc_beta
from src.risk.metrics import hist_var_cvar
from src.risk.risk_gate import GateLimits, Portfolio, Position, Proposal, evaluate

DEFAULT_BETA = 1.0
DEFAULT_CVAR = 0.05
MARKET = "BTC/USDT"
MIN_RETURNS = 10


def _sign(side: str) -> float:
    return 1.0 if side in ("buy", "long") else -1.0


def asset_risk(
    symbol: str, returns: Callable[[str], Optional[np.ndarray]]
) -> Dict[str, Any]:
    """(beta, cvar, источник) по доходностям окна; без данных — значения по умолчанию."""
    r = returns(symbol)
    if r is None or len(r) < MIN_RETURNS:
        return {"beta": DEFAULT_BETA, "cvar": DEFAULT_CVAR, "source": "default"}
    cvar = hist_var_cvar(r, 0.95)[1]
    m = returns(MARKET)
    if symbol == MARKET:
        b = 1.0
    elif m is not None and len(m) >= MIN_RETURNS:
        n = min(len(r), len(m))
        b = calc_beta(r[-n:], m[-n:])
    else:
        return {"beta": DEFAULT_BETA, "cvar": cvar, "source": "computed_cvar_default_beta"}
    return {"beta": b, "cvar": cvar, "source": "computed"}


def shadow_verdict(
    *,
    symbol: str,
    side: str,
    qty: float,
    entry: float,
    positions_before: Dict[str, Dict[str, Any]],
    equity: float,
    peak_equity: float,
    day_start_equity: float,
    consecutive_losses: int,
    returns: Callable[[str], Optional[np.ndarray]],
    max_corr: Callable[[str, List[str]], float],
    limits: GateLimits,
    kill_switch: bool = False,
    ood: bool = False,
) -> Dict[str, Any]:
    """Вердикт гейта по фактическому входу. Чистая функция: всё состояние приходит аргументами."""
    held: List[Position] = []
    sources = set()
    for sym, p in positions_before.items():
        risk = asset_risk(sym, returns)
        sources.add(risk["source"])
        notional = _sign(p.get("side", "buy")) * float(p.get("qty") or 0) * float(p.get("entry") or 0)
        held.append(Position(sym, notional, risk["beta"], risk["cvar"]))
    new = asset_risk(symbol, returns)
    sources.add(new["source"])
    notional = _sign(side) * float(qty) * float(entry)
    open_syms = [s for s in positions_before if s != symbol]
    proposal = Proposal(
        symbol=symbol,
        notional=notional,
        beta=new["beta"],
        cvar=new["cvar"],
        max_corr_with_open=max_corr(symbol, open_syms) if open_syms else 0.0,
        ood=ood,
    )
    portfolio = Portfolio(
        equity=equity,
        peak_equity=max(peak_equity, equity),
        day_start_equity=day_start_equity,
        positions=held,
        consecutive_losses=consecutive_losses,
        kill_switch=kill_switch,
    )
    v = evaluate(proposal, portfolio, limits)
    return {
        "proposal": {
            "side": side,
            "notional": round(notional, 4),
            "beta": round(new["beta"], 4),
            "cvar": round(new["cvar"], 6),
            "max_corr": round(proposal.max_corr_with_open, 4),
        },
        "inputs": {
            "equity": round(equity, 4),
            "peak": round(portfolio.peak_equity, 4),
            "day_start": round(day_start_equity, 4),
            "consecutive_losses": consecutive_losses,
            "n_open": len(held),
            "source": "computed" if sources == {"computed"} else "mixed_or_default",
        },
        "verdict": v.to_dict(),
        "would_block": v.action != "allow",
    }
