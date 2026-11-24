"""Продвинутые риск-инструменты (Phase 9, реестр идей №18–22): чистая математика, numpy.

* `dynamic_cvar` — CVaR по фильтрованной исторической симуляции: доходности делятся на EWMA-
  волатильность, хвост берётся из стандартизированных остатков и масштабируется ТЕКУЩЕЙ
  волатильностью. Статическое окно в спокойном режиме занижает риск, этот вариант реагирует на
  рост волатильности сразу (тяжёлые хвосты остатков сохраняются).
* `stress_btc_shock` — стресс общей BTC-беты: все активы падают на beta·шок (корреляция → 1) +
  неблагоприятный идиосинкратический компонент.
* Black–Litterman: апостериорные ожидания из равновесия и СТРУКТУРИРОВАННЫХ взглядов (LLM даёт
  проверяемый JSON, не текст → вес). Это модель ожиданий/весов, а не замена Kelly/CVaR; в бою
  не используется (исследование, нужно доказать OOS-улучшение после издержек).
* `kelly_from_meta` — размер по Kelly от КАЛИБРОВАННОЙ вероятности Meta-Model: только при
  `usable=True`, с поправкой на неопределённость оценки и потолком.
Riskfolio-Lib не подключён: зависит от scipy/cvxpy (локально scipy сломан), а нужные формулы здесь.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# ── Динамический CVaR ─────────────────────────────────────────────────────────
def ewma_sigma(returns: Sequence[float], lam: float = 0.94, warmup: int = 20) -> np.ndarray:
    """Условная σ_t по RiskMetrics: σ_t² = λσ_{t-1}² + (1−λ)r_{t-1}²; старт — выборочная σ первых `warmup`."""
    r = np.asarray(returns, float)
    s2 = np.empty(len(r) + 1)
    s2[0] = np.var(r[:warmup]) if len(r) >= warmup else np.var(r)
    for t in range(len(r)):
        s2[t + 1] = lam * s2[t] + (1 - lam) * r[t] ** 2
    return np.sqrt(np.maximum(s2, 1e-18))  # [t] — прогноз σ для шага t; последний — на следующий шаг


def dynamic_cvar(returns: Sequence[float], alpha: float = 0.95, lam: float = 0.94) -> Dict[str, float]:
    r = np.asarray(returns, float)
    if len(r) < 30:
        raise ValueError("нужно ≥ 30 наблюдений")
    sig = ewma_sigma(r, lam)
    z = r / sig[:-1]  # стандартизированные остатки
    losses = -z
    var_z = float(np.quantile(losses, alpha))
    tail = losses[losses >= var_z]
    nxt = float(sig[-1])
    return {
        "var": max(var_z, 0.0) * nxt,
        "cvar": max(float(tail.mean()), 0.0) * nxt,
        "sigma_next": nxt,
        "sigma_hist": float(np.std(r)),
    }


# ── Стресс общей беты ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class StressPosition:
    symbol: str
    notional: float  # знаковая (минус — шорт)
    beta: float = 1.0
    idio_vol: float = 0.0  # идиосинкратическая σ за горизонт стресса (доля)


def stress_btc_shock(
    positions: Sequence[StressPosition],
    equity: float,
    shock: float = -0.20,
    idio_mult: float = 1.0,
    max_loss: Optional[float] = None,
) -> Dict[str, Any]:
    """Потеря портфеля при шоке BTC на `shock` (корреляция → 1) + неблагоприятная идио-часть."""
    if equity <= 0:
        raise ValueError("equity > 0")
    systematic = sum(p.notional * p.beta * shock for p in positions)
    idio = sum(abs(p.notional) * p.idio_vol * idio_mult for p in positions)  # всегда против нас
    pnl = systematic - idio
    loss_pct = max(0.0, -pnl / equity)
    return {
        "pnl": pnl,
        "systematic": systematic,
        "idio": -idio,
        "loss_pct": loss_pct,
        "breach": (loss_pct > max_loss) if max_loss is not None else None,
    }


# ── Black–Litterman ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class View:
    assets: Dict[str, float]  # строка матрицы P: {символ: вес}, напр. {"ETH": 1} или {"ETH": 1, "BTC": -1}
    expected_return: float  # q: ожидаемая доходность портфеля взгляда за горизонт
    confidence: float  # (0, 1]


def _finite_number(x: Any) -> bool:
    return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x)


def validate_views(raw: Any, universe: Sequence[str], max_abs_q: float = 0.5) -> List[View]:
    """Строгая проверка структурированных взглядов (в т.ч. от LLM): неизвестный актив/NaN/уверенность вне (0,1] — отказ."""
    if not isinstance(raw, list):
        raise ValueError("взгляды должны быть списком")
    known, out = set(universe), []
    for i, v in enumerate(raw):
        if not isinstance(v, dict) or set(v) - {"assets", "expected_return", "confidence"}:
            raise ValueError(f"взгляд {i}: неизвестные поля или не объект")
        assets, q, c = v.get("assets"), v.get("expected_return"), v.get("confidence")
        if not isinstance(assets, dict) or not assets or not set(assets) <= known:
            raise ValueError(f"взгляд {i}: активы вне universe")
        if not all(_finite_number(w) for w in assets.values()):
            raise ValueError(f"взгляд {i}: веса должны быть конечными числами")
        if not _finite_number(q) or not _finite_number(c):
            raise ValueError(f"взгляд {i}: expected_return и confidence — конечные числа")
        if abs(q) > max_abs_q or not 0 < c <= 1:
            raise ValueError(f"взгляд {i}: |q| ≤ {max_abs_q}, 0 < confidence ≤ 1")
        out.append(View({k: float(w) for k, w in assets.items()}, float(q), float(c)))
    return out


def black_litterman(
    universe: Sequence[str],
    cov: np.ndarray,
    market_weights: Sequence[float],
    views: Sequence[View],
    tau: float = 0.05,
    risk_aversion: float = 2.5,
) -> Dict[str, Any]:
    """Апостериорные ожидаемые доходности и ковариация (формула Бидермана; Ω по уверенности взглядов)."""
    S, w = np.asarray(cov, float), np.asarray(market_weights, float)
    n = len(universe)
    if S.shape != (n, n) or w.shape != (n,):
        raise ValueError("форма cov/market_weights не совпадает с universe")
    pi = risk_aversion * S @ w  # равновесные доходности
    if not views:
        return {"mu": pi, "cov": S, "pi": pi}
    idx = {s: i for i, s in enumerate(universe)}
    P = np.zeros((len(views), n))
    q = np.array([v.expected_return for v in views])
    for k, v in enumerate(views):
        for a, wt in v.assets.items():
            P[k, idx[a]] = wt
    tS = tau * S
    omega = np.diag(
        [((1 - v.confidence) / v.confidence) * float(P[k] @ tS @ P[k]) + 1e-12 for k, v in enumerate(views)]
    )
    inv_tS, inv_om = np.linalg.inv(tS), np.linalg.inv(omega)
    M = np.linalg.inv(inv_tS + P.T @ inv_om @ P)
    mu = M @ (inv_tS @ pi + P.T @ inv_om @ q)
    return {"mu": mu, "cov": S + M, "pi": pi}


def mv_weights(
    mu: np.ndarray, cov: np.ndarray, risk_aversion: float = 2.5, cap: float = 0.3, long_only: bool = True
) -> np.ndarray:
    """w ∝ Σ⁻¹μ/λ с потолком на актив и суммарной экспозицией ≤ 1. Исследовательская функция."""
    w = np.linalg.solve(np.asarray(cov, float), np.asarray(mu, float)) / risk_aversion
    w = np.clip(w, 0.0 if long_only else -cap, cap)
    gross = float(np.abs(w).sum())
    return w / gross if gross > 1.0 else w


# ── Kelly от калиброванной вероятности ────────────────────────────────────────
def kelly_from_meta(
    p: float,
    reward_risk: float,
    *,
    usable: bool,
    n_cal: int,
    fraction: float = 0.5,
    cap: float = 0.20,
    se_shrink: float = 1.0,
) -> Dict[str, Any]:
    """Доля капитала по Kelly от вероятности Meta-Model.

    Только если Meta-Model прошла проверку (`usable`): иначе 0 — вероятность без доказанной
    калибровки в размер не идёт. Неопределённость оценки: p_eff = p − se_shrink·√(p(1−p)/n_cal).
    """
    if not usable:
        return {"fraction": 0.0, "p_eff": None, "reason": "meta_model_not_usable"}
    if reward_risk <= 0 or n_cal <= 0 or not 0.0 <= p <= 1.0:
        return {"fraction": 0.0, "p_eff": None, "reason": "bad_input"}
    p_eff = max(0.0, p - se_shrink * math.sqrt(p * (1 - p) / n_cal))
    full = (p_eff * reward_risk - (1 - p_eff)) / reward_risk
    return {
        "fraction": float(min(max(full * fraction, 0.0), cap)),
        "p_eff": p_eff,
        "reason": "ok" if full > 0 else "no_edge",
    }
