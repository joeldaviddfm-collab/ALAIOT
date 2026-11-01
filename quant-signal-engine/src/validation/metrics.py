"""Métricas de trading: Sharpe, Sortino, Calmar, drawdown, profit factor,
expectativa, Deflated Sharpe Ratio (Bailey & López de Prado) y PBO (CSCV).
Todos los cálculos verificados contra casos conocidos en tests/test_metrics.py.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import norm

PPY = 252  # períodos por año (diario)


def equity_metrics(equity: pd.Series, trades: list | None = None) -> dict:
    equity = pd.Series(equity).astype(float).dropna()
    out: dict = {"n_days": int(len(equity))}
    if len(equity) < 2:
        out.update(sharpe=0.0, sortino=0.0, calmar=0.0, max_drawdown=0.0, cagr=0.0, total_return=0.0)
        return out
    rets = equity.pct_change().dropna()
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    ann = PPY / max(1, len(rets))
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** ann - 1) if equity.iloc[0] > 0 else 0.0
    std = float(rets.std())
    sharpe = float(rets.mean() / std * np.sqrt(PPY)) if std > 0 else 0.0
    downside = rets[rets < 0]
    dd_std = float(np.sqrt((downside**2).mean())) if len(downside) else 0.0
    sortino = float(rets.mean() / dd_std * np.sqrt(PPY)) if dd_std > 0 else 0.0
    running_max = equity.cummax()
    max_dd = float((equity / running_max - 1).min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else float("inf")
    out.update(
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_dd,
        cagr=cagr,
        total_return=total_return,
        volatility=float(std * np.sqrt(PPY)),
        skew=float(rets.skew()) if len(rets) > 2 else 0.0,
        kurt=float(rets.kurt()) if len(rets) > 3 else 0.0,
    )
    if trades:
        pnls = np.array([t.pnl for t in trades], dtype=float)
        rs = np.array([t.r_multiple for t in trades], dtype=float)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        out.update(
            n_trades=int(len(trades)),
            win_rate=float((pnls > 0).mean()) if len(pnls) else 0.0,
            profit_factor=float(wins.sum() / abs(losses.sum()))
            if len(losses) and losses.sum() != 0
            else (float("inf") if len(wins) else 0.0),
            expectancy=float(pnls.mean()) if len(pnls) else 0.0,
            avg_rr=float(np.abs(wins).mean() / np.abs(losses).mean())
            if len(wins) and len(losses) and np.abs(losses).mean() > 0
            else 0.0,
            avg_r_multiple=float(rs.mean()) if len(rs) else 0.0,
            avg_hold_days=float(np.mean([(t.exit_date - t.entry_date).days for t in trades]))
            if trades
            else 0.0,
            avg_win=float(wins.mean()) if len(wins) else 0.0,
            avg_loss=float(losses.mean()) if len(losses) else 0.0,
        )
    return out


def deflated_sharpe(sr: float, skew: float, kurt: float, T: int, n_trials: int = 1) -> tuple[float, float]:
    """Deflated Sharpe Ratio bajo H0 (Bailey, Borak, López de Prado 2014) con
    corrección de Šidák por `n_trials` configuraciones probadas.
    Devuelve (DSR, p_value)."""
    if T <= 1 or not np.isfinite(sr):
        return 0.0, 1.0
    var = (1 - skew * sr + (kurt - 1) / 4.0 * sr**2) / (T - 1)
    if var <= 0:
        return 0.0, 1.0
    ds = sr / np.sqrt(var)
    p_single = float(1 - norm.cdf(ds))
    n = max(1, int(n_trials))
    p = 1 - (1 - p_single) ** n
    return float(ds), float(min(1.0, p))


def pbo_cscv(config_returns: dict[str, pd.Series], n_segments: int = 8) -> float:
    """Probability of Backtest Overfitting con CSCV (combinatorially symmetric CV).

    config_returns: {nombre_config: serie de retornos diarios de la estrategia}.
    Divide el histórico en S segmentos; para cada combinación mitad IS/mitad OOS
    elige la config con mejor Sharpe IS y registra su Sharpe OOS.
    PBO = fracción de combinaciones con Sharpe OOS < 0.
    """
    series = {k: pd.Series(v).dropna() for k, v in config_returns.items()}
    if not series:
        return 1.0
    common: pd.Index | None = None
    for s in series.values():
        common = s.index if common is None else common.intersection(s.index)
    if common is None or len(common) < n_segments * 5:
        return 1.0
    mat = pd.DataFrame({k: s.reindex(common) for k, s in series.items()}).dropna()
    n = len(mat)
    seg_len = n // n_segments
    if seg_len < 5:
        return 1.0
    seg_rets = [mat.iloc[i * seg_len : (i + 1) * seg_len] for i in range(n_segments)]

    def _sharpe(rets: pd.DataFrame) -> pd.Series:
        std = rets.std()
        std = std.replace(0, np.nan)
        return rets.mean() / std * np.sqrt(PPY)

    oos_neg = 0
    total = 0
    half = n_segments // 2
    for combo in itertools.combinations(range(n_segments), half):
        is_idx = list(combo)
        oos_idx = [i for i in range(n_segments) if i not in combo]
        is_rets = pd.concat([seg_rets[i] for i in is_idx])
        oos_rets = pd.concat([seg_rets[i] for i in oos_idx])
        is_sharpe = _sharpe(is_rets).fillna(-np.inf)
        best = is_sharpe.idxmax()
        oos_sharpe = float(_sharpe(oos_rets).get(best, np.nan))
        total += 1
        if not np.isfinite(oos_sharpe) or oos_sharpe < 0:
            oos_neg += 1
    return oos_neg / total if total else 1.0
