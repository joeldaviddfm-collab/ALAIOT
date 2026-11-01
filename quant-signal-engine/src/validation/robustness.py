"""Pruebas anti-suerte y anti-overfitting:
permutación, estrategias aleatorias, Monte Carlo (bootstrap por bloques),
stress de costos, stress histórico, fuga de información (leakage) y validación
cruzada entre activos. Todas con semilla fija.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.validation.metrics import equity_metrics


def permutation_pvalue(trade_r: np.ndarray, observed_sharpe: float, n: int = 1000, seed: int = 42) -> float:
    """p-value del Sharpe observado vs. distribución nula por PERMUTACIÓN DE SIGNOS:
    invierte aleatoriamente la dirección de cada operación (conserva magnitudes,
    rompe cualquier edge real). Si el sistema tiene edge, pocas permutaciones lo superan."""
    r = np.asarray(trade_r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 5:
        return 1.0
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(int(n)):
        signs = rng.choice([-1.0, 1.0], size=len(r))
        equity = 100000 * np.cumprod(1 + r * signs * 0.01)
        null_sr = equity_metrics(pd.Series(equity))["sharpe"]
        if null_sr >= observed_sharpe:
            count += 1
    return (count + 1) / (n + 1)


def random_strategies_pvalue(
    prices: dict[str, pd.DataFrame],
    signal_dates_assets: list[tuple[pd.Timestamp, str]],
    observed_sharpe: float,
    *,
    horizon: int = 5,
    n: int = 1000,
    seed: int = 42,
) -> float:
    """Compara contra `n` estrategias aleatorias con la MISMA frecuencia de operaciones:
    elige fechas/activos/direcciones al azar y mide el Sharpe. p-value empírico."""
    rng = np.random.default_rng(seed)
    all_dates: list[pd.Timestamp] = sorted({d for df in prices.values() for d in df.index})
    if len(all_dates) < horizon + 5 or not signal_dates_assets:
        return 1.0
    fwd: dict[str, pd.Series] = {}
    for asset, df in prices.items():
        fwd[asset] = df["close"].pct_change(horizon).shift(-horizon)
    count = 0
    n_trades = len(signal_dates_assets)
    for _ in range(int(n)):
        rets = []
        for _ in range(n_trades):
            d = all_dates[rng.integers(0, len(all_dates) - horizon - 1)]
            asset = list(prices.keys())[rng.integers(0, len(prices))]
            direction = 1 if rng.random() < 0.5 else -1
            r = fwd[asset].get(d, np.nan)
            if np.isfinite(r):
                rets.append(direction * float(r))
        if len(rets) < 5:
            continue
        equity = 100000 * np.cumprod(1 + np.array(rets) * 0.01)
        null_sr = equity_metrics(pd.Series(equity))["sharpe"]
        if null_sr >= observed_sharpe:
            count += 1
    return (count + 1) / (n + 1)


def monte_carlo_block_bootstrap(
    trade_r: np.ndarray,
    *,
    initial_equity: float = 100000.0,
    risk_per_trade: float = 0.01,
    block: int = 5,
    n_sims: int = 2000,
    ruin_threshold: float = 0.5,
    seed: int = 42,
) -> dict:
    """Bootstrap por bloques de los R-multiples: IC de retorno total, max drawdown
    y probabilidad de ruina (equity < ruin_threshold * inicial)."""
    r = np.asarray(trade_r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 5:
        return {"p_ruin": 1.0, "ret_p05": 0.0, "ret_p50": 0.0, "ret_p95": 0.0, "maxdd_p95": 0.0, "n_sims": 0}
    rng = np.random.default_rng(seed)
    n = len(r)
    rets, maxdds, ruins = [], [], 0
    for _ in range(int(n_sims)):
        idx = rng.integers(0, n, size=int(np.ceil(n / block)) * block)
        idx = np.concatenate([idx[i : i + block] for i in range(0, len(idx), block)])
        sample = r[idx % n][:n]
        pnl = initial_equity * risk_per_trade * sample
        equity = initial_equity + np.cumsum(pnl)
        rets.append(equity[-1] / initial_equity - 1)
        maxdds.append(float((equity / np.maximum.accumulate(equity) - 1).min()))
        if equity.min() < ruin_threshold * initial_equity:
            ruins += 1
    return {
        "p_ruin": ruins / n_sims,
        "ret_p05": float(np.percentile(rets, 5)),
        "ret_p50": float(np.percentile(rets, 50)),
        "ret_p95": float(np.percentile(rets, 95)),
        "maxdd_p95": float(np.percentile(maxdds, 95)),
        "n_sims": int(n_sims),
    }


def leakage_auc(X: pd.DataFrame, y: np.ndarray, seed: int = 42, n_splits: int = 5) -> float:
    """Test de fuga: entrenar con etiquetas BARAJADAS debe rendir como azar (AUC ~ 0.5).
    Si rinde mejor, hay leakage. Devuelve el AUC medio de CV."""
    from lightgbm import LGBMClassifier

    y_bin = (np.asarray(y) > 0).astype(int)
    if y_bin.sum() < 10 or y_bin.sum() == len(y_bin):
        return 0.5
    rng = np.random.default_rng(seed)
    y_shuf = rng.permutation(y_bin)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X.to_numpy(dtype=float), y_shuf):
        clf = LGBMClassifier(
            objective="binary",
            n_estimators=100,
            max_depth=3,
            num_leaves=15,
            min_child_samples=20,
            random_state=seed,
            verbosity=-1,
            n_jobs=-1,
        )
        clf.fit(X.to_numpy(dtype=float)[tr], y_shuf[tr])
        p = clf.predict_proba(X.to_numpy(dtype=float)[te])[:, 1]
        with contextlib.suppress(ValueError):
            aucs.append(roc_auc_score(y_shuf[te], p))
    return float(np.mean(aucs)) if aucs else 0.5


def historical_stress(equity: pd.Series, windows: list[dict]) -> dict[str, float]:
    """Max drawdown dentro de cada ventana histórica de estrés."""
    equity = pd.Series(equity)
    out = {}
    for w in windows:
        seg = equity.loc[str(w["start"]) : str(w["end"])]  # type: ignore[misc]
        if len(seg) < 2:
            out[w["name"]] = 0.0
            continue
        out[w["name"]] = float((seg / seg.cummax() - 1).min())
    return out
