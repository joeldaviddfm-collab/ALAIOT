"""Baseline (tendencia SMA50/200 + volatilidad objetivo) y buy & hold, para comparar."""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_baseline(prices: dict[str, pd.DataFrame], cfg, *, initial_equity: float = 100000.0) -> pd.Series:
    """Estrategia simple de referencia: long solo si SMA50 > SMA200, ponderada por
    volatilidad objetivo (target_vol_ann / vol realizada), máx. 1 por activo, sin apalancamiento."""
    target_vol = float(cfg.get("risk.target_vol_ann", 0.10))
    rets_all = []
    for df in prices.values():
        c = df["close"]
        sma50 = c.rolling(50, min_periods=50).mean()
        sma200 = c.rolling(200, min_periods=200).mean()
        signal = (sma50 > sma200).astype(float)
        vol = c.pct_change().rolling(20, min_periods=20).std() * np.sqrt(252)
        weight = (target_vol / vol.replace(0, np.nan)).clip(0, 1) * signal
        rets_all.append(weight * c.pct_change())
    port = pd.concat(rets_all, axis=1).mean(axis=1).fillna(0.0)
    equity = initial_equity * (1 + port).cumprod()
    equity.index.name = "date"
    return equity


def run_buy_hold(prices: dict[str, pd.DataFrame], *, initial_equity: float = 100000.0) -> pd.Series:
    """Buy & hold: pesos iguales fijos desde el inicio, sin rebalanceo."""
    norm = {a: df["close"] / df["close"].iloc[0] for a, df in prices.items()}
    df = pd.concat(norm, axis=1).ffill()
    equity = initial_equity * df.mean(axis=1)
    equity.index.name = "date"
    return equity
