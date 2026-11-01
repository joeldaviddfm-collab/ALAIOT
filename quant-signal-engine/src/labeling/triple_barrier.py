"""Etiquetado triple barrier basado en ATR (take profit / stop loss / tiempo máx.).

La etiqueta en t usa datos FUTUROS (hasta t + max_h) — eso es correcto: las etiquetas
miden el resultado, no son features. Lo que nunca puede usar futuro son las FEATURES
(ver tests/test_no_lookahead.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.technical import atr


def triple_barrier_labels(
    df: pd.DataFrame,
    *,
    atr_mult_tp: float = 3.0,
    atr_mult_sl: float = 2.0,
    max_h: int = 10,
) -> pd.DataFrame:
    """Etiqueta por cada día i (entrada al close[i]):
    +1 si toca TP antes que SL, -1 si toca SL antes que TP, 0 en barrera temporal
    (con el signo del retorno al vencimiento si no es ~0).
    Devuelve DataFrame con columnas label, touch, horizon (días)."""
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    atr_v = atr(df, 14).to_numpy(dtype=float)
    n = len(df)
    labels = np.zeros(n, dtype=int)
    touch = np.array(["time"] * n, dtype=object)
    horizon = np.zeros(n, dtype=int)
    for i in range(n - 1):
        entry = close[i]
        a = atr_v[i]
        if not np.isfinite(a) or a <= 0 or not np.isfinite(entry) or entry <= 0:
            continue
        up = entry + atr_mult_tp * a
        dn = entry - atr_mult_sl * a
        j_end = min(n, i + 1 + max_h)
        lab, t, hj = 0, "time", j_end - 1 - i
        for j in range(i + 1, j_end):
            if high[j] >= up:
                lab, t, hj = 1, "tp", j - i
                break
            if low[j] <= dn:
                lab, t, hj = -1, "sl", j - i
                break
        else:
            r = close[j_end - 1] / entry - 1
            lab = 1 if r > 0 else (-1 if r < 0 else 0)
        labels[i], touch[i], horizon[i] = lab, t, hj
    return pd.DataFrame({"label": labels, "touch": touch, "horizon": horizon}, index=df.index)


def realized_r_multiple(
    df: pd.DataFrame,
    i: int,
    direction: int,
    *,
    atr_mult_sl: float = 2.0,
    atr_mult_tp: float = 3.0,
    max_h: int = 10,
) -> float:
    """R-multiple realizado de una posición abierta al close[i] en `direction` (+1/-1)."""
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    atr_v = atr(df, 14).to_numpy(dtype=float)
    n = len(df)
    if i >= n - 1 or direction == 0:
        return 0.0
    entry = close[i]
    a = atr_v[i]
    if not np.isfinite(a) or a <= 0:
        return 0.0
    up = entry + atr_mult_tp * a
    dn = entry - atr_mult_sl * a
    j_end = min(n, i + 1 + max_h)
    exit_price = close[j_end - 1]
    for j in range(i + 1, j_end):
        if direction == 1:
            if high[j] >= up:
                exit_price = up
                break
            if low[j] <= dn:
                exit_price = dn
                break
        else:
            if low[j] <= dn:
                exit_price = dn
                break
            if high[j] >= up:
                exit_price = up
                break
    pnl = direction * (exit_price - entry)
    return pnl / (atr_mult_sl * a)
