"""Etiquetado triple barrier sobre caminos de precios construidos a mano.

Se usa una vela "plana" de relleno (15 filas) para que el ATR(14) esté definido en el
punto de entrada (índice 15)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.labeling.triple_barrier import realized_r_multiple, triple_barrier_labels

ENTRY = 15  # índice de entrada tras el relleno


def _df_from_path(path_after_entry: list[float], atr_val: float = 2.0) -> pd.DataFrame:
    """ATR(14) ≈ atr_val: con velas simétricas, TR = high - low = 2 * half."""
    half = atr_val / 2.0
    closes = [100.0] * ENTRY + list(path_after_entry)
    idx = pd.bdate_range("2024-01-01", periods=len(closes))
    close = np.array(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + half,
            "low": close - half,
            "close": close,
            "volume": np.full(len(closes), 1e6),
        },
        index=idx,
    )


def test_tp_hit_first():
    # entry 100, atr≈2 => tp≈106, sl≈96; el precio sube a 107 antes de bajar
    df = _df_from_path([100.5, 107.0, 105.0, 104.0])
    lab = triple_barrier_labels(df, atr_mult_tp=3.0, atr_mult_sl=2.0, max_h=10)
    assert lab["label"].iloc[ENTRY] == 1
    assert lab["touch"].iloc[ENTRY] == "tp"


def test_sl_hit_first():
    df = _df_from_path([99.5, 95.0, 98.0, 99.0])
    lab = triple_barrier_labels(df, atr_mult_tp=3.0, atr_mult_sl=2.0, max_h=10)
    assert lab["label"].iloc[ENTRY] == -1
    assert lab["touch"].iloc[ENTRY] == "sl"


def test_time_barrier_sign():
    # barreras lejanas: no se tocan; label = signo del retorno al vencimiento.
    # entry = close[ENTRY] = 100.1; el camino termina en 100.2 > entry => label +1
    df = _df_from_path([100.1, 100.2, 100.3, 100.4, 100.5, 100.4, 100.3, 100.25, 100.22, 100.21, 100.2])
    lab = triple_barrier_labels(df, atr_mult_tp=30.0, atr_mult_sl=30.0, max_h=10)
    assert lab["touch"].iloc[ENTRY] == "time"
    assert lab["label"].iloc[ENTRY] == 1  # retorno positivo al vencimiento


def test_realized_r_multiple():
    df = _df_from_path([100.5, 107.0, 105.0, 104.0])
    r = realized_r_multiple(df, ENTRY, 1, atr_mult_sl=2.0, atr_mult_tp=3.0, max_h=10)
    # tp = 100 + 3*atr; con atr≈2 => tp≈106; r = (106-100)/(2*2) ≈ 1.5
    assert abs(r - 1.5) < 0.2
