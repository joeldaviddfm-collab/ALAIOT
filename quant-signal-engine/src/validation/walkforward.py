"""Walk-forward con purging y embargo (López de Prado).

- embargo >= horizonte máximo de etiqueta (las etiquetas usan hasta max_h días futuros,
  por eso train termina `embargo_days` antes del test: purging).
- split ciego final: los últimos `blind_days` nunca se tocan durante el desarrollo.
NUNCA se usa train/test aleatorio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def wf_splits(
    dates: pd.DatetimeIndex,
    *,
    train_days: int,
    embargo_days: int,
    test_days: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Devuelve lista de (posiciones_train, posiciones_test) sobre el índice de fechas."""
    dates = pd.DatetimeIndex(dates)
    n = len(dates)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    start = train_days
    while start + test_days <= n:
        train_end = max(0, start - embargo_days)  # purging: se poda el embargo
        tr = np.arange(0, train_end)
        te = np.arange(start, start + test_days)
        if len(tr) > 0:
            splits.append((tr, te))
        start += test_days
    return splits


def blind_split(dates: pd.DatetimeIndex, blind_days: int) -> tuple[np.ndarray, np.ndarray]:
    """Separa los últimos `blind_days` como test ciego intocable."""
    dates = pd.DatetimeIndex(dates)
    n = len(dates)
    blind = min(blind_days, max(0, n // 5)) if blind_days <= 0 else min(blind_days, n - 1)
    return np.arange(0, n - blind), np.arange(n - blind, n)
