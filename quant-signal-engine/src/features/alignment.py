"""Alineación point-in-time con rezagos de publicación (anti look-ahead).

Cada columna global se incorpora a la fecha t SOLO si fue publicada en <= t - lag.
Los rezagos por tipo de dato vienen de config.yaml (publication_lags).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def lag_for_column(col: str, lags: dict) -> int:
    lags = lags or {}
    if col in ("cpi_yoy", "unrate", "fedfunds"):
        return int(lags.get("fred_monthly", 30))
    if col in ("dgs10", "dgs2", "curve", "y10", "y5", "curve_proxy"):
        return int(lags.get("fred_daily", 1))
    if col in ("vix", "vix_z", "dxy", "dxy_ret5"):
        return int(lags.get("market", 1))
    if col.startswith("fear_greed"):
        return int(lags.get("fear_greed", 1))
    if col.startswith("trends"):
        return int(lags.get("trends", 7))
    if col.startswith("news"):
        return int(lags.get("news", 1))
    if any(k in col for k in ("funding", "oi_", "ls_ratio")):
        return int(lags.get("derivatives", 1))
    if any(k in col for k in ("mcap", "hash", "n_tx", "mvrv", "price_cg", "vol_cg")):
        return int(lags.get("onchain", 1))
    return 1


def asof_with_lag(series: pd.Series, base: pd.DatetimeIndex, lag_days: int) -> pd.Series:
    """Para cada fecha de `base`, el último valor de `series` con timestamp <= fecha - lag."""
    s = series.dropna()
    if s.empty:
        return pd.Series(np.nan, index=base, dtype=float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    tmp = pd.DataFrame({"v": s.to_numpy()}, index=s.index)
    target = pd.DataFrame({"target": (base - pd.Timedelta(days=lag_days)).to_numpy()}, index=base)
    merged = pd.merge_asof(
        target.sort_values("target"), tmp, left_on="target", right_index=True, direction="backward"
    )
    merged.index = base
    return pd.Series(merged["v"].to_numpy(), index=base, dtype=float)


def merge_global(base_index: pd.DatetimeIndex, global_df: pd.DataFrame, lags: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=base_index)
    for col in global_df.columns:
        out[col] = asof_with_lag(global_df[col], base_index, lag_for_column(col, lags))
    return out
