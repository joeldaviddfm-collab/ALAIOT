"""TEST OBLIGATORIO: detectar look-ahead bias.

1) Las features en t calculadas sobre datos <= t deben ser idénticas a las calculadas
   sobre el dataset completo (un spike en t+1 no puede cambiar las features de t).
2) El alineamiento as-of con rezago: un dato publicado en t solo entra en t + lag.
3) Las features de t no dependen de la etiqueta (futuro) de t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.features import build_feature_matrix, feature_columns
from src.features.alignment import asof_with_lag
from src.features.technical import compute_technical


def test_features_no_future_leakage(tiny_df):
    full = compute_technical(tiny_df)
    t = tiny_df.index[80]
    truncated = compute_technical(tiny_df.loc[:t])
    cols = [c for c in full.columns]
    pd.testing.assert_frame_equal(
        full.loc[[t], cols].reset_index(drop=True),
        truncated.loc[[t], cols].reset_index(drop=True),
        check_exact=False,
        rtol=1e-9,
        atol=1e-12,
    )


def test_spike_tomorrow_does_not_change_today(tiny_df):
    t = tiny_df.index[60]
    modified = tiny_df.copy()
    # spike enorme en t+1 y siguientes
    modified.loc[modified.index > t, "close"] *= 3.0
    modified.loc[modified.index > t, ["high", "low", "open"]] *= 3.0
    f1 = compute_technical(tiny_df).loc[t]
    f2 = compute_technical(modified).loc[t]
    pd.testing.assert_series_equal(f1, f2, check_exact=False, rtol=1e-9, atol=1e-12)


def test_asof_lag_blocks_future_publication():
    base = pd.bdate_range("2024-01-01", periods=10)  # 1,2,3,4,5,8,9,10,11,12 de enero
    pub = pd.Series({pd.Timestamp("2024-01-05"): 999.0})  # publicado el viernes 5
    out = asof_with_lag(pub, pd.DatetimeIndex(base), lag_days=1)
    # lag de 1 día calendario: el dato publicado el 5 solo es visible cuando
    # fecha - 1d >= 5, es decir, desde el día hábil 8 (8-1=7 >= 5)
    assert np.isnan(out.loc["2024-01-05"])
    assert out.loc["2024-01-08"] == 999.0
    first_visible = out.dropna().index[0]
    assert first_visible > pd.Timestamp("2024-01-05") + pd.Timedelta(days=1)


def test_feature_matrix_independent_of_labels(cfg, small_market):
    X1, _ = build_feature_matrix(small_market.prices, small_market.global_df, cfg)
    # si se añaden filas futuras a un activo, las features de fechas antiguas no cambian

    prices2 = {k: v.copy() for k, v in small_market.prices.items()}
    extra_idx = pd.bdate_range(small_market.prices["SPY"].index[-1] + pd.Timedelta(days=1), periods=5)
    extra = small_market.prices["SPY"].iloc[[-1] * 5].copy()
    extra.index = extra_idx
    extra["close"] *= 5.0
    prices2["SPY"] = pd.concat([prices2["SPY"], extra])
    X2, _ = build_feature_matrix(prices2, small_market.global_df, cfg)
    cols = feature_columns(X1)
    X1s = X1.sort_values(["asset", "date"]).reset_index(drop=True)
    X2s = (
        X2[X2["date"] <= small_market.prices["SPY"].index[-1]]
        .sort_values(["asset", "date"])
        .reset_index(drop=True)
    )
    X2s = X2s[X2s["asset"].isin(X1s["asset"].unique())]
    common = X1s.merge(X2s, on=["date", "asset"], suffixes=("_1", "_2"))
    for c in cols:
        a = pd.to_numeric(common[f"{c}_1"], errors="coerce")
        b = pd.to_numeric(common[f"{c}_2"], errors="coerce")
        both = a.notna() & b.notna()
        if both.any():
            assert np.allclose(a[both], b[both], rtol=1e-8, atol=1e-10), f"look-ahead en feature {c}"
