"""Construcción del dataset de entrenamiento (features + etiquetas triple barrier),
alineado por (date, asset). Las etiquetas SÍ pueden mirar el futuro (miden resultado);
las features NO (ver tests/test_no_lookahead.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import build_feature_matrix
from src.labeling.triple_barrier import triple_barrier_labels


def build_training_set(cfg, market) -> tuple[pd.DataFrame, np.ndarray, dict[str, str]]:
    X, groups = build_feature_matrix(market.prices, market.global_df, cfg)
    # columnas auxiliares para stops/tp y sizing (no son features del modelo)
    aux_rows = []
    for asset, df in market.prices.items():
        from src.features.technical import atr

        aux = pd.DataFrame(
            {
                "date": df.index,
                "asset": asset,
                "close_ref": df["close"].to_numpy(),
                "atr14": atr(df, 14).to_numpy(),
            }
        )
        aux_rows.append(aux)
    aux = pd.concat(aux_rows, ignore_index=True)
    X = X.merge(aux, on=["date", "asset"], how="left")
    tb = cfg.get("model.triple_barrier", {})
    lab_rows = []
    for asset, df in market.prices.items():
        lab = triple_barrier_labels(
            df,
            atr_mult_tp=float(tb.get("atr_mult_tp", 3.0)),
            atr_mult_sl=float(tb.get("atr_mult_sl", 2.0)),
            max_h=int(tb.get("max_horizon_days", 10)),
        )
        lab = lab.reset_index()
        if "index" in lab.columns:
            lab = lab.rename(columns={"index": "date"})
        lab["asset"] = asset
        lab_rows.append(lab[["date", "asset", "label"]])
    lab_df = pd.concat(lab_rows, ignore_index=True)
    X = X.merge(lab_df, on=["date", "asset"], how="left")
    y = X["label"].fillna(0).to_numpy(dtype=int)
    X = X.drop(columns=["label"])
    X = X.sort_values("date").reset_index(drop=True)
    return X, y, groups


def signals_from_predictions(X_rows: pd.DataFrame, preds: pd.DataFrame, cfg) -> pd.DataFrame:
    """Convierte predicciones del modelo en filas de señal con stop/TP (RR mínima)."""
    tb = cfg.get("model.triple_barrier", {})
    sl_mult = float(tb.get("atr_mult_sl", 2.0))
    tp_mult = float(tb.get("atr_mult_tp", 3.0))
    min_rr = float(cfg.get("risk.min_rr", 1.5))
    out = X_rows[["date", "asset", "close_ref", "atr14", "regime_id"]].copy().reset_index(drop=True)
    out["direction"] = preds["direction"].to_numpy()
    out["prob"] = preds["prob"].to_numpy()
    out["prob_pos"] = preds["prob_pos"].to_numpy()
    out["prob_neg"] = preds["prob_neg"].to_numpy()
    out["prob_ens_pos"] = preds["prob_ens_pos"].to_numpy()
    out["prob_ens_neg"] = preds["prob_ens_neg"].to_numpy()
    out["meta_prob"] = preds["meta_prob"].to_numpy()
    out["agreement"] = preds["agreement"].to_numpy()
    close = out["close_ref"].to_numpy(dtype=float)
    atr_v = out["atr14"].to_numpy(dtype=float)
    direction = out["direction"].to_numpy()
    rr_needed = max(tp_mult, min_rr * sl_mult)
    stop = np.where(
        direction == 1, close - sl_mult * atr_v, np.where(direction == -1, close + sl_mult * atr_v, np.nan)
    )
    tp = np.where(
        direction == 1,
        close + rr_needed * atr_v,
        np.where(direction == -1, close - rr_needed * atr_v, np.nan),
    )
    out["stop"] = stop
    out["tp"] = tp
    return out
