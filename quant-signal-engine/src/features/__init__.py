"""Construcción de la matriz de features (sin look-ahead: todo rolling/as-of con rezago)."""

from __future__ import annotations

import pandas as pd

from src.data.prices import asset_class
from src.features.alignment import merge_global
from src.features.regime import RegimeDetector
from src.features.technical import compute_technical

# grupos de features para el ablation study
FEATURE_GROUPS: dict[str, list[str]] = {
    "trend": [
        "close_sma_10",
        "close_sma_20",
        "close_sma_50",
        "close_sma_200",
        "ema_ratio",
        "sma20_slope",
        "adx_14",
    ],
    "momentum": ["rsi_14", "macd_hist", "roc_10", "roc_20"],
    "volatility": ["atr_pct", "atr_pct_z", "bb_pct_b", "bb_width", "realized_vol_20", "range_pct_20"],
    "volume": ["vol_z_20", "obv_slope_20", "dist_vwap_20"],
    "structure_seasonal": [
        "ret_1",
        "ret_5",
        "ret_10",
        "ret_20",
        "structure_20",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    ],
    "regime": ["regime_id", "regime_prob"],
    "macro": [
        "vix",
        "vix_z",
        "dxy",
        "dxy_ret5",
        "y10",
        "y5",
        "curve",
        "curve_proxy",
        "cpi_yoy",
        "unrate",
        "fedfunds",
        "dgs10",
        "dgs2",
    ],
    "sentiment": ["fear_greed", "fear_greed_chg5", "trends_btc", "news_sent", "news_sent_ma5"],
    "derivatives": [
        "funding_btc",
        "oi_btc",
        "oi_btc_chg7",
        "ls_ratio_btc",
        "funding_eth",
        "oi_eth",
        "oi_eth_chg7",
        "ls_ratio_eth",
    ],
    "onchain": [
        "btc_price_cg",
        "btc_mcap",
        "btc_mcap_chg7",
        "btc_vol_cg",
        "eth_price_cg",
        "eth_mcap",
        "hash_rate",
        "hash_rate_chg7",
        "n_tx",
        "n_tx_chg7",
        "mvrv_btc",
    ],
    "asset_meta": ["is_crypto", "is_fx", "is_equity"],
}


def build_feature_matrix(
    prices: dict[str, pd.DataFrame], global_df: pd.DataFrame, cfg
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Devuelve X con columnas [date, asset, features...] y el mapa feature->grupo."""
    lags = dict(cfg.get("publication_lags", {}))
    rows: list[pd.DataFrame] = []
    detector = RegimeDetector(seed=cfg.seed)
    for asset, df in prices.items():
        tech = compute_technical(df)
        reg = detector.fit_predict(df)
        if global_df is None or global_df.empty:
            glob = pd.DataFrame(index=df.index)
        else:
            glob = merge_global(df.index, global_df, lags)
        feat = pd.concat([tech, reg, glob], axis=1)
        # derivados de las globales ya rezagadas
        if "vix" in feat:
            feat["vix_z"] = (feat["vix"] - feat["vix"].rolling(60, min_periods=20).mean()) / feat[
                "vix"
            ].rolling(60, min_periods=20).std()
        if "fear_greed" in feat:
            feat["fear_greed_chg5"] = feat["fear_greed"].diff(5)
        if "news_sent" in feat:
            feat["news_sent_ma5"] = feat["news_sent"].rolling(5, min_periods=1).mean()
        for c in ("oi_btc", "oi_eth", "btc_mcap", "hash_rate", "n_tx"):
            if c in feat:
                feat[f"{c}_chg7"] = feat[c].pct_change(7)
        klass = asset_class(asset, list(cfg.get("assets.crypto", [])))
        feat["is_crypto"] = 1.0 if klass == "crypto" else 0.0
        feat["is_fx"] = 1.0 if klass == "fx" else 0.0
        feat["is_equity"] = 1.0 if klass == "equity" else 0.0
        feat["asset"] = asset
        feat["date"] = feat.index
        rows.append(feat.reset_index(drop=True))
    if not rows:
        return pd.DataFrame(), {}
    X = pd.concat(rows, ignore_index=True)
    groups: dict[str, str] = {}
    for group, cols in FEATURE_GROUPS.items():
        for c in cols:
            if c in X.columns:
                groups[c] = group
    for c in X.columns:
        if c not in groups and c not in ("date", "asset"):
            groups[c] = "other"
    return X, groups


def feature_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in ("date", "asset")]


def lag_map_note() -> str:
    return "rezagos aplicados por columna vía publication_lags (features/alignment.lag_for_column)"
