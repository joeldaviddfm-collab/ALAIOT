"""Indicadores técnicos. TODOS son rolling/rezagados: el valor en t solo usa datos <= t
(test de look-ahead en tests/test_no_lookahead.py lo verifica)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hi, lo, c = df["high"], df["low"], df["close"]
    tr = pd.concat([hi - lo, (hi - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hi, lo = df["high"], df["low"]
    up = hi.diff()
    dn = -lo.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = pd.concat([hi - lo, (hi - df["close"].shift()).abs(), (lo - df["close"].shift()).abs()], axis=1).max(
        axis=1
    )
    atr_n = tr.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, min_periods=n, adjust=False).mean() / atr_n.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, min_periods=n, adjust=False).mean() / atr_n.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def rolling_vwap(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).rolling(n, min_periods=n).sum() / df["volume"].rolling(
        n, min_periods=n
    ).sum().replace(0, np.nan)


def compute_technical(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c, hi, lo, v = df["close"], df["high"], df["low"], df["volume"]
    out["ret_1"] = c.pct_change(1)
    out["ret_5"] = c.pct_change(5)
    out["ret_10"] = c.pct_change(10)
    out["ret_20"] = c.pct_change(20)
    for w in (10, 20, 50, 200):
        out[f"close_sma_{w}"] = c / c.rolling(w, min_periods=w).mean() - 1
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    out["ema_ratio"] = ema12 / ema26 - 1
    sma20 = c.rolling(20, min_periods=20).mean()
    out["sma20_slope"] = sma20 / sma20.shift(5) - 1
    out["adx_14"] = adx(df) / 100.0
    out["rsi_14"] = rsi(c) / 100.0
    out["macd_hist"] = macd_hist(c) / c
    out["roc_10"] = c.pct_change(10)
    out["roc_20"] = c.pct_change(20)
    atr14 = atr(df)
    atr_pct = atr14 / c
    out["atr_pct"] = atr_pct
    out["atr_pct_z"] = (atr_pct - atr_pct.rolling(60, min_periods=20).mean()) / atr_pct.rolling(
        60, min_periods=20
    ).std()
    mid = c.rolling(20, min_periods=20).mean()
    std = c.rolling(20, min_periods=20).std()
    upper, lower = mid + 2 * std, mid - 2 * std
    out["bb_pct_b"] = (c - lower) / (upper - lower).replace(0, np.nan)
    out["bb_width"] = (upper - lower) / mid
    out["realized_vol_20"] = c.pct_change().rolling(20, min_periods=20).std() * np.sqrt(252)
    o = obv(c, v)
    out["obv_slope_20"] = (o / o.shift(20).abs().replace(0, np.nan) - 1).clip(-5, 5)
    out["vol_z_20"] = (v - v.rolling(20, min_periods=20).mean()) / v.rolling(20, min_periods=20).std()
    out["dist_vwap_20"] = c / rolling_vwap(df) - 1
    hh = (hi > hi.shift(1)).astype(float)
    ll = (lo < lo.shift(1)).astype(float)
    out["structure_20"] = (hh.rolling(20, min_periods=20).sum() - ll.rolling(20, min_periods=20).sum()) / 20.0
    out["range_pct_20"] = ((hi - lo) / c).rolling(20, min_periods=20).mean()
    dow = out.index.dayofweek
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    month = out.index.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    return out
