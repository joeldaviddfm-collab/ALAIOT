"""Datos SINTÉTICOS deterministas (solo para --offline / tests / CI).

NUNCA se usan en producción: en modo real, si fallan las fuentes la señal es NADA.
Generan un proceso con cambio de régimen (3 estados), saltos y volumen procíclico,
para poder ejercitar todo el pipeline sin internet.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def _rng_for(seed: int, *parts: str) -> np.random.Generator:
    digest = hashlib.sha256("|".join([str(seed), *parts]).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


# transición de régimen (bear, lateral, bull) y parámetros
_TRANS = np.array([[0.90, 0.08, 0.02], [0.10, 0.85, 0.05], [0.02, 0.08, 0.90]])
_MU = np.array([-0.0012, 0.0002, 0.0010])  # retorno diario por régimen
_SIG = np.array([0.030, 0.010, 0.016])


def synthetic_ohlcv(asset: str, start: str, end: str | None, seed: int = 42) -> pd.DataFrame:
    rng = _rng_for(seed, "ohlcv", asset)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.now().normalize()
    idx = pd.bdate_range(start, end_ts, freq="B")
    if len(idx) < 30:
        idx = pd.bdate_range(start, pd.Timestamp(start) + pd.Timedelta(days=400), freq="B")
    n = len(idx)
    states = np.zeros(n, dtype=int)
    states[0] = 1
    for i in range(1, n):
        states[i] = rng.choice(3, p=_TRANS[states[i - 1]])
    mu = _MU[states]
    sig = _SIG[states]
    rets = rng.normal(mu, sig)
    jumps = rng.poisson(0.004, n)
    rets = rets + jumps * rng.normal(-0.03, 0.05, n) * (states == 0)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.roll(close, 1)
    open_[0] = 100.0
    open_ = open_ * (1 + rng.normal(0, 0.002, n))
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    vol = (1e6 * (1 + 4 * np.abs(rets)) * (1 + 0.5 * (states == 0))).astype(float)
    df = pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close, "volume": vol}, index=idx)
    df.index.name = "date"
    return df


def synthetic_global(start: str, end: str | None, seed: int = 42) -> pd.DataFrame:
    """Serie global sintética: macro, sentimiento, derivados y on-chain."""
    rng = _rng_for(seed, "global")
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.now().normalize()
    idx = pd.bdate_range(start, end_ts, freq="B")
    n = len(idx)
    t = np.arange(n)
    vix = 18 + 8 * np.sin(t / 90.0) + rng.normal(0, 1.5, n).cumsum() * 0.05
    vix = np.clip(vix, 10, 80)
    dxy = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.004, n)))
    curve = 1.5 + 0.8 * np.sin(t / 250.0) + rng.normal(0, 0.05, n)
    cpi_yoy = 2.5 + 1.5 * np.sin(t / 300.0 + 1) + rng.normal(0, 0.1, n)
    unrate = 4.5 + 0.5 * np.sin(t / 400.0) + rng.normal(0, 0.02, n)
    fear = np.clip(50 + 25 * np.sin(t / 60.0) + rng.normal(0, 6, n), 0, 100)
    trends = np.clip(50 + 30 * np.sin(t / 45.0) + rng.normal(0, 5, n), 0, 100)
    news = np.clip(rng.normal(0.0, 0.15, n).cumsum() * 0.1 + 0.0, -1, 1)
    funding = rng.normal(0.0001, 0.0004, n)
    oi = np.exp(np.cumsum(rng.normal(0.0005, 0.01, n))) * 1e10
    ls = np.clip(1.0 + rng.normal(0, 0.15, n), 0.3, 3.0)
    mcap = np.exp(np.cumsum(rng.normal(0.001, 0.03, n))) * 5e11
    hrate = np.exp(np.cumsum(rng.normal(0.0002, 0.02, n))) * 3e8
    ntx = np.clip(300000 * (1 + rng.normal(0, 0.1, n)), 1e5, None)
    df = pd.DataFrame(
        {
            "vix": vix,
            "dxy": dxy,
            "curve": curve,
            "cpi_yoy": cpi_yoy,
            "unrate": unrate,
            "fear_greed": fear,
            "trends_btc": trends,
            "news_sent": news,
            "funding_btc": funding,
            "oi_btc": oi,
            "ls_ratio_btc": ls,
            "funding_eth": funding * 0.8,
            "oi_eth": oi * 0.4,
            "ls_ratio_eth": ls * 0.9,
            "btc_mcap": mcap,
            "hash_rate": hrate,
            "n_tx": ntx,
        },
        index=idx,
    )
    df.index.name = "date"
    return df
