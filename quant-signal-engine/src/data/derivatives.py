"""Derivados cripto (APIs públicas de Binance, sin key):
funding rate, open interest y ratio long/short de los perpetuos BTC/ETH.
"""

from __future__ import annotations

import functools

import pandas as pd
import requests
import structlog

from src.data.base import safe_call
from src.data.cache import DiskCache
from src.utils.alerts import AlertBus

log = structlog.get_logger("qse.derivatives")
FAPI = "https://fapi.binance.com"
SYMBOLS = {"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT"}


def _get_json(url: str, params: dict) -> list[dict]:
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def _series_from(rows: list[dict], ts_key: str, val_key: str) -> pd.Series:
    s = pd.Series({pd.Timestamp(int(r[ts_key]), unit="ms").normalize(): float(r[val_key]) for r in rows})
    s.index = pd.DatetimeIndex(s.index)
    return s.sort_index()


def _funding(sym: str):
    return _series_from(
        _get_json(f"{FAPI}/fapi/v1/fundingRate", {"symbol": sym, "limit": 1000}), "fundingTime", "fundingRate"
    )


def _oi_hist(sym: str):
    return _series_from(
        _get_json(f"{FAPI}/futures/data/openInterestHist", {"symbol": sym, "period": "1d", "limit": 1000}),
        "timestamp",
        "sumOpenInterestValue",
    )


def _ls_ratio(sym: str):
    return _series_from(
        _get_json(
            f"{FAPI}/futures/data/globalLongShortAccountRatio", {"symbol": sym, "period": "1d", "limit": 1000}
        ),
        "timestamp",
        "longShortRatio",
    )


def get_derivatives(start: str, end: str | None, cache: DiskCache, alerts: AlertBus) -> pd.DataFrame:
    key = ("derivatives", str(start), str(end))
    cached = cache.get(key, max_age_days=1.0)
    if cached is not None and not cached.empty:
        return cached
    frames: dict[str, pd.Series] = {}
    for asset, sym in SYMBOLS.items():
        tag = asset.split("-")[0].lower()
        funding = safe_call(f"funding:{sym}", functools.partial(_funding, sym), alerts)
        oi = safe_call(f"oi_hist:{sym}", functools.partial(_oi_hist, sym), alerts)
        ls = safe_call(f"ls_ratio:{sym}", functools.partial(_ls_ratio, sym), alerts)
        if funding is not None:
            frames[f"funding_{tag}"] = funding.resample("B").last()
        if oi is not None:
            frames[f"oi_{tag}"] = oi.resample("B").last()
        if ls is not None:
            frames[f"ls_ratio_{tag}"] = ls.resample("B").last()
    if not frames:
        alerts.add("ERROR", "Derivados cripto no disponibles (Binance fapi)")
        return pd.DataFrame()
    df = pd.DataFrame(frames).sort_index()
    df.index.name = "date"
    df = df.resample("B").ffill()
    cache.put(key, df)
    log.info("derivatives_ok", cols=list(df.columns), rows=len(df))
    return df
