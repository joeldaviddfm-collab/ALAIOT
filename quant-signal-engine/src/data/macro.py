"""Datos macro: FRED API (con FRED_API_KEY) con fallback a yfinance para VIX/DXY/tasas.

Cada serie se anota con su fecha de publicación real (point-in-time); el rezago se
aplica después en features/alignment.py (publication_lags en config.yaml).
"""

from __future__ import annotations

import os

import pandas as pd
import structlog

from src.data.base import safe_call
from src.data.cache import DiskCache
from src.data.prices import YFinanceSource
from src.utils.alerts import AlertBus

log = structlog.get_logger("qse.macro")

FRED_SERIES = {
    "DGS10": "dgs10",
    "DGS2": "dgs2",
    "T10Y2Y": "curve",
    "CPIAUCSL": "cpi",
    "UNRATE": "unrate",
    "FEDFUNDS": "fedfunds",
}
YF_FALLBACK = {"^VIX": "vix", "DX-Y.NYB": "dxy", "^TNX": "y10", "^FVX": "y5"}


def _fred_series(fred, ticker: str, start: str) -> pd.Series | None:
    s = fred.get_series(ticker, observation_start=start)
    if s is None or s.empty:
        return None
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).normalize()
    return s.astype(float)


def get_macro(start: str, end: str | None, cache: DiskCache, alerts: AlertBus) -> pd.DataFrame:
    key = ("macro", str(start), str(end))
    cached = cache.get(key, max_age_days=1.0)
    if cached is not None and not cached.empty:
        return cached

    frames: dict[str, pd.Series] = {}
    fred_key = os.getenv("FRED_API_KEY", "").strip()
    if fred_key:

        def _fred_all() -> dict[str, pd.Series]:
            from fredapi import Fred

            fred = Fred(api_key=fred_key)
            out = {}
            for ticker, col in FRED_SERIES.items():
                s = safe_call(f"fred:{ticker}", lambda t=ticker: _fred_series(fred, t, start), alerts)  # type: ignore[misc]
                if s is not None:
                    out[col] = s
            return out

        frames = safe_call("fred", _fred_all, alerts) or {}
        if not frames:
            alerts.add("WARNING", "FRED no respondió; se usa fallback yfinance para VIX/DXY/tasas")
    else:
        alerts.add(
            "INFO", "FRED_API_KEY no definida; VIX/DXY/tasas vía yfinance (^VIX, DX-Y.NYB, ^TNX, ^FVX)"
        )

    yf = YFinanceSource()
    for ticker, col in YF_FALLBACK.items():
        if col in frames:
            continue
        df = safe_call(f"yf:{ticker}", lambda t=ticker: yf.fetch(t, start, end), alerts)  # type: ignore[misc]
        if df is not None and not df.empty:
            frames[col] = df["close"]

    if not frames:
        alerts.add("ERROR", "Macro no disponible en ninguna fuente (FRED/yfinance)")
        return pd.DataFrame()

    df = pd.DataFrame(frames).sort_index()
    df.index.name = "date"
    # derivados point-in-time
    if "cpi" in df:
        df["cpi_yoy"] = df["cpi"].pct_change(12) * 100
        df = df.drop(columns=["cpi"])
    if {"y10", "y5"} <= set(df.columns) and "curve" not in df:
        df["curve_proxy"] = df["y10"] - df["y5"]  # proxy 10y-5y si FRED no dio T10Y2Y
    if "dxy" in df:
        df["dxy_ret5"] = df["dxy"].pct_change(5)
    df = df.resample("B").ffill()
    cache.put(key, df)
    log.info("macro_ok", cols=list(df.columns), rows=len(df))
    return df
