"""Métricas on-chain públicas: CoinGecko (free, sin key), blockchain.info y
Glassnode (free tier, solo si GLASSNODE_API_KEY está definida).
"""

from __future__ import annotations

import os
from collections.abc import Callable

import pandas as pd
import requests
import structlog

from src.data.base import safe_call
from src.data.cache import DiskCache
from src.utils.alerts import AlertBus

log = structlog.get_logger("qse.onchain")
CG = "https://api.coingecko.com/api/v3"
BC = "https://api.blockchain.info"


def _cg_market_chart(coin: str, days: int = 2000) -> Callable[[], pd.DataFrame]:
    def _fetch() -> pd.DataFrame:
        r = requests.get(
            f"{CG}/coins/{coin}/market_chart",
            params={"vs_currency": "usd", "days": days, "interval": "daily"},  # type: ignore[arg-type]
            headers={"accept": "application/json"},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        idx = (
            pd.DatetimeIndex(pd.to_datetime([p[0] for p in data["prices"]], unit="ms", utc=True))
            .tz_localize(None)
            .normalize()
        )
        return pd.DataFrame(
            {
                f"{coin}_price_cg": [p[1] for p in data["prices"]],
                f"{coin}_mcap": [m[1] for m in data["market_caps"]],
                f"{coin}_vol_cg": [v[1] for v in data["total_volumes"]],
            },
            index=idx,
        )

    return _fetch


def _bc_chart(name: str, col: str, timespan: str = "2000days") -> Callable[[], pd.Series]:
    def _fetch() -> pd.Series:
        r = requests.get(
            f"{BC}/charts/{name}",
            params={"timespan": timespan, "format": "json", "cors": "true"},
            timeout=20,
        )
        r.raise_for_status()
        vals = r.json()["values"]
        s = pd.Series({pd.Timestamp(v["x"], unit="s").normalize(): float(v["y"]) for v in vals})
        s.index = pd.DatetimeIndex(s.index)
        return s.sort_index().rename(col)

    return _fetch


def get_onchain(start: str, end: str | None, cache: DiskCache, alerts: AlertBus) -> pd.DataFrame:
    key = ("onchain", str(start), str(end))
    cached = cache.get(key, max_age_days=1.0)
    if cached is not None and not cached.empty:
        return cached
    frames: dict[str, pd.Series] = {}
    for coin, tag in (("bitcoin", "btc"), ("ethereum", "eth")):
        df = safe_call(f"coingecko:{coin}", _cg_market_chart(coin), alerts)  # type: ignore[arg-type]
        if df is not None and not df.empty:
            for c in df.columns:
                frames[c.replace(coin, tag)] = df[c]
    hr = safe_call("blockchain:hash-rate", _bc_chart("hash-rate", "hash_rate"), alerts)
    if hr is not None:
        frames["hash_rate"] = hr
    ntx = safe_call("blockchain:n-transactions", _bc_chart("n-transactions", "n_tx"), alerts)
    if ntx is not None:
        frames["n_tx"] = ntx
    glass_key = os.getenv("GLASSNODE_API_KEY", "").strip()
    if glass_key:

        def _glass() -> pd.Series:
            r = requests.get(
                "https://api.glassnode.com/v1/metrics/market/mvrv",
                params={"a": "BTC", "api_key": glass_key},
                timeout=15,
            )
            r.raise_for_status()
            vals = r.json()
            s = pd.Series({pd.Timestamp(v["t"], unit="s").normalize(): float(v["v"]) for v in vals})
            s.index = pd.DatetimeIndex(s.index)
            return s.sort_index().rename("mvrv_btc")

        mvrv = safe_call("glassnode:mvrv", _glass, alerts)
        if mvrv is not None:
            frames["mvrv_btc"] = mvrv
    if not frames:
        alerts.add("ERROR", "On-chain no disponible en ninguna fuente")
        return pd.DataFrame()
    df = pd.DataFrame(frames).sort_index()
    df.index.name = "date"
    df = df.resample("B").ffill()
    cache.put(key, df)
    log.info("onchain_ok", cols=list(df.columns), rows=len(df))
    return df
