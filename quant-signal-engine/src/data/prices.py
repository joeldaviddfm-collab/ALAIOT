"""Precios OHLCV diarios: yfinance (acciones/ETF/índices/forex) y ccxt (cripto),
con fallback cruzado y cache local. La validación cruzada bloquea discrepancias.
"""

from __future__ import annotations

import pandas as pd
import structlog

from src.data.base import DataUnavailableError, safe_call
from src.data.cache import DiskCache
from src.data.validator import cross_source_check, validate_ohlcv
from src.utils.alerts import AlertBus

log = structlog.get_logger("qse.prices")

CRYPTO_BASES = {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "LINK", "LTC", "DOT", "UNI", "MATIC"}
OHLCV = ["open", "high", "low", "close", "volume"]


def asset_class(asset: str, crypto_list: list[str] | None = None) -> str:
    if asset in (crypto_list or []):
        return "crypto"
    if asset.endswith("=X") or "/" in asset:
        return "fx"
    base = asset.split("-")[0].split("/")[0]
    if base in CRYPTO_BASES:
        return "crypto"
    return "equity"


def normalize_ohlcv(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={"adj close": "close"})
    keep = [c for c in OHLCV if c in df.columns]
    df = df[keep]
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True)).tz_localize(None).normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["close"])
    for col in OHLCV:
        if col not in df.columns:
            df[col] = 0.0 if col == "volume" else df["close"]
    return df[OHLCV].astype(float)


class YFinanceSource:
    name = "yfinance"

    def fetch(self, asset: str, start: str, end: str | None) -> pd.DataFrame:
        import yfinance as yf

        df = yf.download(asset, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
        if df is None or df.empty:
            raise DataUnavailableError(f"yfinance devolvió vacío para {asset}")
        return normalize_ohlcv(df, self.name)


class CCXTSource:
    """Binance (spot) vía ccxt, paginado. Solo datos públicos, sin API key."""

    name = "ccxt:binance"

    def __init__(self) -> None:
        import ccxt

        self.ex = ccxt.binance({"enableRateLimit": True})

    def fetch(self, asset: str, start: str, end: str | None) -> pd.DataFrame:
        symbol = asset.replace("-", "/") + ":USDT" if "/" not in asset else asset + ":USDT"
        since = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) if end else since + 4000 * 86400000
        rows: list[list[float]] = []
        while since < end_ms:
            batch = self.ex.fetch_ohlcv(symbol, "1d", since=since, limit=1000)
            if not batch:
                break
            rows.extend(batch)
            since = batch[-1][0] + 86400000
            if len(batch) < 1000:
                break
        if not rows:
            raise DataUnavailableError(f"ccxt devolvió vacío para {asset}")
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df.index = (
            pd.DatetimeIndex(pd.to_datetime(df["ts"], unit="ms", utc=True)).tz_localize(None).normalize()
        )
        df.index.name = "date"
        return normalize_ohlcv(df, self.name)


def get_prices(
    asset: str,
    start: str,
    end: str | None,
    cache: DiskCache,
    alerts: AlertBus,
    *,
    crypto_list: list[str] | None = None,
    max_stale_bdays: int = 5,
    cross_tol: float = 0.02,
    cache_max_age_days: float = 1.0,
) -> pd.DataFrame:
    """Obtiene OHLCV con cadena de fallback y validación. Lanza DataUnavailableError si todo falla."""
    klass = asset_class(asset, crypto_list)
    key = ("prices", klass, asset, str(start), str(end))
    cached = cache.get(key, max_age_days=cache_max_age_days)
    if cached is not None and not cached.empty:
        rep = validate_ohlcv(cached, max_stale_bdays=max_stale_bdays)
        if rep.ok:
            log.info("prices_from_cache", asset=asset, rows=len(cached))
            return cached
        alerts.add("WARNING", f"{asset}: cache stale/inválido ({rep.issues}); se refresca")

    klass_src = asset_class(asset, crypto_list)
    chain: list = [CCXTSource(), YFinanceSource()] if klass_src == "crypto" else [YFinanceSource()]
    primary: pd.DataFrame | None = None
    primary_src = ""
    for src in chain:
        df = safe_call(src.name, lambda s=src: s.fetch(asset, start, end), alerts)  # type: ignore[misc]
        if df is None or df.empty:
            continue
        rep = validate_ohlcv(df, max_stale_bdays=max_stale_bdays)
        for w in rep.warnings:
            alerts.add("WARNING", f"{asset} [{src.name}]: {w}")
        if not rep.ok:
            alerts.add("ERROR", f"{asset} [{src.name}] rechazado por validación: {rep.issues}")
            continue
        if primary is None:
            primary, primary_src = df, src.name
        else:
            ok, diff = cross_source_check(primary, df, tol=cross_tol)
            if not ok:
                raise DataUnavailableError(
                    f"{asset}: fuentes {primary_src} y {src.name} difieren {diff:.2%} > {cross_tol:.0%}; activo bloqueado"
                )
            alerts.add("INFO", f"{asset}: cruzado {primary_src} vs {src.name} (diff máx {diff:.2%})")
        cache.put(key, df)
    if primary is None:
        raise DataUnavailableError(f"{asset}: ninguna fuente de precios disponible ({klass})")
    log.info("prices_ok", asset=asset, source=primary_src, rows=len(primary))
    return primary
