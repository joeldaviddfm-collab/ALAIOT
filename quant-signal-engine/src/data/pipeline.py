"""Pipeline de datos: descarga (o lee cache) todas las fuentes, valida y ensambla.

Política fail-safe: un activo sin datos válidos queda bloqueado (señal = NADA).
En modo offline (--offline, solo demos/tests) se usan datos sintéticos marcadaos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import structlog

from src.data.base import DataUnavailableError
from src.data.cache import DiskCache
from src.data.derivatives import get_derivatives
from src.data.macro import get_macro
from src.data.onchain import get_onchain
from src.data.prices import get_prices
from src.data.sentiment import get_sentiment
from src.data.synthetic import synthetic_global, synthetic_ohlcv
from src.utils.alerts import AlertBus

log = structlog.get_logger("qse.pipeline")


@dataclass
class MarketData:
    prices: dict[str, pd.DataFrame]
    global_df: pd.DataFrame
    blocked: dict[str, str] = field(default_factory=dict)
    data_mode: str = "real"  # "real" | "synthetic"


def load_market_data(
    cfg,
    assets: list[str],
    *,
    offline: bool = False,
    alerts: AlertBus | None = None,
) -> MarketData:
    alerts = alerts or AlertBus()
    start = str(cfg.get("data.start", "2018-01-01"))
    end = cfg.get("data.end")
    end = str(end) if end else None
    cache = DiskCache(cfg.cache_dir)
    crypto_list = list(cfg.get("assets.crypto", []))

    if offline:
        if not cfg.allow_synthetic and not offline:
            raise RuntimeError("synthetic data requiere --offline o QSE_ALLOW_SYNTHETIC=1")
        alerts.add("WARNING", "MODO OFFLINE: datos SINTÉTICOS (solo demo/test, no son datos de mercado)")
        synth_prices = {a: synthetic_ohlcv(a, start, end, cfg.seed) for a in assets}
        return MarketData(
            prices=synth_prices, global_df=synthetic_global(start, end, cfg.seed), data_mode="synthetic"
        )

    prices: dict[str, pd.DataFrame] = {}
    blocked: dict[str, str] = {}
    for asset in assets:
        try:
            prices[asset] = get_prices(
                asset,
                start,
                end,
                cache,
                alerts,
                crypto_list=crypto_list,
                max_stale_bdays=int(cfg.get("data.max_stale_bdays", 5)),
                cross_tol=float(cfg.get("data.cross_source_tol", 0.02)),
                cache_max_age_days=float(cfg.get("data.cache_max_age_days", 1.0)),
            )
        except DataUnavailableError as exc:
            blocked[asset] = str(exc)
            alerts.add("CRITICAL", f"{asset} BLOQUEADO (sin datos válidos): {exc} => señal NADA")
        except Exception as exc:  # noqa: BLE001 - fail-safe
            blocked[asset] = f"error inesperado: {exc!r}"
            alerts.add("CRITICAL", f"{asset} BLOQUEADO por error inesperado: {exc!r} => señal NADA")

    def _global(name: str, fn) -> pd.DataFrame:
        try:
            df = fn(start, end, cache, alerts)
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        except Exception as exc:  # noqa: BLE001 - fail-safe: una fuente global caída no tumba el pipeline
            alerts.add("ERROR", f"Fuente global '{name}' lanzó {exc!r}; se continúa sin ella")
            return pd.DataFrame()

    globals_parts = [
        _global("macro", get_macro),
        _global("sentiment", get_sentiment),
        _global("derivatives", get_derivatives),
        _global("onchain", get_onchain),
    ]
    parts = [p for p in globals_parts if p is not None and not p.empty]
    if parts:
        global_df = pd.concat(parts, axis=1).sort_index()
        global_df = global_df[~global_df.index.duplicated(keep="last")]
        global_df = global_df.resample("B").ffill()
    else:
        alerts.add(
            "CRITICAL", "Sin datos globales (macro/sentimiento/derivados/on-chain); señales serán NADA"
        )
        global_df = pd.DataFrame()
    log.info("market_data_loaded", assets_ok=list(prices), blocked=list(blocked), mode="real")
    return MarketData(prices=prices, global_df=global_df, blocked=blocked, data_mode="real")
