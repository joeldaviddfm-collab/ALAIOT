"""Filtros de contexto: calendario económico (FOMC/CPI/NFP) y condiciones de mercado
peligrosas (VIX extremo, ATR anómalo, iliquidez, régimen malo)."""

from __future__ import annotations

import pandas as pd

from src.config import ROOT


def load_event_dates() -> dict[pd.Timestamp, str]:
    path = ROOT / "calendars" / "economic.csv"
    events: dict[pd.Timestamp, str] = {}
    if not path.exists():
        return events
    df = pd.read_csv(path, parse_dates=["date"])
    for _, r in df.iterrows():
        if str(r.get("impact", "high")) == "high":
            events[pd.Timestamp(r["date"])] = str(r["event"])
    return events


def event_block(
    date: pd.Timestamp, events: dict[pd.Timestamp, str], before: int = 0, after: int = 1
) -> str | None:
    """Devuelve el nombre del evento si `date` cae en la ventana bloqueada.

    Un evento en e bloquea las fechas [e - before, e + after]; para la fecha q se
    comprueban los eventos en [q - after, q + before].
    """
    for offset in range(-after, before + 1):
        d = date + pd.Timedelta(days=offset)
        if d in events:
            return events[d]
    return None


def context_block(row: pd.Series, cfg, events: dict[pd.Timestamp, str], date: pd.Timestamp) -> str | None:
    """Devuelve la razón de bloqueo o None si el contexto permite operar."""
    cf = cfg.get("context_filters", {})
    if bool(cf.get("block_events", True)):
        ev = event_block(
            date, events, int(cf.get("block_event_days_before", 0)), int(cf.get("block_event_days_after", 1))
        )
        if ev:
            return f"evento de alto impacto ({ev})"
    vix_z = row.get("vix_z")
    if pd.notna(vix_z) and float(vix_z) > float(cf.get("vix_z_max", 3.0)):
        return f"VIX extremo (z={vix_z:.2f})"
    atr_z = row.get("atr_pct_z")
    if pd.notna(atr_z) and float(atr_z) > float(cf.get("atr_z_max", 5.0)):
        return f"ATR anómalo (z={atr_z:.2f})"
    vol_z = row.get("vol_z_20")
    if pd.notna(vol_z) and float(vol_z) < float(cf.get("volume_z_min", -2.0)):
        return f"liquidez baja (vol z={vol_z:.2f})"
    return None
