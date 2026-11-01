"""Validación de calidad de datos (pydantic + chequeos de DataFrame).

Detecta: filas inválidas (high<low, precios<=0), duplicados, timestamps no
monotónicos, datos stale, huecos, saltos anómalos, volumen cero sostenido.
Además: validación cruzada entre dos fuentes para bloquear discrepancias.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class Bar(BaseModel):
    """Esquema de una vela OHLCV (validación por fila en streaming)."""

    model_config = ConfigDict(extra="forbid")
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @field_validator("open", "high", "low", "close", "volume")
    @classmethod
    def finite_nonnegative(cls, v: float) -> float:
        if not math.isfinite(v) or v < 0:
            raise ValueError("valor no finito o negativo")
        return v

    @model_validator(mode="after")
    def consistent(self) -> Bar:
        if self.high < self.low:
            raise ValueError("high < low")
        if self.high <= 0 or self.low <= 0:
            raise ValueError("precios <= 0")
        if max(self.open, self.close) > self.high or min(self.open, self.close) < self.low:
            raise ValueError("open/close fuera de [low, high]")
        return self


def validate_bar_row(row: dict[str, float]) -> list[str]:
    try:
        Bar(
            **{k: float(row[k]) for k in ("open", "high", "low", "close") if k in row},
            **({"volume": float(row["volume"])} if "volume" in row else {}),
        )
        return []
    except Exception as exc:  # noqa: BLE001
        return [str(exc)]


@dataclass
class ValidationReport:
    ok: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stale: bool = False
    n_rows: int = 0

    @property
    def critical(self) -> list[str]:
        return self.issues


def validate_ohlcv(
    df: pd.DataFrame | None,
    *,
    max_stale_bdays: int = 5,
    jump_sigma: float = 10.0,
    today: pd.Timestamp | None = None,
) -> ValidationReport:
    """Valida un DataFrame OHLCV. `ok=False` => el activo queda bloqueado (fail-safe)."""
    if df is None or df.empty:
        return ValidationReport(ok=False, issues=["sin datos"], stale=True, n_rows=0)
    issues: list[str] = []
    warnings: list[str] = []
    n = len(df)
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        issues.append(f"columnas faltantes: {sorted(missing)}")
        return ValidationReport(ok=False, issues=issues, n_rows=n)
    if not isinstance(df.index, pd.DatetimeIndex):
        issues.append("índice no es DatetimeIndex")
    else:
        if not df.index.is_monotonic_increasing:
            issues.append("timestamps no monotónicos")
        if df.index.has_duplicates:
            issues.append(f"timestamps duplicados: {int(df.index.duplicated().sum())}")
    numeric = df[["open", "high", "low", "close", "volume"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        issues.append("valores no finitos (NaN/inf)")
    bad = (df["high"] < df["low"]) | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if bad.any():
        issues.append(f"{int(bad.sum())} velas inválidas (high<low o precio<=0)")
    if (df["volume"] < 0).any():
        issues.append("volumen negativo")
    # datos desactualizados (stale)
    today = today or pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    last = df.index[-1]
    stale_bdays = int(np.busday_count(last.date(), today.date())) if last < today else 0
    stale = stale_bdays > max_stale_bdays
    if stale:
        issues.append(f"datos stale: último dato {last.date()} ({stale_bdays} días hábiles atrás)")
    # huecos
    if isinstance(df.index, pd.DatetimeIndex) and n > 5:
        expected = len(pd.bdate_range(df.index[0], df.index[-1]))
        if expected > 0 and n / expected < 0.5:
            warnings.append(f"huecos: {n} filas vs {expected} días hábiles esperados")
    # saltos anómalos
    ret = df["close"].pct_change()
    med = ret.abs().rolling(60, min_periods=20).median()
    jumps = (ret.abs() > jump_sigma * med) & med.notna()
    if jumps.any():
        worst = float(ret.abs()[jumps].max())
        warnings.append(f"{int(jumps.sum())} saltos anómalos (máx |ret|={worst:.1%})")
    # volumen cero sostenido
    zero_streak = (df["volume"] <= 0).rolling(5, min_periods=5).sum() >= 5
    if zero_streak.any():
        warnings.append("volumen cero sostenido (>=5 velas)")
    ok = not issues
    return ValidationReport(ok=ok, issues=issues, warnings=warnings, stale=stale, n_rows=n)


def cross_source_check(df_a: pd.DataFrame, df_b: pd.DataFrame, *, tol: float = 0.02) -> tuple[bool, float]:
    """Compara el close de dos fuentes en fechas comunes. Devuelve (ok, max_diff_rel)."""
    if df_a is None or df_b is None or df_a.empty or df_b.empty:
        return False, float("inf")
    common = df_a.index.intersection(df_b.index)
    if len(common) < 20:
        return True, 0.0  # pocas fechas comunes: no se puede cruzar, no bloquear
    a = df_a.loc[common, "close"]
    b = df_b.loc[common, "close"]
    diff = ((a - b).abs() / a.abs()).max()
    return bool(diff <= tol), float(diff)
