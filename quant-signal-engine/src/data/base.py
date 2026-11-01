"""Contratos base de las fuentes de datos y ejecución fail-safe."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import pandas as pd
import structlog

from src.utils.alerts import AlertBus
from src.utils.retry import with_retry

log = structlog.get_logger("qse.data")
T = TypeVar("T")


class DataUnavailableError(RuntimeError):
    """Se lanza cuando ninguna fuente pudo entregar datos válidos para un activo."""


@dataclass
class SourceResult:
    df: pd.DataFrame
    source: str
    fetched_at: dt.datetime


def safe_call(
    name: str,
    fn: Callable[[], T],
    alerts: AlertBus,
    *,
    attempts: int = 3,
) -> T | None:
    """Ejecuta fn() con reintentos. Si falla todo, registra alerta y devuelve None.

    Política fail-safe: ante un error de API el dato se considera NO disponible
    (nunca se inventa ni se rellena con suposiciones).
    """
    result = with_retry(fn, attempts=attempts, what=name)
    if result is None:
        alerts.add(
            "ERROR", f"Fuente '{name}' no disponible tras {attempts} intentos; se activa fallback/NADA"
        )
    return result
