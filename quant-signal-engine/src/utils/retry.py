"""Reintentos con backoff exponencial y timeouts explícitos (fail-safe: nunca propaga)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

import structlog

log = structlog.get_logger("qse.retry")
T = TypeVar("T")


def with_retry(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_wait: float = 2.0,
    max_wait: float = 30.0,
    what: str = "call",
) -> T | None:
    """Ejecuta fn() hasta `attempts` veces con backoff exponencial.

    Si falla todas las veces devuelve None (el llamador debe tratar el dato como
    no disponible: fail-safe, nunca rellenar con suposiciones).
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - fail-safe explícito, se registra
            wait = min(base_wait * (2 ** (attempt - 1)), max_wait)
            log.warning("retry_failed", what=what, attempt=attempt, error=repr(exc), next_wait_s=wait)
            if attempt < attempts:
                time.sleep(wait)
    log.error("source_unavailable", what=what, attempts=attempts)
    return None
