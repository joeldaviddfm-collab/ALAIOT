"""Carga de configuración: config.yaml + overrides + variables de entorno.

Nunca se leen API keys desde el código: solo vía variables de entorno (ver .env.example).
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Config:
    """Acceso tipado a config.yaml con creación de directorios de trabajo."""

    raw: dict[str, Any]
    root: Path
    data_dir: Path
    cache_dir: Path
    signals_dir: Path
    reports_dir: Path
    models_dir: Path

    def __init__(self, path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> None:
        path = Path(path) if path is not None else ROOT / "config.yaml"
        with open(path, encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)
        if overrides:
            raw = _deep_merge(raw, overrides)
        self.raw = raw
        self.root = ROOT
        for key in ("data_dir", "cache_dir", "signals_dir", "reports_dir", "models_dir"):
            target = ROOT / raw["paths"][key]
            target.mkdir(parents=True, exist_ok=True)
            setattr(self, key, target)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def seed(self) -> int:
        return int(self.get("seed", 42))

    @property
    def kill_switch(self) -> bool:
        """Kill switch global: archivo KILL_SWITCH en la raíz del repo o env QSE_KILL_SWITCH."""
        if (self.root / "KILL_SWITCH").exists():
            return True
        return bool(os.getenv("QSE_KILL_SWITCH", "").strip())

    @property
    def allow_synthetic(self) -> bool:
        """Datos sintéticos SOLO para demos/tests locales explícitos (--offline)."""
        return bool(os.getenv("QSE_ALLOW_SYNTHETIC", "").strip())
