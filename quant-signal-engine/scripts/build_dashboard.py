"""Genera reports/dashboard.html: un dashboard estático auto-contenido (un solo
archivo HTML, sin servidor) a partir de signals/ + reports/.

Uso: python scripts/build_dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config  # noqa: E402
from src.reporting.dashboard import build_dashboard  # noqa: E402


def main() -> int:
    cfg = Config()
    path = build_dashboard(cfg)
    print(f"Dashboard generado: {path}")
    print("Ábrelo con doble clic en el navegador (no necesita servidor).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
