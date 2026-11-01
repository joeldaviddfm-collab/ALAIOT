"""Healthcheck diario.

Comprueba: (1) que existan señales recientes en signals/history.csv,
(2) que el cache de datos no esté viejo, (3) que no haya alertas CRITICAL de hoy.
Si falla: imprime el problema, deja constancia en reports/ y, en GitHub Actions
(con GITHUB_TOKEN y GITHUB_REPOSITORY), abre un Issue automático.

Uso: python scripts/healthcheck.py   (exit 0 = OK, 1 = fallo)
"""

from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from src.config import Config  # noqa: E402


def main() -> int:
    cfg = Config()
    issues: list[str] = []

    hist = cfg.signals_dir / "history.csv"
    if not hist.exists():
        issues.append("signals/history.csv no existe (no se han generado señales)")
    else:
        df = pd.read_csv(hist)
        if df.empty:
            issues.append("signals/history.csv está vacío")
        else:
            last = pd.Timestamp(df["date"].max())
            age = (pd.Timestamp.utcnow().normalize() - last).days
            if age > 2:
                issues.append(f"última señal tiene {age} días (stale; esperado <= 2)")

    alerts_log = cfg.reports_dir / "alerts.log"
    if alerts_log.exists():
        today = dt.datetime.now(dt.UTC).date().isoformat()
        for line in alerts_log.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"[{today}") and "CRITICAL" in line:
                issues.append(f"alerta CRITICAL hoy: {line[:140]}")

    cache_files = list(cfg.cache_dir.rglob("*.csv")) if cfg.cache_dir.exists() else []
    if not cache_files:
        issues.append("cache de datos vacía (data/cache sin CSVs)")
    else:
        newest = max(f.stat().st_mtime for f in cache_files)
        age_h = (dt.datetime.now().timestamp() - newest) / 3600.0
        if age_h > 30:
            issues.append(f"datos en cache con {age_h:.0f}h de antigüedad (> 30h)")

    if issues:
        msg = "HEALTHCHECK FALLÓ:\n- " + "\n- ".join(issues)
        print(msg)
        (cfg.reports_dir / "healthcheck.log").parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.reports_dir / "healthcheck.log", "a", encoding="utf-8") as fh:
            fh.write(f"[{dt.datetime.now(dt.UTC).isoformat()}] {msg}\n")
        token = os.getenv("GITHUB_TOKEN", "")
        repo = os.getenv("GITHUB_REPOSITORY", "")
        if token and repo:
            try:
                subprocess.run(
                    [
                        "gh",
                        "api",
                        f"repos/{repo}/issues",
                        "-f",
                        f"title=🚨 Healthcheck diario falló ({dt.date.today()})",
                        "-f",
                        f"body={msg}",
                    ],
                    check=False,
                    capture_output=True,
                    env={**os.environ, "GH_TOKEN": token},
                )
            except Exception as exc:  # noqa: BLE001
                print(f"No se pudo abrir el issue automático: {exc!r}")
        return 1
    print("Healthcheck OK: señales y datos recientes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
