"""Regresión de métricas en CI: compara reports/metrics.json (run actual) contra
reports/baseline_metrics.json (versión de referencia commiteada).

Falla (exit 1) si el Sharpe ciego cae > 20% relativo, el max drawdown empeora > 5 pp,
el nº de operaciones cae > 50%, o el profit factor cae > 20%.
Solo aplica cuando ambos archivos son del mismo modo de datos (data_mode).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    cur_path = ROOT / "reports" / "metrics.json"
    base_path = ROOT / "reports" / "baseline_metrics.json"
    if not cur_path.exists() or not base_path.exists():
        print("Sin métricas actuales o de baseline; se omite la comparación.")
        return 0
    cur = json.loads(cur_path.read_text())
    base = json.loads(base_path.read_text())
    cb, bb = cur.get("blind", {}), base.get("blind", {})
    if cur.get("data_mode") != base.get("data_mode"):
        print(f"Modo de datos distinto ({cur.get('data_mode')} vs {base.get('data_mode')}); se omite.")
        return 0
    checks = [
        ("sharpe", bb.get("sharpe", 0), cb.get("sharpe", 0), 0.20, "caída relativa"),
        ("profit_factor", bb.get("profit_factor", 1), cb.get("profit_factor", 1), 0.20, "caída relativa"),
        ("n_trades", bb.get("n_trades", 1), cb.get("n_trades", 0), 0.50, "caída relativa"),
    ]
    failed = []
    for name, old, new, tol, kind in checks:
        if old in (None, 0) or new is None:
            continue
        if old > 0 and new < old * (1 - tol):
            failed.append(f"{name}: {old:.3f} -> {new:.3f} ({kind} > {tol:.0%})")
    old_dd, new_dd = abs(bb.get("max_drawdown", 0) or 0), abs(cb.get("max_drawdown", 0) or 0)
    if new_dd > old_dd + 0.05:
        failed.append(f"max_drawdown: {old_dd:.2%} -> {new_dd:.2%} (empeora > 5 pp)")
    if failed:
        print("REGRESIÓN DETECTADA:\n- " + "\n- ".join(failed))
        return 1
    print("Regresión OK: métricas dentro del margen respecto al baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
