"""Evaluador en vivo: compara las señales pasadas de history.csv contra los resultados
reales (mismo criterio que el backtest) y calcula la precisión por umbral de confianza.

Uso: python -m src.signals.evaluator [--history signals/history.csv]
"""

from __future__ import annotations

import argparse

import pandas as pd
import structlog

from src.config import ROOT, Config
from src.data.pipeline import load_market_data
from src.labeling.triple_barrier import realized_r_multiple
from src.utils.alerts import AlertBus

log = structlog.get_logger("qse.evaluator")


def evaluate_signals(history: pd.DataFrame, prices: dict[str, pd.DataFrame], cfg) -> pd.DataFrame:
    """Para cada señal long/short calcula el resultado realizado (R-multiple) y si fue
    acierto (R > 0). Devuelve DataFrame con resultados por señal."""
    sl_mult = float(cfg.get("model.triple_barrier.atr_mult_sl", 2.0))
    tp_mult = float(cfg.get("model.triple_barrier.atr_mult_tp", 3.0))
    max_h = int(cfg.get("model.triple_barrier.max_horizon_days", 10))
    rows = []
    for _, s in history.iterrows():
        direction = str(s.get("direction", "nada"))
        if direction not in ("long", "short"):
            continue
        asset = s["asset"]
        if asset not in prices:
            continue
        df = prices[asset]
        d = pd.Timestamp(s["date"])
        pos = df.index.get_indexer([d])
        if len(pos) == 0 or pos[0] < 0 or pos[0] >= len(df) - 1:
            continue
        i = int(pos[0])
        side = 1 if direction == "long" else -1
        r = realized_r_multiple(df, i, side, atr_mult_sl=sl_mult, atr_mult_tp=tp_mult, max_h=max_h)
        rows.append(
            {
                "date": s["date"],
                "asset": asset,
                "direction": direction,
                "prob": float(s.get("prob", 0.0)),
                "realized_r": round(float(r), 4),
                "hit": int(r > 0),
                "entry_ref": s.get("entry_ref"),
                "stop": s.get("stop"),
                "tp": s.get("tp"),
            }
        )
    return pd.DataFrame(rows)


def accuracy_by_threshold(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=["bucket", "n", "accuracy"])
    bins = [0.0, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01]
    labels = ["<0.70", "0.70-0.75", "0.75-0.80", "0.80-0.85", "0.85-0.90", ">=0.90"]
    results = results.copy()
    results["bucket"] = pd.cut(results["prob"], bins=bins, labels=labels, right=False, include_lowest=True)
    g = (
        results.groupby("bucket", observed=True)
        .agg(n=("hit", "size"), accuracy=("hit", "mean"))
        .reset_index()
    )
    return g


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evalúa señales pasadas contra resultados reales (paper)")
    parser.add_argument("--history", default=str(ROOT / "signals" / "history.csv"))
    parser.add_argument("--offline", action="store_true", help="usa datos sintéticos (demo/test)")
    parser.add_argument("--assets", nargs="+", default=None)
    args = parser.parse_args(argv)
    cfg = Config()
    alerts = AlertBus(cfg.reports_dir)
    assets = args.assets or list(cfg.get("assets.default", []))
    market = load_market_data(cfg, assets, offline=args.offline, alerts=alerts)
    hist = pd.read_csv(args.history)
    results = evaluate_signals(hist, market.prices, cfg)
    if results.empty:
        print("No hay señales evaluables en el histórico.")
        return 0
    acc = accuracy_by_threshold(results)
    overall = float(results["hit"].mean())
    out_csv = cfg.reports_dir / "live_accuracy.csv"
    results.to_csv(out_csv, index=False)
    acc.to_csv(cfg.reports_dir / "live_accuracy_by_threshold.csv", index=False)
    md = f"# Precisión en vivo (paper trading)\n\n- Señales evaluadas: {len(results)}\n- Precisión global: **{overall:.2%}**\n\n## Por umbral de confianza\n\n{acc.to_markdown(index=False)}\n"
    (cfg.reports_dir / "live_accuracy.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Guardado en {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
