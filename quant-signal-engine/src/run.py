"""Entrada principal — SOLO MODO PAPER TRADING.

Uso:
    python -m src.run --assets BTC-USD ETH-USD SPY QQQ EURUSD=X --mode paper
    python -m src.run --mode paper --offline        # demo/test con datos sintéticos
    python -m src.run --mode paper --retrain        # reentrena el modelo

El modo live está desactivado y fuera de alcance: no se envía ninguna orden real.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.config import ROOT, Config
from src.data.pipeline import load_market_data
from src.models.pipeline import SignalModel
from src.signals.generator import generate_signals, write_signals
from src.training import build_training_set
from src.utils.alerts import AlertBus
from src.utils.logging import setup_logging
from src.utils.seeds import set_seeds
from src.utils.telegram import notify_signals


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="src.run", description="quant-signal-engine (solo PAPER TRADING)")
    p.add_argument("--assets", nargs="+", default=None, help="lista de activos (default: config.yaml)")
    p.add_argument(
        "--mode", default="paper", choices=["paper"], help="solo se admite 'paper'; live está desactivado"
    )
    p.add_argument("--date", default=None, help="fecha de la señal (YYYY-MM-DD); default: hoy")
    p.add_argument("--offline", action="store_true", help="datos sintéticos (solo demo/test local)")
    p.add_argument("--retrain", action="store_true", help="fuerza reentrenamiento del modelo")
    p.add_argument("--config", default=None, help="ruta alternativa de config.yaml")
    return p.parse_args(argv)


def train_model(cfg, market, alerts: AlertBus) -> SignalModel:
    X, y, _groups = build_training_set(cfg, market)
    feat_cols = [c for c in X.columns if c not in ("date", "asset", "close_ref", "atr14")]
    model = SignalModel(cfg)
    alerts.add("INFO", f"Entrenando modelo sobre {len(X)} filas ({feat_cols.__len__()} features)...")
    model.fit(X[["date", "asset"] + feat_cols], y, fast_calibration=False)
    model_path = cfg.models_dir / "signal_model.joblib"
    model.save(model_path)
    alerts.add("INFO", f"Modelo guardado en {model_path}")
    return model


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode != "paper":  # argparse ya lo restringe; doble seguro
        print("ERROR: solo existe el modo 'paper'. El modo live está desactivado.", file=sys.stderr)
        return 2
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except Exception:
        pass
    cfg = Config(args.config)
    log = setup_logging(ROOT / "logs" / "run.log")
    set_seeds(cfg.seed)
    alerts = AlertBus(cfg.reports_dir)
    log.info("run_start", mode="paper", offline=args.offline, assets=args.assets)

    assets = args.assets or list(cfg.get("assets.default", []))
    market = load_market_data(cfg, assets, offline=args.offline, alerts=alerts)

    model_path = cfg.models_dir / "signal_model.joblib"
    model = None
    if not args.retrain and model_path.exists():
        try:
            model = SignalModel.load(model_path)
            log.info("model_loaded", path=str(model_path))
        except Exception as exc:  # noqa: BLE001
            alerts.add("ERROR", f"No se pudo cargar el modelo ({exc!r}); se reentrena")
            model = None
    if model is None:
        if not market.prices:
            alerts.add("CRITICAL", "Sin datos de precios: no se puede entrenar; todas las señales serán NADA")
            model = None
        else:
            model = train_model(cfg, market, alerts)

    asof = pd.Timestamp(args.date) if args.date else None
    signals = generate_signals(cfg, market, model, alerts, asof_date=asof)
    write_signals(cfg, signals)

    n_sig = int((signals["direction"] != "nada").sum())
    print("\n=== SEÑALES (PAPER TRADING) ===")
    print(signals.to_string(index=False))
    print(f"\nSeñales emitidas: {n_sig}/{len(signals)} | Alertas: {alerts.summary()}")
    print("Esto no es asesoría financiera ni garantiza rentabilidad.")
    notify_signals(cfg, alerts, f"Señales emitidas: {n_sig}/{len(signals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
