"""TEST OBLIGATORIO: idempotencia. Correr dos veces la generación de señales del mismo
día NO debe duplicar filas en signals/history.csv."""

from __future__ import annotations

import pandas as pd
from src.models.pipeline import SignalModel
from src.signals.generator import generate_signals, write_signals
from src.training import build_training_set
from src.utils.alerts import AlertBus


def test_history_idempotent(cfg, small_market, tmp_path, monkeypatch):
    monkeypatch.setattr(type(cfg), "signals_dir", tmp_path / "signals", raising=False)
    cfg.signals_dir = tmp_path / "signals"
    cfg.signals_dir.mkdir(parents=True, exist_ok=True)
    X, y, _ = build_training_set(cfg, small_market)
    feat_cols = [c for c in X.columns if c not in ("date", "asset", "close_ref", "atr14")]
    model = SignalModel(cfg)
    model.fit(X[["date", "asset"] + feat_cols], y, fast_calibration=True)

    s1 = generate_signals(cfg, small_market, model, AlertBus(tmp_path))
    write_signals(cfg, s1)
    hist1 = pd.read_csv(cfg.signals_dir / "history.csv")
    s2 = generate_signals(cfg, small_market, model, AlertBus(tmp_path))
    write_signals(cfg, s2)
    hist2 = pd.read_csv(cfg.signals_dir / "history.csv")
    assert len(hist1) == len(hist2), "history.csv creció al correr dos veces el mismo día"
    assert not hist2.duplicated(subset=["date", "asset"]).any()
    pd.testing.assert_frame_equal(hist1, hist2)
