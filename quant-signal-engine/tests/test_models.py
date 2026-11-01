"""Modelos: primario (fit/predict/SHAP), calibración, meta, ensamble y pipeline completo."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.models.calibrate import IsotonicCalibrator, PlattCalibrator
from src.models.pipeline import SignalModel
from src.training import build_training_set


def _trained_model(cfg, small_market) -> SignalModel:
    X, y, _ = build_training_set(cfg, small_market)
    cols = [c for c in X.columns if c not in ("date", "asset", "close_ref", "atr14")]
    m = SignalModel(cfg)
    m.fit(X[["date", "asset"] + cols], y, fast_calibration=True)
    return m, X, cols, y


def test_primary_predict_proba_shape_and_top_features(cfg, small_market):
    m, X, cols, y = _trained_model(cfg, small_market)
    preds = m.predict(X[["date", "asset"] + cols].head(20))
    assert preds.shape[0] == 20
    for c in ("direction", "prob", "prob_pos", "prob_neg", "prob_ens_pos", "meta_prob", "agreement"):
        assert c in preds.columns
    assert set(preds["direction"].unique()) <= {-1, 0, 1}
    assert preds["prob"].between(0, 1).all()
    row = X.iloc[[0]]
    top = m.top_features(row, 1, k=3)
    assert len(top) == 3 and all(isinstance(t, str) for t in top)


def test_calibrators_normalize(cfg, small_market):
    m, X, cols, y = _trained_model(cfg, small_market)
    Xf = X[cols]
    proba = m.primary.predict_proba(Xf.head(50))
    for cal_cls in (IsotonicCalibrator, PlattCalibrator):
        cal = cal_cls().fit(proba, y[:50])
        out = cal.transform(proba)
        assert out.shape == proba.shape
        assert np.allclose(out.sum(axis=1), 1.0)
        assert (out >= 0).all() and (out <= 1).all()


def test_model_save_load(cfg, small_market, tmp_path):
    m, X, cols, y = _trained_model(cfg, small_market)
    path = tmp_path / "model.joblib"
    m.save(path)
    m2 = SignalModel.load(path)
    p1 = m.predict(X[["date", "asset"] + cols].head(10))
    p2 = m2.predict(X[["date", "asset"] + cols].head(10))
    pd.testing.assert_frame_equal(p1, p2)


def test_threshold_gating_zero_direction(cfg, small_market):
    m, X, cols, y = _trained_model(cfg, small_market)
    m.threshold = 0.999  # umbral inalcanzable => todo NADA
    preds = m.predict(X[["date", "asset"] + cols])
    assert (preds["direction"] == 0).all()
    assert (preds["prob"] == 0).all()
