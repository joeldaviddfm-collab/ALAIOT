"""TEST OBLIGATORIO: fuga de información. Entrenar con etiquetas barajadas debe
rendir como azar (AUC ~ 0.5). Si rinde mejor, hay leakage."""

from __future__ import annotations

import numpy as np
from src.training import build_training_set
from src.validation.robustness import leakage_auc


def test_shuffled_labels_auc_near_chance(cfg, small_market):
    X, y, _ = build_training_set(cfg, small_market)
    feat_cols = [c for c in X.columns if c not in ("date", "asset", "close_ref", "atr14")]
    auc = leakage_auc(X[feat_cols], y, seed=cfg.seed, n_splits=3)
    assert auc < 0.65, f"AUC con etiquetas barajadas = {auc:.3f} (esperado ~0.5); hay fuga de información"


def test_real_labels_beat_shuffled(cfg, small_market):
    """Saneamiento: con etiquetas reales el AUC debe superar al azar claramente."""
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    X, y, _ = build_training_set(cfg, small_market)
    feat_cols = [c for c in X.columns if c not in ("date", "asset", "close_ref", "atr14")]
    Xv = X[feat_cols].to_numpy(dtype=float)
    y_bin = (y > 0).astype(int)
    if y_bin.sum() < 10:
        import pytest

        pytest.skip("pocas etiquetas positivas en el set sintético pequeño")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg.seed)
    aucs = []
    for tr, te in skf.split(Xv, y_bin):
        clf = LGBMClassifier(
            objective="binary",
            n_estimators=40,
            max_depth=3,
            num_leaves=7,
            min_child_samples=10,
            random_state=cfg.seed,
            verbosity=-1,
        )
        clf.fit(Xv[tr], y_bin[tr])
        aucs.append(roc_auc_score(y_bin[te], clf.predict_proba(Xv[te])[:, 1]))
    assert float(np.mean(aucs)) > 0.55
