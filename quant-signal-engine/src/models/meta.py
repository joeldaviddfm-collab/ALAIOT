"""Meta-labeling (López de Prado): segundo modelo binario que decide si tomar o no
la señal del primario, prediciendo P(resultado > 0) dados los features de contexto
y las probabilidades del primario.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

META_FEATURES = [
    "p_pos",
    "p_neg",
    "p_zero",
    "agreement",
    "atr_pct",
    "vix_z",
    "regime_id",
    "regime_prob",
    "vol_z_20",
    "is_crypto",
]


class MetaModel:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.seed = cfg.seed
        self.clf: Any = None

    def _new_clf(self):
        from lightgbm import LGBMClassifier

        p = dict(self.cfg.get("model.lgbm", {}))
        return LGBMClassifier(
            objective="binary",
            n_estimators=min(200, int(p.get("n_estimators", 400))),
            learning_rate=float(p.get("learning_rate", 0.05)),
            max_depth=3,
            num_leaves=15,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.seed,
            n_jobs=-1,
            verbosity=-1,
        )

    @staticmethod
    def build_meta_X(X: pd.DataFrame, proba: np.ndarray, agreement: np.ndarray) -> pd.DataFrame:
        meta = pd.DataFrame(
            {
                "p_pos": proba[:, 2],
                "p_neg": proba[:, 0],
                "p_zero": proba[:, 1],
                "agreement": agreement,
            }
        )
        for c in ("atr_pct", "vix_z", "regime_id", "regime_prob", "vol_z_20", "is_crypto"):
            if c in X.columns:
                meta[c] = X[c].to_numpy(dtype=float)
            else:
                meta[c] = 0.0
        return meta

    def fit(self, meta_X: pd.DataFrame, meta_y: np.ndarray) -> MetaModel:
        self.clf = self._new_clf()
        self.clf.fit(meta_X.to_numpy(dtype=float), np.asarray(meta_y, dtype=int))
        return self

    def predict_proba(self, meta_X: pd.DataFrame) -> np.ndarray:
        if self.clf is None:
            return np.full(len(meta_X), 0.5)
        p = self.clf.predict_proba(meta_X.to_numpy(dtype=float))
        return p[:, 1] if p.shape[1] == 2 else np.full(len(meta_X), 0.5)
