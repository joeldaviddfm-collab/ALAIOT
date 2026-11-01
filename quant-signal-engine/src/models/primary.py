"""Modelo primario: clasificador multiclase {-1, 0, +1} (triple barrier).
LightGBM por defecto (SHAP nativo vía pred_contrib); XGBoost opcional como backend.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

CLASSES = np.array([-1, 0, 1])


class PrimaryModel:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.backend = str(cfg.get("model.backend", "lightgbm"))
        self.seed = cfg.seed
        self.clf: Any = None
        self.feature_names_: list[str] = []

    def _params(self) -> dict:
        p = dict(self.cfg.get("model.lgbm", {}))
        return {
            "n_estimators": int(p.get("n_estimators", 400)),
            "learning_rate": float(p.get("learning_rate", 0.05)),
            "max_depth": int(p.get("max_depth", 4)),
            "num_leaves": int(p.get("num_leaves", 31)),
            "min_child_samples": int(p.get("min_child_samples", 40)),
            "subsample": float(p.get("subsample", 0.8)),
            "colsample_bytree": float(p.get("colsample_bytree", 0.8)),
        }

    def _new_clf(self):
        params = self._params()
        if self.backend == "xgboost":
            from xgboost import XGBClassifier

            return XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                random_state=self.seed,
                n_jobs=-1,
                eval_metric="mlogloss",
            )
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="multiclass",
            num_class=3,
            random_state=self.seed,
            n_jobs=-1,
            verbosity=-1,
            **params,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.feature_names_ = list(X.columns)
        self.clf = self._new_clf()
        self.clf.fit(X.to_numpy(dtype=float), np.asarray(y, dtype=int))
        self.classes_ = np.asarray(self.clf.classes_, dtype=int)  # type: ignore[union-attr]
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve matriz (n, 3) con columnas ordenadas como CLASSES [-1, 0, 1]."""
        raw = self.clf.predict_proba(X.to_numpy(dtype=float))
        out = np.zeros((len(X), 3))
        for j, c in enumerate(self.classes_):
            out[:, int(np.where(c == CLASSES)[0][0])] = raw[:, j]
        return out

    def shap_top(self, X_row: pd.DataFrame, chosen_class_idx: int, k: int = 3) -> list[str]:
        """Top-k features por contribución absoluta (SHAP nativo de LightGBM).
        Con backend xgboost se usa importancia por ganancia como aproximación."""
        if self.backend == "lightgbm" and hasattr(self.clf, "predict"):
            contrib = self.clf.predict(X_row.to_numpy(dtype=float), pred_contrib=True)
            n_feat = len(self.feature_names_)
            if contrib.ndim == 3:
                vec = np.abs(contrib[0, chosen_class_idx, :n_feat])
            else:
                # multiclase: (n, n_class * (n_feat + 1)) con bloques por clase
                block = contrib[0, chosen_class_idx * (n_feat + 1) : (chosen_class_idx + 1) * (n_feat + 1)]
                vec = np.abs(block[:n_feat])
            order = np.argsort(-vec)[:k]
            return [self.feature_names_[i] for i in order]
        imp = np.asarray(getattr(self.clf, "feature_importances_", np.zeros(len(self.feature_names_))))
        order = np.argsort(-imp)[:k]
        return [self.feature_names_[i] for i in order]
