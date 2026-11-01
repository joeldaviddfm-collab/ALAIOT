"""Ensamble: LightGBM + regresión logística + random forest (soft vote).
`agreement` = fracción de miembros cuyo argmax coincide con el del ensamble;
el pipeline exige un acuerdo mínimo para emitir señal (anti-overfitting).
Los miembros que no aceptan NaN reciben imputación por mediana.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.primary import CLASSES


class EnsembleModel:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.seed = cfg.seed
        self.imputer = SimpleImputer(strategy="median")
        lgbm_params = dict(cfg.get("model.lgbm", {}))
        from lightgbm import LGBMClassifier

        self.members = [
            LGBMClassifier(
                objective="multiclass",
                num_class=3,
                n_estimators=int(lgbm_params.get("n_estimators", 400)),
                learning_rate=float(lgbm_params.get("learning_rate", 0.05)),
                max_depth=int(lgbm_params.get("max_depth", 4)),
                num_leaves=int(lgbm_params.get("num_leaves", 31)),
                min_child_samples=int(lgbm_params.get("min_child_samples", 40)),
                subsample=float(lgbm_params.get("subsample", 0.8)),
                colsample_bytree=float(lgbm_params.get("colsample_bytree", 0.8)),
                random_state=self.seed,
                n_jobs=-1,
                verbosity=-1,
            ),
            make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=self.seed),
            ),
            RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=self.seed,
                n_jobs=-1,
            ),
        ]
        self.classes_ = CLASSES

    def _prep(self, X: pd.DataFrame) -> np.ndarray:
        return self.imputer.transform(X.to_numpy(dtype=float))

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> EnsembleModel:
        Xv = self.imputer.fit_transform(X.to_numpy(dtype=float))
        for m in self.members:
            m.fit(Xv, np.asarray(y, dtype=int))
        return self

    def _proba_aligned(self, m, Xv: np.ndarray) -> np.ndarray:
        raw = m.predict_proba(Xv)
        out = np.zeros((Xv.shape[0], 3))
        for j, c in enumerate(m.classes_):
            out[:, int(np.where(c == CLASSES)[0][0])] = raw[:, j]
        return out

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xv = self._prep(X)
        return np.mean([self._proba_aligned(m, Xv) for m in self.members], axis=0)

    def agreement(self, X: pd.DataFrame) -> np.ndarray:
        Xv = self._prep(X)
        ens = self.predict_proba(X).argmax(axis=1)
        votes = np.mean([(self._proba_aligned(m, Xv).argmax(axis=1) == ens) for m in self.members], axis=0)
        return votes
