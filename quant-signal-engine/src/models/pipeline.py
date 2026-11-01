"""SignalModel: orquesta primario + calibración + ensamble + meta-labeling.

fit(): walk-forward interno para obtener predicciones OOF (out-of-fold) con las que
calibrar probabilidades y entrenar el meta-modelo (sin leakage). Luego reentrena
todos los modelos sobre el total de los datos disponibles.
predict(): señal solo si prob. calibrada >= threshold, meta_prob >= meta_min y
acuerdo del ensamble >= min_agreement; si no, direction = 0 (NADA).
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.models.calibrate import build_calibrator
from src.models.ensemble import EnsembleModel
from src.models.meta import MetaModel
from src.models.primary import PrimaryModel
from src.validation.walkforward import wf_splits


class SignalModel:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.primary = PrimaryModel(cfg)
        self.ensemble = EnsembleModel(cfg)
        self.meta = MetaModel(cfg)
        self.calibrator = build_calibrator(cfg.get("model.calibration", "isotonic"))
        self.threshold = float(cfg.get("model.threshold", 0.70))
        self.meta_min = float(cfg.get("model.meta_min", 0.55))
        self.min_agreement = float(cfg.get("model.min_agreement", 0.67))
        self.feature_names_: list[str] = []
        self.regime_perf: dict[int, float] = {}
        self.n_configs_tried = int(cfg.get("model.n_configs_tried", 1))
        self.trained = False

    # ------------------------------------------------------------- fit
    def fit(self, X: pd.DataFrame, y: np.ndarray, *, fast_calibration: bool = False) -> SignalModel:
        feat_cols = [c for c in X.columns if c not in ("date", "asset")]
        self.feature_names_ = feat_cols
        Xf = X[feat_cols]
        y = np.asarray(y, dtype=int)
        dates = pd.DatetimeIndex(X["date"])

        # OOF para calibrar y entrenar el meta-modelo. Con fast_calibration se usa
        # una única partición (para no anidar walk-forward dentro de walk-forward).
        oof_proba = np.full((len(X), 3), np.nan)
        if fast_calibration and len(X) > 200:
            cut = int(len(X) * 0.8)
            splits = [(np.arange(0, cut), np.arange(cut, len(X)))]
        else:
            splits = wf_splits(
                dates,
                train_days=int(self.cfg.get("model.walk_forward.train_days", 504)),
                embargo_days=int(self.cfg.get("model.walk_forward.embargo_days", 21)),
                test_days=int(self.cfg.get("model.walk_forward.test_days", 63)),
            )
        for tr_pos, te_pos in splits:
            if len(tr_pos) < 100:
                continue
            m = PrimaryModel(self.cfg)
            m.fit(Xf.iloc[tr_pos], y[tr_pos])
            oof_proba[te_pos] = m.predict_proba(Xf.iloc[te_pos])
        mask = ~np.isnan(oof_proba[:, 0])
        if mask.sum() > 50:
            self.calibrator.fit(oof_proba[mask], y[mask])
            # meta-labeling: ¿el primario acertó en OOF?
            proba_cal = self.calibrator.transform(oof_proba[mask])
            pos, neg = proba_cal[:, 2], proba_cal[:, 0]
            pred_dir = np.where(
                (pos >= self.threshold) & (pos > neg),
                1,
                np.where((neg >= self.threshold) & (neg > pos), -1, 0),
            )
            meta_y = (y[mask] == pred_dir).astype(int)
            Xm = Xf.iloc[mask].reset_index(drop=True)
            ens_tmp = EnsembleModel(self.cfg)
            ens_tmp.fit(Xm, y[mask])
            agree = ens_tmp.agreement(Xm)
            meta_X = MetaModel.build_meta_X(Xm, proba_cal, agree)
            self.meta.fit(meta_X, meta_y)
            # rendimiento histórico por régimen (para el filtro de contexto)
            if "regime_id" in X.columns:
                regimes = X["regime_id"].to_numpy(dtype=float)[mask]
                for r in np.unique(regimes[~np.isnan(regimes)]):
                    sel = regimes == r
                    if sel.sum() > 20 and (pred_dir[sel] != 0).any():
                        self.regime_perf[int(r)] = float(
                            np.mean((y[mask][sel] == pred_dir[sel])[pred_dir[sel] != 0])
                        )
        else:
            # sin OOF suficiente: calibrador trivial (uniforme) del tamaño de y
            self.calibrator.fit(np.tile([0.33, 0.34, 0.33], (len(y), 1)), y)

        # reentrenar sobre TODO el histórico disponible
        self.primary.fit(Xf, y)
        self.ensemble.fit(Xf, y)
        self.trained = True
        return self

    # ---------------------------------------------------------- predict
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        feat_cols = self.feature_names_ or [c for c in X.columns if c not in ("date", "asset")]
        Xf = X[feat_cols]
        proba = self.primary.predict_proba(Xf)
        proba_cal = self.calibrator.transform(proba)
        ens_proba = self.ensemble.predict_proba(Xf)
        agree = self.ensemble.agreement(Xf)
        meta_X = MetaModel.build_meta_X(Xf, proba_cal, agree)
        meta_prob = (
            self.meta.predict_proba(meta_X) if getattr(self, "meta_reliable", True) else np.ones(len(X))
        )

        pos = proba_cal[:, 2]
        neg = proba_cal[:, 0]
        direction = np.zeros(len(X), dtype=int)
        prob = np.maximum(pos, neg)
        long_ok = (pos >= self.threshold) & (pos > neg)
        short_ok = (neg >= self.threshold) & (neg > pos)
        direction[long_ok] = 1
        direction[short_ok] = -1
        prob[long_ok] = pos[long_ok]
        prob[short_ok] = neg[short_ok]
        # gates: meta-labeling + acuerdo del ensamble
        gate = (meta_prob >= self.meta_min) & (agree >= self.min_agreement)
        direction[~gate] = 0
        prob = np.where(direction != 0, prob, 0.0)

        return pd.DataFrame(
            {
                "direction": direction,
                "prob": prob,
                "prob_pos": proba_cal[:, 2],
                "prob_neg": proba_cal[:, 0],
                "prob_zero": proba_cal[:, 1],
                "prob_ens": np.maximum(ens_proba[:, 2], ens_proba[:, 0]),
                "prob_ens_pos": ens_proba[:, 2],
                "prob_ens_neg": ens_proba[:, 0],
                "meta_prob": meta_prob,
                "agreement": agree,
            }
        )

    def top_features(self, X_row: pd.DataFrame, direction: int, k: int = 3) -> list[str]:
        if not self.trained:
            return []
        cls_idx = 2 if direction == 1 else 0
        return self.primary.shap_top(X_row[self.feature_names_], cls_idx, k=k)

    # ------------------------------------------------------------ io
    def save(self, path) -> None:
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path) -> SignalModel:
        return joblib.load(path)
