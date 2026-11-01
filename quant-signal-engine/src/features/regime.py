"""Régimen de mercado con Gaussian Mixture (3 estados) ajustado en ventana expansiva:
el régimen en t solo usa datos <= t (sin look-ahead). Estados etiquetados por retorno
medio histórico: 0=bajista, 1=lateral, 2=alcista.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


class RegimeDetector:
    def __init__(self, n_states: int = 3, refit_days: int = 60, min_train: int = 250, seed: int = 42) -> None:
        self.n_states = n_states
        self.refit_days = refit_days
        self.min_train = min_train
        self.seed = seed

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]
        return pd.DataFrame(
            {
                "ret_5": c.pct_change(5),
                "vol_20": c.pct_change().rolling(20).std(),
                "trend_50": c / c.rolling(50).mean() - 1,
            },
            index=df.index,
        )

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = self._features(df)
        n = len(df)
        regime_id = np.full(n, 1.0)  # 1 = lateral/desconocido por defecto
        regime_prob = np.full(n, 1.0 / self.n_states)
        if n < self.min_train:
            return pd.DataFrame({"regime_id": regime_id, "regime_prob": regime_prob}, index=df.index)
        refit_positions = list(range(self.min_train, n, self.refit_days))
        if not refit_positions or refit_positions[-1] != n - 1:
            refit_positions.append(n - 1)

        def _fit_at(pos: int):
            """Ajusta el GMM con datos <= pos. Devuelve (gm, mu, sd, mapping) o None."""
            train = feats.iloc[: pos + 1].dropna()
            if len(train) < self.min_train:
                return None
            mu = train.mean()
            sd = train.std().replace(0, 1.0)
            z = (train - mu) / sd
            gm = GaussianMixture(n_components=self.n_states, random_state=self.seed, n_init=2, max_iter=200)
            gm.fit(z.to_numpy())
            labels = gm.predict(z.to_numpy())
            order = np.argsort(
                [
                    train["ret_5"].to_numpy()[labels == k].mean() if (labels == k).any() else 0
                    for k in range(self.n_states)
                ]
            )
            mapping = {old: new for new, old in enumerate(order)}  # 0=peor retorno, 2=mejor
            return gm, mu, sd, mapping

        def _assign(seg_from: int, seg_to: int, model) -> None:
            """Asigna el segmento (seg_from, seg_to] con un modelo entrenado <= seg_from
            (estrictamente point-in-time: sin datos futuros respecto a cada fila)."""
            gm, mu, sd, mapping = model
            seg = feats.iloc[seg_from + 1 : seg_to + 1].dropna()
            if seg.empty:
                return
            zseg = ((seg - mu) / sd).to_numpy()
            proba = gm.predict_proba(zseg)
            lab = np.array([mapping[cl] for cl in gm.predict(zseg)])
            positions = df.index.get_indexer(seg.index)
            regime_id[positions] = lab
            regime_prob[positions] = proba.max(axis=1)

        prev_model = None
        prev_pos = self.min_train - 1
        for pos in refit_positions:
            model = _fit_at(pos)
            if model is None:
                continue
            if prev_model is not None:
                _assign(prev_pos, pos, prev_model)
            prev_model, prev_pos = model, pos
        # cola tras el último refit, con el último modelo (entrenado <= prev_pos)
        if prev_model is not None and prev_pos < n - 1:
            _assign(prev_pos, n - 1, prev_model)
        return pd.DataFrame({"regime_id": regime_id, "regime_prob": regime_prob}, index=df.index)
