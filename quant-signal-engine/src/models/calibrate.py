"""Calibración de probabilidades: isotónica (one-vs-rest por clase) o Platt (logística)."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CLASSES = np.array([-1, 0, 1])


class IsotonicCalibrator:
    name = "isotonic"

    def __init__(self) -> None:
        self.models: list[IsotonicRegression | None] = []

    def fit(self, proba: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
        self.models = []
        for i in range(proba.shape[1]):
            target = (y == CLASSES[i]).astype(int)
            if target.sum() < 5 or target.sum() == len(target):
                self.models.append(None)  # clase degenerate: sin calibrar
                continue
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(proba[:, i], target)
            self.models.append(iso)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        out = proba.copy()
        for i, m in enumerate(self.models):
            if m is not None:
                out[:, i] = m.predict(proba[:, i])
        s = out.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return out / s


class PlattCalibrator:
    name = "platt"

    def __init__(self) -> None:
        self.models: list[LogisticRegression | None] = []

    def fit(self, proba: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        self.models = []
        eps = 1e-6
        logits = np.log(np.clip(proba, eps, 1 - eps) / (1 - np.clip(proba, eps, 1 - eps)))
        for i in range(proba.shape[1]):
            target = (y == CLASSES[i]).astype(int)
            if target.sum() < 5 or target.sum() == len(target):
                self.models.append(None)
                continue
            lr = LogisticRegression(max_iter=1000)
            lr.fit(logits[:, i].reshape(-1, 1), target)
            self.models.append(lr)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        eps = 1e-6
        logits = np.log(np.clip(proba, eps, 1 - eps) / (1 - np.clip(proba, eps, 1 - eps)))
        out = proba.copy()
        for i, m in enumerate(self.models):
            if m is not None:
                out[:, i] = m.predict_proba(logits[:, i].reshape(-1, 1))[:, 1]
        s = out.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return out / s


def build_calibrator(kind: str):
    if str(kind).lower() == "platt":
        return PlattCalibrator()
    return IsotonicCalibrator()
