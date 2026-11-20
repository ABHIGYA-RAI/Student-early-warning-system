
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class PlattCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=300)

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        x = _safe_logit(np.asarray(y_prob)).reshape(-1, 1)
        self.model.fit(x, np.asarray(y_true))
        return self

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        x = _safe_logit(np.asarray(y_prob)).reshape(-1, 1)
        return self.model.predict_proba(x)[:, 1]


class IsotonicCalibrator:
    def __init__(self) -> None:
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        self.model.fit(np.asarray(y_prob), np.asarray(y_true))
        return self

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        return self.model.predict(np.asarray(y_prob))
