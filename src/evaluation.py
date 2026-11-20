
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .fairness import confusion_rates, expected_calibration_error


def summarize_binary_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    rates = confusion_rates(y_true, y_pred)
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "prauc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": float(expected_calibration_error(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        **rates,
    }


def make_prediction_frame(ids, y_true, y_prob, extra: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = pd.DataFrame({"id_student": ids, "y_true": y_true, "y_prob": y_prob})
    if extra is not None:
        cols = [c for c in extra.columns if c not in df.columns]
        df = df.merge(extra[["id_student"] + cols].drop_duplicates(), on="id_student", how="left")
    return df


def roc_points(y_true, y_prob):
    return roc_curve(y_true, y_prob)


def pr_points(y_true, y_prob):
    return precision_recall_curve(y_true, y_prob)
