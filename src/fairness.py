
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += abs(acc - conf) * (mask.sum() / len(y_true))
    return float(ece)


def confusion_rates(y_true, y_pred) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    selection_rate = (tp + fp) / max(1, (tn + fp + fn + tp))
    return {
        "fnr": float(fnr),
        "fpr": float(fpr),
        "tpr": float(tpr),
        "tnr": float(tnr),
        "selection_rate": float(selection_rate),
    }


def subgroup_fairness_report(
    df: pd.DataFrame,
    sensitive_attribute: str,
    y_true_col: str = "y_true",
    y_prob_col: str = "y_prob",
    threshold: float = 0.5,
) -> pd.DataFrame:
    rows = []
    for group, group_df in df.groupby(sensitive_attribute):
        y_true = group_df[y_true_col].to_numpy()
        y_prob = group_df[y_prob_col].to_numpy()
        y_pred = (y_prob >= threshold).astype(int)
        rates = confusion_rates(y_true, y_pred)
        rows.append({
            "attribute": sensitive_attribute,
            "group": group,
            "n": len(group_df),
            **rates,
            "ece": expected_calibration_error(y_true, y_prob),
        })
    return pd.DataFrame(rows)


def fairness_gap_summary(subgroup_df: pd.DataFrame) -> Dict[str, float]:
    result = {}
    for col in ["fnr", "fpr", "selection_rate", "ece"]:
        result[f"max_{col}_gap"] = float(subgroup_df[col].max() - subgroup_df[col].min()) if len(subgroup_df) > 0 else 0.0
    return result


def kamiran_calders_reweighing(y: Iterable[int], a: Iterable[str | int]) -> np.ndarray:
    y = pd.Series(y).astype(str)
    a = pd.Series(a).astype(str)
    joint = pd.crosstab(a, y, normalize=True)
    pa = a.value_counts(normalize=True)
    py = y.value_counts(normalize=True)

    weights = []
    for ai, yi in zip(a, y):
        p_joint = joint.loc[ai, yi]
        w = (pa.loc[ai] * py.loc[yi]) / p_joint if p_joint > 0 else 1.0
        weights.append(float(w))
    return np.array(weights, dtype=float)
