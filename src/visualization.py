
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_dir


def plot_reliability_diagram(y_true, y_prob, path: str | Path, n_bins: int = 10) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        xs.append(np.mean(y_prob[mask]))
        ys.append(np.mean(y_true[mask]))
    plt.figure(figsize=(5.6, 4.2))
    plt.plot([0, 1], [0, 1])
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event rate")
    plt.title("Reliability diagram")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_model_comparison(df: pd.DataFrame, metric: str, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    plot_df = df.sort_values(metric, ascending=False)
    plt.figure(figsize=(6.5, 4.2))
    plt.bar(plot_df["model"].astype(str), plot_df[metric].astype(float))
    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.title(f"Model comparison: {metric}")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_subgroup_gap(df: pd.DataFrame, gap_col: str, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    temp = df.copy()
    temp["label"] = temp["model"].astype(str) + " | " + temp["attribute"].astype(str)
    plt.figure(figsize=(7.2, 4.5))
    plt.bar(temp["label"], temp[gap_col])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(gap_col)
    plt.title(f"Subgroup fairness gap: {gap_col}")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
