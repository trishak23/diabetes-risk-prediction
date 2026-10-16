"""Metrics (recall-focused), calibration utilities and plots."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width-bin ECE."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def threshold_for_recall(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """Highest threshold whose recall on (y, p) is at least ``target``."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target)[0]
    return float(thr[ok[-1]]) if len(ok) else float(thr[0])


def score_probs(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Threshold-free and thresholded metrics."""
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "recall": float(recall_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred)),
        "specificity": float(tn / (tn + fp)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_missingness(rates, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.barh(rates.index[::-1], rates.values[::-1] * 100, color="#4c72b0")
    ax.set_xlabel("% of rows recorded as 0 (missing)")
    ax.set_title("Zero-encoded missing values per column")
    _save(fig, path)


def plot_imputation_comparison(table: Dict[str, Dict[str, float]], path: Path) -> None:
    names = list(table)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, (m, c) in enumerate([("recall", "#c44e52"), ("pr_auc", "#4c72b0"), ("roc_auc", "#55a868")]):
        ax.bar(x + (i - 1) * 0.26, [table[n][m] for n in names], 0.26, label=m, color=c)
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Cross-validated comparison (training split)")
    ax.legend()
    _save(fig, path)


def plot_pr_curve(y, p, threshold, path: Path) -> None:
    prec, rec, _ = precision_recall_curve(y, p)
    pt = (recall_score(y, p >= threshold), precision_score(y, p >= threshold))
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(rec, prec, color="#4c72b0")
    ax.axhline(y.mean(), ls="--", color="grey", label="prevalence baseline")
    ax.scatter(*pt, color="#c44e52", zorder=3, label=f"chosen threshold={threshold:.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend()
    ax.set_title("Precision-recall (test set)")
    _save(fig, path)


def plot_calibration(y, p_raw, p_cal, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    for p, lab, c in [(p_raw, "uncalibrated (class-weighted)", "#c44e52"), (p_cal, "calibrated (sigmoid)", "#4c72b0")]:
        fr, mp = calibration_curve(y, p, n_bins=8, strategy="quantile")
        ax.plot(mp, fr, "o-", label=lab, color=c)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram (test set)"); ax.legend(fontsize=8)
    _save(fig, path)


def plot_confusion(cm: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["No diabetes", "Diabetes"]); ax.set_yticks([0, 1], ["No diabetes", "Diabetes"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title, fontsize=10)
    _save(fig, path)


def plot_importance(names, means, stds, path: Path) -> None:
    order = np.argsort(means)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(np.array(names)[order], np.array(means)[order], xerr=np.array(stds)[order], color="#55a868")
    ax.set_xlabel("Permutation importance (drop in PR-AUC)")
    ax.set_title("Feature importance (test set)")
    _save(fig, path)
