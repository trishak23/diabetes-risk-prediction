"""Training orchestration: compare imputers, pick one, calibrate, tune threshold."""
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict, train_test_split

from .data import FEATURES, TARGET
from .evaluate import score_probs, threshold_for_recall
from .features import all_candidates, make_baseline

LOG = logging.getLogger(__name__)


def split(df: pd.DataFrame, test_size: float, seed: int):
    """Stratified train/test split."""
    return train_test_split(df[FEATURES], df[TARGET], test_size=test_size,
                            stratify=df[TARGET], random_state=seed)


def compare_candidates(X: pd.DataFrame, y: pd.Series, folds: int, seed: int) -> Dict[str, Dict[str, float]]:
    """Cross-validated comparison on the training split only."""
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    scoring = {"recall": "recall", "precision": "precision", "pr_auc": "average_precision",
               "roc_auc": "roc_auc", "f1": "f1"}
    out: Dict[str, Dict[str, float]] = {}
    cands = {"baseline/prior": make_baseline(seed), **all_candidates(seed)}
    for name, est in cands.items():
        res = cross_validate(est, X, y, cv=cv, scoring=scoring, n_jobs=1)
        out[name] = {k: float(res[f"test_{k}"].mean()) for k in scoring}
        out[name].update({f"{k}_std": float(res[f"test_{k}"].std()) for k in ("recall", "pr_auc")})
        LOG.info("%-28s recall=%.3f pr_auc=%.3f", name, out[name]["recall"], out[name]["pr_auc"])
    return out


def pick_best(table: Dict[str, Dict[str, float]]) -> str:
    """Pick the non-baseline candidate with the best CV PR-AUC (threshold-free)."""
    return max((k for k in table if not k.startswith("baseline")), key=lambda k: table[k]["pr_auc"])


def fit_final(X_tr, y_tr, name: str, cfg: Dict[str, Any], seed: int
              ) -> Tuple[Any, Any, float]:
    """Fit raw + calibrated model and choose a threshold on out-of-fold probabilities."""
    est = all_candidates(seed)[name]
    raw = clone(est).fit(X_tr, y_tr)
    cal = CalibratedClassifierCV(clone(est), method=cfg["calibration_method"],
                                 cv=cfg["cv_folds"])
    cv = StratifiedKFold(cfg["cv_folds"], shuffle=True, random_state=seed + 1)
    oof = cross_val_predict(cal, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    thr = threshold_for_recall(y_tr.to_numpy(), oof, cfg["target_recall"])
    cal.fit(X_tr, y_tr)
    return raw, cal, thr


def importance(model, X_te, y_te, seed: int):
    """Permutation importance of the final model on held-out data (PR-AUC)."""
    r = permutation_importance(model, X_te, y_te, scoring="average_precision",
                               n_repeats=15, random_state=seed)
    return list(X_te.columns), r.importances_mean.tolist(), r.importances_std.tolist()
