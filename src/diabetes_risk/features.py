"""Imputation strategies and model pipelines."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .data import FEATURES, ZERO_AS_MISSING, zero_to_nan

STRATEGIES = ["zeros_as_values", "median", "median+indicators", "knn", "iterative"]


def _to_nan(X):
    import pandas as pd
    X = pd.DataFrame(X, columns=FEATURES) if not hasattr(X, "columns") else X
    return zero_to_nan(X)[FEATURES].to_numpy(dtype=float)


def make_imputer(strategy: str) -> BaseEstimator:
    """Return an (unfitted) imputer that expects the raw zero-encoded frame."""
    if strategy == "zeros_as_values":
        return FunctionTransformer(lambda X: np.asarray(X, dtype=float))
    to_nan = FunctionTransformer(_to_nan)
    if strategy == "median":
        return Pipeline([("nan", to_nan), ("imp", SimpleImputer(strategy="median"))])
    if strategy == "median+indicators":
        return Pipeline([("nan", to_nan),
                         ("imp", SimpleImputer(strategy="median", add_indicator=True))])
    if strategy == "knn":
        return Pipeline([("nan", to_nan), ("sc", StandardScaler()),
                         ("imp", KNNImputer(n_neighbors=7))])
    if strategy == "iterative":
        return Pipeline([("nan", to_nan),
                         ("imp", IterativeImputer(max_iter=10, random_state=0, sample_posterior=False))])
    raise ValueError(f"Unknown imputation strategy: {strategy}")


def make_logreg_pipeline(strategy: str, seed: int = 42) -> Pipeline:
    """Imputer -> scaler -> class-weighted logistic regression."""
    return Pipeline([
        ("impute", make_imputer(strategy)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
    ])


def make_hgb_pipeline(seed: int = 42) -> Pipeline:
    """Gradient boosting that handles NaN natively (no explicit imputation)."""
    return Pipeline([
        ("nan", FunctionTransformer(_to_nan)),
        ("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                                class_weight="balanced", random_state=seed)),
    ])


def make_baseline(seed: int = 42) -> DummyClassifier:
    """Prior-probability baseline."""
    return DummyClassifier(strategy="prior", random_state=seed)


def all_candidates(seed: int = 42) -> Dict[str, BaseEstimator]:
    """Every imputation strategy with logistic regression, plus native-NaN HGB."""
    cands: Dict[str, BaseEstimator] = {f"logreg/{s}": make_logreg_pipeline(s, seed) for s in STRATEGIES}
    cands["hgb/native_nan"] = make_hgb_pipeline(seed)
    return cands
