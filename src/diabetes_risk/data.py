"""Synthetic Pima-style diabetes data with zero-encoded missing values.

The real Pima Indians dataset encodes physiologically impossible values (e.g.
glucose = 0) to mean "not measured". This generator reproduces that quirk: the
true values are drawn from a correlated latent model, and then a realistic
fraction of entries is replaced with 0. Missingness is not completely random:
insulin is more often missing for people with normal glucose (less likely to
have been tested), and skin-fold thickness is more often missing at lower BMI.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET = "Outcome"
#: columns where 0 is physiologically impossible and therefore means "missing"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_diabetes_data(n_samples: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Generate a seeded synthetic dataset with zero-encoded missing values."""
    rng = np.random.default_rng(seed)
    age = np.clip(rng.gamma(shape=6.0, scale=5.5, size=n_samples) + 21, 21, 81).round()
    preg = np.clip(rng.poisson(np.clip((age - 18) / 9.0, 0.3, None)), 0, 17)
    bmi = np.clip(rng.normal(32.0, 6.5, n_samples) + 0.03 * (age - 33), 18, 67)
    pedigree = np.clip(rng.lognormal(-0.9, 0.7, n_samples), 0.08, 2.4)
    # glucose depends on age, bmi and family history
    glucose = np.clip(
        85 + 0.35 * (age - 33) + 0.9 * (bmi - 32) + 14 * pedigree + rng.normal(0, 20, n_samples),
        55, 199,
    )
    bp = np.clip(60 + 0.25 * (age - 33) + 0.6 * (bmi - 32) + rng.normal(0, 10, n_samples), 30, 122)
    skin = np.clip(20 + 0.8 * (bmi - 32) + rng.normal(0, 7, n_samples), 7, 99)
    insulin = np.clip(
        np.exp(4.1 + 0.012 * (glucose - 100) + 0.01 * (bmi - 32) + rng.normal(0, 0.55, n_samples)),
        14, 846,
    )
    logit = (
        -8.3 + 0.036 * glucose + 0.075 * (bmi - 20) * 0.5 + 0.03 * age
        + 0.07 * preg + 0.9 * pedigree + 0.003 * bp
    )
    outcome = rng.binomial(1, _sigmoid(logit + rng.normal(0, 0.35, n_samples)))
    df = pd.DataFrame({
        "Pregnancies": preg.astype(int), "Glucose": glucose, "BloodPressure": bp,
        "SkinThickness": skin, "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree, "Age": age.astype(int), TARGET: outcome,
    })
    # --- zero-as-missing injection (informative, partly MNAR) ---
    p_ins = np.where(df["Glucose"] < 110, 0.58, 0.38)
    p_skin = np.where(df["BMI"] < 30, 0.38, 0.22)
    masks = {
        "Insulin": rng.random(n_samples) < p_ins,
        "SkinThickness": rng.random(n_samples) < p_skin,
        "BloodPressure": rng.random(n_samples) < 0.05,
        "BMI": rng.random(n_samples) < 0.015,
        "Glucose": rng.random(n_samples) < 0.01,
    }
    for col, m in masks.items():
        df.loc[m, col] = 0.0
    df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = df[
        ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    ].round(1)
    df["DiabetesPedigreeFunction"] = df["DiabetesPedigreeFunction"].round(3)
    LOG.info("Generated %d rows, prevalence %.3f", n_samples, df[TARGET].mean())
    return df


def zero_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace impossible zeros with NaN in the designated columns."""
    out = df.copy()
    for col in ZERO_AS_MISSING:
        out[col] = out[col].replace(0, np.nan)
    return out


def missing_rate(df: pd.DataFrame) -> pd.Series:
    """Fraction of zero-encoded (i.e. missing) entries per column."""
    return (df[ZERO_AS_MISSING] == 0).mean().sort_values(ascending=False)


def load_or_generate(path: str | Path, n_samples: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Load the CSV at ``path`` if present, otherwise generate and save it."""
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    df = generate_diabetes_data(n_samples, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    d = generate_diabetes_data()
    Path("data").mkdir(exist_ok=True)
    d.to_csv("data/diabetes_sample.csv", index=False)
    print(missing_rate(d))
