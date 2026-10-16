"""Prediction helper and CLI. Research/education only - not a medical device."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .data import FEATURES

DISCLAIMER = ("NOT FOR CLINICAL USE. Model trained on synthetic data for demonstration; "
              "it must not inform diagnosis or treatment.")


def load_model(path: str | Path):
    """Load the persisted bundle ``{"model", "threshold", ...}``."""
    return joblib.load(path)


def predict_one(bundle: dict, record: dict) -> dict:
    """Score a single record. Use 0 (or omit) for unmeasured values."""
    row = {f: record.get(f, 0) for f in FEATURES}
    p = float(bundle["model"].predict_proba(pd.DataFrame([row]))[0, 1])
    return {"probability": round(p, 4), "flag_high_risk": p >= bundle["threshold"],
            "threshold": bundle["threshold"], "disclaimer": DISCLAIMER}


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a patient record (demo only).")
    ap.add_argument("--model", default="models/diabetes_model.joblib")
    ap.add_argument("--json", required=True, help='e.g. \'{"Glucose": 148, "BMI": 33.6, "Age": 50}\'')
    a = ap.parse_args()
    print(json.dumps(predict_one(load_model(a.model), json.loads(a.json)), indent=2))


if __name__ == "__main__":
    main()
