"""Single entry point: data -> compare imputers -> calibrate -> evaluate -> save."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import yaml
from sklearn.metrics import confusion_matrix

from src.diabetes_risk import evaluate as ev
from src.diabetes_risk.data import FEATURES, TARGET, load_or_generate, missing_rate
from src.diabetes_risk.train import compare_candidates, fit_final, importance, pick_best, split

LOG = logging.getLogger("pipeline")


def main() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    seed = cfg["seed"]
    df = load_or_generate(ROOT / cfg["data"]["sample_path"], cfg["data"]["n_samples"], seed)
    fig_dir = ROOT / cfg["paths"]["figures"]
    X_tr, X_te, y_tr, y_te = split(df, cfg["data"]["test_size"], seed)

    ev.plot_missingness(missing_rate(df), fig_dir / "missingness.png")
    table = compare_candidates(X_tr, y_tr, cfg["model"]["cv_folds"], seed)
    ev.plot_imputation_comparison({k: v for k, v in table.items() if not k.startswith("baseline")},
                                  fig_dir / "imputation_comparison.png")
    best = pick_best(table)
    LOG.info("Selected: %s", best)
    raw, cal, thr = fit_final(X_tr, y_tr, best, cfg["model"], seed)

    y = y_te.to_numpy()
    p_raw = raw.predict_proba(X_te)[:, 1]
    p_cal = cal.predict_proba(X_te)[:, 1]
    test_default = ev.score_probs(y, p_cal, 0.5)
    test_tuned = ev.score_probs(y, p_cal, thr)
    test_raw = ev.score_probs(y, p_raw, 0.5)
    ev.plot_pr_curve(y, p_cal, thr, fig_dir / "pr_curve.png")
    ev.plot_calibration(y, p_raw, p_cal, fig_dir / "calibration.png")
    ev.plot_confusion(confusion_matrix(y, p_cal >= thr), fig_dir / "confusion_matrix.png",
                      f"Test set, recall-tuned threshold {thr:.2f}")
    names, means, stds = importance(cal, X_te, y_te, seed)
    ev.plot_importance(names, means, stds, fig_dir / "feature_importance.png")

    metrics = {
        "data": {"synthetic": True, "n_rows": int(len(df)), "prevalence": float(df[TARGET].mean()),
                 "zero_missing_rate": {k: float(v) for k, v in missing_rate(df).items()},
                 "n_train": int(len(X_tr)), "n_test": int(len(X_te))},
        "cv_comparison_train": table,
        "selected_model": best,
        "target_recall": cfg["model"]["target_recall"],
        "test_uncalibrated_threshold_0.5": test_raw,
        "test_calibrated_threshold_0.5": test_default,
        "test_calibrated_recall_tuned": test_tuned,
        "permutation_importance": dict(zip(names, [float(m) for m in means])),
        "disclaimer": "Synthetic data; not for clinical use.",
    }
    (ROOT / cfg["paths"]["metrics"]).write_text(json.dumps(metrics, indent=2))
    mp = ROOT / cfg["paths"]["model"]
    mp.parent.mkdir(exist_ok=True)
    joblib.dump({"model": cal, "threshold": thr, "features": FEATURES, "name": best}, mp)
    LOG.info("Saved metrics and model.")
    return metrics


if __name__ == "__main__":
    main()
