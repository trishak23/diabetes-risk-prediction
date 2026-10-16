import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.diabetes_risk.data import FEATURES, ZERO_AS_MISSING, generate_diabetes_data, zero_to_nan
from src.diabetes_risk.evaluate import expected_calibration_error, threshold_for_recall
from src.diabetes_risk.features import STRATEGIES, make_logreg_pipeline


class TestData(unittest.TestCase):
    def test_seeded_and_shape(self):
        a = generate_diabetes_data(300, 1); b = generate_diabetes_data(300, 1)
        self.assertTrue(a.equals(b))
        self.assertEqual(list(a.columns[:-1]), FEATURES)

    def test_zero_as_missing_present_and_converted(self):
        df = generate_diabetes_data(800, 3)
        self.assertGreater((df["Insulin"] == 0).mean(), 0.2)
        self.assertEqual(zero_to_nan(df)[ZERO_AS_MISSING].isna().sum().sum(),
                         (df[ZERO_AS_MISSING] == 0).sum().sum())
        self.assertTrue((df["Age"] > 0).all())


class TestEval(unittest.TestCase):
    def test_threshold_meets_recall(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 500); p = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
        t = threshold_for_recall(y, p, 0.9)
        self.assertGreaterEqual(((p >= t) & (y == 1)).sum() / y.sum(), 0.9)

    def test_ece_perfect_and_bad(self):
        y = np.array([0, 1] * 50)
        self.assertGreater(expected_calibration_error(y, np.where(y == 1, 0.05, 0.95)), 0.8)
        self.assertLess(expected_calibration_error(y, np.full(100, 0.5)), 1e-9)


class TestPipelines(unittest.TestCase):
    def test_every_strategy_fits_and_predicts_no_nan(self):
        df = generate_diabetes_data(400, 5)
        X, y = df[FEATURES], df["Outcome"]
        for s in STRATEGIES:
            p = make_logreg_pipeline(s).fit(X, y).predict_proba(X)[:, 1]
            self.assertFalse(np.isnan(p).any(), s)
            self.assertTrue(((p >= 0) & (p <= 1)).all())


if __name__ == "__main__":
    unittest.main()
