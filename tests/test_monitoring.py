import numpy as np
import pandas as pd

from knowledge_tracing.monitoring.data_quality import quality_report
from knowledge_tracing.monitoring.drift import drift_report


def test_no_drift_for_identical_distributions(mini_cfg):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    report = drift_report(df, df.copy(), ["a", "b"], mini_cfg)
    assert report["overall_status"] == "OK"
    assert report["n_significant_drift"] == 0


def test_drift_detected_for_shifted_distribution(mini_cfg):
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"a": rng.normal(0, 1, size=500)})
    cur = pd.DataFrame({"a": rng.normal(5, 1, size=500)})  # large shift
    report = drift_report(ref, cur, ["a"], mini_cfg)
    assert report["features"]["a"]["status"] == "significant_drift"


def test_quality_report_passes_on_clean_frame():
    long = pd.DataFrame(
        {
            "user_id": [0, 0, 1],
            "order_idx": [0, 1, 0],
            "skill_idx": [0, 1, 0],
            "correct": [1, 0, 1],
            "split": ["train", "train", "test"],
        }
    )
    assert quality_report(long)["passed"] is True


def test_quality_report_flags_bad_labels():
    long = pd.DataFrame(
        {
            "user_id": [0],
            "order_idx": [0],
            "skill_idx": [0],
            "correct": [5],  # invalid
            "split": ["train"],
        }
    )
    report = quality_report(long)
    assert report["passed"] is False
    assert report["checks"]["binary_labels"]["passed"] is False
