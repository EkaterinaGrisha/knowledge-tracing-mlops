import logging

import numpy as np
import pandas as pd
import pytest

from knowledge_tracing.monitoring.data_quality import (
    DataQualityError,
    enforce_quality_gate,
    quality_report,
)
from knowledge_tracing.monitoring.drift import drift_report


def test_no_drift_for_identical_distributions():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    report = drift_report(df, df.copy(), ["a", "b"], psi_warn=0.1, psi_alert=0.25)
    assert report["overall_status"] == "OK"
    assert report["n_significant_drift"] == 0


def test_drift_detected_for_shifted_distribution():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"a": rng.normal(0, 1, size=500)})
    cur = pd.DataFrame({"a": rng.normal(5, 1, size=500)})  # large shift
    report = drift_report(ref, cur, ["a"], psi_warn=0.1, psi_alert=0.25)
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


def _report_with_failure() -> dict:
    long = pd.DataFrame(
        {"user_id": [0], "order_idx": [0], "skill_idx": [0], "correct": [5], "split": ["train"]}
    )
    return quality_report(long)


def test_quality_gate_stops_on_failure():
    with pytest.raises(DataQualityError, match="binary_labels"):
        enforce_quality_gate(_report_with_failure(), fail=True)


def test_quality_gate_can_only_warn(caplog):
    with caplog.at_level(logging.WARNING):
        enforce_quality_gate(_report_with_failure(), fail=False)
    assert "binary_labels" in caplog.text


def test_quality_gate_passes_clean_report():
    enforce_quality_gate({"passed": True, "checks": {}}, fail=True)
