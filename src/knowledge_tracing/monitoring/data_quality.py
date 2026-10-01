"""Data-quality gate: schema, ranges, nulls, duplicates.

Run on the cleaned long frame before training. Returns a report with per-check
pass/fail; `passed` is False if any hard check fails (the pipeline logs this to
MLflow and surfaces it in the run summary).
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["user_id", "order_idx", "skill_idx", "correct", "split"]


def quality_report(long: pd.DataFrame) -> dict:
    checks: dict[str, dict] = {}

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in long.columns]
    checks["schema"] = {"passed": not missing_cols, "missing_columns": missing_cols}

    null_counts = {c: int(long[c].isna().sum()) for c in long.columns}
    checks["no_nulls"] = {"passed": sum(null_counts.values()) == 0, "null_counts": null_counts}

    bad_labels = int((~long["correct"].isin([0, 1])).sum()) if "correct" in long else -1
    checks["binary_labels"] = {"passed": bad_labels == 0, "invalid_label_rows": bad_labels}

    neg_skill = int((long["skill_idx"] < 0).sum()) if "skill_idx" in long else -1
    checks["skill_range"] = {"passed": neg_skill == 0, "negative_skill_rows": neg_skill}

    dup = int(long.duplicated(subset=["user_id", "order_idx"]).sum()) if "order_idx" in long else -1
    checks["unique_order"] = {"passed": dup == 0, "duplicate_rows": dup}

    passed = all(c["passed"] for c in checks.values())
    return {"passed": passed, "checks": checks}
