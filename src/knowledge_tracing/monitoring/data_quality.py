"""Data-quality gate: schema, ranges, nulls, duplicates.

Run on the cleaned long frame before training. ``quality_report`` returns a
report with per-check pass/fail; ``passed`` is False if any hard check fails.
``enforce_quality_gate`` turns a failed report into an error, so models are
never trained on data that violates the schema.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ..errors import DataQualityError

__all__ = ["DataQualityError", "enforce_quality_gate", "failed_checks", "quality_report"]

LOG = logging.getLogger(__name__)

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


def failed_checks(report: dict[str, Any]) -> list[str]:
    """Names of the checks that did not pass."""
    return [name for name, check in report["checks"].items() if not check["passed"]]


def enforce_quality_gate(report: dict[str, Any], *, fail: bool) -> None:
    """Stop the pipeline (or only warn when ``fail`` is False) if any check failed.

    Raises:
        DataQualityError: Some checks failed and ``fail`` is True.
    """
    if report["passed"]:
        return
    message = f"data-quality checks failed: {', '.join(failed_checks(report))}"
    if fail:
        raise DataQualityError(message)
    LOG.warning("%s — continuing because monitoring.fail_on_data_quality is false", message)
