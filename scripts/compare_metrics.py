"""Regression check: compare a pipeline run's metrics.json with a baseline.

Deterministic models (BKT, DKT, DKT+Optuna) must reproduce every baseline
metric within a tight tolerance. FLAML AutoML always spends its whole
wall-clock budget, so the number of trials it completes depends on machine
speed: its metrics are stable between runs of the same code on the same
machine but legitimately shift with anything that changes speed. Such
time-budgeted models are compared with a looser tolerance.

Usage:
    python scripts/compare_metrics.py docs/baseline/metrics_sample_quick.json artifacts/metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _same(expected: float, actual: float, tolerance: float) -> bool:
    if math.isnan(expected) or math.isnan(actual):
        return math.isnan(expected) and math.isnan(actual)
    return abs(expected - actual) <= tolerance


def compare(baseline: dict, current: dict, tolerance: float, loose: dict[str, float]) -> list[str]:
    """Return human-readable mismatches (empty when metrics match).

    Args:
        baseline: Parsed baseline metrics.json.
        current: Parsed metrics.json of the run under test.
        tolerance: Allowed absolute difference for deterministic models.
        loose: Model name -> allowed absolute difference for time-budgeted models.
    """
    problems: list[str] = []
    for model, expected_metrics in baseline["results"].items():
        actual_metrics = current["results"].get(model)
        if actual_metrics is None:
            problems.append(f"{model}: missing in the current run")
            continue
        model_tolerance = loose.get(model, tolerance)
        names = ["auc"] if model in loose else list(expected_metrics)
        for name in names:
            expected, actual = expected_metrics[name], actual_metrics.get(name)
            if actual is None or not _same(float(expected), float(actual), model_tolerance):
                problems.append(
                    f"{model}.{name}: baseline={expected} current={actual} "
                    f"(tolerance {model_tolerance:g})"
                )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument(
        "--loose",
        action="append",
        metavar="MODEL",
        help="time-budgeted model compared on AUC with --loose-tolerance (default: AutoML)",
    )
    parser.add_argument("--loose-tolerance", type=float, default=0.02)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    current = json.loads(args.current.read_text(encoding="utf-8"))
    loose = dict.fromkeys(args.loose or ["AutoML"], args.loose_tolerance)
    problems = compare(baseline, current, args.tolerance, loose)

    print(f"{'model':12s} {'baseline AUC':>13s} {'current AUC':>12s}  check")
    for model, metrics in baseline["results"].items():
        current_auc = current["results"].get(model, {}).get("auc", float("nan"))
        check = f"AUC ±{loose[model]:g}" if model in loose else "all metrics exact"
        print(f"{model:12s} {metrics['auc']:13.6f} {current_auc:12.6f}  {check}")
    if problems:
        print(f"\nREGRESSION: {len(problems)} metric(s) differ from the baseline:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nOK: metrics match the baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
