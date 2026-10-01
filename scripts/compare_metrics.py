"""Regression check: compare a pipeline run's metrics.json with a baseline.

Every metric of every model in the baseline must be reproduced within the
tolerance; the default tolerance is tight because the quick sample run is
deterministic on a given platform.

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


def compare(baseline: dict, current: dict, tolerance: float) -> list[str]:
    """Return a list of human-readable mismatches (empty when metrics match)."""
    problems: list[str] = []
    for model, expected_metrics in baseline["results"].items():
        actual_metrics = current["results"].get(model)
        if actual_metrics is None:
            problems.append(f"{model}: missing in the current run")
            continue
        for name, expected in expected_metrics.items():
            actual = actual_metrics.get(name)
            if actual is None or not _same(float(expected), float(actual), tolerance):
                problems.append(f"{model}.{name}: baseline={expected} current={actual}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    current = json.loads(args.current.read_text(encoding="utf-8"))
    problems = compare(baseline, current, args.tolerance)

    print(f"{'model':12s} {'baseline AUC':>13s} {'current AUC':>12s}")
    for model, metrics in baseline["results"].items():
        current_auc = current["results"].get(model, {}).get("auc", float("nan"))
        print(f"{model:12s} {metrics['auc']:13.6f} {current_auc:12.6f}")
    if problems:
        print(f"\nREGRESSION: {len(problems)} metric(s) differ from the baseline:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"\nOK: all metrics match the baseline (tolerance {args.tolerance:g}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
