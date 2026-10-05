"""Disaggregated evaluation of the packaged DKT model, for the Model Card.

Model Cards report quality on relevant subgroups, not only overall. For
knowledge tracing the natural factors are how much of the student's history
the model has seen (cold start) and how often the skill occurs in training.

The script rebuilds the same deterministic student split as the training run
(same config and seed), predicts every test attempt one step ahead with the
packaged model and reports metrics per group.

Usage:
    python scripts/sliced_metrics.py --model artifacts/model --data-source full \
        --output artifacts/sliced_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch

from knowledge_tracing.config import DEFAULT_CONFIG_PATH, load_config, resolve_path
from knowledge_tracing.etl.datasets import StudentSequence, build_sequences
from knowledge_tracing.etl.extract import extract
from knowledge_tracing.etl.transform import transform
from knowledge_tracing.evaluation.metrics import compute_metrics
from knowledge_tracing.inference import DKTPredictor
from knowledge_tracing.runtime import configure_runtime

# Number of attempts the model has already seen for the student.
HISTORY_BINS = [1, 5, 10, 20, 50, np.inf]
HISTORY_LABELS = ["1–4", "5–9", "10–19", "20–49", "50+"]
# Skills are split into three equally sized groups by training frequency.
FREQUENCY_LABELS = ["редкие", "средние", "частые"]
_BATCH_SIZE = 64


@torch.no_grad()
def step_predictions(predictor: DKTPredictor, sequences: list[StudentSequence]) -> pd.DataFrame:
    """One-step-ahead prediction of every attempt after the first, with its position."""
    network = predictor.network.eval()
    rows = []
    usable = [s for s in sequences if len(s) >= 2]  # noqa: PLR2004 - one transition at least
    for start in range(0, len(usable), _BATCH_SIZE):
        batch = usable[start : start + _BATCH_SIZE]
        lengths = torch.tensor([len(s) for s in batch])
        x = torch.zeros((len(batch), int(lengths.max())), dtype=torch.int64)
        for b, seq in enumerate(batch):
            x[b, : len(seq)] = torch.from_numpy(2 * seq.skills + seq.correct)
        probs = torch.sigmoid(network(x, lengths)).numpy()
        for b, seq in enumerate(batch):
            n_next = len(seq) - 1
            rows.append(
                pd.DataFrame(
                    {
                        "user_id": seq.user_id,
                        "prior_attempts": np.arange(1, len(seq)),
                        "skill_idx": seq.skills[1:],
                        "y_true": seq.correct[1:],
                        "y_pred": probs[b, np.arange(n_next), seq.skills[1:]],
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def _metrics_by(frame: pd.DataFrame, column: str) -> list[dict]:
    groups = []
    for label, group in frame.groupby(column, observed=True, sort=True):
        metrics = compute_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        groups.append(
            {
                "group": str(label),
                "n": metrics["n"],
                "auc": round(metrics["auc"], 4),
                "accuracy": round(metrics["accuracy"], 4),
                "share_correct": round(float(group["y_true"].mean()), 4),
            }
        )
    return groups


def main() -> int:
    """Compute the sliced metrics and print them as Markdown tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="artifacts/model", help="packaged model directory")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--data-source", choices=["sample", "full"], default="full")
    parser.add_argument("--output", default="artifacts/sliced_metrics.json")
    args = parser.parse_args()

    configure_runtime()
    cfg = load_config(args.config)
    processed = transform(extract(cfg.data, args.data_source), cfg.data, seed=cfg.seed)
    predictor = DKTPredictor.load(resolve_path(args.model))
    expected_skills = [s for s, _ in sorted(processed.skill_remap.items(), key=lambda kv: kv[1])]
    if predictor.skill_ids != expected_skills:
        print("The model was trained on different data (skill mapping differs).", file=sys.stderr)
        return 1

    frame = step_predictions(predictor, build_sequences(processed.long, "test"))
    frame["history"] = pd.cut(
        frame["prior_attempts"], bins=HISTORY_BINS, right=False, labels=HISTORY_LABELS
    )
    train = processed.long[processed.long["split"] == "train"]
    train_counts = (
        train["skill_idx"].value_counts().reindex(range(processed.n_skills), fill_value=0)
    )
    skill_group = pd.qcut(train_counts.rank(method="first"), 3, labels=FREQUENCY_LABELS)
    frame["skill_frequency"] = frame["skill_idx"].map(skill_group)

    overall = compute_metrics(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy())
    report = {
        "model": predictor.metadata.get("model"),
        "data_source": args.data_source,
        "overall": {"n": overall["n"], "auc": round(overall["auc"], 4)},
        "by_history_length": _metrics_by(frame, "history"),
        "by_skill_frequency": _metrics_by(frame, "skill_frequency"),
        "skill_frequency_thresholds": {
            label: [
                int(train_counts[skill_group == label].min()),
                int(train_counts[skill_group == label].max()),
            ]
            for label in FREQUENCY_LABELS
        },
    }
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Overall: AUC {report['overall']['auc']} on {report['overall']['n']} attempts\n")
    for title, key in (
        ("История студента (сколько попыток уже видела модель)", "by_history_length"),
        ("Частота навыка в обучающей выборке", "by_skill_frequency"),
    ):
        print(
            f"{title}\n\n| Группа | Попыток | AUC | Accuracy | Доля верных |\n|---|---|---|---|---|"
        )
        for g in report[key]:
            print(
                f"| {g['group']} | {g['n']} | {g['auc']:.4f} | {g['accuracy']:.4f} | {g['share_correct']:.4f} |"
            )
        print()
    print(f"Written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
