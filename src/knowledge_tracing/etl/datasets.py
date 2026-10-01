"""Model-ready views of the processed data.

Sequence models (BKT, DKT) consume per-student chronological sequences; the
tabular AutoML model consumes the causal feature frame. ``TrainingData``
bundles both views for the train / validation / test splits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .transform import ProcessedData


@dataclass(frozen=True, eq=False)
class StudentSequence:
    """Chronological interactions of one student."""

    user_id: int
    skills: np.ndarray  # dense skill index of every interaction (int64)
    correct: np.ndarray  # 1 if the answer was correct, else 0 (int64)

    def __len__(self) -> int:
        return len(self.skills)


def build_sequences(long: pd.DataFrame, split: str) -> list[StudentSequence]:
    """Per-student chronological sequences of one split, for BKT and DKT."""
    sub = long[long["split"] == split].sort_values(["user_id", "order_idx"])
    return [
        StudentSequence(
            # pandas-stubs types groupby keys as a generic scalar; user_id is an int column.
            user_id=int(uid),  # type: ignore[arg-type]
            skills=grp["skill_idx"].to_numpy(dtype=np.int64),
            correct=grp["correct"].to_numpy(dtype=np.int64),
        )
        for uid, grp in sub.groupby("user_id")
    ]


@dataclass(frozen=True, eq=False)
class SplitData:
    """One split (train / val / test) in both representations the models use."""

    sequences: list[StudentSequence]  # per-student sequences (BKT, DKT)
    features: pd.DataFrame  # causal tabular features (AutoML)
    target: pd.Series  # correctness of every interaction in ``features``


@dataclass(frozen=True, eq=False)
class TrainingData:
    """Train / validation / test splits shared by all models."""

    train: SplitData
    val: SplitData
    test: SplitData
    n_skills: int


def build_training_data(processed: ProcessedData) -> TrainingData:
    """Assemble the model inputs of every split from the processed data."""

    def split(name: str) -> SplitData:
        features, target = processed.split_xy(name)
        return SplitData(build_sequences(processed.long, name), features, target)

    return TrainingData(split("train"), split("val"), split("test"), processed.n_skills)
