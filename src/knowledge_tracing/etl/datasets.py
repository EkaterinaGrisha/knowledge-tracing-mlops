"""Model-ready views of the processed data.

Sequence models (BKT, DKT) consume per-student chronological sequences built
from the cleaned long frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


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
