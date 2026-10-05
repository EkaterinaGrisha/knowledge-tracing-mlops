"""ETL — Transform.

Takes the raw long DataFrame (user_id, order_idx, skill_id, correct) and:
  1. cleans it (valid labels, contiguous skill index, min sequence length, length cap);
  2. splits students into train/val/test (split is BY STUDENT to avoid leakage);
  3. engineers causal features for the tabular AutoML model (only past information
     is ever used to predict the current `correct` label).

The output `ProcessedData` carries both the tabular feature frame and the cleaned
long frame (used to build per-student sequences for DKT/BKT downstream).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import DataConfig

LOG = logging.getLogger(__name__)

FEATURE_COLS = [
    "skill_idx",
    "order_idx",
    "user_prior_attempts",
    "user_prior_correct_rate",
    "skill_prior_attempts",
    "skill_prior_correct_rate",
    "recent3_correct_rate",
    "skill_difficulty",
]
TARGET = "correct"
# recent3_correct_rate: share of correct answers among the last attempts
RECENT_WINDOW = 3


@dataclass
class ProcessedData:
    """Output of the Transform step, shared by all downstream stages."""

    features: pd.DataFrame  # feature frame + correct + split + user_id
    long: pd.DataFrame  # cleaned long frame (user_id, order_idx, skill_idx, correct, split)
    feature_cols: list[str]
    n_skills: int
    skill_remap: dict[int, int]
    stats: dict = field(default_factory=dict)

    def split_xy(self, split: str) -> tuple[pd.DataFrame, pd.Series]:
        """Feature matrix and target of one split (``train``, ``val`` or ``test``)."""
        sub = self.features[self.features["split"] == split]
        return sub[self.feature_cols].copy(), sub[TARGET].copy()


def _assign_splits(
    user_ids: np.ndarray, val_frac: float, test_frac: float, seed: int
) -> dict[int, str]:
    rng = np.random.default_rng(seed)
    users = np.array(sorted({int(u) for u in user_ids}))
    rng.shuffle(users)
    n = len(users)
    n_test = round(n * test_frac)
    n_val = round(n * val_frac)
    split_of: dict[int, str] = {}
    for u in users[:n_test]:
        split_of[int(u)] = "test"
    for u in users[n_test : n_test + n_val]:
        split_of[int(u)] = "val"
    for u in users[n_test + n_val :]:
        split_of[int(u)] = "train"
    return split_of


def transform(df_raw: pd.DataFrame, data_cfg: DataConfig, seed: int) -> ProcessedData:
    """Clean the interactions, split students and engineer causal features.

    Args:
        df_raw: Long frame with columns user_id, order_idx, skill_id, correct.
        data_cfg: Sequence-length limits and split fractions.
        seed: Seed of the student split.

    Returns:
        Features, cleaned interactions, the skill index mapping and statistics.
    """
    min_len = data_cfg.min_seq_len
    max_len = data_cfg.max_seq_len

    df = df_raw.copy()
    # --- clean ---
    df = df[df["correct"].isin([0, 1])]
    df = df.dropna(subset=["user_id", "skill_id", "correct"])
    df = df.astype({"user_id": int, "order_idx": int, "skill_id": int, "correct": int})
    df = df.sort_values(["user_id", "order_idx"]).reset_index(drop=True)

    # cap sequence length per student
    df["_rank"] = df.groupby("user_id").cumcount()
    df = df[df["_rank"] < max_len].drop(columns="_rank")

    # drop short students
    seq_len = df.groupby("user_id")["correct"].transform("size")
    df = df[seq_len >= min_len].reset_index(drop=True)

    # contiguous skill index 0..K-1
    skills = sorted(df["skill_id"].unique())
    skill_remap = {int(s): i for i, s in enumerate(skills)}
    df["skill_idx"] = df["skill_id"].map(skill_remap).astype(int)
    n_skills = len(skills)

    # re-rank order within user after cleaning so order_idx is dense
    df["order_idx"] = df.groupby("user_id").cumcount()

    # --- split by student ---
    split_of = _assign_splits(df["user_id"].to_numpy(), data_cfg.val_frac, data_cfg.test_frac, seed)
    df["split"] = df["user_id"].map(split_of)

    # --- causal feature engineering ---
    global_mean = float(df.loc[df["split"] == "train", "correct"].mean())

    g_user = df.groupby("user_id")
    df["user_prior_attempts"] = g_user.cumcount()
    user_cum = g_user["correct"].cumsum() - df["correct"]
    df["user_prior_correct_rate"] = np.where(
        df["user_prior_attempts"] > 0,
        user_cum / df["user_prior_attempts"].clip(lower=1),
        global_mean,
    )

    g_us = df.groupby(["user_id", "skill_idx"])
    df["skill_prior_attempts"] = g_us.cumcount()
    skill_cum = g_us["correct"].cumsum() - df["correct"]
    df["skill_prior_correct_rate"] = np.where(
        df["skill_prior_attempts"] > 0,
        skill_cum / df["skill_prior_attempts"].clip(lower=1),
        global_mean,
    )

    df["recent3_correct_rate"] = (
        g_user["correct"]
        .transform(lambda s: s.shift(1).rolling(RECENT_WINDOW, min_periods=1).mean())
        .fillna(global_mean)
    )

    # static skill difficulty estimated on TRAIN ONLY (1 - mean correct = harder)
    skill_diff = df.loc[df["split"] == "train"].groupby("skill_idx")["correct"].mean()
    df["skill_difficulty"] = (1.0 - df["skill_idx"].map(skill_diff)).fillna(1.0 - global_mean)

    features = df[["user_id", "split", TARGET, *FEATURE_COLS]].copy()
    # skill_idx kept as int so every AutoML estimator (lgbm/xgboost/rf) handles it;
    # the real per-skill signal is carried by skill_difficulty + skill_prior_correct_rate.
    features["skill_idx"] = features["skill_idx"].astype(int)

    stats = {
        "n_interactions": len(df),
        "n_students": int(df["user_id"].nunique()),
        "n_skills": int(n_skills),
        "global_correct_rate": round(global_mean, 4),
        "split_students": {
            s: int(df.loc[df["split"] == s, "user_id"].nunique()) for s in ("train", "val", "test")
        },
        "split_interactions": {s: int((df["split"] == s).sum()) for s in ("train", "val", "test")},
    }
    LOG.info("Transform complete: %s", stats)

    long = df[["user_id", "order_idx", "skill_idx", TARGET, "split"]].copy()
    return ProcessedData(
        features=features,
        long=long,
        feature_cols=FEATURE_COLS,
        n_skills=n_skills,
        skill_remap=skill_remap,
        stats=stats,
    )
