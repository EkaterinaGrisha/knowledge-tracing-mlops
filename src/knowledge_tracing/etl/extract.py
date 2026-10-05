"""ETL — Extract.

Source: ASSISTments 2009 "skill-builder" knowledge-tracing benchmark, distilled
triplet format (one student = 3 lines: length, skill-id list, correctness list).

This module turns the raw on-disk source into a single tidy *long* DataFrame:

    user_id | order_idx | skill_id | correct

Two source modes:
  * "full"   — download the full train+test triplet files (network required, cached).
  * "sample" — load the committed long-format CSV (offline; used by tests/CI).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

from ..config import DataConfig, resolve_path

LOG = logging.getLogger(__name__)
LONG_COLUMNS = ["user_id", "order_idx", "skill_id", "correct"]
DataSource = Literal["sample", "full"]


def _download(url: str, dest: Path, retries: int = 4) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: requests.RequestException | None = None
    for attempt in range(retries):
        try:
            LOG.info("Downloading %s -> %s", url, dest)
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return
        except requests.RequestException as exc:  # network and HTTP errors are retried
            last_err = exc
            wait = 2 ** (attempt + 1)
            LOG.warning(
                "Download failed (attempt %d): %s — retrying in %ds", attempt + 1, exc, wait
            )
            time.sleep(wait)
    raise RuntimeError(f"Could not download {url} after {retries} attempts") from last_err


def parse_triplet_file(path: Path, user_offset: int = 0) -> pd.DataFrame:
    """Parse the 3-lines-per-student triplet format into a long DataFrame."""
    rows: list[tuple[int, int, int, int]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    user_id = user_offset
    i = 0
    while i + 2 < len(lines):
        n_raw = lines[i].strip()
        skills_raw = lines[i + 1].strip()
        correct_raw = lines[i + 2].strip()
        i += 3
        if not n_raw:
            continue
        skills = [s for s in skills_raw.split(",") if s != ""]
        corrects = [c for c in correct_raw.split(",") if c != ""]
        n = min(len(skills), len(corrects))
        if n == 0:
            continue
        for order_idx in range(n):
            rows.append((user_id, order_idx, int(skills[order_idx]), int(corrects[order_idx])))
        user_id += 1
    return pd.DataFrame(rows, columns=LONG_COLUMNS)


def extract(data_cfg: DataConfig, data_source: DataSource = "sample") -> pd.DataFrame:
    """Return the raw interactions as a long DataFrame.

    Args:
        data_cfg: Data section of the configuration.
        data_source: ``"sample"`` — the committed subset (offline);
            ``"full"`` — the full dataset, downloaded and cached on first use.
    """
    if data_source == "sample":
        sample_path = resolve_path(data_cfg.sample_path)
        if not sample_path.exists():
            raise FileNotFoundError(
                f"Sample dataset not found at {sample_path}. "
                "Regenerate it with: kt etl --data-source full --write-sample 300"
            )
        LOG.info("Loading committed sample dataset: %s", sample_path)
        df = pd.read_csv(sample_path)
        return df[LONG_COLUMNS].copy()

    if data_source == "full":
        raw_dir = resolve_path(data_cfg.raw_dir)
        train_path = raw_dir / "assist2009_train.csv"
        test_path = raw_dir / "assist2009_test.csv"
        if not train_path.exists():
            _download(data_cfg.source_url_train, train_path)
        if not test_path.exists():
            _download(data_cfg.source_url_test, test_path)

        df_train = parse_triplet_file(train_path, user_offset=0)
        next_uid = int(df_train["user_id"].max()) + 1 if len(df_train) else 0
        df_test = parse_triplet_file(test_path, user_offset=next_uid)
        df = pd.concat([df_train, df_test], ignore_index=True)
        LOG.info(
            "Extracted %d interactions from %d students (%d skills).",
            len(df),
            df["user_id"].nunique(),
            df["skill_id"].nunique(),
        )
        return df

    raise ValueError(f"Unknown data_source: {data_source!r} (expected 'sample' or 'full')")


def write_sample(df: pd.DataFrame, sample_path: Path, n_students: int = 300) -> Path:
    """Persist the first ``n_students`` students as the committed sample CSV."""
    keep = sorted(df["user_id"].unique())[:n_students]
    sample = df[df["user_id"].isin(keep)].copy()
    out = resolve_path(sample_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False)
    LOG.info(
        "Wrote sample (%d rows, %d students) -> %s", len(sample), sample["user_id"].nunique(), out
    )
    return out
