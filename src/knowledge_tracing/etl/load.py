"""ETL — Load.

Persists the processed feature frame, the cleaned long frame, and a JSON of
dataset statistics into `data/processed/` so downstream training is decoupled
from extraction/transformation and the artifacts are inspectable.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import resolve_path
from ..utils import get_logger
from .transform import ProcessedData

LOG = get_logger()


def load(processed: ProcessedData, processed_dir: Path) -> dict[str, Path]:
    out_dir = resolve_path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features_path = out_dir / "features.parquet"
    long_path = out_dir / "interactions_long.parquet"
    stats_path = out_dir / "dataset_stats.json"

    processed.features.to_parquet(features_path, index=False)
    processed.long.to_parquet(long_path, index=False)
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(processed.stats, fh, ensure_ascii=False, indent=2)

    LOG.info("Loaded processed artifacts into %s", out_dir)
    return {"features": features_path, "long": long_path, "stats": stats_path}
