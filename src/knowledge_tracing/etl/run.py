"""ETL stage: Extract -> Transform -> Load."""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import Config
from .extract import DataSource, extract, write_sample
from .load import load
from .transform import ProcessedData, transform

LOG = logging.getLogger(__name__)


def run_etl(
    cfg: Config, data_source: DataSource, *, sample_students: int | None = None
) -> tuple[ProcessedData, dict[str, Path]]:
    """Run the ETL stage and return the processed data with the written files.

    Args:
        cfg: Validated configuration.
        data_source: ``"sample"`` or ``"full"``.
        sample_students: If set, also refresh the committed sample CSV with the
            first ``sample_students`` students of the extracted data.
    """
    df_raw = extract(cfg.data, data_source=data_source)
    if sample_students is not None:
        write_sample(df_raw, cfg.data.sample_path, n_students=sample_students)
    processed = transform(df_raw, cfg.data, seed=cfg.seed)
    paths = load(processed, cfg.data.processed_dir)
    LOG.info("ETL artifacts: %s", {name: str(path) for name, path in paths.items()})
    return processed, paths
