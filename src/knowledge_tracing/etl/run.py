"""Standalone ETL entrypoint: Extract -> Transform -> Load."""

from __future__ import annotations

import argparse

from ..utils import get_logger, load_config
from .extract import extract
from .load import load
from .transform import transform

LOG = get_logger()


def run_etl(config_path: str, data_source: str):
    cfg = load_config(config_path)
    df_raw = extract(cfg, data_source=data_source)
    processed = transform(df_raw, cfg)
    paths = load(processed, cfg)
    return processed, paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the ETL stage only.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--data-source", choices=["sample", "full"], default="sample")
    args = ap.parse_args()
    _, paths = run_etl(args.config, args.data_source)
    LOG.info("ETL artifacts: %s", {k: str(v) for k, v in paths.items()})


if __name__ == "__main__":
    main()
