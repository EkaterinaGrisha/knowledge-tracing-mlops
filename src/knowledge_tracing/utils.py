"""Shared helpers: config loading, logging, paths."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

# Project root: src/knowledge_tracing/utils.py -> repository root.
ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | os.PathLike = "config/config.yaml") -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with cfg_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(path: str | os.PathLike) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def get_logger(name: str = "kt-pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
