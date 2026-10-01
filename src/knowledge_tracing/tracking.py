"""MLflow experiment tracking setup."""

from __future__ import annotations

import os

import mlflow

from .config import MLflowConfig, resolve_path

TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
_FILE_SCHEME = "file:"


def tracking_uri(cfg: MLflowConfig) -> str:
    """Return the tracking URI: ``MLFLOW_TRACKING_URI`` wins over the config.

    A relative local store (``file:./mlruns``) is anchored at the project root,
    so the location does not depend on the directory the pipeline is run from.
    """
    uri = os.environ.get(TRACKING_URI_ENV) or cfg.tracking_uri
    if uri.startswith(_FILE_SCHEME):
        return _FILE_SCHEME + str(resolve_path(uri.removeprefix(_FILE_SCHEME)))
    return uri


def configure_tracking(cfg: MLflowConfig) -> None:
    """Point MLflow at the configured tracking store and experiment."""
    mlflow.set_tracking_uri(tracking_uri(cfg))
    mlflow.set_experiment(cfg.experiment_name)
