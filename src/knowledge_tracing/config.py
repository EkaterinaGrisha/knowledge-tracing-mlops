"""Typed pipeline configuration.

``config/config.yaml`` is validated against the schema below as soon as it is
loaded: unknown keys (typos), missing values and out-of-range numbers fail at
start-up instead of in the middle of a long training run. Overlay files (for
example ``config/quick.yaml``) are deep-merged on top of the base file.

Relative paths are resolved against the project root: the current working
directory, or the directory given in ``KT_PROJECT_ROOT``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator

PROJECT_ROOT_ENV = "KT_PROJECT_ROOT"
DEFAULT_CONFIG_PATH = Path("config/config.yaml")
QUICK_OVERLAY_NAME = "quick.yaml"

PathLike = str | os.PathLike[str]


def project_root() -> Path:
    """Return the directory that relative paths in the configuration refer to."""
    return Path(os.environ.get(PROJECT_ROOT_ENV) or Path.cwd()).resolve()


def resolve_path(path: PathLike) -> Path:
    """Resolve a project-relative path; absolute paths are returned unchanged."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root() / candidate


class _Section(BaseModel):
    """Base for configuration sections: immutable, unknown keys are errors."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Section):
    """Data sources, storage locations and preprocessing parameters."""

    source_url_train: str
    source_url_test: str
    raw_dir: Path
    processed_dir: Path
    sample_path: Path
    min_seq_len: PositiveInt
    max_seq_len: PositiveInt
    val_frac: float = Field(gt=0, lt=1)
    test_frac: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.min_seq_len > self.max_seq_len:
            raise ValueError("min_seq_len must not exceed max_seq_len")
        if self.val_frac + self.test_frac >= 1:
            raise ValueError("val_frac + test_frac must leave students for training")
        return self


class BKTConfig(_Section):
    """Bayesian Knowledge Tracing."""

    em_iters: PositiveInt


class DKTConfig(_Section):
    """Deep Knowledge Tracing (LSTM) with a hand-picked architecture."""

    embed_dim: PositiveInt
    hidden_dim: PositiveInt
    dropout: float = Field(ge=0, lt=1)
    epochs: PositiveInt
    batch_size: PositiveInt
    lr: PositiveFloat


class DKTOptunaConfig(_Section):
    """Optuna search over the DKT architecture."""

    n_trials: PositiveInt
    epochs_per_trial: PositiveInt


class AutoMLConfig(_Section):
    """FLAML AutoML on the engineered tabular features."""

    time_budget_s: PositiveInt
    metric: str
    estimator_list: list[str] = Field(min_length=1)


class ModelsConfig(_Section):
    """Hyperparameters of every trained model."""

    bkt: BKTConfig
    dkt: DKTConfig
    dkt_optuna: DKTOptunaConfig
    automl_flaml: AutoMLConfig


class MonitoringConfig(_Section):
    """Data-quality gate behaviour and drift thresholds (PSI)."""

    fail_on_data_quality: bool
    psi_warn: PositiveFloat
    psi_alert: PositiveFloat

    @model_validator(mode="after")
    def _check_thresholds(self) -> Self:
        if self.psi_warn >= self.psi_alert:
            raise ValueError("psi_warn must be below psi_alert")
        return self


class MLflowConfig(_Section):
    """Experiment tracking."""

    experiment_name: str = Field(min_length=1)
    tracking_uri: str = Field(min_length=1)


class OutputConfig(_Section):
    """Where a run writes its reports."""

    figures_dir: Path
    metrics_path: Path


class Config(_Section):
    """Complete pipeline configuration (``config/config.yaml``)."""

    seed: int
    data: DataConfig
    models: ModelsConfig
    monitoring: MonitoringConfig
    mlflow: MLflowConfig
    output: OutputConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        content = yaml.safe_load(fh) or {}
    if not isinstance(content, dict):
        raise ValueError(f"{path}: the top level must be a mapping")
    return content


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def load_config(path: PathLike = DEFAULT_CONFIG_PATH, overlays: Sequence[PathLike] = ()) -> Config:
    """Load and validate the configuration.

    Args:
        path: Base YAML file, relative to the project root unless absolute.
        overlays: YAML files deep-merged on top of the base file, in order.

    Returns:
        The validated, immutable configuration.

    Raises:
        pydantic.ValidationError: The merged configuration does not match the schema.
    """
    data = _read_yaml(resolve_path(path))
    for overlay in overlays:
        data = _deep_merge(data, _read_yaml(resolve_path(overlay)))
    return Config.model_validate(data)


def quick_overlay_path(config_path: PathLike = DEFAULT_CONFIG_PATH) -> Path:
    """Return the quick-run overlay that lives next to the given base config."""
    return resolve_path(config_path).with_name(QUICK_OVERLAY_NAME)
