"""MLflow experiment tracking and Model Registry operations."""

from __future__ import annotations

import logging
import math
import os

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from .config import MLflowConfig, resolve_path
from .errors import PromotionError

LOG = logging.getLogger(__name__)

TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
# Registry aliases: a freshly trained version that passed the quality gate is
# the "challenger"; the version approved for use is the "champion".
CHALLENGER = "challenger"
CHAMPION = "champion"
TEST_AUC_TAG = "test_auc"

_FILE_SCHEME = "file:"
_SQLITE_SCHEME = "sqlite:///"


def tracking_uri(cfg: MLflowConfig) -> str:
    """Return the tracking URI: ``MLFLOW_TRACKING_URI`` wins over the config.

    Local stores (``sqlite:///…`` or ``file:…`` with a relative path) are anchored
    at the project root, so their location does not depend on the directory the
    command is run from.
    """
    uri = os.environ.get(TRACKING_URI_ENV) or cfg.tracking_uri
    if uri.startswith(_SQLITE_SCHEME):
        db_path = resolve_path(uri.removeprefix(_SQLITE_SCHEME))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return _SQLITE_SCHEME + str(db_path)
    if uri.startswith(_FILE_SCHEME):
        # MLflow 3 refuses the file store unless explicitly allowed; using it
        # is a deliberate choice of the configuration.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        return _FILE_SCHEME + str(resolve_path(uri.removeprefix(_FILE_SCHEME)))
    return uri


def use_tracking_store(cfg: MLflowConfig) -> None:
    """Point MLflow (tracking and registry) at the configured store."""
    mlflow.set_tracking_uri(tracking_uri(cfg))


def configure_tracking(cfg: MLflowConfig) -> None:
    """Use the configured store and experiment; create the experiment on first use."""
    use_tracking_store(cfg)
    if mlflow.get_experiment_by_name(cfg.experiment_name) is None:
        artifacts = resolve_path(cfg.artifact_location)
        mlflow.create_experiment(cfg.experiment_name, artifact_location=artifacts.as_uri())
    mlflow.set_experiment(cfg.experiment_name)


def mark_challenger(name: str, version: str, tags: dict[str, str]) -> None:
    """Tag a freshly registered version and point the challenger alias at it."""
    client = MlflowClient()
    for key, value in tags.items():
        client.set_model_version_tag(name, version, key, value)
    client.set_registered_model_alias(name, CHALLENGER, version)


def _version_by_alias(client: MlflowClient, name: str, alias: str) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def _test_auc(version: ModelVersion) -> float:
    return float(version.tags.get(TEST_AUC_TAG, "nan"))


def promote(
    name: str, *, source: str = CHALLENGER, target: str = CHAMPION, force: bool = False
) -> str:
    """Point the ``target`` alias at the version currently under ``source``.

    This is the deployment step of the retraining process (Model_retrain_BPMN.md):
    it runs after the model owner has approved the new version. Unless ``force``
    is set, a candidate with a lower test AUC than the current ``target`` is rejected.

    Returns:
        The promoted version.

    Raises:
        PromotionError: There is no ``source`` version, or it is worse than ``target``.
    """
    client = MlflowClient()
    candidate = _version_by_alias(client, name, source)
    if candidate is None:
        raise PromotionError(f"model '{name}' has no version with alias '{source}'")
    current = _version_by_alias(client, name, target)
    if current is not None and current.version == candidate.version:
        LOG.info("Version %s of '%s' is already '%s'", candidate.version, name, target)
        return str(candidate.version)
    if current is not None and not force:
        candidate_auc, current_auc = _test_auc(candidate), _test_auc(current)
        if not math.isnan(current_auc) and not candidate_auc >= current_auc:
            raise PromotionError(
                f"'{source}' version {candidate.version} (test AUC {candidate_auc:.4f}) is worse "
                f"than '{target}' version {current.version} ({current_auc:.4f}); "
                "use --force to promote anyway"
            )
    client.set_registered_model_alias(name, target, candidate.version)
    LOG.info("Model '%s' version %s is now '%s'", name, candidate.version, target)
    return str(candidate.version)
