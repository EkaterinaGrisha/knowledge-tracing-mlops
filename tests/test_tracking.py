import os

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from knowledge_tracing.errors import PromotionError
from knowledge_tracing.tracking import (
    CHALLENGER,
    CHAMPION,
    TEST_AUC_TAG,
    promote,
    tracking_uri,
)

NAME = "kt-test"


@pytest.fixture
def registry(tmp_path):
    """An empty MLflow registry in a temporary SQLite store."""
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    client = MlflowClient()
    client.create_registered_model(NAME)
    yield client
    mlflow.set_tracking_uri(previous)


def _version(client: MlflowClient, auc: float, alias: str | None = None) -> str:
    version = client.create_model_version(NAME, source="file:///nonexistent").version
    client.set_model_version_tag(NAME, version, TEST_AUC_TAG, str(auc))
    if alias:
        client.set_registered_model_alias(NAME, alias, version)
    return str(version)


def test_first_challenger_becomes_champion(registry):
    version = _version(registry, 0.80, CHALLENGER)
    assert promote(NAME) == version
    assert str(registry.get_model_version_by_alias(NAME, CHAMPION).version) == version


def test_better_challenger_replaces_champion(registry):
    _version(registry, 0.78, CHAMPION)
    better = _version(registry, 0.80, CHALLENGER)
    assert promote(NAME) == better


def test_worse_challenger_is_rejected_unless_forced(registry):
    champion = _version(registry, 0.80, CHAMPION)
    worse = _version(registry, 0.75, CHALLENGER)
    with pytest.raises(PromotionError, match="worse"):
        promote(NAME)
    assert str(registry.get_model_version_by_alias(NAME, CHAMPION).version) == champion
    assert promote(NAME, force=True) == worse


def test_missing_challenger_is_an_error(registry):
    with pytest.raises(PromotionError, match="no version"):
        promote(NAME)


def test_promoting_the_current_champion_is_a_no_op(registry):
    version = _version(registry, 0.80, CHAMPION)
    registry.set_registered_model_alias(NAME, CHALLENGER, version)
    assert promote(NAME) == version


def test_local_stores_are_anchored_at_the_project_root(config, tmp_path, monkeypatch):
    monkeypatch.setenv("KT_PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    sqlite_uri = tracking_uri(config.mlflow)
    assert sqlite_uri == f"sqlite:///{tmp_path.resolve() / 'mlruns' / 'mlflow.db'}"

    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    file_cfg = config.mlflow.model_copy(update={"tracking_uri": "file:./mlruns"})
    assert tracking_uri(file_cfg) == f"file:{tmp_path.resolve() / 'mlruns'}"
    assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "true"


def test_environment_variable_overrides_the_config(config, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.example:5000")
    assert tracking_uri(config.mlflow) == "http://mlflow.example:5000"
