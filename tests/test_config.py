from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from knowledge_tracing.config import (
    PROJECT_ROOT_ENV,
    load_config,
    quick_overlay_path,
    resolve_path,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def _write_config(tmp_path: Path, mutate) -> Path:
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_project_config_is_valid(config):
    assert config.data.min_seq_len <= config.data.max_seq_len
    assert config.monitoring.psi_warn < config.monitoring.psi_alert
    assert config.models.automl_flaml.estimator_list


def test_quick_overlay_only_reduces_budgets(config):
    quick = load_config(CONFIG_PATH, [quick_overlay_path(CONFIG_PATH)])
    assert quick.models.dkt.epochs < config.models.dkt.epochs
    assert quick.models.dkt_optuna.n_trials < config.models.dkt_optuna.n_trials
    assert quick.models.automl_flaml.time_budget_s < config.models.automl_flaml.time_budget_s
    # everything not listed in quick.yaml keeps its value
    assert quick.models.dkt.lr == config.models.dkt.lr
    assert quick.data == config.data


def test_typo_in_key_is_rejected(tmp_path):
    def rename_epochs(data):
        data["models"]["dkt"]["epoch"] = data["models"]["dkt"].pop("epochs")

    with pytest.raises(ValidationError, match="epoch"):
        load_config(_write_config(tmp_path, rename_epochs))


def test_out_of_range_value_is_rejected(tmp_path):
    def negative_lr(data):
        data["models"]["dkt"]["lr"] = -0.01

    with pytest.raises(ValidationError, match="lr"):
        load_config(_write_config(tmp_path, negative_lr))


def test_inconsistent_split_fractions_are_rejected(tmp_path):
    def oversized_holdout(data):
        data["data"]["val_frac"] = 0.5
        data["data"]["test_frac"] = 0.6

    with pytest.raises(ValidationError, match="val_frac"):
        load_config(_write_config(tmp_path, oversized_holdout))


def test_relative_paths_resolve_against_project_root(tmp_path, monkeypatch):
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(tmp_path))
    assert resolve_path("data/raw") == tmp_path.resolve() / "data" / "raw"
    assert resolve_path("/abs/path") == Path("/abs/path")


def test_min_length_above_max_length_is_rejected(tmp_path):
    def inverted_lengths(data):
        data["data"]["min_seq_len"] = 500

    with pytest.raises(ValidationError, match="min_seq_len"):
        load_config(_write_config(tmp_path, inverted_lengths))


def test_drift_thresholds_must_be_ordered(tmp_path):
    def swapped_thresholds(data):
        data["monitoring"]["psi_warn"] = 0.3

    with pytest.raises(ValidationError, match="psi_warn"):
        load_config(_write_config(tmp_path, swapped_thresholds))


def test_top_level_must_be_a_mapping(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)
