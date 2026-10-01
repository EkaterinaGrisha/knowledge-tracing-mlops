import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml
from mlflow.tracking import MlflowClient

from knowledge_tracing.cli import EXIT_DATA_QUALITY, EXIT_INVALID_CONFIG, main

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.startswith("kt ")


def test_etl_writes_processed_data(project_dir):
    assert main(["etl", "--data-source", "sample"]) == 0
    processed = project_dir / "data" / "processed"
    assert (processed / "features.parquet").exists()
    assert (processed / "interactions_long.parquet").exists()
    stats = json.loads((processed / "dataset_stats.json").read_text(encoding="utf-8"))
    assert stats["n_students"] > 0


def test_invalid_config_returns_error_code(project_dir):
    config = project_dir / "config" / "config.yaml"
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["models"]["dkt"]["epoch"] = data["models"]["dkt"].pop("epochs")  # typo
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert main(["train", "--config", str(config)]) == EXIT_INVALID_CONFIG


def test_failed_quality_gate_stops_before_training(project_dir, monkeypatch):
    failing = {"passed": False, "checks": {"binary_labels": {"passed": False}}}
    monkeypatch.setattr("knowledge_tracing.pipeline.quality_report", lambda _: failing)
    assert main(["train", "--quick"]) == EXIT_DATA_QUALITY
    assert not (project_dir / "reports" / "metrics.json").exists()


@pytest.mark.slow
def test_train_register_predict_promote(project_dir):
    tiny = project_dir / "config" / "tiny.yaml"
    tiny.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "bkt": {"em_iters": 2},
                    "dkt": {"epochs": 1},
                    "dkt_optuna": {"n_trials": 1, "epochs_per_trial": 1},
                    "automl_flaml": {"time_budget_s": 3},
                },
                "serving": {"min_test_auc": 0.5},
            }
        ),
        encoding="utf-8",
    )
    assert main(["train", "--quick", "--override", str(tiny)]) == 0

    metrics = json.loads((project_dir / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["results"]) == {"BKT", "DKT", "DKT+Optuna", "AutoML"}
    assert len(list((project_dir / "reports" / "figures").glob("*.png"))) == 7
    serving = metrics["serving"]
    assert serving["registered"] is True
    assert serving["alias"] == "challenger"
    assert (project_dir / "artifacts" / "model" / "model.pt").exists()

    # the packaged model and the registered version give the same predictions
    shutil.copy(REPO_ROOT / "examples" / "history.csv", project_dir / "history.csv")
    local_args = ["predict", "--input", "history.csv", "--output", "out/local.csv"]
    assert main(local_args) == 0
    registry_args = ["predict", "--model", "models:/kt-dkt@challenger", "--input", "history.csv"]
    assert main([*registry_args, "--output", "out/registry.csv"]) == 0
    local = pd.read_csv(project_dir / "out" / "local.csv")
    assert list(local.columns) == ["user_id", "skill_id", "p_correct"]
    assert local["p_correct"].between(0, 1).all()
    pd.testing.assert_frame_equal(pd.read_csv(project_dir / "out" / "registry.csv"), local)

    # approval step: the challenger becomes the champion
    assert main(["promote"]) == 0
    client = MlflowClient()
    champion = client.get_model_version_by_alias("kt-dkt", "champion")
    assert str(champion.version) == serving["version"]
