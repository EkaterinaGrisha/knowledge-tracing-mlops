"""End-to-end automated ML pipeline for Knowledge Tracing.

Stages: ETL (extract/transform/load) -> data-quality gate -> train and evaluate
every model (BKT, DKT, DKT+Optuna, FLAML AutoML) -> drift monitoring ->
MLflow logging -> figures -> packaged serving model (registered as the
"challenger" if it passes the quality gate) -> metrics.json.

Run from the command line:
    kt train --data-source sample          # fast, offline (CI)
    kt train --data-source full            # full ASSISTments
    kt train --data-source sample --quick  # minimal budgets (CI smoke)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from . import __version__
from .config import Config, MonitoringConfig, resolve_path
from .etl.datasets import TrainingData, build_training_data
from .etl.extract import DataSource
from .etl.run import run_etl
from .etl.transform import ProcessedData
from .evaluation import visualize as viz
from .evaluation.metrics import compute_metrics
from .inference import DKTPredictor, log_predictor
from .models.base import KnowledgeTracingModel, Predictions
from .models.dkt import DKTModel
from .models.registry import build_models
from .monitoring.data_quality import enforce_quality_gate, quality_report
from .monitoring.drift import drift_report
from .monitoring.resources import ResourceMonitor
from .tracking import CHALLENGER, TEST_AUC_TAG, configure_tracking, mark_challenger

LOG = logging.getLogger(__name__)


@dataclass(eq=False)
class ModelRun:
    """A trained model with its test predictions, metrics and resource usage."""

    model: KnowledgeTracingModel
    predictions: Predictions
    metrics: dict[str, float]
    resources: dict[str, float]


def etl_stage(cfg: Config, data_source: DataSource) -> ProcessedData:
    """Extract -> transform -> load; dataset statistics go to MLflow."""
    processed, _ = run_etl(cfg, data_source)
    mlflow.log_params(
        {key: processed.stats[key] for key in ("n_students", "n_skills", "n_interactions")}
    )
    return processed


def check_data_quality(processed: ProcessedData, cfg: MonitoringConfig) -> dict[str, Any]:
    """Run the data-quality gate on the cleaned interactions.

    The report is logged to MLflow before the gate is enforced, so a failed
    run still shows which checks failed.

    Raises:
        DataQualityError: A check failed and ``cfg.fail_on_data_quality`` is set.
    """
    report = quality_report(processed.long)
    LOG.info("Data quality: passed=%s", report["passed"])
    mlflow.log_metric("data_quality_passed", int(report["passed"]))
    mlflow.log_dict(report, "monitoring/data_quality.json")
    enforce_quality_gate(report, fail=cfg.fail_on_data_quality)
    return report


def train_and_evaluate(
    models: list[KnowledgeTracingModel], data: TrainingData
) -> dict[str, ModelRun]:
    """Fit every model, predict the test split and measure the cost."""
    runs: dict[str, ModelRun] = {}
    for model in models:
        with ResourceMonitor(model.name) as monitor:
            model.fit(data)
            predictions = model.predict(data.test)
        runs[model.name] = ModelRun(
            model=model,
            predictions=predictions,
            metrics=compute_metrics(*predictions),
            resources=monitor.stats.as_dict(),
        )
        if params := model.mlflow_params():
            mlflow.log_params(params)
        for key, value in model.mlflow_metrics().items():
            mlflow.log_metric(key, value)
    return runs


def monitor_drift(processed: ProcessedData, cfg: MonitoringConfig) -> dict[str, Any]:
    """Compare feature distributions of the test split against the train split."""
    features = processed.features
    drift = drift_report(
        features[features["split"] == "train"],
        features[features["split"] == "test"],
        processed.feature_cols,
        psi_warn=cfg.psi_warn,
        psi_alert=cfg.psi_alert,
    )
    LOG.info(
        "Drift: %s (%d/%d features drifted)",
        drift["overall_status"],
        drift["n_significant_drift"],
        drift["n_features"],
    )
    mlflow.log_dict(drift, "monitoring/drift_report.json")
    return drift


def _metric_prefix(model_name: str) -> str:
    return model_name.lower().replace("+", "_")


def log_model_runs(runs: dict[str, ModelRun]) -> None:
    """Log test metrics and resource usage of every model to MLflow."""
    for name, run in runs.items():
        for key, value in run.metrics.items():
            if not math.isnan(value):
                mlflow.log_metric(f"{_metric_prefix(name)}__{key}", value)
    for name, run in runs.items():
        for key, value in run.resources.items():
            mlflow.log_metric(f"{_metric_prefix(name)}__{key}", value)


def select_best(runs: dict[str, ModelRun]) -> str:
    """Name of the model with the highest test AUC (an undefined AUC ranks last)."""

    def auc(name: str) -> float:
        value = runs[name].metrics["auc"]
        return -1.0 if math.isnan(value) else value

    return max(runs, key=auc)


def build_figures(
    processed: ProcessedData, runs: dict[str, ModelRun], best_name: str, figures_dir: Path
) -> list[Path]:
    """Render the report figures; returns the written files."""
    results = {name: run.metrics for name, run in runs.items()}
    predictions = {name: run.predictions for name, run in runs.items()}
    figures = [
        viz.plot_dataset_overview(processed.long, processed.stats, figures_dir),
        viz.plot_model_comparison(results, figures_dir),
        viz.plot_roc(predictions, figures_dir),
    ]
    if "DKT" in runs:
        figures.append(viz.plot_dkt_loss(runs["DKT"].model.training_curve(), figures_dir))
    y_true, y_pred = runs[best_name].predictions
    figures.append(viz.plot_confusion(y_true, y_pred, best_name, figures_dir))
    figures.append(viz.plot_calibration(y_true, y_pred, best_name, figures_dir))
    for run in runs.values():
        importance = run.model.feature_importance()
        if importance is not None:
            figures.append(
                viz.plot_feature_importance(
                    list(importance.index), importance.to_numpy(), figures_dir
                )
            )
            break
    else:
        LOG.warning("Feature importance unavailable")
    return figures


def _history_example(
    processed: ProcessedData, skill_ids: list[int], n_rows: int = 10
) -> pd.DataFrame:
    """A few test interactions in the serving input format (original skill ids)."""
    test = processed.long[processed.long["split"] == "test"]
    rows = test[test["user_id"] == test["user_id"].iloc[0]].head(n_rows)
    return pd.DataFrame(
        {
            "user_id": rows["user_id"].to_numpy(),
            "order_idx": rows["order_idx"].to_numpy(),
            "skill_id": [skill_ids[i] for i in rows["skill_idx"]],
            "correct": rows["correct"].to_numpy(),
        }
    )


def package_model(
    cfg: Config,
    runs: dict[str, ModelRun],
    processed: ProcessedData,
    data_source: DataSource,
    quick: bool,
) -> dict[str, Any]:
    """Save the serving model and register it in MLflow if it passes the quality gate.

    The model configured in ``serving.model`` is always written to
    ``output.model_dir`` and logged to the run; it becomes a new registry version
    with the ``challenger`` alias only if its test AUC reaches
    ``serving.min_test_auc`` and the run is not a smoke run: reduced ``quick``
    budgets do not produce a release candidate. Promotion to ``champion`` is a
    separate, approved step (``kt promote``).
    """
    run = runs[cfg.serving.model]
    if not isinstance(run.model, DKTModel):
        raise TypeError(f"serving.model must be a DKT model, got {run.model.name}")
    test_auc = run.metrics["auc"]
    active = mlflow.active_run()
    predictor = DKTPredictor.from_model(
        run.model,
        processed.skill_remap,
        metadata={
            "model": run.model.name,
            "package_version": __version__,
            "data_source": data_source,
            "quick": quick,
            "test_metrics": run.metrics,
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "mlflow_run_id": active.info.run_id if active else None,
        },
    )
    model_dir = predictor.save(resolve_path(cfg.output.model_dir))

    name = cfg.serving.registered_model_name
    passed = not math.isnan(test_auc) and test_auc >= cfg.serving.min_test_auc
    register = passed and not quick
    info = log_predictor(
        model_dir, _history_example(processed, predictor.skill_ids), name if register else None
    )
    serving: dict[str, Any] = {
        "model": run.model.name,
        "path": str(cfg.output.model_dir),
        "test_auc": test_auc,
        "min_test_auc": cfg.serving.min_test_auc,
        "registered": register,
    }
    if register:
        version = str(info.registered_model_version)
        mark_challenger(
            name,
            version,
            {TEST_AUC_TAG: f"{test_auc:.6f}", "data_source": data_source},
        )
        serving.update(registered_model=name, version=version, alias=CHALLENGER)
        LOG.info(
            "Registered %s version %s as '%s' (test AUC %.4f)", name, version, CHALLENGER, test_auc
        )
    elif quick:
        LOG.info("Smoke run (--quick): model saved to %s, not registered", cfg.output.model_dir)
    else:
        LOG.warning(
            "%s test AUC %.4f is below serving.min_test_auc %.2f: saved to %s, not registered",
            run.model.name,
            test_auc,
            cfg.serving.min_test_auc,
            cfg.output.model_dir,
        )
    return serving


def write_summary(summary: dict[str, Any], metrics_path: Path) -> Path:
    """Write the run summary as JSON and return its location."""
    path = resolve_path(metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return path


def run_pipeline(cfg: Config, data_source: DataSource, *, quick: bool = False) -> dict[str, Any]:
    """Run every stage of the pipeline inside one MLflow run.

    Args:
        cfg: Validated configuration.
        data_source: ``"sample"`` (committed subset) or ``"full"`` (full dataset).
        quick: Marks the run as a smoke run with reduced budgets (already applied
            to ``cfg`` via the quick overlay); used for the run name.

    Returns:
        The summary also written to ``cfg.output.metrics_path``.
    """
    configure_tracking(cfg.mlflow)
    with mlflow.start_run(run_name=f"kt-{data_source}{'-quick' if quick else ''}"):
        mlflow.log_params({"data_source": data_source, "quick": quick, "seed": cfg.seed})

        processed = etl_stage(cfg, data_source)
        dq = check_data_quality(processed, cfg.monitoring)
        data = build_training_data(processed)
        runs = train_and_evaluate(build_models(cfg.models, cfg.seed), data)
        drift = monitor_drift(processed, cfg.monitoring)
        log_model_runs(runs)

        best_name = select_best(runs)
        for figure in build_figures(processed, runs, best_name, cfg.output.figures_dir):
            mlflow.log_artifact(str(figure), artifact_path="figures")
        serving = package_model(cfg, runs, processed, data_source, quick)

        summary: dict[str, Any] = {
            "data_source": data_source,
            "dataset": processed.stats,
            "results": {name: run.metrics for name, run in runs.items()},
            "resources": {name: run.resources for name, run in runs.items()},
            "best_model": best_name,
            "drift": drift,
            "data_quality_passed": dq["passed"],
            "serving": serving,
        }
        for run in runs.values():
            summary.update(run.model.report())
        mlflow.log_artifact(str(write_summary(summary, cfg.output.metrics_path)))

    _print_summary(summary)
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    LOG.info("=" * 64)
    LOG.info("RESULTS (test, one-step-ahead correctness prediction)")
    LOG.info("%-12s %8s %8s %8s %8s", "model", "AUC", "ACC", "F1", "RMSE")
    for model, m in summary["results"].items():
        LOG.info(
            "%-12s %8.4f %8.4f %8.4f %8.4f", model, m["auc"], m["accuracy"], m["f1"], m["rmse"]
        )
    LOG.info(
        "Best model: %s | AutoML estimator: %s",
        summary["best_model"],
        summary.get("automl_best_estimator"),
    )
    LOG.info("=" * 64)
