"""End-to-end automated ML pipeline for Knowledge Tracing.

Stages: ETL (extract/transform/load) -> data-quality gate -> train and evaluate
every model (BKT, DKT, DKT+Optuna, FLAML AutoML) -> drift monitoring ->
MLflow logging -> figures -> metrics.json.

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
from pathlib import Path
from typing import Any

import mlflow

from .config import Config, MonitoringConfig, resolve_path
from .etl.datasets import TrainingData, build_training_data
from .etl.extract import DataSource
from .etl.run import run_etl
from .etl.transform import ProcessedData
from .evaluation import visualize as viz
from .evaluation.metrics import compute_metrics
from .models.base import KnowledgeTracingModel, Predictions
from .models.registry import build_models
from .monitoring.data_quality import enforce_quality_gate, quality_report
from .monitoring.drift import drift_report
from .monitoring.resources import ResourceMonitor
from .tracking import configure_tracking

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

        summary: dict[str, Any] = {
            "data_source": data_source,
            "dataset": processed.stats,
            "results": {name: run.metrics for name, run in runs.items()},
            "resources": {name: run.resources for name, run in runs.items()},
            "best_model": best_name,
            "drift": drift,
            "data_quality_passed": dq["passed"],
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
