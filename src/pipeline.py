"""End-to-end automated ML pipeline for Knowledge Tracing.

Stages: ETL (extract/transform/load) -> data-quality gate -> train 4 models
(BKT, DKT, DKT+Optuna, FLAML AutoML) -> evaluate -> drift monitoring ->
visualisations -> MLflow logging -> metrics.json.

Run:
    python -m src.pipeline --data-source sample        # fast, offline (CI)
    python -m src.pipeline --data-source full          # full ASSISTments
    python -m src.pipeline --data-source sample --quick # minimal budgets (CI smoke)
"""

from __future__ import annotations

import argparse
import json

import mlflow
import numpy as np

from .etl.extract import extract
from .etl.load import load
from .etl.transform import build_sequences, transform
from .evaluation import visualize as viz
from .evaluation.metrics import compute_metrics
from .models import bkt as bkt_mod
from .models import dkt as dkt_mod
from .models.automl_flaml import evaluate_automl, train_automl
from .models.dkt_optuna import search_dkt
from .monitoring.data_quality import quality_report
from .monitoring.drift import drift_report
from .monitoring.resources import ResourceMonitor
from .utils import get_logger, load_config, resolve

LOG = get_logger()


def _apply_quick(cfg: dict) -> None:
    cfg["models"]["dkt"]["epochs"] = 5
    cfg["models"]["dkt_optuna"]["n_trials"] = 4
    cfg["models"]["dkt_optuna"]["epochs_per_trial"] = 3
    cfg["models"]["automl_flaml"]["time_budget_s"] = 15


def run(config_path: str, data_source: str, quick: bool = False) -> dict:
    cfg = load_config(config_path)
    if quick:
        _apply_quick(cfg)
    seed = cfg["seed"]

    mlflow.set_tracking_uri(f"file:{resolve('mlruns')}")
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    results: dict[str, dict] = {}
    preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    resources: dict[str, dict] = {}

    with mlflow.start_run(run_name=f"kt-{data_source}{'-quick' if quick else ''}"):
        mlflow.log_params({"data_source": data_source, "quick": quick, "seed": seed})

        # ---------- ETL ----------
        df_raw = extract(cfg, data_source=data_source)
        processed = transform(df_raw, cfg)
        load(processed, cfg)
        mlflow.log_params(
            {
                "n_students": processed.stats["n_students"],
                "n_skills": processed.stats["n_skills"],
                "n_interactions": processed.stats["n_interactions"],
            }
        )

        # ---------- data quality gate ----------
        dq = quality_report(processed.long)
        LOG.info("Data quality: passed=%s", dq["passed"])
        mlflow.log_metric("data_quality_passed", int(dq["passed"]))
        mlflow.log_dict(dq, "monitoring/data_quality.json")
        if not dq["passed"]:
            LOG.warning("Data quality checks failed: %s", dq["checks"])

        # ---------- sequences for BKT/DKT ----------
        train_seq = build_sequences(processed.long, "train")
        val_seq = build_sequences(processed.long, "val")
        test_seq = build_sequences(processed.long, "test")
        n_skills = processed.n_skills

        # ---------- 1) BKT baseline ----------
        with ResourceMonitor("bkt") as rm:
            bkt_params = bkt_mod.fit_bkt_per_skill(
                train_seq, n_skills, cfg["models"]["bkt"]["em_iters"]
            )
            _, _, _, yt_bkt, yp_bkt = bkt_mod.evaluate_bkt(bkt_params, test_seq, n_skills)
        results["BKT"] = compute_metrics(yt_bkt, yp_bkt)
        preds["BKT"] = (yt_bkt, yp_bkt)
        resources["BKT"] = rm.stats.as_dict()

        # ---------- 2) DKT (default config) ----------
        device = dkt_mod.pick_device()
        train_traces = dkt_mod.traces_from_sequences(train_seq)
        val_traces = dkt_mod.traces_from_sequences(val_seq)
        test_traces = dkt_mod.traces_from_sequences(test_seq)
        dcfg = cfg["models"]["dkt"]
        with ResourceMonitor("dkt") as rm:
            dkt_model, losses = dkt_mod.train_dkt(
                train_traces,
                n_skills,
                device=device,
                embed_dim=dcfg["embed_dim"],
                hidden_dim=dcfg["hidden_dim"],
                dropout=dcfg["dropout"],
                epochs=dcfg["epochs"],
                batch_size=dcfg["batch_size"],
                lr=dcfg["lr"],
                seed=seed,
            )
            _, _, _, yt_dkt, yp_dkt = dkt_mod.evaluate_dkt(dkt_model, test_traces, device)
        results["DKT"] = compute_metrics(yt_dkt, yp_dkt)
        preds["DKT"] = (yt_dkt, yp_dkt)
        resources["DKT"] = rm.stats.as_dict()

        # ---------- 3) DKT + Optuna (automated HPO) ----------
        ocfg = cfg["models"]["dkt_optuna"]
        with ResourceMonitor("dkt_optuna") as rm:
            search = search_dkt(
                train_traces,
                val_traces,
                n_skills,
                n_trials=ocfg["n_trials"],
                epochs_per_trial=ocfg["epochs_per_trial"],
                seed=seed,
            )
            best = search["best_params"]
            best_model, _ = dkt_mod.train_dkt(
                train_traces,
                n_skills,
                device=device,
                embed_dim=best["embed_dim"],
                hidden_dim=best["hidden_dim"],
                dropout=best["dropout"],
                lr=best["lr"],
                batch_size=best["batch_size"],
                epochs=dcfg["epochs"],
                seed=seed,
            )
            _, _, _, yt_opt, yp_opt = dkt_mod.evaluate_dkt(best_model, test_traces, device)
        results["DKT+Optuna"] = compute_metrics(yt_opt, yp_opt)
        preds["DKT+Optuna"] = (yt_opt, yp_opt)
        resources["DKT+Optuna"] = rm.stats.as_dict()
        mlflow.log_params({f"dkt_optuna_{k}": v for k, v in best.items()})
        mlflow.log_metric("dkt_optuna_best_val_auc", search["best_val_auc"])

        # ---------- 4) FLAML AutoML ----------
        Xtr, ytr = processed.split_xy("train")
        Xva, yva = processed.split_xy("val")
        Xte, yte = processed.split_xy("test")
        acfg = cfg["models"]["automl_flaml"]
        with ResourceMonitor("automl_flaml") as rm:
            automl = train_automl(
                Xtr,
                ytr,
                Xva,
                yva,
                time_budget_s=acfg["time_budget_s"],
                metric=acfg["metric"],
                estimator_list=acfg["estimator_list"],
                seed=seed,
            )
            _, _, _, yt_aml, yp_aml = evaluate_automl(automl, Xte, yte)
        results["AutoML"] = compute_metrics(yt_aml, yp_aml)
        preds["AutoML"] = (yt_aml, yp_aml)
        resources["AutoML"] = rm.stats.as_dict()
        mlflow.log_param("automl_best_estimator", automl.best_estimator)

        # ---------- drift monitoring (train vs test features) ----------
        ref = processed.features[processed.features["split"] == "train"]
        cur = processed.features[processed.features["split"] == "test"]
        drift = drift_report(ref, cur, processed.feature_cols, cfg)
        LOG.info(
            "Drift: %s (%d/%d features drifted)",
            drift["overall_status"],
            drift["n_significant_drift"],
            drift["n_features"],
        )
        mlflow.log_dict(drift, "monitoring/drift_report.json")

        # ---------- log metrics + resources ----------
        for model, m in results.items():
            tag = model.lower().replace("+", "_")
            for k, v in m.items():
                if v == v:  # skip NaN
                    mlflow.log_metric(f"{tag}__{k}", v)
        for model, r in resources.items():
            tag = model.lower().replace("+", "_")
            for k, v in r.items():
                mlflow.log_metric(f"{tag}__{k}", v)

        # ---------- visualisations ----------
        figs = []
        figs.append(viz.plot_dataset_overview(processed.long, processed.stats, cfg))
        figs.append(viz.plot_model_comparison(results, cfg))
        figs.append(viz.plot_roc(preds, cfg))
        figs.append(viz.plot_dkt_loss(losses, cfg))
        best_name = max(
            results,
            key=lambda k: results[k]["auc"] if results[k]["auc"] == results[k]["auc"] else -1,
        )
        y_true_best, y_pred_best = preds[best_name]
        figs.append(viz.plot_confusion(y_true_best, y_pred_best, best_name, cfg))
        figs.append(viz.plot_calibration(y_true_best, y_pred_best, best_name, cfg))
        try:
            est = automl.model.estimator
            importances = getattr(est, "feature_importances_", None)
            if importances is not None:
                figs.append(viz.plot_feature_importance(list(Xtr.columns), importances, cfg))
        except Exception as exc:
            LOG.warning("Feature importance unavailable: %s", exc)
        for f in figs:
            mlflow.log_artifact(str(f), artifact_path="figures")

        # ---------- write metrics.json ----------
        summary = {
            "data_source": data_source,
            "dataset": processed.stats,
            "results": results,
            "resources": resources,
            "best_model": best_name,
            "drift": drift,
            "data_quality_passed": dq["passed"],
            "automl_best_estimator": automl.best_estimator,
            "dkt_optuna_best_params": best,
        }
        metrics_path = resolve(cfg["output"]["metrics_path"])
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        mlflow.log_artifact(str(metrics_path))

    _print_summary(summary)
    return summary


def _print_summary(summary: dict) -> None:
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
        summary["automl_best_estimator"],
    )
    LOG.info("=" * 64)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the end-to-end KT ML pipeline.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--data-source", choices=["sample", "full"], default="sample")
    ap.add_argument("--quick", action="store_true", help="minimal budgets for CI smoke runs")
    args = ap.parse_args()
    run(args.config, args.data_source, quick=args.quick)


if __name__ == "__main__":
    main()
