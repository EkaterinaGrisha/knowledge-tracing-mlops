"""Automated architecture/hyperparameter search for DKT using Optuna.

This is the "automation of individual architecture elements" track: instead of
hand-picking the LSTM size, embedding width, dropout and learning rate, Optuna
runs a TPE search that maximises validation AUC, then we retrain the best
configuration and report it on the test split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import optuna

from ..config import DKTConfig, DKTOptunaConfig
from ..etl.datasets import StudentSequence, TrainingData
from ..evaluation.metrics import roc_auc
from .base import NotFittedError
from .dkt import DKTHyperparameters, DKTModel, pick_device, predict_dkt, train_dkt

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """Outcome of the architecture search."""

    best_params: dict[str, Any]
    best_val_auc: float
    study: optuna.Study


def search_dkt(
    train_sequences: list[StudentSequence],
    val_sequences: list[StudentSequence],
    n_concepts: int,
    *,
    n_trials: int,
    epochs_per_trial: int,
    seed: int,
) -> SearchResult:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    device = pick_device()

    def objective(trial: optuna.Trial) -> float:
        embed_dim = trial.suggest_categorical("embed_dim", [32, 64, 128])
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-3, 2e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        model, _ = train_dkt(
            train_sequences,
            n_concepts,
            device=device,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs_per_trial,
            seed=seed,
        )
        y_true, y_pred = predict_dkt(model, val_sequences, device)
        return roc_auc(y_true, y_pred)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    LOG.info("Optuna best val AUC=%.4f params=%s", study.best_value, study.best_params)
    return SearchResult(study.best_params, float(study.best_value), study)


class DKTOptunaModel(DKTModel):
    """DKT whose architecture is chosen by an Optuna search on the validation split."""

    name = "DKT+Optuna"

    def __init__(self, search_cfg: DKTOptunaConfig, dkt_cfg: DKTConfig, seed: int) -> None:
        super().__init__(dkt_cfg, seed)
        self.search_cfg = search_cfg
        self.search_: SearchResult | None = None

    def fit(self, data: TrainingData) -> None:
        self.search_ = search_dkt(
            data.train.sequences,
            data.val.sequences,
            data.n_skills,
            n_trials=self.search_cfg.n_trials,
            epochs_per_trial=self.search_cfg.epochs_per_trial,
            seed=self.seed,
        )
        best = self.search_.best_params
        # the winning architecture is retrained for the full number of epochs
        self._fit_with(
            data,
            DKTHyperparameters(
                embed_dim=best["embed_dim"],
                hidden_dim=best["hidden_dim"],
                dropout=best["dropout"],
                lr=best["lr"],
                batch_size=best["batch_size"],
            ),
        )

    def _search(self) -> SearchResult:
        if self.search_ is None:
            raise NotFittedError(f"{self.name} is not fitted")
        return self.search_

    def mlflow_params(self) -> dict[str, Any]:
        return {f"dkt_optuna_{k}": v for k, v in self._search().best_params.items()}

    def mlflow_metrics(self) -> dict[str, float]:
        return {"dkt_optuna_best_val_auc": self._search().best_val_auc}

    def report(self) -> dict[str, Any]:
        return {"dkt_optuna_best_params": self._search().best_params}
