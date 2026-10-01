"""Automated architecture/hyperparameter search for DKT using Optuna.

This is the "automation of individual architecture elements" track: instead of
hand-picking the LSTM size, embedding width, dropout and learning rate, Optuna
runs a TPE search that maximises validation AUC, then we retrain the best
configuration and report it on the test split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import optuna

from ..etl.datasets import StudentSequence
from ..evaluation.metrics import roc_auc
from ..utils import get_logger
from .dkt import pick_device, predict_dkt, train_dkt

LOG = get_logger()


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
