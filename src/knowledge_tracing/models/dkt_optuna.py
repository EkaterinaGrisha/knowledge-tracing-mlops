"""Automated architecture/hyperparameter search for DKT using Optuna.

This is the "automation of individual architecture elements" track: instead of
hand-picking the LSTM size, embedding width, dropout and learning rate, Optuna
runs a TPE search that maximises validation AUC, then we retrain the best
configuration and report it on the test split.
"""

from __future__ import annotations

import optuna

from ..utils import get_logger
from .dkt import StudentTrace, evaluate_dkt, pick_device, train_dkt

LOG = get_logger()
optuna.logging.set_verbosity(optuna.logging.WARNING)


def search_dkt(
    train_traces: list[StudentTrace],
    val_traces: list[StudentTrace],
    n_concepts: int,
    *,
    n_trials: int = 12,
    epochs_per_trial: int = 6,
    seed: int = 42,
) -> dict:
    device = pick_device()

    def objective(trial: optuna.Trial) -> float:
        embed_dim = trial.suggest_categorical("embed_dim", [32, 64, 128])
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-3, 2e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        model, _ = train_dkt(
            train_traces,
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
        auc, _, _, _, _ = evaluate_dkt(model, val_traces, device)
        return auc

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    LOG.info("Optuna best val AUC=%.4f params=%s", study.best_value, study.best_params)
    return {
        "best_params": study.best_params,
        "best_val_auc": float(study.best_value),
        "study": study,
    }
