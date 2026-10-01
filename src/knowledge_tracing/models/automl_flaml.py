"""AutoML model via FLAML.

FLAML automatically selects the estimator (LightGBM / XGBoost / RandomForest /
ExtraTrees) and tunes its hyperparameters under a wall-clock budget to maximise
ROC-AUC for predicting next-attempt correctness from the engineered causal
features. This is the "ready AutoML framework" track of the assignment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

from ..utils import get_logger

LOG = get_logger()


def train_automl(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    time_budget_s: int = 60,
    metric: str = "roc_auc",
    estimator_list: list[str] | None = None,
    seed: int = 42,
) -> AutoML:
    automl = AutoML()
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        task="classification",
        metric=metric,
        time_budget=time_budget_s,
        estimator_list=estimator_list or ["lgbm", "xgboost", "rf", "extra_tree"],
        seed=seed,
        verbose=0,
        early_stop=True,
    )
    LOG.info("FLAML best estimator=%s | config=%s", automl.best_estimator, automl.best_config)
    return automl


def evaluate_automl(automl: AutoML, X: pd.DataFrame, y: pd.Series):
    proba = automl.predict_proba(X)[:, 1]
    yt = y.to_numpy().astype(int)
    auc = float(roc_auc_score(yt, proba))
    acc = float(accuracy_score(yt, (proba > 0.5).astype(int)))
    rmse = float(np.sqrt(mean_squared_error(yt, proba)))
    return auc, acc, rmse, yt, proba
