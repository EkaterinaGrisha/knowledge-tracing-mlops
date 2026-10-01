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

from ..utils import get_logger

LOG = get_logger()


def train_automl(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    time_budget_s: int,
    metric: str,
    estimator_list: list[str],
    seed: int,
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
        estimator_list=estimator_list,
        seed=seed,
        verbose=0,
        early_stop=True,
    )
    LOG.info("FLAML best estimator=%s | config=%s", automl.best_estimator, automl.best_config)
    return automl


def predict_automl(automl: AutoML, X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Predictions on a feature frame: (observed correctness, P(correct))."""
    proba = automl.predict_proba(X)[:, 1]
    return y.to_numpy().astype(int), proba
