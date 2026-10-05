"""AutoML model via FLAML.

FLAML automatically selects the estimator (LightGBM / XGBoost / RandomForest /
ExtraTrees) and tunes its hyperparameters under a wall-clock budget to maximise
ROC-AUC for predicting next-attempt correctness from the engineered causal
features. This is the "ready AutoML framework" track of the assignment.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from flaml import AutoML
from typing_extensions import override

from ..config import AutoMLConfig
from ..etl.datasets import SplitData, TrainingData
from .base import KnowledgeTracingModel, NotFittedError, Predictions

LOG = logging.getLogger(__name__)


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
    """Let FLAML choose and tune an estimator under a wall-clock budget.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features used for model selection and early stopping.
        y_val: Validation labels.
        time_budget_s: Search budget in seconds; FLAML spends all of it, so the
            number of trials depends on machine speed.
        metric: Metric FLAML optimises (e.g. ``"roc_auc"``).
        estimator_list: Candidate estimators (e.g. ``["lgbm", "xgboost"]``).
        seed: Seed of the search.

    Returns:
        The fitted FLAML ``AutoML`` object.
    """
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


class AutoMLModel(KnowledgeTracingModel):
    """FLAML picks and tunes a tabular estimator on the causal features."""

    name = "AutoML"

    def __init__(self, cfg: AutoMLConfig, seed: int) -> None:
        self.cfg = cfg
        self.seed = seed
        self.automl_: AutoML | None = None
        self.feature_names_: list[str] = []

    @override
    def fit(self, data: TrainingData) -> None:
        self.feature_names_ = list(data.train.features.columns)
        self.automl_ = train_automl(
            data.train.features,
            data.train.target,
            data.val.features,
            data.val.target,
            time_budget_s=self.cfg.time_budget_s,
            metric=self.cfg.metric,
            estimator_list=list(self.cfg.estimator_list),
            seed=self.seed,
        )

    def _automl(self) -> AutoML:
        if self.automl_ is None:
            raise NotFittedError(f"{self.name} is not fitted")
        return self.automl_

    @override
    def predict(self, split: SplitData) -> Predictions:
        return predict_automl(self._automl(), split.features, split.target)

    @override
    def mlflow_params(self) -> dict[str, Any]:
        return {"automl_best_estimator": self._automl().best_estimator}

    @override
    def report(self) -> dict[str, Any]:
        return {"automl_best_estimator": self._automl().best_estimator}

    @override
    def feature_importance(self) -> pd.Series | None:
        estimator = getattr(getattr(self.automl_, "model", None), "estimator", None)
        importances = getattr(estimator, "feature_importances_", None)
        if importances is None:
            return None
        return pd.Series(np.asarray(importances), index=self.feature_names_)
