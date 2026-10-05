"""Common interface of the knowledge-tracing models trained by the pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from ..etl.datasets import SplitData, TrainingData

# (y_true, y_pred): observed correctness and the predicted P(correct) for every
# interaction the model makes a one-step-ahead prediction for.
Predictions = tuple[np.ndarray, np.ndarray]


class NotFittedError(RuntimeError):
    """A model was asked to predict before ``fit`` was called."""


class KnowledgeTracingModel(ABC):
    """Predicts whether a student answers the next exercise correctly.

    The pipeline treats every model the same way: ``fit`` on the training data,
    ``predict`` the test split, then compute metrics from the predictions.
    Adding a model means adding one subclass and one line in ``registry.py``.
    """

    name: str

    @abstractmethod
    def fit(self, data: TrainingData) -> None:
        """Train on ``data.train`` (``data.val`` is available for tuning)."""

    @abstractmethod
    def predict(self, split: SplitData) -> Predictions:
        """One-step-ahead predictions for every interaction of the split."""

    def mlflow_params(self) -> dict[str, Any]:
        """Model-specific parameters to log to MLflow (e.g. tuned hyperparameters)."""
        return {}

    def mlflow_metrics(self) -> dict[str, float]:
        """Model-specific metrics to log to MLflow (e.g. best validation score)."""
        return {}

    def report(self) -> dict[str, Any]:
        """Model-specific fields for metrics.json."""
        return {}

    def training_curve(self) -> list[float]:
        """Training loss per epoch, for models trained by gradient descent."""
        return []

    def feature_importance(self) -> pd.Series | None:
        """Importance of every input feature, for models that expose it."""
        return None
