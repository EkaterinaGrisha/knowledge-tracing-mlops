"""Standard classification metrics for one-step-ahead correctness prediction."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

# A predicted P(correct) above the threshold counts as "will answer correctly".
DECISION_THRESHOLD = 0.5
# Probabilities are clipped away from 0 and 1 so log-loss stays finite.
PROBA_EPS = 1e-7


def has_both_classes(y_true: np.ndarray) -> bool:
    """True when the labels contain both outcomes, so AUC and log-loss are defined."""
    return np.unique(np.asarray(y_true).astype(int)).size > 1


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC of raw scores; NaN when only one class is present."""
    if not has_both_classes(y_true):
        return float("nan")
    return float(roc_auc_score(np.asarray(y_true).astype(int), y_score))


def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict[str, float]:
    """Classification metrics of P(correct) predictions against observed answers.

    AUC and log-loss are NaN when ``y_true`` contains a single class; ``n`` is
    the number of evaluated interactions.
    """
    yt = np.asarray(y_true).astype(int)
    yp = np.clip(np.asarray(y_pred_proba, dtype=float), PROBA_EPS, 1 - PROBA_EPS)
    yhat = (yp > DECISION_THRESHOLD).astype(int)
    has_both = has_both_classes(yt)
    return {
        "auc": float(roc_auc_score(yt, yp)) if has_both else float("nan"),
        "accuracy": float(accuracy_score(yt, yhat)),
        "f1": float(f1_score(yt, yhat, zero_division=0)),
        "precision": float(precision_score(yt, yhat, zero_division=0)),
        "recall": float(recall_score(yt, yhat, zero_division=0)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "log_loss": float(log_loss(yt, yp)) if has_both else float("nan"),
        "n": len(yt),
    }
