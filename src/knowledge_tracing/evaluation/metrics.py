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


def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true).astype(int)
    yp = np.clip(np.asarray(y_pred_proba, dtype=float), 1e-7, 1 - 1e-7)
    yhat = (yp > 0.5).astype(int)
    has_both = len(set(yt.tolist())) > 1
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
