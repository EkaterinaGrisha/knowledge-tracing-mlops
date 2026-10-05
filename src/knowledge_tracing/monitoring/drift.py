"""Data-drift monitoring via Population Stability Index (PSI) + KS test.

Compares a reference distribution (train) against a current one (test, or a
future production batch). PSI interpretation: <0.1 stable, 0.1-0.25 moderate
drift, >0.25 significant drift. Used to guard the model against silent input
distribution shift before predictions are trusted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Empty bins get this share so the logarithm in PSI stays finite.
_PSI_EPS = 1e-6


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    quantiles = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if quantiles.size <= 1:  # constant reference values: no bins to compare
        return 0.0
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, bins=quantiles)[0] / max(len(ref), 1)
    cur_pct = np.histogram(cur, bins=quantiles)[0] / max(len(cur), 1)
    ref_pct = np.clip(ref_pct, _PSI_EPS, None)
    cur_pct = np.clip(cur_pct, _PSI_EPS, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list[str],
    *,
    psi_warn: float,
    psi_alert: float,
) -> dict:
    """Compare the distribution of every feature in ``current`` against ``reference``.

    Args:
        reference: Baseline data (the training split).
        current: Data to check (the test split or a new production batch).
        feature_cols: Numeric columns to compare.
        psi_warn: PSI from which a feature counts as moderately drifted.
        psi_alert: PSI from which a feature counts as significantly drifted.

    Returns:
        Per-feature PSI, KS statistic and status, the number of significantly
        drifted features and an overall status (``"OK"`` or ``"ALERT"``).
    """
    warn = psi_warn
    alert = psi_alert
    features: dict[str, dict] = {}
    n_alert = 0
    for col in feature_cols:
        ref = pd.to_numeric(reference[col], errors="coerce").dropna().to_numpy()
        cur = pd.to_numeric(current[col], errors="coerce").dropna().to_numpy()
        if len(ref) == 0 or len(cur) == 0:
            continue
        psi = _psi(ref, cur)
        ks_stat, ks_p = stats.ks_2samp(ref, cur)
        status = (
            "stable" if psi < warn else "moderate_drift" if psi < alert else "significant_drift"
        )
        if status == "significant_drift":
            n_alert += 1
        features[col] = {
            "psi": round(psi, 4),
            "ks_stat": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_p), 4),
            "status": status,
        }
    return {
        "n_features": len(features),
        "n_significant_drift": n_alert,
        "overall_status": "ALERT" if n_alert else "OK",
        "features": features,
    }
