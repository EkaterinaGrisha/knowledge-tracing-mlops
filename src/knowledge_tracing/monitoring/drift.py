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


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    quantiles = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 2:
        return 0.0
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, bins=quantiles)[0] / max(len(ref), 1)
    cur_pct = np.histogram(cur, bins=quantiles)[0] / max(len(cur), 1)
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, feature_cols: list[str], cfg: dict
) -> dict:
    warn = cfg["monitoring"]["psi_warn"]
    alert = cfg["monitoring"]["psi_alert"]
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
