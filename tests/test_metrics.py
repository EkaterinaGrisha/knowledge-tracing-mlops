import numpy as np

from knowledge_tracing.evaluation.metrics import compute_metrics


def test_perfect_predictions():
    yt = np.array([0, 0, 1, 1])
    yp = np.array([0.01, 0.02, 0.98, 0.99])
    m = compute_metrics(yt, yp)
    assert m["auc"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["rmse"] < 0.05


def test_metrics_ranges():
    rng = np.random.default_rng(0)
    yt = rng.integers(0, 2, size=200)
    yp = rng.random(200)
    m = compute_metrics(yt, yp)
    for key in ("auc", "accuracy", "f1", "precision", "recall"):
        assert 0.0 <= m[key] <= 1.0
    assert m["n"] == 200


def test_single_class_auc_is_nan():
    yt = np.zeros(10, dtype=int)
    yp = np.full(10, 0.3)
    m = compute_metrics(yt, yp)
    assert np.isnan(m["auc"])
