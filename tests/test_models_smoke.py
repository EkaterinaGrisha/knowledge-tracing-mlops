"""Slow smoke tests: each model trains for a couple of steps without error."""

import numpy as np
import pandas as pd
import pytest

from knowledge_tracing.models import dkt as dkt_mod


@pytest.fixture
def tiny_sequences():
    rng = np.random.default_rng(0)
    seqs = []
    for _ in range(40):
        n = rng.integers(5, 12)
        seqs.append({"skills": rng.integers(0, 4, size=n), "correct": rng.integers(0, 2, size=n)})
    return seqs


@pytest.mark.slow
def test_dkt_trains_and_predicts(tiny_sequences):
    device = dkt_mod.pick_device()
    traces = dkt_mod.traces_from_sequences(tiny_sequences)
    model, losses = dkt_mod.train_dkt(
        traces, n_concepts=4, device=device, epochs=2, hidden_dim=16, embed_dim=16
    )
    assert len(losses) == 2
    _auc, acc, _rmse, yt, yp = dkt_mod.evaluate_dkt(model, traces, device)
    assert 0.0 <= acc <= 1.0
    assert len(yt) == len(yp)


@pytest.mark.slow
def test_flaml_trains():
    from knowledge_tracing.models.automl_flaml import evaluate_automl, train_automl

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=300), "f2": rng.normal(size=300)})
    y = pd.Series((X["f1"] + rng.normal(0, 0.3, size=300) > 0).astype(int))
    automl = train_automl(
        X.iloc[:200], y.iloc[:200], X.iloc[200:250], y.iloc[200:250], time_budget_s=5
    )
    auc, _acc, _rmse, _yt, _yp = evaluate_automl(automl, X.iloc[250:], y.iloc[250:])
    assert 0.0 <= auc <= 1.0
