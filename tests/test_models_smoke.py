"""Slow smoke tests: each model trains for a couple of steps without error."""

import numpy as np
import pandas as pd
import pytest

from knowledge_tracing.etl.datasets import StudentSequence
from knowledge_tracing.evaluation.metrics import roc_auc
from knowledge_tracing.models import dkt as dkt_mod
from knowledge_tracing.models.dkt import DKTHyperparameters


@pytest.fixture
def tiny_sequences() -> list[StudentSequence]:
    rng = np.random.default_rng(0)
    seqs = []
    for uid in range(40):
        n = rng.integers(5, 12)
        seqs.append(
            StudentSequence(
                user_id=uid, skills=rng.integers(0, 4, size=n), correct=rng.integers(0, 2, size=n)
            )
        )
    return seqs


@pytest.mark.slow
def test_dkt_trains_and_predicts(tiny_sequences):
    device = dkt_mod.pick_device()
    hp = DKTHyperparameters(embed_dim=16, hidden_dim=16, dropout=0.2, lr=5e-3, batch_size=32)
    model, losses = dkt_mod.train_dkt(tiny_sequences, 4, hp, epochs=2, device=device, seed=0)
    assert len(losses) == 2
    y_true, y_pred = dkt_mod.predict_dkt(model, tiny_sequences, device)
    # one prediction per transition (every interaction except the first of each student)
    assert len(y_true) == len(y_pred) == sum(len(s) - 1 for s in tiny_sequences)
    assert ((y_pred >= 0.0) & (y_pred <= 1.0)).all()


@pytest.mark.slow
def test_flaml_trains():
    from knowledge_tracing.models.automl_flaml import predict_automl, train_automl

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=300), "f2": rng.normal(size=300)})
    y = pd.Series((X["f1"] + rng.normal(0, 0.3, size=300) > 0).astype(int))
    automl = train_automl(
        X.iloc[:200],
        y.iloc[:200],
        X.iloc[200:250],
        y.iloc[200:250],
        time_budget_s=5,
        metric="roc_auc",
        estimator_list=["lgbm"],
        seed=0,
    )
    y_true, y_pred = predict_automl(automl, X.iloc[250:], y.iloc[250:])
    assert 0.0 <= roc_auc(y_true, y_pred) <= 1.0
