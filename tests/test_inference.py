import logging

import numpy as np
import pandas as pd
import pytest
import torch

from knowledge_tracing.inference import PREDICTION_COLUMNS, DKTPredictor
from knowledge_tracing.models.dkt import DKTHyperparameters, DKTNetwork

SKILL_IDS = [10, 20, 30]


@pytest.fixture
def predictor() -> DKTPredictor:
    torch.manual_seed(0)
    hp = DKTHyperparameters(embed_dim=8, hidden_dim=8, dropout=0.0, lr=0.01, batch_size=4)
    network = DKTNetwork(len(SKILL_IDS), hp.embed_dim, hp.hidden_dim, hp.dropout)
    return DKTPredictor(network, hp, SKILL_IDS, {"model": "DKT"})


@pytest.fixture
def history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "order_idx": [0, 1, 2, 0, 1],
            "skill_id": [10, 20, 10, 30, 30],
            "correct": [0, 1, 1, 1, 0],
        }
    )


def test_predicts_every_skill_for_every_student(predictor, history):
    out = predictor.predict(history)
    assert list(out.columns) == PREDICTION_COLUMNS
    assert len(out) == 2 * len(SKILL_IDS)
    assert set(out["skill_id"]) == set(SKILL_IDS)
    assert out["p_correct"].between(0, 1).all()


def test_students_are_independent_of_batching(predictor, history):
    together = predictor.predict(history)
    alone = predictor.predict(history[history["user_id"] == 2])
    np.testing.assert_allclose(
        together[together["user_id"] == 2]["p_correct"].to_numpy(),
        alone["p_correct"].to_numpy(),
        rtol=1e-6,
    )


def test_row_order_does_not_matter(predictor, history):
    shuffled = history.sample(frac=1, random_state=0)
    pd.testing.assert_frame_equal(predictor.predict(history), predictor.predict(shuffled))


def test_save_and_load_roundtrip(predictor, history, tmp_path):
    predictor.save(tmp_path / "model")
    loaded = DKTPredictor.load(tmp_path / "model")
    assert loaded.skill_ids == SKILL_IDS
    assert loaded.metadata == {"model": "DKT"}
    pd.testing.assert_frame_equal(loaded.predict(history), predictor.predict(history))


def test_unknown_skills_are_ignored(predictor, history, caplog):
    unknown = pd.DataFrame({"user_id": [1], "order_idx": [3], "skill_id": [999], "correct": [1]})
    with caplog.at_level(logging.WARNING):
        out = predictor.predict(pd.concat([history, unknown], ignore_index=True))
    pd.testing.assert_frame_equal(out, predictor.predict(history))
    assert "unknown to the model" in caplog.text


def test_invalid_history_is_rejected(predictor, history):
    with pytest.raises(ValueError, match="missing columns"):
        predictor.predict(history.drop(columns=["correct"]))
    with pytest.raises(ValueError, match="0 and 1"):
        predictor.predict(history.assign(correct=2))
