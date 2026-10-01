import numpy as np

from knowledge_tracing.etl.datasets import StudentSequence
from knowledge_tracing.models.bkt import (
    DEFAULT_PARAMS,
    fit_bkt_per_skill,
    fit_em,
    predict_bkt,
    predict_next_correct,
)


def test_predict_next_correct_in_range():
    seq = [0, 1, 1, 0, 1, 1, 1]
    preds = predict_next_correct(seq, DEFAULT_PARAMS)
    assert len(preds) == len(seq)
    assert all(0.0 <= p <= 1.0 for p in preds)


def test_em_learns_from_improving_sequences():
    # sequences that start wrong then become consistently right -> learning signal
    seqs = [[0, 0, 1, 1, 1, 1] for _ in range(40)]
    p = fit_em(seqs, n_iter=20)
    assert 0.0 < p.p_init < 1.0
    assert 0.0 < p.p_learn <= 0.5


def test_per_skill_fit_and_predict():
    rng = np.random.default_rng(0)
    seqs = [
        StudentSequence(
            user_id=uid,
            skills=rng.integers(0, 3, size=12),
            correct=rng.integers(0, 2, size=12),
        )
        for uid in range(30)
    ]
    params = fit_bkt_per_skill(seqs, n_skills=3, em_iters=10)
    assert set(params.keys()) == {0, 1, 2}
    y_true, y_pred = predict_bkt(params, seqs, n_skills=3)
    assert len(y_true) == len(y_pred) == 30 * 12
    assert ((y_pred >= 0.0) & (y_pred <= 1.0)).all()
