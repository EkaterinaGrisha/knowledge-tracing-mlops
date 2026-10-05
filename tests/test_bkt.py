import numpy as np

from src.models.bkt import (
    DEFAULT_PARAMS,
    evaluate_bkt,
    fit_bkt_per_skill,
    fit_em,
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


def test_per_skill_fit_and_eval():
    rng = np.random.default_rng(0)
    seqs = []
    for _ in range(30):
        skills = rng.integers(0, 3, size=12)
        correct = rng.integers(0, 2, size=12)
        seqs.append({"skills": skills, "correct": correct})
    params = fit_bkt_per_skill(seqs, n_skills=3, em_iters=10)
    assert set(params.keys()) == {0, 1, 2}
    _auc, acc, _rmse, yt, yp = evaluate_bkt(params, seqs, n_skills=3)
    assert 0.0 <= acc <= 1.0
    assert len(yt) == len(yp)
