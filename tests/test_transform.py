import pandas as pd

from knowledge_tracing.etl.transform import FEATURE_COLS, build_sequences, transform


def test_transform_schema_and_splits(raw_long, data_cfg):
    pd_data = transform(raw_long, data_cfg, seed=0)
    # all feature columns present
    for c in FEATURE_COLS:
        assert c in pd_data.features.columns
    # splits are disjoint by student
    by_split = pd_data.features.groupby("split")["user_id"].agg(set)
    users = list(by_split)
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            assert users[i].isdisjoint(users[j])
    # skill index is contiguous from 0
    assert pd_data.long["skill_idx"].min() == 0
    assert pd_data.long["skill_idx"].max() == pd_data.n_skills - 1


def test_features_are_causal(data_cfg):
    # one student, alternating correctness — the first row must not leak its own label
    df = pd.DataFrame(
        {
            "user_id": [0, 0, 0, 0, 1, 1, 1, 1],
            "order_idx": [0, 1, 2, 3, 0, 1, 2, 3],
            "skill_id": [5, 5, 5, 5, 5, 5, 5, 5],
            "correct": [1, 0, 1, 0, 1, 1, 1, 1],
        }
    )
    pd_data = transform(df, data_cfg, seed=0)
    first_rows = pd_data.features[pd_data.features["order_idx"] == 0]
    # at the very first interaction there is no prior history
    assert (first_rows["user_prior_attempts"] == 0).all()
    assert (first_rows["skill_prior_attempts"] == 0).all()


def test_build_sequences(raw_long, data_cfg):
    pd_data = transform(raw_long, data_cfg, seed=0)
    seqs = build_sequences(pd_data.long, "train")
    assert all(len(s["skills"]) == len(s["correct"]) for s in seqs)
    assert all(len(s["skills"]) >= 1 for s in seqs)
