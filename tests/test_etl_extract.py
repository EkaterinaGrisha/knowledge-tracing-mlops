from knowledge_tracing.etl.extract import LONG_COLUMNS, parse_triplet_file


def test_parse_triplet_file(tmp_path):
    content = "\n".join(
        [
            "3",
            "1,2,1",
            "1,0,1",
            "2",
            "5,5",
            "0,1",
        ]
    )
    f = tmp_path / "mini.csv"
    f.write_text(content, encoding="utf-8")
    df = parse_triplet_file(f)
    assert list(df.columns) == LONG_COLUMNS
    assert df["user_id"].nunique() == 2
    assert len(df) == 5
    # first student's first interaction: skill 1, correct 1
    first = df.iloc[0]
    assert int(first["skill_id"]) == 1 and int(first["correct"]) == 1


def test_user_offset(tmp_path):
    content = "2\n1,1\n1,1\n"
    f = tmp_path / "mini.csv"
    f.write_text(content, encoding="utf-8")
    df = parse_triplet_file(f, user_offset=100)
    assert df["user_id"].min() == 100
