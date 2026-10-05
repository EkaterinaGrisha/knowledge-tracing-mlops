import pandas as pd
import pytest
import requests

from knowledge_tracing.etl.extract import (
    LONG_COLUMNS,
    _download,
    extract,
    parse_triplet_file,
    write_sample,
)


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


TRAIN_TRIPLETS = "3\n1,2,1\n1,0,1\n"  # one student, three attempts
TEST_TRIPLETS = "2\n5,5\n0,1\n"  # one student, two attempts


class _Response:
    def __init__(self, text: str) -> None:
        self.content = text.encode("utf-8")

    def raise_for_status(self) -> None:
        pass


def test_full_source_downloads_once_and_merges(tmp_path, data_cfg, monkeypatch):
    cfg = data_cfg.model_copy(update={"raw_dir": tmp_path / "raw"})
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        return _Response(TRAIN_TRIPLETS if url == cfg.source_url_train else TEST_TRIPLETS)

    monkeypatch.setattr("knowledge_tracing.etl.extract.requests.get", fake_get)
    df = extract(cfg, "full")
    assert calls == [cfg.source_url_train, cfg.source_url_test]
    # test-file students get ids after the train-file students
    assert df["user_id"].tolist() == [0, 0, 0, 1, 1]
    extract(cfg, "full")  # the cached files are reused
    assert len(calls) == 2


def test_download_retries_then_gives_up(tmp_path, monkeypatch):
    attempts = []

    def failing_get(url, timeout):
        attempts.append(url)
        raise requests.ConnectionError("offline")

    monkeypatch.setattr("knowledge_tracing.etl.extract.requests.get", failing_get)
    monkeypatch.setattr("knowledge_tracing.etl.extract.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        _download("https://example.invalid/data.csv", tmp_path / "data.csv", retries=3)
    assert len(attempts) == 3


def test_write_sample_keeps_the_first_students(tmp_path, raw_long):
    out = write_sample(raw_long, tmp_path / "sample.csv", n_students=3)
    sample = pd.read_csv(out)
    assert sorted(sample["user_id"].unique()) == [0, 1, 2]
    assert len(sample) == (raw_long["user_id"] < 3).sum()


def test_missing_sample_points_to_the_regeneration_command(tmp_path, data_cfg):
    cfg = data_cfg.model_copy(update={"sample_path": tmp_path / "missing.csv"})
    with pytest.raises(FileNotFoundError, match="kt etl --data-source full --write-sample"):
        extract(cfg, "sample")


def test_unknown_source_is_rejected(data_cfg):
    with pytest.raises(ValueError, match="Unknown data_source"):
        extract(data_cfg, "remote")  # type: ignore[arg-type]
