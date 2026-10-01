import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from knowledge_tracing.config import PROJECT_ROOT_ENV, Config, DataConfig, load_config
from knowledge_tracing.runtime import configure_runtime

# Same platform settings as the `kt` command, before torch/lightgbm are imported.
configure_runtime()

REPO_ROOT = Path(__file__).resolve().parents[1]
# Relative paths in config/config.yaml refer to the repository root,
# whatever directory pytest is started from.
os.environ.setdefault(PROJECT_ROOT_ENV, str(REPO_ROOT))


@pytest.fixture
def project_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated project root: config and committed sample copied, outputs stay in tmp."""
    shutil.copytree(REPO_ROOT / "config", tmp_path / "config")
    shutil.copytree(REPO_ROOT / "data" / "sample", tmp_path / "data" / "sample")
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(tmp_path))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    return tmp_path


@pytest.fixture(autouse=True)
def _reset_package_logging():
    """Undo configure_logging() of CLI tests so caplog sees package records."""
    yield
    logger = logging.getLogger("knowledge_tracing")
    logger.handlers.clear()
    logger.propagate = True


@pytest.fixture(scope="session")
def config() -> Config:
    return load_config(REPO_ROOT / "config" / "config.yaml")


@pytest.fixture
def data_cfg(config: Config) -> DataConfig:
    """Project data config with thresholds suited to tiny synthetic frames."""
    return config.data.model_copy(
        update={"min_seq_len": 2, "max_seq_len": 100, "val_frac": 0.2, "test_frac": 0.2}
    )


@pytest.fixture
def raw_long() -> pd.DataFrame:
    """Small deterministic long frame: 10 students, 3 skills."""
    rng = np.random.default_rng(0)
    rows = []
    for uid in range(10):
        n = rng.integers(8, 16)
        for j in range(n):
            skill = int(rng.integers(0, 3))
            correct = int(rng.random() < 0.5 + 0.1 * skill)
            rows.append((uid, j, skill, correct))
    return pd.DataFrame(rows, columns=["user_id", "order_idx", "skill_id", "correct"])
