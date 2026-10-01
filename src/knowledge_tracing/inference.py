"""Serving a trained DKT model: P(correct) for every skill given a student's history.

A packaged model is a directory with the network weights (``model.pt``) and a
JSON description (``model.json``: architecture, skill-id mapping, metadata).
It is used directly (``DKTPredictor.load``) or through MLflow as a pyfunc
model from the Model Registry (``models:/kt-dkt@champion``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol

import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow.models.model import ModelInfo
from mlflow.pyfunc.model import PythonModel, PythonModelContext

from .config import resolve_path
from .models.base import NotFittedError
from .models.dkt import DKTHyperparameters, DKTModel, DKTNetwork

LOG = logging.getLogger(__name__)

WEIGHTS_FILE = "model.pt"
SPEC_FILE = "model.json"
HISTORY_COLUMNS = ["user_id", "order_idx", "skill_id", "correct"]
PREDICTION_COLUMNS = ["user_id", "skill_id", "p_correct"]
_BATCH_SIZE = 64


class SupportsPredict(Protocol):
    """Anything that maps a history frame to a predictions frame."""

    def predict(self, history: pd.DataFrame) -> pd.DataFrame: ...


@dataclass
class DKTPredictor:
    """A trained DKT network with everything needed to use it outside the pipeline."""

    network: DKTNetwork
    hyperparameters: DKTHyperparameters
    skill_ids: list[int]  # original skill id of every dense skill index
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_model(
        cls, model: DKTModel, skill_remap: dict[int, int], metadata: dict[str, Any]
    ) -> DKTPredictor:
        """Package a fitted DKT model; ``skill_remap`` maps original skill ids to indices."""
        if model.network_ is None or model.hyperparameters_ is None:
            raise NotFittedError(f"{model.name} is not fitted")
        skill_ids = [skill for skill, _ in sorted(skill_remap.items(), key=lambda kv: kv[1])]
        return cls(model.network_.cpu(), model.hyperparameters_, skill_ids, metadata)

    def save(self, directory: Path) -> Path:
        """Write ``model.pt`` and ``model.json`` into ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), directory / WEIGHTS_FILE)
        spec = {
            "hyperparameters": asdict(self.hyperparameters),
            "skill_ids": self.skill_ids,
            "metadata": self.metadata,
        }
        (directory / SPEC_FILE).write_text(
            json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: Path) -> DKTPredictor:
        """Load a model written by :meth:`save`."""
        spec = json.loads((directory / SPEC_FILE).read_text(encoding="utf-8"))
        hp = DKTHyperparameters(**spec["hyperparameters"])
        skill_ids = [int(skill) for skill in spec["skill_ids"]]
        network = DKTNetwork(len(skill_ids), hp.embed_dim, hp.hidden_dim, hp.dropout)
        state = torch.load(directory / WEIGHTS_FILE, map_location="cpu", weights_only=True)
        network.load_state_dict(state)
        network.eval()
        return cls(network, hp, skill_ids, spec.get("metadata", {}))

    @torch.no_grad()
    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        """P(correct) on the next attempt of every known skill, for every student.

        Args:
            history: Answered exercises in the dataset's long format with columns
                ``user_id, order_idx, skill_id, correct`` (original skill ids).

        Returns:
            One row per student and skill: ``user_id, skill_id, p_correct``.

        Raises:
            ValueError: Required columns are missing or ``correct`` is not 0/1.
        """
        missing = sorted(set(HISTORY_COLUMNS) - set(history.columns))
        if missing:
            raise ValueError(f"history is missing columns: {missing}")
        if not history["correct"].isin([0, 1]).all():
            raise ValueError("'correct' must contain only 0 and 1")

        index_of = {skill: i for i, skill in enumerate(self.skill_ids)}
        known = history["skill_id"].isin(index_of)
        if not known.all():
            LOG.warning("Ignoring %d interactions with skills unknown to the model", (~known).sum())
        rows = history[known].sort_values(["user_id", "order_idx"])

        users: list[Any] = []
        inputs: list[np.ndarray] = []
        for user_id, group in rows.groupby("user_id"):
            skills = group["skill_id"].map(index_of).to_numpy(dtype=np.int64)
            inputs.append(2 * skills + group["correct"].to_numpy(dtype=np.int64))
            users.append(user_id)

        self.network.eval()
        frames = []
        for start in range(0, len(inputs), _BATCH_SIZE):
            batch = inputs[start : start + _BATCH_SIZE]
            lengths = torch.tensor([len(seq) for seq in batch])
            x = torch.zeros((len(batch), int(lengths.max())), dtype=torch.int64)
            for b, seq in enumerate(batch):
                x[b, : len(seq)] = torch.from_numpy(seq)
            logits = self.network(x, lengths)
            # the state after the last answered exercise predicts the next attempt
            probs = torch.sigmoid(logits[torch.arange(len(batch)), lengths - 1]).numpy()
            for user_id, p in zip(users[start : start + _BATCH_SIZE], probs, strict=True):
                frames.append(
                    pd.DataFrame({"user_id": user_id, "skill_id": self.skill_ids, "p_correct": p})
                )
        if not frames:
            return pd.DataFrame(columns=PREDICTION_COLUMNS)
        return pd.concat(frames, ignore_index=True)


class DKTPyfuncModel(PythonModel):
    """MLflow pyfunc flavour of :class:`DKTPredictor` for the Model Registry."""

    def load_context(self, context: PythonModelContext) -> None:
        self._predictor = DKTPredictor.load(Path(context.artifacts["model_dir"]))

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        return self._predictor.predict(model_input)


def _pip_requirements() -> list[str]:
    return [f"{name}=={version(name)}" for name in ("torch", "numpy", "pandas", "mlflow")]


def log_predictor(
    model_dir: Path, input_example: pd.DataFrame, registered_model_name: str | None
) -> ModelInfo:
    """Log a packaged model to the active MLflow run, optionally registering it.

    The knowledge_tracing package is bundled with the model (``code_paths``),
    so the registered model loads in any environment with the pinned libraries.
    """
    return mlflow.pyfunc.log_model(
        name="model",
        python_model=DKTPyfuncModel(),
        artifacts={"model_dir": str(model_dir)},
        code_paths=[str(Path(__file__).resolve().parent)],
        pip_requirements=_pip_requirements(),
        input_example=input_example,
        registered_model_name=registered_model_name,
    )


def load_model(reference: str) -> SupportsPredict:
    """Load a packaged model from a directory or an MLflow model URI.

    Args:
        reference: A directory written by :meth:`DKTPredictor.save` (relative to
            the project root) or an MLflow URI such as ``models:/kt-dkt@champion``;
            for a URI the tracking store must already be configured.
    """
    directory = resolve_path(reference)
    if directory.is_dir():
        return DKTPredictor.load(directory)
    model: SupportsPredict = mlflow.pyfunc.load_model(reference)
    return model
